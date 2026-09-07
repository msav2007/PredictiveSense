"""Analysis-path parameter sweep (Phase 1.5, in-process / loopback).

    python scripts\\benchmark_analysis_path.py --label loopback --seconds 6

Sweeps ``analysis_fps`` x ``analysis_width x analysis_height`` x
``analysis_jpeg_quality`` and, for each combination, drives the **real**
``BrowserSource`` -> ``LatestFrameMailbox`` -> ``AnalysisLoop`` pipeline in one
process with synthetic frames at the target rate. It reports, per combination:

    - frame age p50 / p95  (StateSnapshot.frame_age_ms - backend processing lag)
    - drop rate             (single-slot newest-wins drops / frames seen)
    - backend decode ms     (cv2.imdecode on the ingest thread)
    - encode ms (proxy)     (cv2.imencode here; the browser's OffscreenCanvas
                             convertToBlob time is NOT measurable from Python)
    - ingest bytes / s
    - process CPU %

Choose defaults that MINIMISE frame age and drops at acceptable quality - not
the highest fps / resolution / quality numbers.

This is a loopback measurement: there is no browser, no camera, and the RTT is
~0. It measures the backend half of the analysis path only. The worker
newest-wins + ``bufferedAmount`` skip rules are browser-side and are covered by
``tests/unit/test_worker_backpressure_contract.py``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import psutil

from predictivesense.camera.browser import BrowserSource
from predictivesense.camera.framing import ClockOffset
from predictivesense.camera.mailbox import LatestFrameMailbox
from predictivesense.config.settings import load_config
from predictivesense.core.types import IngestHeader
from predictivesense.logging_setup import configure_logging, get_logger
from predictivesense.pipeline.loop import AnalysisLoop

_LOG = get_logger("predictivesense.scripts.benchmark_analysis_path")

_DEFAULT_FPS = "5,10,15"
_DEFAULT_SIZES = "480x360,640x480,800x600"
_DEFAULT_QUALITIES = "0.5,0.7,0.85"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep the analysis-path parameters.")
    p.add_argument("--label", required=True)
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--profile", default="dev", help="profile for consumer sample rate")
    p.add_argument("--fps-list", default=_DEFAULT_FPS)
    p.add_argument("--sizes", default=_DEFAULT_SIZES)
    p.add_argument("--qualities", default=_DEFAULT_QUALITIES)
    p.add_argument("--results-dir", default="results")
    return p.parse_args(argv)


def _pct(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    frac = rank - low
    if low + 1 >= len(ordered):
        return ordered[-1]
    return ordered[low] + frac * (ordered[low + 1] - ordered[low])


def _feed(
    source: BrowserSource,
    stop_evt: threading.Event,
    *,
    width: int,
    height: int,
    fps: float,
    quality: int,
    encode_ms: list[float],
) -> None:
    # A mildly structured, camera-like frame: horizontal gradient + a few blocks
    # + light noise. Compresses to a realistic JPEG size (a flat or pure-noise
    # image would under- or over-state ingest bytes/s and decode cost).
    rng = np.random.default_rng(1234)
    xs = np.linspace(0, 220, width, dtype=np.uint8)
    img = np.repeat(xs[None, :, None], height, axis=0)
    img = np.repeat(img, 3, axis=2).copy()
    img[height // 4 : height // 2, width // 5 : width // 2] = (60, 140, 200)
    img[height // 2 : 3 * height // 4, width // 2 : 4 * width // 5] = (200, 120, 40)
    img = np.clip(
        img.astype(np.int16) + rng.integers(-12, 13, img.shape, dtype=np.int16),
        0, 255,
    ).astype(np.uint8)
    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    period = 1.0 / fps
    seq = 0
    next_due = time.monotonic()
    while not stop_evt.is_set():
        now = time.monotonic()
        if now < next_due:
            stop_evt.wait(next_due - now)
            continue
        next_due += period
        e0 = time.perf_counter()
        ok, buf = cv2.imencode(".jpg", img, params)
        e1 = time.perf_counter()
        if not ok:
            continue
        encode_ms.append((e1 - e0) * 1000.0)
        header = IngestHeader(
            client_ts_ms=time.time() * 1000.0, seq=seq, w=width, h=height
        )
        source.submit(header, buf.tobytes())
        seq += 1


def _measure_combo(
    cfg, *, width: int, height: int, fps: float, quality_frac: float, seconds: float
) -> dict[str, object]:
    source = BrowserSource(analysis_width=width, analysis_height=height)
    mailbox = LatestFrameMailbox()
    loop = AnalysisLoop(cfg, source, mailbox)

    # Map the synthetic client epoch clock onto the server monotonic timeline so
    # StateSnapshot.frame_age_ms is a real processing lag (RTT is 0 in loopback).
    offset = time.monotonic() - time.time()
    source.begin_client_session(ClockOffset(offset_s=offset, rtt_ms=0.0))

    proc = psutil.Process()
    proc.cpu_percent(None)
    encode_ms: list[float] = []
    stop_evt = threading.Event()

    loop.start()
    feeder = threading.Thread(
        target=_feed,
        name="ap-feeder",
        args=(source, stop_evt),
        kwargs=dict(
            width=width, height=height, fps=fps,
            quality=int(round(quality_frac * 100)), encode_ms=encode_ms,
        ),
        daemon=True,
    )
    feeder.start()

    ages: list[float] = []
    drop_rates: list[float] = []
    decode_ms: list[float] = []
    bytes_per_s: list[float] = []
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < seconds:
            time.sleep(0.1)
            snap = loop.latest
            if snap is None:
                continue
            if snap.frame_age_ms is not None and not snap.stale:
                ages.append(snap.frame_age_ms)
            m = snap.metrics
            drop_rates.append(m.get("drop_rate", 0.0))
            if m.get("decode_ms"):
                decode_ms.append(m["decode_ms"])
            if m.get("ingest_bytes_per_s"):
                bytes_per_s.append(m["ingest_bytes_per_s"])
    finally:
        stop_evt.set()
        feeder.join(timeout=2.0)
        loop.stop()

    cpu_pct = proc.cpu_percent(None)
    info = source.info()
    total_submitted = info["frames_submitted"]
    stats = mailbox.stats()
    seen = stats.consumed + stats.dropped
    row = {
        "requested": {
            "analysis_fps": fps,
            "analysis_width": width,
            "analysis_height": height,
            "analysis_jpeg_quality": quality_frac,
        },
        "frames_submitted": total_submitted,
        "achieved_submit_fps": round(total_submitted / seconds, 2),
        "frame_age_ms_p50": round(_pct(ages, 50), 1),
        "frame_age_ms_p95": round(_pct(ages, 95), 1),
        "drop_rate": round(statistics.fmean(drop_rates), 3) if drop_rates else 0.0,
        "mailbox_drop_rate_final": round(stats.dropped / seen, 3) if seen else 0.0,
        "buffer_dropped": info["buffer_dropped"],
        "decode_ms_p50": round(_pct(decode_ms, 50), 2),
        "encode_ms_proxy_p50": round(_pct(encode_ms, 50), 2),
        "ingest_bytes_per_s": round(statistics.fmean(bytes_per_s)) if bytes_per_s else 0,
        "process_cpu_percent": round(cpu_pct, 1),
        "loop_error": repr(loop.error) if loop.error else None,
    }
    _LOG.info(
        "fps %-2s %dx%d q%.2f -> age p50/p95 %s/%s ms | drop %s | "
        "decode %s ms | enc(proxy) %s ms | %d B/s | cpu %.0f%%",
        fps, width, height, quality_frac,
        row["frame_age_ms_p50"], row["frame_age_ms_p95"], row["drop_rate"],
        row["decode_ms_p50"], row["encode_ms_proxy_p50"],
        row["ingest_bytes_per_s"], row["process_cpu_percent"],
    )
    return row


def _markdown(label: str, rows: list[dict[str, object]]) -> str:
    head = (
        f"# Analysis-path sweep - {label} (loopback, backend half only)\n\n"
        "`encode ms (proxy)` is cv2.imencode here, **not** the browser worker's "
        "convertToBlob time. RTT ~ 0 (in-process).\n\n"
        "| req fps | size | quality | age p50 ms | age p95 ms | drop rate | "
        "decode ms p50 | enc proxy ms p50 | ingest B/s | CPU % |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for r in rows:
        rq = r["requested"]  # type: ignore[index]
        lines.append(
            f"| {rq['analysis_fps']} | {rq['analysis_width']}x{rq['analysis_height']} "
            f"| {rq['analysis_jpeg_quality']} | {r['frame_age_ms_p50']} | "
            f"{r['frame_age_ms_p95']} | {r['drop_rate']} | {r['decode_ms_p50']} | "
            f"{r['encode_ms_proxy_p50']} | {r['ingest_bytes_per_s']} | "
            f"{r['process_cpu_percent']} |"
        )
    return head + "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    configure_logging("INFO")
    cfg = load_config(args.profile)

    fps_list = [float(x) for x in args.fps_list.split(",")]
    sizes: list[tuple[int, int]] = []
    for token in args.sizes.split(","):
        w, _, h = token.strip().lower().partition("x")
        sizes.append((int(w), int(h)))
    qualities = [float(x) for x in args.qualities.split(",")]

    combos = [
        (w, h, fps, q)
        for fps in fps_list
        for (w, h) in sizes
        for q in qualities
    ]
    _LOG.info("analysis sweep: %d combos x %.0fs", len(combos), args.seconds)

    rows = [
        _measure_combo(
            cfg, width=w, height=h, fps=fps, quality_frac=q, seconds=args.seconds
        )
        for (w, h, fps, q) in combos
    ]

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"analysis_sweep_{args.label}.json"
    md_path = results_dir / f"analysis_sweep_{args.label}.md"
    json_path.write_text(
        json.dumps({"label": args.label, "seconds": args.seconds, "combos": rows}, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(_markdown(args.label, rows), encoding="utf-8")
    _LOG.info("wrote %s", json_path)
    _LOG.info("wrote %s", md_path)
    return 1 if any(r["loop_error"] for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
