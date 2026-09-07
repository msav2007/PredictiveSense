"""Backend-owned camera transport benchmark.

    python scripts\\benchmark_transport.py --index 0 --seconds 30 --label laptop-cam

Opens one device index backend-owned and reports achieved resolution, achieved
FPS, measured FPS, per-``read()`` blocking time, frame-interval p50/p95/max,
time to first frame, read-failure / reconnect counts, and process RSS/CPU delta.
Written to ``results/transport_<label>.json``. Run once per transport; the
comparison table is assembled by hand in the phase report.

Coarse timing (the --seconds deadline, RSS sampling) uses ``time.monotonic``;
sub-millisecond intervals use ``time.perf_counter`` (Phase 0 clock decision).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import psutil

from predictivesense.camera.device import DeviceSource
from predictivesense.logging_setup import configure_logging, get_logger

_LOG = get_logger("predictivesense.scripts.benchmark_transport")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark one camera transport.")
    parser.add_argument("--index", type=int, required=True, help="OpenCV device index")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--label", required=True, help="transport label for the output filename")
    parser.add_argument("--backend", choices=("auto", "msmf", "dshow"), default="auto")
    parser.add_argument("--request-width", type=int, default=1280)
    parser.add_argument("--request-height", type=int, default=720)
    parser.add_argument("--request-fps", type=float, default=30.0)
    parser.add_argument("--fourcc", default="auto", help="'auto' or a 4-char code e.g. MJPG")
    parser.add_argument("--buffer-size", type=int, default=1)
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args(argv)


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


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "p50": round(_pct(values, 50), 3),
        "p95": round(_pct(values, 95), 3),
        "max": round(max(values), 3) if values else 0.0,
        "min": round(min(values), 3) if values else 0.0,
        "mean": round(statistics.fmean(values), 3) if values else 0.0,
        "count": len(values),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    configure_logging("INFO")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"transport_{args.label}.json"

    source = DeviceSource(
        args.index,
        backend=args.backend,
        request_width=args.request_width,
        request_height=args.request_height,
        request_fps=args.request_fps,
        fourcc=args.fourcc,
        buffer_size=args.buffer_size,
        open_timeout_s=5.0,
    )

    record: dict[str, object] = {
        "label": args.label,
        "index": args.index,
        "requested": {
            "width": args.request_width,
            "height": args.request_height,
            "fps": args.request_fps,
            "backend": args.backend,
            "fourcc": args.fourcc,
            "buffer_size": args.buffer_size,
        },
        "seconds_requested": args.seconds,
        "opened": False,
    }

    proc = psutil.Process()
    rss_start_mb = proc.memory_info().rss / (1024 * 1024)
    proc.cpu_percent(None)  # prime the CPU meter

    try:
        source.start()
        record["opened"] = True
    except RuntimeError as exc:
        _LOG.error("could not open device index %d: %s", args.index, exc)
        record["error"] = str(exc)
        out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        _LOG.info("wrote %s (device did not open)", out_path)
        return 1

    intervals: list[float] = []
    read_call_ms: list[float] = []
    frames = 0
    read_failures = 0
    first_frame_s: float | None = None
    prev: float | None = None
    t_start = time.monotonic()
    try:
        while time.monotonic() - t_start < args.seconds:
            r0 = time.perf_counter()
            frame = source.read()
            r1 = time.perf_counter()
            if frame is None:
                read_failures += 1
                continue
            read_call_ms.append((r1 - r0) * 1000.0)
            if first_frame_s is None:
                first_frame_s = time.monotonic() - t_start
            if prev is not None:
                intervals.append((r1 - prev) * 1000.0)
            prev = r1
            frames += 1
    finally:
        info = source.info()
        source.stop()

    elapsed = time.monotonic() - t_start
    cpu_pct = proc.cpu_percent(None)
    rss_end_mb = proc.memory_info().rss / (1024 * 1024)
    achieved_fps = info.get("achieved_fps") or 0.0
    expected = achieved_fps * elapsed if achieved_fps else 0.0

    record.update(
        {
            "seconds_elapsed": round(elapsed, 3),
            "backend_used": info.get("backend"),
            "fourcc_used": info.get("fourcc"),
            "achieved_width": info.get("achieved_width"),
            "achieved_height": info.get("achieved_height"),
            "achieved_fps_reported": achieved_fps or None,
            "measured_fps": round(frames / elapsed, 3) if elapsed > 0 else 0.0,
            "frames": frames,
            "frames_expected_at_achieved_fps": round(expected, 1) if expected else None,
            "dropped_estimate": round(expected - frames, 1) if expected else None,
            "read_failures": read_failures,
            "reconnects": info.get("reconnects"),
            "reconnect_attempts": info.get("reconnect_attempts"),
            "time_to_first_frame_s": round(first_frame_s, 4) if first_frame_s is not None else None,
            "frame_interval_ms": _summary(intervals),
            "read_call_ms": _summary(read_call_ms),
            "process_cpu_percent": round(cpu_pct, 1),
            "process_rss_mb_start": round(rss_start_mb, 1),
            "process_rss_mb_end": round(rss_end_mb, 1),
            "process_rss_mb_delta": round(rss_end_mb - rss_start_mb, 2),
        }
    )
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    _LOG.info("=== transport benchmark: %s ===", args.label)
    _LOG.info("backend / fourcc    : %s / %s", record["backend_used"], record["fourcc_used"])
    _LOG.info("achieved resolution : %sx%s @ %s fps (reported)",
              record["achieved_width"], record["achieved_height"], record["achieved_fps_reported"])
    _LOG.info("measured fps        : %s  (%d frames / %.1fs)", record["measured_fps"], frames, elapsed)
    _LOG.info("dropped estimate    : %s", record["dropped_estimate"])
    _LOG.info("read() call ms      : p50 %s / p95 %s / min %s",
              record["read_call_ms"]["p50"], record["read_call_ms"]["p95"], record["read_call_ms"]["min"])
    _LOG.info("frame interval ms   : p50 %s / p95 %s / max %s",
              record["frame_interval_ms"]["p50"], record["frame_interval_ms"]["p95"], record["frame_interval_ms"]["max"])
    _LOG.info("time to first frame : %s s", record["time_to_first_frame_s"])
    _LOG.info("read failures / reconnects : %d / %s", read_failures, record["reconnects"])
    _LOG.info("process CPU / RSS d : %s%% / %s MB", record["process_cpu_percent"], record["process_rss_mb_delta"])
    _LOG.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
