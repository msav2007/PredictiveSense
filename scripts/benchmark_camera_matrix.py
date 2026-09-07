"""Backend-owned camera matrix benchmark (Phase 1.5).

    python scripts\\benchmark_camera_matrix.py --index 0 --label integrated --seconds 10

Sweeps, for one device index:

    resolution  x  requested FPS  x  capture backend (MSMF, DSHOW)  x  FOURCC (auto, MJPG)

and measures each combination over ``--seconds`` (>= 10 recommended):

    - time to first frame
    - achieved resolution reported by the device      (vs the requested value)
    - achieved FPS reported by the device + measured FPS (vs the requested value)
    - frame-interval p50 / p95 / max
    - read-failure count
    - process CPU%  and  process RSS delta (psutil)

Writes ``results/camera_matrix_<label>.json`` (every row) and a readable
``results/camera_matrix_<label>.md`` table, and merges the winning backend for
this index into ``results/camera_backends.json`` so ``capture.device_backend:
auto`` can open the known-good backend first.

**Requested is not achieved.** Every resolution / FPS column shows the requested
value and the value the device actually returned, side by side.

Coarse timing (the ``--seconds`` deadline, RSS sampling) uses ``time.monotonic``;
sub-millisecond frame intervals use ``time.perf_counter`` (Phase 0 clock rule).
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
from predictivesense.camera.enumerate import save_backend_hint
from predictivesense.logging_setup import configure_logging, get_logger

_LOG = get_logger("predictivesense.scripts.benchmark_camera_matrix")

_DEFAULT_RESOLUTIONS = "640x480,1280x720,1920x1080"
_DEFAULT_FPS = "30"
_DEFAULT_BACKENDS = "msmf,dshow"
_DEFAULT_FOURCCS = "auto,MJPG"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep one camera's capture matrix.")
    p.add_argument("--index", type=int, required=True, help="OpenCV device index")
    p.add_argument("--label", required=True, help="label for the output filenames")
    p.add_argument("--seconds", type=float, default=10.0, help="measure window per combo")
    p.add_argument("--resolutions", default=_DEFAULT_RESOLUTIONS, help="WxH,WxH,...")
    p.add_argument("--fps", default=_DEFAULT_FPS, help="comma-separated requested FPS")
    p.add_argument("--backends", default=_DEFAULT_BACKENDS, help="msmf,dshow")
    p.add_argument("--fourccs", default=_DEFAULT_FOURCCS, help="auto,MJPG,...")
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


def _interval_summary(values: list[float]) -> dict[str, float]:
    return {
        "p50": round(_pct(values, 50), 2),
        "p95": round(_pct(values, 95), 2),
        "max": round(max(values), 2) if values else 0.0,
        "count": len(values),
    }


def _measure_combo(
    index: int,
    *,
    width: int,
    height: int,
    fps: float,
    backend: str,
    fourcc: str,
    seconds: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        "requested": {
            "width": width,
            "height": height,
            "fps": fps,
            "backend": backend,
            "fourcc": fourcc,
        },
        "opened": False,
    }
    source = DeviceSource(
        index,
        backend=backend,
        request_width=width,
        request_height=height,
        request_fps=fps,
        fourcc=fourcc,
        buffer_size=1,
        open_timeout_s=5.0,
    )

    proc = psutil.Process()
    rss_start_mb = proc.memory_info().rss / (1024 * 1024)
    proc.cpu_percent(None)  # prime

    try:
        source.start()
    except RuntimeError as exc:
        row["error"] = str(exc)
        _LOG.warning(
            "combo %dx%d @%s %s/%s did not open: %s",
            width, height, fps, backend, fourcc, exc,
        )
        return row

    row["opened"] = True
    intervals: list[float] = []
    frames = 0
    read_failures = 0
    prev: float | None = None
    first_frame_s: float | None = None
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < seconds:
            r = source.read()
            if r is None:
                read_failures += 1
                continue
            mark = time.perf_counter()
            if first_frame_s is None:
                first_frame_s = time.monotonic() - t0
            if prev is not None:
                intervals.append((mark - prev) * 1000.0)
            prev = mark
            frames += 1
    finally:
        info = source.info()
        source.stop()

    elapsed = time.monotonic() - t0
    cpu_pct = proc.cpu_percent(None)
    rss_end_mb = proc.memory_info().rss / (1024 * 1024)
    achieved_fps = info.get("achieved_fps") or 0.0
    expected = achieved_fps * elapsed if achieved_fps else 0.0

    row.update(
        {
            "seconds_elapsed": round(elapsed, 2),
            "backend_used": info.get("backend"),
            "fourcc_used": info.get("fourcc"),
            "achieved": {
                "width": info.get("achieved_width"),
                "height": info.get("achieved_height"),
                "fps_reported": achieved_fps or None,
            },
            "measured_fps": round(frames / elapsed, 2) if elapsed > 0 else 0.0,
            "frames": frames,
            "dropped_estimate": round(expected - frames, 1) if expected else None,
            "read_failures": read_failures,
            "reconnects": info.get("reconnects"),
            "time_to_first_frame_s": (
                round(first_frame_s, 4) if first_frame_s is not None else None
            ),
            "frame_interval_ms": _interval_summary(intervals),
            "process_cpu_percent": round(cpu_pct, 1),
            "process_rss_mb_delta": round(rss_end_mb - rss_start_mb, 1),
        }
    )
    _LOG.info(
        "combo req %dx%d@%s %s/%-4s -> got %sx%s@%s meas %.1f fps | "
        "iv p50/p95/max %s/%s/%s ms | rf %d | ttff %s s | cpu %.0f%%",
        width, height, fps, backend, fourcc,
        row["achieved"]["width"], row["achieved"]["height"],
        row["achieved"]["fps_reported"], row["measured_fps"],
        row["frame_interval_ms"]["p50"], row["frame_interval_ms"]["p95"],
        row["frame_interval_ms"]["max"], read_failures,
        row["time_to_first_frame_s"], row["process_cpu_percent"],
    )
    return row


def _pick_winning_backend(rows: list[dict[str, object]]) -> str | None:
    """Lowest median frame-interval p95 among rows that read cleanly, tie -> msmf."""

    clean: dict[str, list[float]] = {"msmf": [], "dshow": []}
    for row in rows:
        if not row.get("opened") or row.get("read_failures", 1):
            continue
        used = row.get("backend_used")
        if used in clean:
            clean[used].append(row["frame_interval_ms"]["p95"])  # type: ignore[index]
    scored = {b: statistics.median(v) for b, v in clean.items() if v}
    if not scored:
        return None
    best = min(scored, key=lambda b: (scored[b], 0 if b == "msmf" else 1))
    return best


def _markdown_table(label: str, index: int, rows: list[dict[str, object]]) -> str:
    head = (
        f"# Camera matrix - {label} (index {index})\n\n"
        "`req` = requested, `got` = the value the device actually returned.\n\n"
        "| req res | got res | req fps | got fps (rep) | meas fps | backend | fourcc | "
        "iv p50/p95/max ms | read fail | TTFF s | CPU % | RSS d MB |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for r in rows:
        rq = r["requested"]  # type: ignore[index]
        if not r.get("opened"):
            lines.append(
                f"| {rq['width']}x{rq['height']} | - | {rq['fps']} | - | - | "
                f"{rq['backend']} | {rq['fourcc']} | - | did not open | - | - | - |"
            )
            continue
        ac = r["achieved"]  # type: ignore[index]
        iv = r["frame_interval_ms"]  # type: ignore[index]
        lines.append(
            f"| {rq['width']}x{rq['height']} | {ac['width']}x{ac['height']} | "
            f"{rq['fps']} | {ac['fps_reported']} | {r['measured_fps']} | "
            f"{r['backend_used']} | {r['fourcc_used']} | "
            f"{iv['p50']}/{iv['p95']}/{iv['max']} | {r['read_failures']} | "
            f"{r['time_to_first_frame_s']} | {r['process_cpu_percent']} | "
            f"{r['process_rss_mb_delta']} |"
        )
    return head + "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    configure_logging("INFO")

    resolutions: list[tuple[int, int]] = []
    for token in args.resolutions.split(","):
        w, _, h = token.strip().lower().partition("x")
        resolutions.append((int(w), int(h)))
    fps_list = [float(x) for x in args.fps.split(",")]
    backends = [b.strip() for b in args.backends.split(",")]
    fourccs = [f.strip() for f in args.fourccs.split(",")]

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    combos = [
        (w, h, fps, b, fc)
        for (w, h) in resolutions
        for fps in fps_list
        for b in backends
        for fc in fourccs
    ]
    _LOG.info(
        "camera matrix: index %d, %d combos x %.0fs (~%.0f min total)",
        args.index, len(combos), args.seconds,
        len(combos) * (args.seconds + 3) / 60.0,
    )

    rows: list[dict[str, object]] = []
    for (w, h, fps, b, fc) in combos:
        rows.append(
            _measure_combo(
                args.index, width=w, height=h, fps=fps, backend=b,
                fourcc=fc, seconds=args.seconds,
            )
        )

    any_opened = any(r.get("opened") for r in rows)
    winning_backend = _pick_winning_backend(rows)
    if winning_backend:
        save_backend_hint(args.index, winning_backend, results_dir)

    record = {
        "label": args.label,
        "index": args.index,
        "seconds_per_combo": args.seconds,
        "winning_backend": winning_backend,
        "combos": rows,
    }
    json_path = results_dir / f"camera_matrix_{args.label}.json"
    md_path = results_dir / f"camera_matrix_{args.label}.md"
    json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_table(args.label, args.index, rows), encoding="utf-8")

    _LOG.info("winning backend for index %d: %s", args.index, winning_backend)
    _LOG.info("wrote %s", json_path)
    _LOG.info("wrote %s", md_path)

    if not any_opened:
        _LOG.error("device index %d did not open on any combo", args.index)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
