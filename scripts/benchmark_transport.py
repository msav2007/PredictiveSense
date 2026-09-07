"""Backend-owned camera transport benchmark.

    python scripts\\benchmark_transport.py --index 0 --seconds 30 --label laptop-cam

Opens one device index backend-owned and reports achieved resolution, achieved
FPS, frame-interval p50/p95/max, time to first frame, and read-failure count,
written to ``results/transport_<label>.json``. Run once per transport; the
comparison table is assembled by hand in the phase report.

Coarse timing (the --seconds deadline) uses ``time.monotonic``; sub-millisecond
frame intervals use ``time.perf_counter`` (Phase 0 clock decision).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

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
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args(argv)


def _percentile(values: list[float], pct: float) -> float:
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
        },
        "seconds_requested": args.seconds,
        "opened": False,
    }

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
    frames = 0
    read_failures = 0
    first_frame_s: float | None = None
    prev = None
    t_start = time.monotonic()
    try:
        while time.monotonic() - t_start < args.seconds:
            frame = source.read()
            mark = time.perf_counter()
            if frame is None:
                read_failures += 1
                continue
            if first_frame_s is None:
                first_frame_s = time.monotonic() - t_start
            if prev is not None:
                intervals.append((mark - prev) * 1000.0)
            prev = mark
            frames += 1
    finally:
        info = source.info()
        source.stop()

    elapsed = time.monotonic() - t_start
    record.update(
        {
            "seconds_elapsed": round(elapsed, 3),
            "backend_used": info.get("backend"),
            "achieved_width": info.get("achieved_width"),
            "achieved_height": info.get("achieved_height"),
            "achieved_fps_reported": info.get("achieved_fps"),
            "measured_fps": round(frames / elapsed, 3) if elapsed > 0 else 0.0,
            "frames": frames,
            "read_failures": read_failures,
            "reconnects": info.get("reconnects"),
            "time_to_first_frame_s": round(first_frame_s, 4) if first_frame_s is not None else None,
            "frame_interval_ms": {
                "p50": round(_percentile(intervals, 50), 3),
                "p95": round(_percentile(intervals, 95), 3),
                "max": round(max(intervals), 3) if intervals else 0.0,
                "mean": round(statistics.fmean(intervals), 3) if intervals else 0.0,
                "count": len(intervals),
            },
        }
    )
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    _LOG.info("=== transport benchmark: %s ===", args.label)
    _LOG.info("backend used        : %s", record["backend_used"])
    _LOG.info("achieved resolution : %sx%s", record["achieved_width"], record["achieved_height"])
    _LOG.info("measured fps        : %s", record["measured_fps"])
    _LOG.info("frame interval p50  : %s ms", record["frame_interval_ms"]["p50"])
    _LOG.info("frame interval p95  : %s ms", record["frame_interval_ms"]["p95"])
    _LOG.info("frame interval max  : %s ms", record["frame_interval_ms"]["max"])
    _LOG.info("time to first frame : %s s", record["time_to_first_frame_s"])
    _LOG.info("read failures       : %d", read_failures)
    _LOG.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
