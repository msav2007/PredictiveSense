"""Instrumented no-op run.

    python scripts\\run_noop.py --profile dev --seconds 60

Runs the analysis loop for N seconds with no intelligence attached, samples
telemetry into a CSV, writes a session manifest, and checks the two Phase 0
assertions: RSS growth under the configured budget, and mailbox depth never
above 1. Exits non-zero if a loop thread errored or an assertion failed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from pathlib import Path

import psutil

from predictivesense.config.settings import (
    ConfigError,
    ValidationError,
    load_config,
)
from predictivesense.logging_setup import configure_logging, get_logger
from predictivesense.pipeline.loop import build_loop
from predictivesense.telemetry.manifest import build_manifest, utc_now_iso
from predictivesense.telemetry.metrics import MetricRegistry
from predictivesense.telemetry.writer import MetricsWriter

_LOG = get_logger("predictivesense.scripts.run_noop")

CSV_FIELDS = [
    "elapsed_s",
    "wall_utc",
    "snapshot_id",
    "frame_id",
    "stale",
    "loop_rate_hz",
    "producer_rate_hz",
    "consumed",
    "dropped",
    "mailbox_depth",
    "iter_latency_ms",
    "rss_mb",
]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the instrumented no-op loop.")
    parser.add_argument("--profile", required=True, help="config profile name (dev, eval)")
    parser.add_argument("--seconds", type=int, default=None, help="run duration (default: config)")
    return parser.parse_args(argv)


def _check_results_dir_writable(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    probe = results_dir / f".write_probe_{os.getpid()}"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise RuntimeError(f"results dir {results_dir} is not writable: {exc}") from exc


def _sample_row(loop, proc: psutil.Process, elapsed_s: float) -> dict[str, object] | None:
    snap = loop.latest
    if snap is None:
        return None
    m = snap.metrics
    return {
        "elapsed_s": round(elapsed_s, 3),
        "wall_utc": utc_now_iso(),
        "snapshot_id": snap.snapshot_id,
        "frame_id": snap.frame_id if snap.frame_id is not None else "",
        "stale": int(snap.stale),
        "loop_rate_hz": round(m.get("loop_rate_hz", 0.0), 3),
        "producer_rate_hz": round(m.get("producer_rate_hz", 0.0), 3),
        "consumed": int(m.get("consumed", 0)),
        "dropped": int(m.get("dropped", 0)),
        "mailbox_depth": int(m.get("mailbox_depth", 0)),
        "iter_latency_ms": round(m.get("iter_latency_ms", 0.0), 4),
        "rss_mb": round(proc.memory_info().rss / 1_000_000, 3),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    configure_logging("INFO")  # measurement runs always log at INFO

    try:
        config = load_config(args.profile)
    except (ConfigError, ValidationError) as exc:
        _LOG.error("configuration error: %s", exc)
        return 2

    # The no-op run measures the Phase 0 synthetic producer/consumer loop and its
    # RSS budget. Phase 2 perception is a separate concern (scripts/benchmark_
    # providers.py) and would load ~60 MB of ONNX Runtime + weights for no signal
    # on synthetic noise - force it off here regardless of the profile.
    config = config.model_copy(
        update={
            "perception": config.perception.model_copy(
                update={"detection_enabled": False, "pose_enabled": False}
            )
        }
    )

    seconds = args.seconds if args.seconds is not None else config.noop.default_seconds
    results_dir = Path(config.results_dir)
    try:
        _check_results_dir_writable(results_dir)
    except RuntimeError as exc:
        _LOG.error("%s", exc)
        return 2

    session_id = uuid.uuid4().hex
    started_utc = utc_now_iso()
    _LOG.info(
        "no-op run start session=%s profile=%s seconds=%d results_dir=%s",
        session_id,
        config.profile,
        seconds,
        results_dir,
    )

    registry = MetricRegistry()
    loop = build_loop(config, registry=registry)
    manifest = build_manifest(config, session_id=session_id, started_utc=started_utc)

    proc = psutil.Process()
    rss_start = proc.memory_info().rss

    csv_path = results_dir / f"noop_{session_id}.csv"
    manifest_path = results_dir / f"manifest_{session_id}.json"
    writer = MetricsWriter(csv_path, CSV_FIELDS)

    loop.start()
    t0 = time.monotonic()
    next_sample = t0
    try:
        while time.monotonic() - t0 < seconds:
            if loop.error is not None:
                break
            time.sleep(0.05)
            now = time.monotonic()
            if now >= next_sample:
                next_sample += config.noop.csv_sample_period_s
                row = _sample_row(loop, proc, now - t0)
                if row is not None:
                    writer.write_row(row)
    finally:
        loop.stop()

    rss_end = proc.memory_info().rss
    rss_growth_mb = (rss_end - rss_start) / 1_000_000
    summary = loop.summary()

    manifest.extra = {
        "run": {
            "kind": "noop",
            "requested_seconds": seconds,
            "actual_seconds": round(time.monotonic() - t0, 3),
            "rss_start_bytes": rss_start,
            "rss_end_bytes": rss_end,
            "rss_growth_mb": round(rss_growth_mb, 3),
            "csv_path": str(csv_path),
            "csv_rows": writer.rows_written,
            **summary,
        }
    }
    manifest.finalize()
    manifest.write(manifest_path)
    writer.close()

    depth_ok = loop.max_mailbox_depth <= 1
    rss_ok = rss_growth_mb <= config.noop.max_rss_growth_mb

    _LOG.info("=== no-op run summary ===")
    _LOG.info("session_id           : %s", session_id)
    _LOG.info("profile / mode       : %s / %s", config.profile, config.mode.value)
    _LOG.info("requested seconds    : %d", seconds)
    _LOG.info("snapshots emitted    : %d", summary["snapshots"])
    _LOG.info("frames consumed      : %d", summary["consumed"])
    _LOG.info("frames dropped       : %d", summary["dropped"])
    _LOG.info("loop rate (Hz)       : %.3f", summary["loop_rate_hz"])
    _LOG.info("producer rate (Hz)   : %.3f", summary["producer_rate_hz"])
    _LOG.info("iter latency p50 (ms): %.4f", summary["iter_latency_ms_p50"])
    _LOG.info("iter latency p95 (ms): %.4f", summary["iter_latency_ms_p95"])
    _LOG.info("iter latency max (ms): %.4f", summary["iter_latency_ms_max"])
    _LOG.info("max mailbox depth    : %d  (<=1 required: %s)", loop.max_mailbox_depth, depth_ok)
    _LOG.info("RSS start (MB)        : %.3f", rss_start / 1_000_000)
    _LOG.info("RSS end (MB)          : %.3f", rss_end / 1_000_000)
    _LOG.info(
        "RSS growth (MB)      : %.3f  (<=%.1f required: %s)",
        rss_growth_mb,
        config.noop.max_rss_growth_mb,
        rss_ok,
    )
    _LOG.info("csv                  : %s (%d rows)", csv_path, writer.rows_written)
    _LOG.info("manifest             : %s", manifest_path)

    if loop.error is not None:
        _LOG.error("FAIL: analysis loop error: %r", loop.error)
        return 1
    if not depth_ok:
        _LOG.error("FAIL: mailbox depth exceeded 1 (max=%d)", loop.max_mailbox_depth)
        return 1
    if not rss_ok:
        _LOG.error("FAIL: RSS grew by %.3f MB (budget %.1f MB)", rss_growth_mb, config.noop.max_rss_growth_mb)
        return 1

    _LOG.info("PASS: no-op run within budget")
    return 0


if __name__ == "__main__":
    sys.exit(main())
