"""Run the recorded-video driver over one file (Mode B).

    python scripts\\run_recorded.py --path data\\videos\\<clip>.mp4 --replay-mode asfast

Validates the path is inside ``video.input_dir``, runs
:class:`RecordedDriver`, prints a summary, and writes
``results/recorded_<run_id>.jsonl`` plus a run manifest. Exits non-zero on any
error (path outside the input dir, unreadable file).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from predictivesense.config.settings import ConfigError, ValidationError, load_config
from predictivesense.logging_setup import configure_logging, get_logger
from predictivesense.pipeline.recorded import RecordedDriver

_LOG = get_logger("predictivesense.scripts.run_recorded")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse a recorded video (Mode B).")
    parser.add_argument("--profile", default="eval", help="config profile (default: eval)")
    parser.add_argument("--path", required=True, help="video file, absolute or relative to video.input_dir")
    parser.add_argument("--replay-mode", choices=("realtime", "asfast"), default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    configure_logging("INFO")

    try:
        config = load_config(args.profile)
    except (ConfigError, ValidationError) as exc:
        _LOG.error("configuration error: %s", exc)
        return 2

    base = Path(config.video.input_dir).resolve()
    raw = Path(args.path)
    target = (raw if raw.is_absolute() else base / raw).resolve()
    if target != base and base not in target.parents:
        _LOG.error("path %s is outside %s", target, base)
        return 2
    if not target.is_file():
        _LOG.error("video file not found: %s", target)
        return 2

    replay_mode = args.replay_mode or config.video.replay_mode
    driver = RecordedDriver(results_dir=config.results_dir)
    try:
        result = driver.run(target, replay_mode=replay_mode, config_profile=config.profile)
    except (RuntimeError, OSError) as exc:
        _LOG.error("recorded run failed: %r", exc)
        return 1

    _LOG.info("=== recorded run summary ===")
    _LOG.info("run_id      : %s", result["run_id"])
    _LOG.info("source      : %s", result["source_path"])
    _LOG.info("replay mode : %s", result["replay_mode"])
    _LOG.info("frames      : %d", result["frames"])
    _LOG.info("jsonl       : %s", result["jsonl_path"])
    _LOG.info("manifest    : %s", result["manifest_path"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
