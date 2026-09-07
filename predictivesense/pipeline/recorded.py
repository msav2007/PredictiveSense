"""``RecordedDriver`` - Mode B: lossless, deterministic recorded-video analysis.

Drives a :class:`~predictivesense.camera.file_source.FileSource` frame by frame
and writes one JSON object per frame to ``results/recorded_<run_id>.jsonl`` plus
a run manifest. It never touches the mailbox, so nothing is dropped: every
decoded frame produces exactly one line.

Determinism: the JSONL content depends only on the file and the config. Fields
are written in a fixed key order with rounded timestamps; ``capture_ts`` /
``pts_s`` come from the file's presentation timestamps, never wall time. Two
runs over the same file produce byte-identical JSONL. (The manifest, which
records wall time and git state, is not byte-identical and is not part of that
guarantee.)

In Phase 1 each line carries only what exists: ``frame_id``, ``capture_ts``,
``pts_s`` and empty ``detections`` / ``poses`` / ``tracks``. Event analysis and
findings reports are later phases.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from predictivesense.camera.file_source import FileSource
from predictivesense.logging_setup import get_logger
from predictivesense.telemetry.manifest import git_state, utc_now_iso

__all__ = ["RecordedDriver", "RecordedResult"]

_LOG = get_logger(__name__)
_LINE_KEYS = ("frame_id", "capture_ts", "pts_s", "detections", "poses", "tracks")


class RecordedResult(dict):
    """``{run_id, jsonl_path, manifest_path, frames, replay_mode, source_path}``."""


class RecordedDriver:
    """Processes every frame of a video file with no drops, deterministically."""

    def __init__(self, results_dir: str | Path = "results") -> None:
        self._results_dir = Path(results_dir)

    def run(
        self,
        path: str | Path,
        *,
        replay_mode: str = "asfast",
        run_id: str | None = None,
        config_profile: str = "unknown",
    ) -> RecordedResult:
        src_path = Path(path)
        run_id = run_id or uuid.uuid4().hex
        self._results_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = self._results_dir / f"recorded_{run_id}.jsonl"
        manifest_path = self._results_dir / f"recorded_{run_id}.manifest.json"

        started = utc_now_iso()
        wall0 = time.monotonic()
        frames = 0
        first_pts: float | None = None
        last_pts: float | None = None

        source = FileSource(src_path, replay_mode=replay_mode)
        source.start()
        try:
            with jsonl_path.open("w", encoding="utf-8", newline="\n") as fh:
                while True:
                    frame = source.read()
                    if frame is None:
                        break
                    if first_pts is None:
                        first_pts = frame.capture_ts
                    last_pts = frame.capture_ts
                    line = {
                        "frame_id": frame.frame_id,
                        "capture_ts": frame.capture_ts,
                        "pts_s": frame.capture_ts,
                        "detections": [],
                        "poses": [],
                        "tracks": [],
                    }
                    fh.write(
                        json.dumps(
                            {k: line[k] for k in _LINE_KEYS},
                            separators=(",", ":"),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    frames += 1
        finally:
            source.stop()

        info = source.info()
        commit, dirty = git_state()
        manifest = {
            "run_id": run_id,
            "kind": "recorded",
            "source_path": str(src_path.resolve()),
            "replay_mode": replay_mode,
            "config_profile": config_profile,
            "started_utc": started,
            "ended_utc": utc_now_iso(),
            "wall_seconds": round(time.monotonic() - wall0, 3),
            "git_commit": commit,
            "git_dirty": dirty,
            "frames": frames,
            "fps": info.get("fps"),
            "width": info.get("width"),
            "height": info.get("height"),
            "frame_count_hint": info.get("frame_count_hint"),
            "pts_fallbacks": info.get("pts_fallbacks"),
            "first_pts_s": first_pts,
            "last_pts_s": last_pts,
            "jsonl_path": str(jsonl_path),
            "jsonl_bytes": jsonl_path.stat().st_size,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        _LOG.info(
            "recorded run %s: %d frames from %s (%s) -> %s",
            run_id,
            frames,
            src_path.name,
            replay_mode,
            jsonl_path,
        )

        return RecordedResult(
            run_id=run_id,
            jsonl_path=str(jsonl_path),
            manifest_path=str(manifest_path),
            frames=frames,
            replay_mode=replay_mode,
            source_path=str(src_path.resolve()),
        )
