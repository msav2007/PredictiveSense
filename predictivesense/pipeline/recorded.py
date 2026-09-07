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

Phase 2: when a :class:`~predictivesense.perception.engine.PerceptionEngine` is
passed, each line carries real ``detections`` and ``poses`` produced by the
**same** perception code the live loop runs. Determinism now includes the model:
the JSONL depends only on the file, the config and the weights, so two runs are
byte-identical (asserted by
``tests/integration/test_recorded_determinism_with_models.py``). Coordinates and
scores are written at fixed precision in a fixed key order. Without an engine the
lists stay empty and the output is exactly the Phase 1 shape. ``tracks`` is
always empty - there is no tracker.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from predictivesense.camera.file_source import FileSource
from predictivesense.logging_setup import get_logger
from predictivesense.telemetry.manifest import git_state, utc_now_iso

if TYPE_CHECKING:
    from predictivesense.core.types import Detection, Pose
    from predictivesense.perception.engine import PerceptionEngine

__all__ = ["RecordedDriver", "RecordedResult"]

_LOG = get_logger(__name__)
_LINE_KEYS = ("frame_id", "capture_ts", "pts_s", "detections", "poses", "tracks")
_COORD_DP = 2   # box / keypoint pixel precision in the JSONL
_SCORE_DP = 4   # confidence precision in the JSONL


def _detection_to_dict(det: "Detection") -> dict[str, object]:
    x1, y1, x2, y2 = det.bbox
    return {
        "bbox": [round(x1, _COORD_DP), round(y1, _COORD_DP),
                 round(x2, _COORD_DP), round(y2, _COORD_DP)],
        "class_id": det.class_id,
        "class_name": det.class_name,
        "score": round(det.score, _SCORE_DP),
    }


def _pose_to_dict(pose: "Pose") -> dict[str, object]:
    x1, y1, x2, y2 = pose.bbox
    return {
        "bbox": [round(x1, _COORD_DP), round(y1, _COORD_DP),
                 round(x2, _COORD_DP), round(y2, _COORD_DP)],
        "score": round(pose.score, _SCORE_DP),
        "keypoints": [
            [round(kx, _COORD_DP), round(ky, _COORD_DP), round(kv, _SCORE_DP)]
            for (kx, ky, kv) in pose.keypoints
        ],
    }


class RecordedResult(dict):
    """``{run_id, jsonl_path, manifest_path, frames, replay_mode, source_path,
    perception_enabled, total_detections, total_poses}``."""


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
        perception: "PerceptionEngine | None" = None,
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
        total_detections = 0
        total_poses = 0
        perception_errors = 0

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

                    dets: list[dict[str, object]] = []
                    poses: list[dict[str, object]] = []
                    if perception is not None:
                        result = perception.infer(frame, frame_index=frame.frame_id)
                        if result.frame_error:
                            perception_errors += 1
                        dets = [_detection_to_dict(d) for d in result.detections]
                        poses = [_pose_to_dict(p) for p in result.poses]
                        total_detections += len(dets)
                        total_poses += len(poses)

                    line = {
                        "frame_id": frame.frame_id,
                        "capture_ts": frame.capture_ts,
                        "pts_s": frame.capture_ts,
                        "detections": dets,
                        "poses": poses,
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
            "perception": (
                {
                    "enabled": True,
                    "total_detections": total_detections,
                    "total_poses": total_poses,
                    "frame_errors": perception_errors,
                    **perception.info(),
                }
                if perception is not None
                else {"enabled": False}
            ),
            "jsonl_path": str(jsonl_path),
            "jsonl_bytes": jsonl_path.stat().st_size,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        _LOG.info(
            "recorded run %s: %d frames from %s (%s), %d detections / %d poses -> %s",
            run_id,
            frames,
            src_path.name,
            replay_mode,
            total_detections,
            total_poses,
            jsonl_path,
        )

        return RecordedResult(
            run_id=run_id,
            jsonl_path=str(jsonl_path),
            manifest_path=str(manifest_path),
            frames=frames,
            replay_mode=replay_mode,
            source_path=str(src_path.resolve()),
            perception_enabled=perception is not None,
            total_detections=total_detections,
            total_poses=total_poses,
        )
