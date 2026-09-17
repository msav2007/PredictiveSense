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
lists stay empty and the output is exactly the Phase 1 shape.

Phase 9: when a :class:`~predictivesense.tracking.Tracker` is passed, each line
also carries real ``tracks`` - the **same** tracker class the live loop runs,
fed frame-by-frame in file order with the file's own presentation timestamps
(never wall time), so two runs of the same file/config/model/tracker produce
byte-identical track ids and states (asserted by
``tests/integration/test_recorded_determinism_with_models.py``). Recorded mode
is lossless (every decoded frame reaches the tracker - see section 13.3 of the
Phase 9 prompt), which is *not* comparable to real-time's lossy, timing-driven
continuity. Without a tracker, ``tracks`` stays empty.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from predictivesense.camera.file_source import FileSource
from predictivesense.logging_setup import get_logger
from predictivesense.pipeline.perception_frame import (
    build_perception_frame,
    resolve_model_version,
)
from predictivesense.telemetry.manifest import git_state, utc_now_iso

if TYPE_CHECKING:
    from predictivesense.core.types import Detection, Pose, Track
    from predictivesense.perception.engine import PerceptionEngine
    from predictivesense.perception.policy import RecognitionPolicy
    from predictivesense.tracking import Tracker

__all__ = ["RecordedDriver", "RecordedResult"]

_LOG = get_logger(__name__)
_LINE_KEYS = ("frame_id", "capture_ts", "pts_s", "detections", "poses", "tracks")
_COORD_DP = 2   # box / keypoint pixel precision in the JSONL
_SCORE_DP = 4   # confidence precision in the JSONL


def _detection_to_dict(det: "Detection", *, annotated: bool) -> dict[str, object]:
    x1, y1, x2, y2 = det.bbox
    out: dict[str, object] = {
        "bbox": [round(x1, _COORD_DP), round(y1, _COORD_DP),
                 round(x2, _COORD_DP), round(y2, _COORD_DP)],
        "class_id": det.class_id,
        "class_name": det.class_name,
        "score": round(det.score, _SCORE_DP),
    }
    # Phase 2.5 additive fields - written only when the recognition policy ran, so
    # a `perception=`-only run (no `policy=`) keeps the exact Phase 2 JSONL shape.
    if annotated:
        out["raw_class_name"] = det.raw_class_name or det.class_name
        out["policy_state"] = det.policy_state or "accepted"
        out["runner_up"] = (
            [det.runner_up[0], round(float(det.runner_up[1]), _SCORE_DP)]
            if det.runner_up is not None
            else None
        )
    return out


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


def _track_to_dict(tr: "Track") -> dict[str, object]:
    x1, y1, x2, y2 = tr.bbox
    vx, vy = tr.velocity
    return {
        "track_id": tr.track_id,
        "status": tr.status.value,
        "class_name": tr.class_name,
        "bbox": [round(x1, _COORD_DP), round(y1, _COORD_DP),
                 round(x2, _COORD_DP), round(y2, _COORD_DP)],
        "observed_class": tr.observed_class,
        "track_class": tr.track_class,
        "class_votes": [[c, n] for c, n in tr.class_votes],
        "velocity": [round(vx, _COORD_DP), round(vy, _COORD_DP)],
        "fresh": tr.fresh,
        "age_frames": tr.age_frames,
        "age_ms": round(tr.age_ms, 1),
        "hits": tr.hits,
        "consecutive_misses": tr.consecutive_misses,
        "last_detector_confidence": (
            round(tr.last_detector_confidence, _SCORE_DP)
            if tr.last_detector_confidence is not None else None
        ),
        "last_detector_confidence_age_ms": round(tr.last_detector_confidence_age_ms, 1),
        "policy_state": tr.policy_state,
        "tier": tr.tier,
        "pose_keypoints": [
            [round(kx, _COORD_DP), round(ky, _COORD_DP), round(kv, _SCORE_DP)]
            for (kx, ky, kv) in tr.pose_keypoints
        ],
        "pose_frame_id": tr.pose_frame_id,
        "pose_age_ms": round(tr.pose_age_ms, 1),
        "pose_fresh": tr.pose_fresh,
    }


class RecordedResult(dict):
    """``{run_id, jsonl_path, manifest_path, frames, replay_mode, source_path,
    perception_enabled, total_detections, total_poses, tracking_enabled,
    total_tracks}``."""


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
        policy: "RecognitionPolicy | None" = None,
        tracker: "Tracker | None" = None,
    ) -> RecordedResult:
        src_path = Path(path)
        run_id = run_id or uuid.uuid4().hex
        self._results_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = self._results_dir / f"recorded_{run_id}.jsonl"
        manifest_path = self._results_dir / f"recorded_{run_id}.manifest.json"
        # Phase 8 section 16: the immutable per-frame PerceptionFrame record,
        # identical in structure to what the real-time loop exposes. Written to a
        # SIDECAR so the main JSONL line shape and its byte-determinism guarantee
        # are unchanged. Only written when perception ran.
        frames_path = self._results_dir / f"recorded_{run_id}.frames.jsonl"
        model_version = resolve_model_version() if perception is not None else "unknown"
        active_ep = "none"
        if perception is not None:
            _pi = perception.info()
            active_ep = str(_pi.get("detector_ep") or _pi.get("provider") or "none")

        started = utc_now_iso()
        wall0 = time.monotonic()
        frames = 0
        first_pts: float | None = None
        last_pts: float | None = None
        total_detections = 0
        total_poses = 0
        total_tracks = 0
        perception_errors = 0
        policy_counts = None

        source = FileSource(src_path, replay_mode=replay_mode)
        source.start()
        frames_fh = (
            frames_path.open("w", encoding="utf-8", newline="\n")
            if perception is not None
            else None
        )
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
                    out_dets: list["Detection"] = []
                    if perception is not None:
                        result = perception.infer(frame, frame_index=frame.frame_id)
                        if result.frame_error:
                            perception_errors += 1
                        out_dets = list(result.detections)
                        if policy is not None:
                            outcome = policy.apply(
                                result.detections,
                                frame_width=int(frame.width),
                                frame_height=int(frame.height),
                            )
                            out_dets = outcome.detections
                            policy_counts = (
                                outcome.counts
                                if policy_counts is None
                                else policy_counts.merged(outcome.counts)
                            )
                        dets = [
                            _detection_to_dict(d, annotated=policy is not None)
                            for d in out_dets
                        ]
                        poses = [_pose_to_dict(p) for p in result.poses]
                        total_detections += len(dets)
                        total_poses += len(poses)

                        # Section 16 sidecar: identical structure to the loop's
                        # PerceptionFrame. Deterministic (frame-derived fields
                        # only) so two runs of this sidecar are byte-identical.
                        if frames_fh is not None:
                            pf = build_perception_frame(
                                frame=frame,
                                detections=out_dets,
                                poses=result.poses,
                                model_version=model_version,
                                provider=active_ep,
                                pose_reused=result.pose_reused,
                                pose_frame_id=result.pose_frame_id,
                                pose_capture_ts=result.pose_capture_ts,
                                pose_age_ms=result.pose_age_ms,
                            )
                            frames_fh.write(
                                pf.model_dump_json() + "\n"
                            )

                    tracks: list[dict[str, object]] = []
                    if tracker is not None:
                        # Recorded mode is lossless - every decoded frame reaches
                        # the tracker, unlike the real-time loop's staleness/
                        # mailbox-driven drops (section 13.3). capture_ts is the
                        # file's own PTS, never wall time, so two runs are
                        # byte-identical (section 13.2).
                        track_objs = tracker.update(
                            out_dets, frame_id=frame.frame_id, capture_ts=frame.capture_ts,
                            frame_width=float(frame.width), frame_height=float(frame.height),
                            poses=result.poses if perception is not None else (),
                            pose_fresh=not result.pose_reused if perception is not None else True,
                        )
                        tracks = [_track_to_dict(t) for t in track_objs]
                        total_tracks += len(tracks)

                    line = {
                        "frame_id": frame.frame_id,
                        "capture_ts": frame.capture_ts,
                        "pts_s": frame.capture_ts,
                        "detections": dets,
                        "poses": poses,
                        "tracks": tracks,
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
            if frames_fh is not None:
                frames_fh.close()

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
            "policy": (
                {
                    "enabled": True,
                    **policy.info(),
                    "counts": (
                        policy_counts.as_metrics("") if policy_counts is not None else {}
                    ),
                }
                if policy is not None
                else {"enabled": False}
            ),
            "tracking": (
                {"enabled": True, "total_tracks": total_tracks, "stats": vars(tracker.stats)}
                if tracker is not None
                else {"enabled": False}
            ),
            "jsonl_path": str(jsonl_path),
            "jsonl_bytes": jsonl_path.stat().st_size,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        _LOG.info(
            "recorded run %s: %d frames from %s (%s), %d detections / %d poses / "
            "%d tracks -> %s",
            run_id,
            frames,
            src_path.name,
            replay_mode,
            total_detections,
            total_poses,
            total_tracks,
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
            tracking_enabled=tracker is not None,
            total_tracks=total_tracks,
        )
