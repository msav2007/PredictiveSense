"""Build the immutable per-frame :class:`PerceptionFrame` record (Phase 8,
section 16) - **tracking-ready output, no tracker**.

One helper, used by both the real-time loop (`pipeline/loop.py`) and the
recorded driver (`pipeline/recorded.py`), so the structure a future tracker
consumes is byte-identical between the two modes. Detections and poses are put
into a deterministic within-frame order (``-score``, then ``class_name`` for
detections, then box top-left) so two runs agree.

This is a **pipeline-internal contract**: the loop exposes the recent records
via ``AnalysisLoop.perception_frames()`` and the recorded driver writes them to a
``results/recorded_<run_id>.frames.jsonl`` sidecar. It is *not* added to the
`StateSnapshot` wire format (that already carries `detections` / `poses` /
`frame_id` / `capture_ts`) and *not* added to the main recorded JSONL line, so
no existing contract or determinism guarantee changes.
"""

from __future__ import annotations

from predictivesense.core.types import Detection, PerceptionFrame, Pose

__all__ = [
    "build_perception_frame",
    "resolve_model_version",
    "UNKNOWN_MODEL_VERSION",
]

UNKNOWN_MODEL_VERSION = "unknown"


def resolve_model_version() -> str:
    """Active model-registry ``version_id``, or ``"unknown"`` when the registry
    is absent / malformed (it degrades gracefully everywhere else too)."""

    try:
        from predictivesense.models.registry import ModelRegistry

        reg = ModelRegistry.load()
        reg.validate()
        return reg.active().version_id
    except Exception:  # noqa: BLE001
        return UNKNOWN_MODEL_VERSION


def _det_key(d: Detection) -> tuple:
    return (-float(d.score), d.class_name, float(d.bbox[0]), float(d.bbox[1]))


def _pose_key(p: Pose) -> tuple:
    return (-float(p.score), float(p.bbox[0]), float(p.bbox[1]))


def build_perception_frame(
    *,
    frame,
    detections,
    poses,
    model_version: str,
    provider: str,
    pose_reused: bool = False,
    pose_frame_id: int | None = None,
    pose_capture_ts: float | None = None,
    pose_age_ms: float | None = None,
) -> PerceptionFrame:
    """Assemble one frame's perception output into the immutable record."""

    return PerceptionFrame(
        frame_id=int(frame.frame_id),
        seq=int(frame.seq),
        capture_ts=float(frame.capture_ts),
        width=int(frame.width),
        height=int(frame.height),
        model_version=model_version or UNKNOWN_MODEL_VERSION,
        provider=provider or "unknown",
        detections=tuple(sorted(detections, key=_det_key)),
        poses=tuple(sorted(poses, key=_pose_key)),
        pose_stale=bool(pose_reused),
        pose_frame_id=pose_frame_id,
        pose_capture_ts=pose_capture_ts,
        pose_age_ms=pose_age_ms,
    )
