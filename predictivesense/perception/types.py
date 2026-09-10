"""Small aggregate returned by :class:`PerceptionEngine.infer`.

The wire contracts ``Detection`` and ``Pose`` are frozen in
``predictivesense/core/types.py`` and unchanged. This is an internal,
non-wire helper that bundles one frame's perception output with its timing so
the analysis loop can populate the snapshot and its metrics in one step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from predictivesense.core.types import Detection, Pose

__all__ = ["PerceptionResult"]


@dataclass(frozen=True)
class PerceptionResult:
    """One frame's detections and poses plus per-model latency (milliseconds).

    Phase 8 (section 9): when ``pose_reused`` is true the ``poses`` tuple is the
    **same immutable objects** from an earlier frame - never mutated, never a
    half-written read - and ``pose_frame_id`` / ``pose_capture_ts`` /
    ``pose_age_ms`` describe where they came from. ``pose_ms`` is ``None`` on a
    reused frame (no inference ran).
    """

    detections: tuple[Detection, ...] = ()
    poses: tuple[Pose, ...] = ()
    detector_ms: float | None = None
    pose_ms: float | None = None
    pose_ran: bool = False
    frame_error: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)
    pose_reused: bool = False
    pose_frame_id: int | None = None
    pose_capture_ts: float | None = None
    pose_age_ms: float | None = None
