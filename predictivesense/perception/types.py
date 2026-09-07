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
    """One frame's detections and poses plus per-model latency (milliseconds)."""

    detections: tuple[Detection, ...] = ()
    poses: tuple[Pose, ...] = ()
    detector_ms: float | None = None
    pose_ms: float | None = None
    pose_ran: bool = False
    frame_error: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)
