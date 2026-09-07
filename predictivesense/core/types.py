"""Frozen contracts shared across every phase.

All contracts are immutable, fully annotated Pydantic v2 models. Later phases may
*add* fields; renaming or reshaping anything here requires a line in
``docs/decisions.md``.

Allowed imports for this module: the standard library, ``typing``, ``enum`` (via
``core.enums``), ``dataclasses``, a numpy type alias, and Pydantic. It must not
import OpenCV, torch, onnxruntime, any HTTP client, or anything from
``predictivesense.api``, ``.pipeline`` or ``.camera``.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from predictivesense.core.enums import Mode, RiskLevel, TrackStatus

__all__ = [
    "BBox",
    "Frame",
    "Detection",
    "Pose",
    "Track",
    "Relation",
    "RiskState",
    "Alert",
    "StateSnapshot",
    "MailboxStats",
]

# (x1, y1, x2, y2) in pixels.
BBox = tuple[float, float, float, float]


class _Frozen(BaseModel):
    """Base for immutable contracts: assignment after construction raises."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Frame(_Frozen):
    """A single captured image plus its provenance."""

    # ``image`` is an arbitrary (non-Pydantic) type, so this model opts in.
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    frame_id: int = Field(description="Monotonic per-source counter from 0.")
    capture_ts: float = Field(description="time.monotonic() at capture.")
    image: np.ndarray = Field(description="HxWx3, BGR, uint8.")
    width: int
    height: int
    source_id: str = Field(description="Opaque id of the producing source.")
    seq: int = Field(
        description="Sequence within the current source session; resets on reconnect."
    )


class Detection(_Frozen):
    """A single object detection. Defined for later phases; unused in Phase 0."""

    bbox: BBox
    class_id: int
    class_name: str
    score: float
    frame_id: int


class Pose(_Frozen):
    """A single body pose. Defined for later phases; unused in Phase 0."""

    keypoints: tuple[tuple[float, float, float], ...] = Field(
        description="Tuples of (x, y, visibility)."
    )
    bbox: BBox
    score: float
    frame_id: int


class Track(_Frozen):
    """A tracked entity. Minimal on purpose; P3 extends it."""

    track_id: int
    status: TrackStatus
    class_name: str
    bbox: BBox
    last_seen_frame_id: int


class Relation(_Frozen):
    """A relation between two tracked entities. Minimal; P3 extends it."""

    subject_id: int
    object_id: int
    kind: str
    value: float


class RiskState(_Frozen):
    """A scored risk scenario. Minimal; P6 extends it."""

    scenario: str
    level: RiskLevel
    confidence: float


class Alert(_Frozen):
    """A user-facing alert. Minimal; P7 extends it."""

    alert_id: int
    level: RiskLevel
    text: str
    recommendation: str
    issued_ts: float


class StateSnapshot(_Frozen):
    """The single object broadcast to the UI. Frozen wire format.

    Its JSON encoding is documented in ``docs/architecture.md``.
    """

    snapshot_id: int = Field(description="Monotonic.")
    mode: Mode
    frame_id: int | None = Field(description="Frame this snapshot describes.")
    capture_ts: float | None = Field(description="Capture time of that frame.")
    emitted_ts: float = Field(description="time.monotonic() at emission.")
    frame_age_ms: float | None = Field(
        description="(emitted_ts - capture_ts) * 1000, or None when no frame."
    )
    detections: list[Detection] = Field(default_factory=list)
    poses: list[Pose] = Field(default_factory=list)
    tracks: list[Track] = Field(default_factory=list)
    risk: RiskState | None = None
    metrics: dict[str, float] = Field(
        description="loop_rate_hz, dropped, consumed, iter_latency_ms."
    )
    stale: bool = Field(
        description="True when no frame arrived within the configured threshold."
    )

    def to_wire_json(self) -> str:
        """Serialise to the frozen wire format (compact JSON)."""

        return self.model_dump_json()

    @classmethod
    def from_wire_json(cls, payload: str) -> "StateSnapshot":
        """Parse the frozen wire format back into a snapshot."""

        return cls.model_validate_json(payload)


class MailboxStats(_Frozen):
    """A point-in-time view of a :class:`~predictivesense.camera.mailbox` counter set."""

    consumed: int
    dropped: int
    depth: int
