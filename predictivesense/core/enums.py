"""Enumerations shared across the pipeline.

Only :class:`Mode` and :data:`SourceKind.SYNTHETIC` carry behaviour in Phase 0.
:class:`RiskLevel` and :class:`TrackStatus` are defined but unused; later phases
consume them without reshaping this module.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "Mode",
    "SourceKind",
    "RiskLevel",
    "TrackStatus",
    "constructible_source_kinds",
    "ensure_source_constructible",
]


class Mode(str, Enum):
    """Input mode. The core never branches on this; only drivers in ``pipeline`` do."""

    REALTIME = "realtime"
    RECORDED = "recorded"


class SourceKind(str, Enum):
    """Kind of frame source. Only ``SYNTHETIC`` is constructible in Phase 0."""

    SYNTHETIC = "synthetic"
    DEVICE = "device"
    FILE = "file"
    WEBRTC = "webrtc"


class RiskLevel(str, Enum):
    """Risk classification. Defined for later phases; unused in Phase 0."""

    SAFE = "safe"
    WARNING = "warning"
    HIGH_RISK = "high_risk"
    UNKNOWN = "unknown"


class TrackStatus(str, Enum):
    """Track lifecycle state. Defined for later phases; unused in Phase 0."""

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    COASTING = "coasting"
    EXPIRED = "expired"


# Which SourceKind values a driver may actually build in this phase.
_CONSTRUCTIBLE: frozenset[SourceKind] = frozenset({SourceKind.SYNTHETIC})

# The phase that will implement each not-yet-constructible kind.
_PLANNED_PHASE: dict[SourceKind, str] = {
    SourceKind.DEVICE: "P1",
    SourceKind.FILE: "P1",
    SourceKind.WEBRTC: "P1",
}


def constructible_source_kinds() -> frozenset[SourceKind]:
    """Return the set of :class:`SourceKind` values buildable in the current phase."""

    return _CONSTRUCTIBLE


def ensure_source_constructible(kind: SourceKind) -> None:
    """Raise :class:`NotImplementedError` naming the owning phase for unsupported kinds."""

    if kind in _CONSTRUCTIBLE:
        return
    phase = _PLANNED_PHASE.get(kind, "a later phase")
    raise NotImplementedError(
        f"SourceKind.{kind.name} is not constructible in Phase 0; it is planned for {phase}."
    )
