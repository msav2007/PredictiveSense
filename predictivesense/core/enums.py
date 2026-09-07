"""Enumerations shared across the pipeline.

Phase 1 makes :data:`SourceKind.DEVICE`, ``FILE`` and ``BROWSER`` constructible
alongside ``SYNTHETIC``. ``WEBRTC`` stays defined but never constructible - the
browser-worker ingest path replaces it (see ``docs/decisions.md``).
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
    """Kind of frame source.

    ``SYNTHETIC``, ``DEVICE``, ``FILE`` and ``BROWSER`` are constructible from
    Phase 1 on. ``WEBRTC`` is retained for wire-format stability only and is
    never constructible.
    """

    SYNTHETIC = "synthetic"
    DEVICE = "device"
    FILE = "file"
    BROWSER = "browser"
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
_CONSTRUCTIBLE: frozenset[SourceKind] = frozenset(
    {
        SourceKind.SYNTHETIC,
        SourceKind.DEVICE,
        SourceKind.FILE,
        SourceKind.BROWSER,
    }
)

# Reason each not-yet-constructible kind is unavailable.
_PLANNED_PHASE: dict[SourceKind, str] = {
    SourceKind.WEBRTC: (
        "never - the browser Web Worker ingest path (WS /ws/ingest) replaces "
        "WebRTC; see docs/decisions.md"
    ),
}


def constructible_source_kinds() -> frozenset[SourceKind]:
    """Return the set of :class:`SourceKind` values buildable in the current phase."""

    return _CONSTRUCTIBLE


def ensure_source_constructible(kind: SourceKind) -> None:
    """Raise :class:`NotImplementedError` naming the owning phase for unsupported kinds."""

    if kind in _CONSTRUCTIBLE:
        return
    reason = _PLANNED_PHASE.get(kind, "not planned")
    raise NotImplementedError(
        f"SourceKind.{kind.name} is not constructible: {reason}."
    )
