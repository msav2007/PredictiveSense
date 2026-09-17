"""Phase 9 multi-object tracker.

Pure, dependency-free (stdlib + numpy only - no third-party tracking
library): consumes a frame's already policy-annotated
:class:`~predictivesense.core.types.Detection` list and produces
:class:`~predictivesense.core.types.Track` objects. No inference, no I/O, no
global state - one :class:`Tracker` instance per analysis run, held by the
caller (the real-time loop or the recorded driver), exactly like
:class:`~predictivesense.perception.policy.RecognitionPolicy`.

See ``docs/architecture.md`` for the lifecycle diagram and the state->visual
mapping table, and ``docs/decisions.md`` for the algorithm choice and every
threshold's derivation.
"""

from __future__ import annotations

from predictivesense.tracking.tracker import TRACK_ELIGIBLE_STATES, Tracker, TrackerStats

__all__ = ["Tracker", "TrackerStats", "TRACK_ELIGIBLE_STATES"]
