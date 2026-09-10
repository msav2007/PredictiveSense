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

from predictivesense.core.enums import Mode, RiskLevel, SourceKind, TrackStatus

__all__ = [
    "BBox",
    "Frame",
    "FrameTrace",
    "Detection",
    "Pose",
    "Track",
    "Relation",
    "RiskState",
    "Alert",
    "StateSnapshot",
    "PerceptionFrame",
    "MailboxStats",
    "SourceInfo",
    "ClipManifest",
    "IngestHeader",
]

# (x1, y1, x2, y2) in pixels.
BBox = tuple[float, float, float, float]


class FrameTrace:
    """Per-frame stage stamps for end-to-end latency attribution (Phase 8).

    Deliberately **not** a frozen contract: it is a mutable scratch object that
    travels attached to one :class:`Frame` and is written once by each pipeline
    stage as that frame passes through it (worker header -> ingest receive ->
    JPEG decode -> browser-source slot -> mailbox -> perception -> snapshot).
    The analysis loop reads it once at snapshot emission, folds each stage delta
    into the existing :class:`~predictivesense.telemetry.metrics.MetricRegistry`
    samples, and freezes the result into an immutable
    :class:`~predictivesense.telemetry.stages.FrameStages` record. Nothing here
    is serialised over the wire; ``Frame.trace`` defaults to ``None`` so every
    non-browser source (synthetic, device, file) is completely unaffected.

    All ``*_ts`` values are ``time.monotonic()`` seconds on the server timeline.
    All ``*_ms`` values are millisecond durations. ``capture_client_ms`` /
    ``send_client_ms`` are the browser's own ``performance`` epoch-ms clock, kept
    raw so a pure client-clock capture->paint age can be computed in the page
    without any cross-clock subtraction.
    """

    __slots__ = (
        "capture_client_ms",
        "send_client_ms",
        "worker_encode_ms",
        "capture_ts",
        "send_ts",
        "recv_ts",
        "decode_ms",
        "src_enqueue_ts",
        "src_dequeue_ts",
        "mbox_enqueue_ts",
        "mbox_dequeue_ts",
        "detector_ms",
        "pose_ms",
        "pose_reused",
        "policy_ms",
        "snapshot_ts",
        "frame_age_at_dequeue_ms",
    )

    def __init__(self) -> None:
        self.capture_client_ms: float | None = None
        self.send_client_ms: float | None = None
        self.worker_encode_ms: float | None = None
        self.capture_ts: float | None = None
        self.send_ts: float | None = None
        self.recv_ts: float | None = None
        self.decode_ms: float | None = None
        self.src_enqueue_ts: float | None = None
        self.src_dequeue_ts: float | None = None
        self.mbox_enqueue_ts: float | None = None
        self.mbox_dequeue_ts: float | None = None
        self.detector_ms: float | None = None
        self.pose_ms: float | None = None
        self.pose_reused: bool = False
        self.policy_ms: float | None = None
        self.snapshot_ts: float | None = None
        self.frame_age_at_dequeue_ms: float | None = None


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
    # Phase 8: optional per-frame latency-attribution scratch object. Mutable,
    # never serialised, populated only on the browser-ingest path. Default None
    # so synthetic / device / file sources and every existing construction site
    # are unaffected. See docs/decisions.md (Phase 8).
    trace: FrameTrace | None = Field(default=None, exclude=True, repr=False)


class Detection(_Frozen):
    """A single object detection.

    Phase 2.5 added four fields **additively** (Phase 0 rule; recorded in
    ``docs/decisions.md``) so the recognition policy layer can annotate what the
    model said versus what the policy decided without a rename or reshape:

    * ``raw_class_name`` - the class the model emitted (mirrors ``class_name``
      until the policy relabels it).
    * ``class_name`` - unchanged field; after the policy it is what the policy
      decided, possibly ``"unknown"``.
    * ``policy_state`` - Phase 5 six-value set: ``accepted`` |
      ``accepted_secondary`` | ``unknown_low_confidence`` | ``unknown_margin`` |
      ``suppressed_implausible`` | ``rejected_size``. (Phase 2.5's
      ``rejected_out_of_domain`` is renamed ``suppressed_implausible`` - the
      model *did* recognise a known class, we chose not to surface it - recorded
      in ``docs/decisions.md``.)
    * ``runner_up`` - ``(class_name, score)`` of the second-best class for this
      detection's anchor, or ``None`` when the model does not expose it.
    * ``tier`` - Phase 5 three-tier vocabulary bucket the raw class falls in:
      ``primary`` | ``secondary`` | ``implausible`` | ``unlisted``. Additive;
      recorded in ``docs/decisions.md``.

    Defaults keep every pre-2.5 construction site valid.
    """

    bbox: BBox
    class_id: int
    class_name: str
    score: float
    frame_id: int
    raw_class_name: str = ""
    policy_state: str = "accepted"
    runner_up: tuple[str, float] | None = None
    tier: str = "primary"


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
        description=(
            "Free-form string->float gauges. Phase 0: loop_rate_hz, "
            "producer_rate_hz, consumed, dropped, mailbox_depth, iter_latency_ms. "
            "Phase 1 adds keys only (no reshape): capture_fps, analysis_fps, "
            "dropped_analysis_frames, drop_rate, frame_age_ms, decode_ms, "
            "ingest_bytes_per_s, reconnects, clock_offset_rtt_ms."
        )
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


class PerceptionFrame(_Frozen):
    """One frame's complete perception output as a single immutable record
    (Phase 8, section 16 - **tracking-ready output, no tracker**).

    A future tracker / relationship layer consumes ``PerceptionFrame`` values
    rather than reassembling parallel ``detections`` / ``poses`` lists off the
    snapshot. Everything a tracker needs to key on is here: a stable per-frame
    identity (``frame_id`` + monotonic ``seq``), the capture clock, the frame
    pixel dimensions the boxes live in, the active model version and execution
    provider, and - per detection - class name, raw class, confidence, box in
    **original frame pixels**, ``policy_state`` and ``tier`` (already on
    :class:`Detection`). ``detections`` and ``poses`` are ordered deterministically
    (``-score``, then ``class_name``, then box top-left) so two runs agree.

    Real-time (`pipeline/loop.py`) and recorded (`pipeline/recorded.py`) modes
    build this with the identical helper, so the structure is byte-identical
    between them. This adds no relationship field that nothing populates and
    invents no track id - ``StateSnapshot.tracks`` stays empty.
    """

    frame_id: int
    seq: int = Field(description="Monotonic frame sequence within the source session.")
    capture_ts: float
    width: int
    height: int
    model_version: str
    provider: str = Field(description="ONNX Runtime EP actually in use this session.")
    detections: tuple[Detection, ...] = ()
    poses: tuple[Pose, ...] = ()
    pose_stale: bool = Field(
        default=False,
        description="True when `poses` were reused from an earlier frame (section 9).",
    )
    pose_frame_id: int | None = Field(
        default=None, description="frame_id the reused pose actually came from."
    )
    pose_capture_ts: float | None = None
    pose_age_ms: float | None = Field(
        default=None, description="capture_ts - pose_capture_ts, in ms, when reused."
    )


class MailboxStats(_Frozen):
    """A point-in-time view of a :class:`~predictivesense.camera.mailbox` counter set."""

    consumed: int
    dropped: int
    depth: int


class SourceInfo(_Frozen):
    """Static + measured description of a frame source (Phase 1).

    ``width``/``height``/``achieved_fps`` are what the device or file *actually*
    returned, never the requested values.
    """

    kind: SourceKind
    source_id: str
    label: str
    width: int
    height: int
    achieved_fps: float | None = None
    backend: str | None = None
    extra: dict[str, str] = Field(default_factory=dict)


class ClipManifest(_Frozen):
    """Sidecar metadata for one raw clip recorded to ``data/raw/`` (Phase 1)."""

    clip_id: str
    session_id: str
    path: str
    scenario_tag: str
    device_label: str
    width: int
    height: int
    nominal_fps: float
    duration_s: float | None
    size_bytes: int
    recorded_utc: str
    git_commit: str
    config_profile: str
    consent_ack: bool
    notes: str


class IngestHeader(_Frozen):
    """The JSON header of one binary ``WS /ws/ingest`` analysis-frame message.

    Phase 8 added two **additive, optional** stage-attribution fields:
    ``cap_ts_ms`` (the worker's ``performance`` epoch-ms clock at the moment it
    drew the frame into the analysis canvas, i.e. the true capture instant) and
    ``enc_ms`` (the worker's own JPEG encode duration). ``client_ts_ms`` keeps
    its meaning - the epoch-ms clock at message send. Both new fields default to
    ``None`` so an old worker, the framing tests, and every non-browser caller
    are valid unchanged.
    """

    client_ts_ms: float
    seq: int
    w: int
    h: int
    cap_ts_ms: float | None = None
    enc_ms: float | None = None
