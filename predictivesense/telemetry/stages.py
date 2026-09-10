"""End-to-end latency attribution (Phase 8).

One analysed frame's journey is stamped stage by stage on a mutable
:class:`~predictivesense.core.types.FrameTrace` that travels attached to the
:class:`~predictivesense.core.types.Frame`. At snapshot emission the analysis
loop calls :func:`record_stages`, which

* folds every stage delta into the **existing**
  :class:`~predictivesense.telemetry.metrics.MetricRegistry` (a ``Samples`` per
  stage named ``stage_<name>_ms`` - no parallel metrics system), and
* returns an immutable :class:`FrameStages` snapshot of that one frame so a test
  can assert "no missing stage" and a benchmark can dump a per-frame record.

Stages, in order, and the two endpoints whose difference defines each:

    worker_encode     browser: JPEG encode duration (from the ingest header)
    ws_transit        send_ts            -> recv_ts
    decode            server: cv2.imdecode duration
    src_buffer_dwell  src_enqueue_ts     -> src_dequeue_ts   (BrowserSource slot)
    producer_handoff  src_dequeue_ts     -> mbox_enqueue_ts
    mailbox_dwell     mbox_enqueue_ts    -> mbox_dequeue_ts   (LatestFrameMailbox)
    detector          server: detector inference duration
    pose              server: pose inference duration (0.0 when reused)
    policy            server: recognition-policy duration
    snapshot_build    (mbox_dequeue_ts + model time) -> snapshot_ts

``ws_out`` (emit -> WebSocket send) is recorded by the broadcaster into the same
registry; ``overlay_paint`` (capture -> canvas paint) is a browser-side number
returned through ``POST /api/metrics/browser``. Neither can be stamped on the
server-side trace, so they are attributed separately and the report says so.
"""

from __future__ import annotations

from dataclasses import dataclass

from predictivesense.core.types import FrameTrace
from predictivesense.telemetry.metrics import MetricRegistry

__all__ = [
    "SERVER_STAGE_NAMES",
    "ALL_STAGE_NAMES",
    "FrameStages",
    "record_stages",
    "stage_metric_key",
]

# Stages the server-side FrameTrace can attribute on its own.
SERVER_STAGE_NAMES: tuple[str, ...] = (
    "worker_encode",
    "ws_transit",
    "decode",
    "src_buffer_dwell",
    "producer_handoff",
    "mailbox_dwell",
    "detector",
    "pose",
    "policy",
    "snapshot_build",
)
# Full end-to-end chain, including the two stages attributed elsewhere.
ALL_STAGE_NAMES: tuple[str, ...] = SERVER_STAGE_NAMES + ("ws_out", "overlay_paint")

_LATENCY_WINDOW = 4096  # per-stage reservoir; bounded like the rest of telemetry


def stage_metric_key(name: str) -> str:
    """Registry ``Samples`` name for one stage (``stage_detector_ms`` ...)."""

    return f"stage_{name}_ms"


@dataclass(frozen=True)
class FrameStages:
    """An immutable per-frame attribution record. ``ms`` values; ``None`` = the
    stage was not reached / not stamped for this frame."""

    frame_id: int
    capture_to_snapshot_ms: float | None
    worker_encode_ms: float | None
    ws_transit_ms: float | None
    decode_ms: float | None
    src_buffer_dwell_ms: float | None
    producer_handoff_ms: float | None
    mailbox_dwell_ms: float | None
    detector_ms: float | None
    pose_ms: float | None
    pose_reused: bool
    policy_ms: float | None
    snapshot_build_ms: float | None
    frame_age_at_dequeue_ms: float | None

    def as_dict(self) -> dict[str, float | int | bool | None]:
        return {
            "frame_id": self.frame_id,
            "capture_to_snapshot_ms": self.capture_to_snapshot_ms,
            "worker_encode_ms": self.worker_encode_ms,
            "ws_transit_ms": self.ws_transit_ms,
            "decode_ms": self.decode_ms,
            "src_buffer_dwell_ms": self.src_buffer_dwell_ms,
            "producer_handoff_ms": self.producer_handoff_ms,
            "mailbox_dwell_ms": self.mailbox_dwell_ms,
            "detector_ms": self.detector_ms,
            "pose_ms": self.pose_ms,
            "pose_reused": self.pose_reused,
            "policy_ms": self.policy_ms,
            "snapshot_build_ms": self.snapshot_build_ms,
            "frame_age_at_dequeue_ms": self.frame_age_at_dequeue_ms,
        }

    def missing_server_stages(self) -> list[str]:
        """Server stages with no value for this frame (used by the 'no missing
        stage' integration test - a fully instrumented browser frame returns
        ``[]``)."""

        got = {
            "worker_encode": self.worker_encode_ms,
            "ws_transit": self.ws_transit_ms,
            "decode": self.decode_ms,
            "src_buffer_dwell": self.src_buffer_dwell_ms,
            "producer_handoff": self.producer_handoff_ms,
            "mailbox_dwell": self.mailbox_dwell_ms,
            "detector": self.detector_ms,
            "pose": self.pose_ms,
            "policy": self.policy_ms,
            "snapshot_build": self.snapshot_build_ms,
        }
        return [name for name, value in got.items() if value is None]


def _ms(a: float | None, b: float | None) -> float | None:
    """``(b - a) * 1000`` when both endpoints exist and are ordered, else None."""

    if a is None or b is None:
        return None
    delta = (b - a) * 1000.0
    return delta if delta >= 0.0 else 0.0


def record_stages(
    trace: FrameTrace, registry: MetricRegistry, *, frame_id: int
) -> FrameStages:
    """Fold ``trace`` into ``registry`` and return the frozen per-frame record."""

    worker_encode = trace.worker_encode_ms
    ws_transit = _ms(trace.send_ts, trace.recv_ts)
    decode = trace.decode_ms
    src_buffer_dwell = _ms(trace.src_enqueue_ts, trace.src_dequeue_ts)
    producer_handoff = _ms(trace.src_dequeue_ts, trace.mbox_enqueue_ts)
    mailbox_dwell = _ms(trace.mbox_enqueue_ts, trace.mbox_dequeue_ts)
    detector = trace.detector_ms
    pose = 0.0 if trace.pose_reused else trace.pose_ms
    policy = trace.policy_ms

    model_ms = sum(v for v in (detector, trace.pose_ms, policy) if v is not None)
    snapshot_build: float | None = None
    if trace.mbox_dequeue_ts is not None and trace.snapshot_ts is not None:
        span_ms = (trace.snapshot_ts - trace.mbox_dequeue_ts) * 1000.0
        snapshot_build = max(0.0, span_ms - model_ms)

    capture_to_snapshot = _ms(trace.capture_ts, trace.snapshot_ts)

    pairs = {
        "worker_encode": worker_encode,
        "ws_transit": ws_transit,
        "decode": decode,
        "src_buffer_dwell": src_buffer_dwell,
        "producer_handoff": producer_handoff,
        "mailbox_dwell": mailbox_dwell,
        "detector": detector,
        "pose": pose,
        "policy": policy,
        "snapshot_build": snapshot_build,
    }
    for name, value in pairs.items():
        if value is not None:
            registry.samples(
                stage_metric_key(name), maxlen=_LATENCY_WINDOW
            ).add(float(value))
    if capture_to_snapshot is not None:
        registry.samples(
            "stage_capture_to_snapshot_ms", maxlen=_LATENCY_WINDOW
        ).add(capture_to_snapshot)

    return FrameStages(
        frame_id=frame_id,
        capture_to_snapshot_ms=capture_to_snapshot,
        worker_encode_ms=worker_encode,
        ws_transit_ms=ws_transit,
        decode_ms=decode,
        src_buffer_dwell_ms=src_buffer_dwell,
        producer_handoff_ms=producer_handoff,
        mailbox_dwell_ms=mailbox_dwell,
        detector_ms=detector,
        pose_ms=pose,
        pose_reused=trace.pose_reused,
        policy_ms=policy,
        snapshot_build_ms=snapshot_build,
        frame_age_at_dequeue_ms=trace.frame_age_at_dequeue_ms,
    )
