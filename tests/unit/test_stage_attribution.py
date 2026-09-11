"""Phase 8 - per-frame latency stage attribution (`telemetry/stages.py`)."""

from __future__ import annotations

import pytest

from predictivesense.core.types import FrameTrace
from predictivesense.telemetry.metrics import MetricRegistry
from predictivesense.telemetry.stages import (
    ALL_STAGE_NAMES,
    SERVER_STAGE_NAMES,
    record_stages,
    stage_metric_key,
)

pytestmark = pytest.mark.unit


def _full_trace() -> FrameTrace:
    """A trace with every server-side stage endpoint stamped, monotonic order."""

    t = FrameTrace()
    t.capture_client_ms = 1_000_000.0
    t.worker_encode_ms = 4.0
    t.capture_ts = 100.000
    t.send_ts = 100.010          # +10 ms worker encode + queue
    t.recv_ts = 100.013          # +3 ms ws transit
    t.decode_ms = 2.5
    t.src_enqueue_ts = 100.014
    t.src_dequeue_ts = 100.015   # +1 ms browser-buffer dwell
    t.mbox_enqueue_ts = 100.0155
    t.mbox_dequeue_ts = 100.180  # +164.5 ms mailbox dwell
    t.detector_ms = 70.0
    t.pose_ms = 55.0
    t.pose_reused = False
    t.policy_ms = 0.1
    t.snapshot_ts = 100.306      # dequeue + 125.1 ms model + ~0.9 ms build
    t.frame_age_at_dequeue_ms = 180.0
    return t


def test_record_stages_folds_every_server_stage_into_the_registry() -> None:
    reg = MetricRegistry()
    stages = record_stages(_full_trace(), reg, frame_id=7)

    assert stages.frame_id == 7
    assert stages.missing_server_stages() == []
    for name in SERVER_STAGE_NAMES:
        s = reg.samples(stage_metric_key(name))
        assert s.count == 1, f"{name} not recorded"

    # deltas line up with the constructed trace
    assert stages.ws_transit_ms == pytest.approx(3.0, abs=0.5)
    assert stages.decode_ms == pytest.approx(2.5)
    assert stages.src_buffer_dwell_ms == pytest.approx(1.0, abs=0.2)
    assert stages.mailbox_dwell_ms == pytest.approx(164.5, abs=0.5)
    assert stages.detector_ms == pytest.approx(70.0)
    assert stages.pose_ms == pytest.approx(55.0)
    assert stages.snapshot_build_ms is not None and stages.snapshot_build_ms >= 0.0
    assert stages.snapshot_build_ms == pytest.approx(0.9, abs=0.3)
    assert stages.capture_to_snapshot_ms == pytest.approx(306.0, abs=1.0)


def test_reused_pose_is_zero_cost_in_the_pose_stage() -> None:
    reg = MetricRegistry()
    t = _full_trace()
    t.pose_reused = True
    stages = record_stages(t, reg, frame_id=1)
    assert stages.pose_ms == 0.0
    assert stages.pose_reused is True
    # the reused-pose frame still has no missing server stage
    assert "pose" not in stages.missing_server_stages()


def test_partial_trace_reports_the_missing_stages_not_a_crash() -> None:
    reg = MetricRegistry()
    t = FrameTrace()
    t.capture_ts = 10.0
    t.mbox_dequeue_ts = 10.2
    t.snapshot_ts = 10.3
    stages = record_stages(t, reg, frame_id=2)
    missing = stages.missing_server_stages()
    assert "worker_encode" in missing and "detector" in missing
    # capture_to_snapshot still computed from the two ends that exist
    assert stages.capture_to_snapshot_ms == pytest.approx(300.0, abs=1.0)


def test_negative_or_reversed_endpoints_floor_at_zero() -> None:
    reg = MetricRegistry()
    t = FrameTrace()
    t.src_enqueue_ts = 5.0
    t.src_dequeue_ts = 4.0  # clock skew / reordering -> must not go negative
    stages = record_stages(t, reg, frame_id=3)
    assert stages.src_buffer_dwell_ms == 0.0


def test_all_stage_names_superset_of_server_stage_names() -> None:
    assert set(SERVER_STAGE_NAMES).issubset(set(ALL_STAGE_NAMES))
    assert "ws_out" in ALL_STAGE_NAMES and "overlay_paint" in ALL_STAGE_NAMES
