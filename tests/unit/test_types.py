"""Contracts: constructible, immutable, and JSON round-tripping."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from predictivesense.core.enums import Mode, RiskLevel, TrackStatus
from predictivesense.core.types import (
    Alert,
    Detection,
    Frame,
    MailboxStats,
    Pose,
    Relation,
    RiskState,
    StateSnapshot,
    Track,
)

pytestmark = pytest.mark.unit


def _frame() -> Frame:
    return Frame(
        frame_id=0,
        capture_ts=1.5,
        image=np.zeros((4, 6, 3), dtype=np.uint8),
        width=6,
        height=4,
        source_id="s",
        seq=0,
    )


def test_every_contract_constructible_with_explicit_types() -> None:
    frame = _frame()
    assert frame.image.shape == (4, 6, 3)

    det = Detection(bbox=(1.0, 2.0, 3.0, 4.0), class_id=1, class_name="cup", score=0.9, frame_id=0)
    pose = Pose(keypoints=((0.0, 0.0, 1.0), (1.0, 1.0, 0.5)), bbox=(0.0, 0.0, 1.0, 1.0), score=0.7, frame_id=0)
    track = Track(track_id=3, status=TrackStatus.CONFIRMED, class_name="person", bbox=(0.0, 0.0, 2.0, 2.0), last_seen_frame_id=0)
    rel = Relation(subject_id=1, object_id=2, kind="near", value=0.4)
    risk = RiskState(scenario="tip_over", level=RiskLevel.WARNING, confidence=0.6)
    alert = Alert(alert_id=1, level=RiskLevel.HIGH_RISK, text="move it", recommendation="step back", issued_ts=2.0)
    stats = MailboxStats(consumed=5, dropped=2, depth=1)

    assert det.class_name == "cup"
    assert pose.keypoints[0] == (0.0, 0.0, 1.0)
    assert track.status is TrackStatus.CONFIRMED
    assert rel.kind == "near"
    assert risk.level is RiskLevel.WARNING
    assert alert.recommendation == "step back"
    assert stats.depth == 1


@pytest.mark.parametrize("field,value", [("frame_id", 9), ("width", 1), ("source_id", "x")])
def test_frame_is_immutable(field: str, value: object) -> None:
    frame = _frame()
    with pytest.raises(ValidationError):
        setattr(frame, field, value)


def test_contracts_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Detection(bbox=(0, 0, 1, 1), class_id=0, class_name="x", score=1.0, frame_id=0, extra=1)


def _snapshot(**overrides) -> StateSnapshot:
    base = dict(
        snapshot_id=1,
        mode=Mode.REALTIME,
        frame_id=10,
        capture_ts=100.0,
        emitted_ts=100.25,
        frame_age_ms=250.0,
        detections=[],
        poses=[],
        tracks=[],
        risk=None,
        metrics={"loop_rate_hz": 20.0, "dropped": 3.0, "consumed": 40.0, "iter_latency_ms": 0.4},
        stale=False,
    )
    base.update(overrides)
    return StateSnapshot(**base)


def test_state_snapshot_round_trips_to_json_identically() -> None:
    snap = _snapshot()
    restored = StateSnapshot.from_wire_json(snap.to_wire_json())
    assert restored == snap
    assert restored.mode is Mode.REALTIME
    assert restored.metrics == snap.metrics


def test_state_snapshot_round_trips_with_populated_collections() -> None:
    snap = _snapshot(
        frame_id=None,
        capture_ts=None,
        frame_age_ms=None,
        stale=True,
        detections=[Detection(bbox=(0.0, 0.0, 1.0, 1.0), class_id=2, class_name="box", score=0.5, frame_id=7)],
        tracks=[Track(track_id=1, status=TrackStatus.TENTATIVE, class_name="box", bbox=(0.0, 0.0, 1.0, 1.0), last_seen_frame_id=7)],
        risk=RiskState(scenario="s", level=RiskLevel.UNKNOWN, confidence=0.0),
    )
    restored = StateSnapshot.from_wire_json(snap.to_wire_json())
    assert restored == snap
    assert restored.detections[0].class_name == "box"
    assert restored.risk is not None and restored.risk.level is RiskLevel.UNKNOWN


def test_state_snapshot_is_immutable() -> None:
    snap = _snapshot()
    with pytest.raises(ValidationError):
        snap.snapshot_id = 2
