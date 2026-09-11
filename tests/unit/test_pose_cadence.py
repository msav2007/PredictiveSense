"""Phase 8 section 9 - pose cadence decoupling + immutable pose reuse.

`PerceptionEngine` is exercised with a fake pose model (no ONNX), so this is a
`unit` test: it checks the cadence decision, the reuse bound, and that a reused
pose is the *same immutable object* (never mutated, never rebuilt).
"""

from __future__ import annotations

import numpy as np
import pytest

from predictivesense.config.settings import PerceptionConfig
from predictivesense.core.types import Frame, Pose
from predictivesense.perception.engine import PerceptionEngine

pytestmark = pytest.mark.unit


class _FakeDetector:
    def infer(self, frame):
        return []


class _FakePose:
    """Returns one distinct Pose per call so reuse is detectable by identity."""

    def __init__(self) -> None:
        self.calls = 0

    def infer(self, frame):
        self.calls += 1
        return [
            Pose(
                keypoints=tuple((float(self.calls), 0.0, 1.0) for _ in range(17)),
                bbox=(0.0, 0.0, 10.0, 10.0),
                score=0.9,
                frame_id=frame.frame_id,
            )
        ]


def _engine(**overrides) -> tuple[PerceptionEngine, _FakePose]:
    cfg = PerceptionConfig(**overrides)
    eng = PerceptionEngine.__new__(PerceptionEngine)
    # bypass __init__'s ONNX session build; set only what infer() touches
    eng._config = cfg
    eng._detector = _FakeDetector()
    fake_pose = _FakePose()
    eng._pose = fake_pose
    eng._frame_counter = -1
    eng._detector_failures = 0
    eng._pose_failures = 0
    eng._pose_gated_skips = 0
    eng._pose_cadence_kind, eng._pose_cadence_value = cfg.pose_cadence_spec()
    eng._pose_max_reuse_ms = float(cfg.pose_max_reuse_ms)
    eng._last_poses = ()
    eng._last_pose_frame_id = None
    eng._last_pose_capture_ts = None
    eng._pose_reuses = 0
    return eng, fake_pose


def _frame(i: int, ts: float) -> Frame:
    return Frame(
        frame_id=i, capture_ts=ts, image=np.zeros((4, 4, 3), np.uint8),
        width=4, height=4, source_id="t", seq=i,
    )


def test_every_frame_runs_pose_every_time_no_reuse() -> None:
    eng, fake = _engine(pose_cadence="every_frame")
    for i in range(5):
        r = eng.infer(_frame(i, i * 0.1))
        assert r.pose_reused is False
        assert r.pose_ms is not None
    assert fake.calls == 5


def test_every_n_reuses_the_same_object_between_inferences() -> None:
    eng, fake = _engine(pose_cadence="every_n:3", pose_max_reuse_ms=1000.0)
    results = [eng.infer(_frame(i, i * 0.05)) for i in range(7)]
    # frames 0, 3, 6 ran pose; 1,2,4,5 reused
    assert fake.calls == 3
    assert [r.pose_reused for r in results] == [False, True, True, False, True, True, False]
    fresh_pose = results[0].poses[0]
    assert results[1].poses[0] is fresh_pose  # identity: not rebuilt, not copied
    assert results[2].poses[0] is fresh_pose
    # reused frames carry the source pose's provenance
    assert results[1].pose_frame_id == 0
    assert results[1].pose_capture_ts == pytest.approx(0.0)
    assert results[1].pose_age_ms == pytest.approx(50.0, abs=1.0)  # frame 1 @ 0.05s
    # the reused Pose object itself is unchanged (frozen contract)
    assert fresh_pose.frame_id == 0


def test_reuse_is_bounded_by_pose_max_reuse_ms() -> None:
    eng, fake = _engine(pose_cadence="every_n:10", pose_max_reuse_ms=120.0)
    eng.infer(_frame(0, 0.0))            # pose runs
    r_ok = eng.infer(_frame(1, 0.10))   # 100 ms later -> reuse
    r_stale = eng.infer(_frame(2, 0.30))  # 300 ms later -> beyond bound -> no pose
    assert r_ok.pose_reused is True and len(r_ok.poses) == 1
    assert r_stale.pose_reused is False and r_stale.poses == ()
    assert fake.calls == 1


def test_interval_ms_cadence_uses_capture_time_not_wall_time() -> None:
    eng, fake = _engine(pose_cadence="interval_ms:200", pose_max_reuse_ms=1000.0)
    # capture timestamps 0, 100, 150, 210, 260 ms
    stamps = [0.0, 0.10, 0.15, 0.21, 0.26]
    results = [eng.infer(_frame(i, ts)) for i, ts in enumerate(stamps)]
    ran = [not r.pose_reused for r in results]
    # pose runs at t=0 (first), then not again until >=200 ms of capture time
    # has elapsed since the last run: t=0.21 (>=0.20) runs; 0.10/0.15 reuse;
    # 0.26 is only 50 ms after 0.21 -> reuse.
    assert ran == [True, False, False, True, False]
    assert fake.calls == 2


def test_legacy_pose_every_n_still_honoured_when_cadence_is_default() -> None:
    eng, fake = _engine(pose_every_n=2)  # pose_cadence left "every_frame"
    kind, value = eng._config.pose_cadence_spec()
    assert (kind, value) == ("every_n", 2.0)
    results = [eng.infer(_frame(i, i * 0.05)) for i in range(4)]
    assert [r.pose_reused for r in results] == [False, True, False, True]
