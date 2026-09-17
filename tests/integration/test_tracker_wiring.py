"""Phase 9 section 14, integration tests 13/15/16 - the tracker wired into the
real-time loop (`AnalysisLoop._iterate`, exercised directly, no threads, no
camera - same style as `tests/unit/test_staleness_guard.py`).

A minimal duck-typed ``_FakePerception`` stands in for the real ONNX engine so
these tests stay fast and need no model weights; ``PerceptionResult`` is a
plain dataclass and `AnalysisLoop` only ever calls ``.infer(frame)`` /
``.info()`` on it. Real-model wiring (recorded-mode determinism with the
actual detector) is covered separately in
``tests/integration/test_recorded_determinism_with_models.py``
(``-m models``).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from predictivesense.camera.mailbox import LatestFrameMailbox
from predictivesense.camera.source import create_frame_source
from predictivesense.config.settings import load_config
from predictivesense.core.enums import TrackStatus
from predictivesense.core.types import Detection, Frame, Track
from predictivesense.perception.types import PerceptionResult
from predictivesense.pipeline.loop import AnalysisLoop
from predictivesense.telemetry.metrics import MetricRegistry
from predictivesense.tracking import Tracker

pytestmark = pytest.mark.integration

_W, _H = 640, 480


class _FakePerception:
    """Duck-typed stand-in for PerceptionEngine: returns a fixed detection set
    on every call, no inference, no model weights."""

    def __init__(self, detections: tuple[Detection, ...] = ()) -> None:
        self.detections = detections
        self.calls = 0

    def infer(self, frame: Frame, *, frame_index: int | None = None) -> PerceptionResult:
        self.calls += 1
        return PerceptionResult(detections=self.detections, poses=())

    def info(self) -> dict:
        return {"detector_ep": "fake", "detector_warmup_ms": 0.0, "pose_warmup_ms": -1.0}


def _det(bbox, *, frame_id=0, score=0.9) -> Detection:
    return Detection(
        bbox=bbox, class_id=0, class_name="cup", score=score, frame_id=frame_id,
        raw_class_name="cup", policy_state="accepted", tier="primary",
    )


def _loop(*, max_frame_age_ms: float, detections=(), tracking_overrides=None) -> AnalysisLoop:
    cfg = load_config("dev")
    tcfg = cfg.tracking.model_copy(update=tracking_overrides or {})
    cfg = cfg.model_copy(update={
        "perception": cfg.perception.model_copy(
            update={"detection_enabled": False, "pose_enabled": False}
        ),
        "analysis": cfg.analysis.model_copy(update={"max_frame_age_ms": max_frame_age_ms}),
        "tracking": tcfg,
        # Phase 12: never auto-resolve whatever is active in the real,
        # developer-machine production classifier registry - this test does
        # not pass classifier= explicitly, so AnalysisLoop would otherwise
        # load and run a REAL ONNX classifier here, breaking these timing
        # assertions (docs/decisions.md Phase 11 Part B's test-isolation gap).
        "training": cfg.training.model_copy(
            update={"classifier_registry_path": Path("__no_classifier_registry_for_tests__.json")}
        ),
    })
    src = create_frame_source(cfg.source)
    perception = _FakePerception(detections)
    tracker = Tracker(tcfg)
    return AnalysisLoop(
        cfg, src, LatestFrameMailbox(), registry=MetricRegistry(),
        perception=perception, tracker=tracker,
    )


def _frame(fid: int, age_ms: float, *, capture_ts: float | None = None) -> Frame:
    ts = capture_ts if capture_ts is not None else time.monotonic() - age_ms / 1000.0
    return Frame(
        frame_id=fid, capture_ts=ts, image=np.zeros((4, 4, 3), np.uint8),
        width=_W, height=_H, source_id="t", seq=fid,
    )


# -- 13. wired into real-time; StateSnapshot.tracks populated ---------------

def test_tracks_populated_in_a_real_time_snapshot() -> None:
    loop = _loop(max_frame_age_ms=0.0, detections=(_det((100, 100, 140, 140)),),
                 tracking_overrides={"n_init": 1})
    loop._iterate(_frame(0, age_ms=0.0))
    snap = loop.latest
    assert snap is not None
    assert len(snap.tracks) == 1
    assert snap.tracks[0].class_name == "cup"
    assert snap.tracks[0].fresh is True


# -- 15. never tracks a frame that failed the staleness guard ---------------

def test_tracker_never_updates_on_a_guard_dropped_frame() -> None:
    loop = _loop(max_frame_age_ms=200.0, detections=(),
                 tracking_overrides={"n_init": 1, "max_age_frames": 50, "max_age_ms": 50_000.0})
    tracker = loop._tracker
    # Seed one confirmed track directly (perception returns no detections, so
    # the loop itself never creates one) - capture_ts anchors "now" for age math.
    now = time.monotonic()
    tracker.update([_det((100, 100, 140, 140))], frame_id=-1, capture_ts=now,
                    frame_width=_W, frame_height=_H)
    assert tracker.active_track_count == 1
    misses_before = next(iter(tracker._tracks.values())).consecutive_misses
    assert misses_before == 0

    # A guard-dropped (stale) frame: must NOT reach the tracker at all.
    # age_ms is computed relative to _frame()'s own time.monotonic() call, an
    # instant before _iterate reads the clock again - old enough to clear the
    # 200ms guard with margin against that small inter-call gap.
    loop._iterate(_frame(0, age_ms=350.0))
    misses_after_stale = next(iter(tracker._tracks.values())).consecutive_misses
    assert misses_after_stale == misses_before  # unchanged - never called

    # A genuinely fresh frame: the loop's own clock drives capture_ts, so just
    # assert the tracker WAS invoked this time (misses advances by exactly 1,
    # proving the prior stale cycle contributed zero calls, not a silent no-op).
    loop._iterate(_frame(1, age_ms=0.0))
    misses_after_fresh = next(iter(tracker._tracks.values())).consecutive_misses
    assert misses_after_fresh == misses_before + 1


# -- 16. tracker_update_ms measured and bounded ------------------------------

def test_tracker_update_ms_is_measured_and_bounded() -> None:
    dets = tuple(_det((50 + i * 60, 100, 90 + i * 60, 140)) for i in range(8))
    loop = _loop(max_frame_age_ms=0.0, detections=dets, tracking_overrides={"n_init": 1})
    ts = 0.0
    for i in range(30):
        loop._iterate(_frame(i, age_ms=0.0, capture_ts=ts))
        ts += 0.05

    samples = loop.metrics.samples("tracker_update_ms")
    assert samples.count == 30
    # Coarse regression guard, not a tight perf budget: a pure-Python tracker
    # over a handful of objects must stay well under this on any dev machine.
    assert samples.p95 < 50.0
    snap = loop.latest
    assert snap is not None
    assert snap.metrics.get("tracker_update_ms", -1.0) >= 0.0
    assert len(snap.tracks) == len(dets)


# -- Phase 11 Part A section 4: displayed_pose_age_ms / displayed_track_age_ms

class _FakeTracker:
    """Duck-typed stand-in returning a fixed, hand-built ``Track`` list -
    isolates ``AnalysisLoop``'s own age computation from real association/
    pose-binding logic (which is covered separately in
    ``tests/unit/test_tracking.py``)."""

    def __init__(self, tracks: tuple[Track, ...]) -> None:
        self._tracks = tracks

    def update(self, detections, *, frame_id, capture_ts, frame_width, frame_height,
               poses=(), pose_fresh=True):
        return list(self._tracks)


def _track(*, last_detection_capture_ts: float, pose_capture_ts: float | None) -> Track:
    return Track(
        track_id=1, status=TrackStatus.CONFIRMED, class_name="person",
        bbox=(0.0, 0.0, 10.0, 10.0), last_seen_frame_id=0,
        last_detection_capture_ts=last_detection_capture_ts,
        pose_capture_ts=pose_capture_ts,
        pose_keypoints=((1.0, 1.0, 0.9),) if pose_capture_ts is not None else (),
    )


def _loop_with_fake_tracker(tracks: tuple[Track, ...]) -> AnalysisLoop:
    cfg = load_config("dev")
    cfg = cfg.model_copy(update={
        "perception": cfg.perception.model_copy(
            update={"detection_enabled": False, "pose_enabled": False}
        ),
        "analysis": cfg.analysis.model_copy(update={"max_frame_age_ms": 0.0}),
        "training": cfg.training.model_copy(
            update={"classifier_registry_path": Path("__no_classifier_registry_for_tests__.json")}
        ),
    })
    src = create_frame_source(cfg.source)
    perception = _FakePerception(())
    return AnalysisLoop(
        cfg, src, LatestFrameMailbox(), registry=MetricRegistry(),
        perception=perception, tracker=_FakeTracker(tracks),
    )


def test_displayed_track_and_pose_age_measure_real_elapsed_time() -> None:
    # A track whose last real hit and last pose binding are each a known
    # elapsed time behind "now" - displayed_*_age_ms must report that real
    # elapsed time (emitted_ts - the field), not this frame's own age.
    now = time.monotonic()
    track = _track(
        last_detection_capture_ts=now - 0.300,  # last real detector hit: 300ms ago
        pose_capture_ts=now - 0.750,            # pose measurement: 750ms ago
    )
    loop = _loop_with_fake_tracker((track,))
    loop._iterate(_frame(0, age_ms=0.0))
    snap = loop.latest
    assert snap is not None
    assert snap.metrics["displayed_track_age_ms"] == pytest.approx(300.0, abs=40.0)
    assert snap.metrics["displayed_pose_age_ms"] == pytest.approx(750.0, abs=40.0)
    # And both are folded into the existing telemetry, not a parallel system.
    assert loop.metrics.samples("displayed_track_age_ms").count == 1
    assert loop.metrics.samples("displayed_pose_age_ms").count == 1


def test_displayed_pose_age_ms_is_absent_when_no_track_carries_a_pose() -> None:
    now = time.monotonic()
    track = _track(last_detection_capture_ts=now, pose_capture_ts=None)
    loop = _loop_with_fake_tracker((track,))
    loop._iterate(_frame(0, age_ms=0.0))
    snap = loop.latest
    assert snap is not None
    assert snap.metrics["displayed_pose_age_ms"] == -1.0
    assert snap.metrics["displayed_track_age_ms"] >= 0.0


def test_displayed_ages_absent_with_no_tracks_at_all() -> None:
    loop = _loop_with_fake_tracker(())
    loop._iterate(_frame(0, age_ms=0.0))
    snap = loop.latest
    assert snap is not None
    assert snap.metrics["displayed_pose_age_ms"] == -1.0
    assert snap.metrics["displayed_track_age_ms"] == -1.0
