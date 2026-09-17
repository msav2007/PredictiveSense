"""Phase 11 Part B section 16 - the active custom classifier wired into the
real-time loop as a SEPARATE, additive pass. A fake, duck-typed classifier
stands in for the real ONNX one (same style as `_FakePerception` in
`test_tracker_wiring.py`) so this stays fast and needs no trained artifact -
the real ONNX path is covered by `tests/integration/test_training_pipeline.py`
(``-m train``).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from predictivesense.camera.mailbox import LatestFrameMailbox
from predictivesense.camera.source import create_frame_source
from predictivesense.config.settings import load_config
from predictivesense.core.types import Detection, Frame
from predictivesense.perception.types import PerceptionResult
from predictivesense.pipeline.loop import AnalysisLoop
from predictivesense.telemetry.metrics import MetricRegistry

pytestmark = pytest.mark.integration

_W, _H = 640, 480


class _FakePerception:
    def __init__(self, detections: tuple[Detection, ...]) -> None:
        self.detections = detections

    def infer(self, frame: Frame, *, frame_index: int | None = None) -> PerceptionResult:
        return PerceptionResult(detections=self.detections, poses=())

    def info(self) -> dict:
        return {"detector_ep": "fake", "detector_warmup_ms": 0.0, "pose_warmup_ms": -1.0}


class _FakeClassifier:
    """Always predicts a fixed class - real inference isn't the point here,
    only that the LOOP wires crop -> classifier -> Detection correctly and
    leaves the policy's own fields untouched."""

    def __init__(self, name: str = "watch", confidence: float = 0.87) -> None:
        self.name = name
        self.confidence = confidence
        self.calls = 0

    def infer_crop(self, image_bgr: np.ndarray) -> tuple[str, float]:
        self.calls += 1
        assert image_bgr.size > 0  # a real, non-empty crop was handed in
        return self.name, self.confidence


def _det(bbox, *, policy_state="accepted", score=0.9, frame_id=0) -> Detection:
    return Detection(
        bbox=bbox, class_id=0, class_name="donut", score=score, frame_id=frame_id,
        raw_class_name="donut", policy_state=policy_state, tier="secondary",
    )


def _loop(*, detections, classifier=None, tmp_path=None) -> AnalysisLoop:
    cfg = load_config("dev")
    # Phase 12: the classifier registry is now config-driven - isolate it to
    # a path that never exists, so this test never depends on whatever a real
    # developer machine happens to have trained/activated in production
    # (models/classifier_registry.json is NOT test-isolated by default, same
    # limitation documented in docs/decisions.md Phase 11 Part B).
    isolated_registry = (tmp_path / "classifier_registry.json") if tmp_path is not None else Path("does-not-exist.json")
    cfg = cfg.model_copy(update={
        "perception": cfg.perception.model_copy(update={"detection_enabled": False, "pose_enabled": False}),
        "analysis": cfg.analysis.model_copy(update={"max_frame_age_ms": 0.0}),
        "tracking": cfg.tracking.model_copy(update={"enabled": False}),
        "training": cfg.training.model_copy(update={"classifier_registry_path": isolated_registry}),
    })
    src = create_frame_source(cfg.source)
    return AnalysisLoop(
        cfg, src, LatestFrameMailbox(), registry=MetricRegistry(),
        perception=_FakePerception(detections), classifier=classifier,
        classifier_version_id="fake-v1" if classifier is not None else None,
    )


def _frame(fid: int = 0) -> Frame:
    image = np.zeros((_H, _W, 3), dtype=np.uint8)
    image[100:140, 100:140] = 255  # a real, non-empty region under the test box
    return Frame(
        frame_id=fid, capture_ts=time.monotonic(), image=image,
        width=_W, height=_H, source_id="t", seq=fid,
    )


def test_no_classifier_leaves_detections_completely_unchanged(tmp_path):
    loop = _loop(detections=(_det((100, 100, 140, 140)),), classifier=None, tmp_path=tmp_path)
    loop._iterate(_frame())
    snap = loop.latest
    assert snap is not None
    d = snap.detections[0]
    assert d.custom_class_name is None
    assert d.custom_class_confidence is None
    assert d.class_name == "donut"  # policy's own decision, untouched
    assert loop.custom_classifier_version == "none"


def test_active_classifier_annotates_accepted_detections_without_changing_policy_fields():
    clf = _FakeClassifier(name="watch", confidence=0.87)
    loop = _loop(detections=(_det((100, 100, 140, 140), policy_state="accepted"),), classifier=clf)
    loop._iterate(_frame())
    snap = loop.latest
    assert snap is not None
    d = snap.detections[0]
    assert d.custom_class_name == "watch"
    assert d.custom_class_confidence == pytest.approx(0.87)
    # The recognition policy's own decision is byte-for-byte unchanged.
    assert d.class_name == "donut"
    assert d.policy_state == "accepted"
    assert d.tier == "secondary"
    assert d.raw_class_name == "donut"
    assert clf.calls == 1
    assert loop.custom_classifier_version == "fake-v1"


def test_classifier_skips_suppressed_and_rejected_detections():
    clf = _FakeClassifier()
    dets = (
        _det((100, 100, 140, 140), policy_state="suppressed_implausible"),
        _det((200, 100, 240, 140), policy_state="rejected_size"),
    )
    loop = _loop(detections=dets, classifier=clf)
    loop._iterate(_frame())
    snap = loop.latest
    assert snap is not None
    assert all(d.custom_class_name is None for d in snap.detections)
    assert clf.calls == 0  # never even invoked for boxes that are never shown


def test_classifier_failure_never_drops_the_frame():
    class _BrokenClassifier:
        def infer_crop(self, image_bgr):
            raise RuntimeError("boom")

    loop = _loop(detections=(_det((100, 100, 140, 140)),), classifier=_BrokenClassifier())
    loop._iterate(_frame())  # must not raise
    snap = loop.latest
    assert snap is not None
    assert snap.detections[0].custom_class_name is None  # degraded, not crashed
    assert snap.detections[0].class_name == "donut"  # policy decision still intact


def test_custom_classifier_ms_is_reported_when_active():
    clf = _FakeClassifier()
    loop = _loop(detections=(_det((100, 100, 140, 140)),), classifier=clf)
    loop._iterate(_frame())
    snap = loop.latest
    assert snap is not None
    assert snap.metrics["custom_classifier_ms"] >= 0.0
