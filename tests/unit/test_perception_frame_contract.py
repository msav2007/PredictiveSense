"""Phase 8 section 16 / unit test 6 - the immutable PerceptionFrame record:
required fields, stable within-frame ordering, and the one shared builder used
by both the real-time loop and the recorded driver.
"""

from __future__ import annotations

import numpy as np
import pytest

from predictivesense.core.types import Detection, Frame, PerceptionFrame, Pose
from predictivesense.pipeline.perception_frame import build_perception_frame

pytestmark = pytest.mark.unit

_REQUIRED = {
    "frame_id", "seq", "capture_ts", "width", "height", "model_version",
    "provider", "detections", "poses", "pose_stale", "pose_frame_id",
    "pose_capture_ts", "pose_age_ms",
}


def _frame() -> Frame:
    return Frame(
        frame_id=42, capture_ts=123.5, image=np.zeros((4, 4, 3), np.uint8),
        width=640, height=480, source_id="s", seq=7,
    )


def _det(name: str, score: float, x: float) -> Detection:
    return Detection(
        bbox=(x, x, x + 10, x + 10), class_id=1, class_name=name, score=score,
        frame_id=42, raw_class_name=name,
    )


def test_all_required_fields_present_and_typed() -> None:
    pf = build_perception_frame(
        frame=_frame(), detections=[_det("cup", 0.9, 5.0)], poses=[],
        model_version="v1", provider="CPUExecutionProvider",
    )
    assert set(pf.model_dump().keys()) == _REQUIRED
    assert pf.frame_id == 42 and pf.seq == 7 and pf.capture_ts == 123.5
    assert pf.width == 640 and pf.height == 480
    assert pf.model_version == "v1" and pf.provider == "CPUExecutionProvider"
    assert isinstance(pf.detections, tuple) and isinstance(pf.poses, tuple)
    # frozen
    with pytest.raises(Exception):
        pf.frame_id = 0


def test_within_frame_ordering_is_deterministic() -> None:
    dets_in = [
        _det("bottle", 0.55, 30.0),
        _det("cup", 0.91, 12.0),
        _det("cup", 0.91, 4.0),   # same score+class, smaller x -> comes first
        _det("laptop", 0.70, 8.0),
    ]
    a = build_perception_frame(frame=_frame(), detections=list(dets_in), poses=[],
                               model_version="v", provider="cpu")
    b = build_perception_frame(frame=_frame(), detections=list(reversed(dets_in)),
                               poses=[], model_version="v", provider="cpu")
    order_a = [(d.class_name, d.score, d.bbox[0]) for d in a.detections]
    order_b = [(d.class_name, d.score, d.bbox[0]) for d in b.detections]
    assert order_a == order_b
    assert order_a == [
        ("cup", 0.91, 4.0), ("cup", 0.91, 12.0),
        ("laptop", 0.70, 8.0), ("bottle", 0.55, 30.0),
    ]


def test_reused_pose_provenance_is_carried() -> None:
    pose = Pose(keypoints=tuple((1.0, 2.0, 0.9) for _ in range(17)),
                bbox=(0, 0, 5, 5), score=0.8, frame_id=40)
    pf = build_perception_frame(
        frame=_frame(), detections=[], poses=[pose],
        model_version="v", provider="cpu",
        pose_reused=True, pose_frame_id=40, pose_capture_ts=123.0, pose_age_ms=500.0,
    )
    assert pf.pose_stale is True
    assert pf.pose_frame_id == 40
    assert pf.pose_age_ms == 500.0
    assert pf.poses[0] is pose  # same immutable object, not rebuilt


def test_json_round_trip() -> None:
    pf = build_perception_frame(
        frame=_frame(), detections=[_det("cup", 0.9, 5.0)],
        poses=[Pose(keypoints=tuple((1.0, 2.0, 0.9) for _ in range(17)),
                    bbox=(0, 0, 5, 5), score=0.8, frame_id=42)],
        model_version="v1", provider="CPUExecutionProvider",
    )
    back = PerceptionFrame.model_validate_json(pf.model_dump_json())
    assert back == pf
