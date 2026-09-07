"""Class-aware NMS on hand-built boxes."""

from __future__ import annotations

import numpy as np
import pytest

from predictivesense.perception.postprocess import (
    class_aware_nms,
    nms,
    xywh_to_xyxy,
)

pytestmark = pytest.mark.unit


def test_xywh_to_xyxy() -> None:
    out = xywh_to_xyxy(np.array([[10.0, 10.0, 4.0, 6.0]]))
    assert np.allclose(out, [[8.0, 7.0, 12.0, 13.0]])


def test_empty_input_is_safe() -> None:
    assert nms(np.zeros((0, 4)), np.zeros((0,)), 0.5) == []
    assert class_aware_nms(np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,)), 0.5, 100) == []


def test_overlapping_same_class_is_suppressed() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [1, 1, 9, 9]], dtype=float)
    scores = np.array([0.9, 0.8, 0.7])
    classes = np.array([0, 0, 0])
    kept = class_aware_nms(boxes, scores, classes, iou_threshold=0.5, max_detections=100)
    assert kept == [0]  # only the highest-scoring of the overlapping cluster


def test_overlapping_different_class_is_kept() -> None:
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=float)
    scores = np.array([0.9, 0.85])
    classes = np.array([0, 1])
    kept = class_aware_nms(boxes, scores, classes, iou_threshold=0.5, max_detections=100)
    assert sorted(kept) == [0, 1]  # different classes never suppress each other


def test_non_overlapping_same_class_both_kept() -> None:
    boxes = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], dtype=float)
    scores = np.array([0.6, 0.7])
    classes = np.array([3, 3])
    kept = class_aware_nms(boxes, scores, classes, iou_threshold=0.5, max_detections=100)
    assert sorted(kept) == [0, 1]


def test_max_detections_is_respected_and_ordered_by_score() -> None:
    boxes = np.array([[i * 50, 0, i * 50 + 10, 10] for i in range(10)], dtype=float)
    scores = np.linspace(0.1, 0.99, 10)
    classes = np.arange(10)
    kept = class_aware_nms(boxes, scores, classes, iou_threshold=0.5, max_detections=3)
    assert len(kept) == 3
    kept_scores = [scores[i] for i in kept]
    assert kept_scores == sorted(kept_scores, reverse=True)
    assert kept_scores[0] == pytest.approx(0.99)


def test_iou_threshold_controls_suppression() -> None:
    # ~0.33 IoU pair: suppressed at 0.3, kept at 0.5
    boxes = np.array([[0, 0, 10, 10], [5, 0, 15, 10]], dtype=float)
    scores = np.array([0.9, 0.8])
    classes = np.array([0, 0])
    assert class_aware_nms(boxes, scores, classes, 0.3, 100) == [0]
    assert sorted(class_aware_nms(boxes, scores, classes, 0.5, 100)) == [0, 1]
