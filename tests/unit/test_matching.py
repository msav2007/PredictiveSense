"""Greedy score-ordered IoU matching on hand-built boxes."""

from __future__ import annotations

import numpy as np
import pytest

from predictivesense.eval.matching import greedy_match, iou_xyxy

pytestmark = pytest.mark.unit


def test_iou_basic() -> None:
    assert iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou_xyxy((0, 0, 10, 10), (10, 10, 20, 20)) == 0.0
    # half overlap on x
    assert iou_xyxy((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_correct_assignment_and_miss_and_duplicate() -> None:
    gt = np.array([[0, 0, 10, 10], [100, 100, 120, 120]], dtype=float)
    # p0 matches gt0 well; p1 is a near-duplicate of gt0 (lower score) -> FP;
    # p2 is far from any gt -> FP; gt1 gets nothing -> miss.
    pred = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [300, 300, 310, 310]], dtype=float)
    scores = np.array([0.9, 0.8, 0.7])

    m = greedy_match(gt, pred, scores, iou_threshold=0.5)
    assert [(g, p) for g, p, _ in m.matches] == [(0, 0)]
    assert m.unmatched_pred == [1, 2]
    assert m.unmatched_gt == [1]


def test_score_order_decides_who_claims_a_shared_gt() -> None:
    gt = np.array([[0, 0, 10, 10]], dtype=float)
    pred = np.array([[0, 0, 10, 10], [0, 0, 9, 9]], dtype=float)
    # lower-score prediction listed first, but the higher score must win the claim
    m = greedy_match(gt, pred, np.array([0.4, 0.95]), iou_threshold=0.5)
    assert m.matches[0][1] == 1
    assert m.unmatched_pred == [0]


def test_empty_inputs_are_safe() -> None:
    empty = np.zeros((0, 4))
    assert greedy_match(empty, empty, np.zeros(0)).matches == []
    m = greedy_match(np.array([[0, 0, 5, 5]], float), empty, np.zeros(0))
    assert m.unmatched_gt == [0]
    m = greedy_match(empty, np.array([[0, 0, 5, 5]], float), np.array([0.5]))
    assert m.unmatched_pred == [0]


def test_threshold_is_respected() -> None:
    gt = np.array([[0, 0, 10, 10]], dtype=float)
    pred = np.array([[0, 0, 6, 10]], dtype=float)  # IoU = 60/100 = 0.6
    assert greedy_match(gt, pred, np.array([0.9]), iou_threshold=0.5).matches
    assert not greedy_match(gt, pred, np.array([0.9]), iou_threshold=0.7).matches
