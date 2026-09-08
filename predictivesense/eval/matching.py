"""IoU and greedy score-ordered assignment for detection evaluation.

Matching is **class-agnostic geometry first**: each prediction, taken in
descending score order, claims the highest-IoU still-unclaimed ground-truth box
at or above ``iou_threshold``. Whether the class was right is a separate question
the metrics layer answers (that is what makes a confusion matrix and a
false-class rate possible). A prediction that claims nothing is a false positive;
a ground-truth box no prediction claims is a miss (BLOCK 3.8 / BLOCK 12.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["MatchResult", "iou_xyxy", "iou_matrix", "greedy_match"]


def iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """IoU of two ``(x1, y1, x2, y2)`` boxes. 0.0 when either is degenerate."""

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def iou_matrix(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    """``(n_gt, n_pred)`` IoU matrix for two ``(N, 4)`` xyxy arrays."""

    gt = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    pr = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4)
    if gt.size == 0 or pr.size == 0:
        return np.zeros((gt.shape[0], pr.shape[0]), dtype=np.float64)

    x1 = np.maximum(gt[:, None, 0], pr[None, :, 0])
    y1 = np.maximum(gt[:, None, 1], pr[None, :, 1])
    x2 = np.minimum(gt[:, None, 2], pr[None, :, 2])
    y2 = np.minimum(gt[:, None, 3], pr[None, :, 3])
    inter = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)
    area_gt = np.clip(gt[:, 2] - gt[:, 0], 0.0, None) * np.clip(gt[:, 3] - gt[:, 1], 0.0, None)
    area_pr = np.clip(pr[:, 2] - pr[:, 0], 0.0, None) * np.clip(pr[:, 3] - pr[:, 1], 0.0, None)
    union = area_gt[:, None] + area_pr[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0.0, inter / union, 0.0)
    return iou


@dataclass(frozen=True)
class MatchResult:
    """One image's assignment.

    ``matches`` are ``(gt_index, pred_index, iou)`` triples ordered by the score
    of the prediction. ``unmatched_gt`` are misses; ``unmatched_pred`` are false
    positives (including duplicate detections of an already-claimed box).
    """

    matches: list[tuple[int, int, float]] = field(default_factory=list)
    unmatched_gt: list[int] = field(default_factory=list)
    unmatched_pred: list[int] = field(default_factory=list)


def greedy_match(
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    *,
    iou_threshold: float = 0.5,
) -> MatchResult:
    """Greedy score-ordered geometric matching for one image."""

    gt = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    pr = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(pred_scores, dtype=np.float64).reshape(-1)
    n_gt, n_pred = gt.shape[0], pr.shape[0]

    if n_pred == 0:
        return MatchResult(unmatched_gt=list(range(n_gt)))
    if n_gt == 0:
        return MatchResult(unmatched_pred=list(range(n_pred)))

    iou = iou_matrix(gt, pr)
    gt_claimed = np.zeros(n_gt, dtype=bool)
    matches: list[tuple[int, int, float]] = []
    unmatched_pred: list[int] = []

    for p in np.argsort(-scores, kind="stable"):
        col = iou[:, p].copy()
        col[gt_claimed] = -1.0
        g = int(np.argmax(col))
        if col[g] >= iou_threshold:
            gt_claimed[g] = True
            matches.append((g, int(p), float(iou[g, p])))
        else:
            unmatched_pred.append(int(p))

    unmatched_gt = [g for g in range(n_gt) if not gt_claimed[g]]
    return MatchResult(
        matches=matches,
        unmatched_gt=sorted(unmatched_gt),
        unmatched_pred=sorted(unmatched_pred),
    )
