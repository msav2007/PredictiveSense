"""Detector head decoding and class-aware non-maximum suppression.

Pure NumPy - unit-tested without weights (``tests/unit/test_postprocess_nms.py``).
Kept separate from ``preprocess`` so the letterbox maths and the box maths are
each small and independently testable.
"""

from __future__ import annotations

import numpy as np

__all__ = ["xywh_to_xyxy", "nms", "class_aware_nms", "yolox_decode"]


def yolox_decode(
    raw: np.ndarray, input_size: int, strides: tuple[int, ...] = (8, 16, 32)
) -> np.ndarray:
    """Grid-decode a YOLOX raw head ``(anchors, 5 + num_classes)``.

    YOLOX exports its ONNX head un-decoded: each row is a feature-map cell with
    ``[cx, cy, w, h]`` relative to its grid and ``obj``/``cls`` already
    sigmoid-activated. This applies the standard ``demo_postprocess`` transform
    so ``[:, :4]`` become ``cx, cy, w, h`` in input pixels. Returns a new array;
    the input is untouched.
    """

    out = np.asarray(raw, dtype=np.float64).copy()
    grids: list[np.ndarray] = []
    expanded: list[np.ndarray] = []
    for stride in strides:
        g = input_size // stride
        yv, xv = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
        grid = np.stack((xv, yv), axis=2).reshape(-1, 2)
        grids.append(grid)
        expanded.append(np.full((grid.shape[0], 1), stride, dtype=np.float64))
    grid_all = np.concatenate(grids, axis=0)
    stride_all = np.concatenate(expanded, axis=0)
    if grid_all.shape[0] != out.shape[0]:
        raise ValueError(
            f"yolox_decode: {out.shape[0]} anchors do not match a {input_size}px "
            f"grid for strides {strides} ({grid_all.shape[0]} cells)"
        )
    out[:, :2] = (out[:, :2] + grid_all) * stride_all
    out[:, 2:4] = np.exp(out[:, 2:4]) * stride_all
    return out


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """``(N, 4)`` centre-x, centre-y, w, h  ->  x1, y1, x2, y2."""

    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    xy = b[:, :2]
    half = b[:, 2:] / 2.0
    return np.concatenate([xy - half, xy + half], axis=1)


def nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Single-class greedy NMS. Returns kept row indices, highest score first."""

    boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if boxes.shape[0] == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(xx2 - xx1, 0.0, None) * np.clip(yy2 - yy1, 0.0, None)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0.0, inter / union, 0.0)
        order = rest[iou < iou_threshold]
    return keep


def class_aware_nms(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
    max_detections: int,
) -> list[int]:
    """NMS applied independently per class; boxes of different classes never
    suppress each other. Returns up to ``max_detections`` row indices ordered by
    score, highest first. Empty input is safe.
    """

    boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    class_ids = np.asarray(class_ids).reshape(-1)
    if boxes.shape[0] == 0:
        return []

    kept: list[int] = []
    for cls in np.unique(class_ids):
        rows = np.nonzero(class_ids == cls)[0]
        local = nms(boxes[rows], scores[rows], iou_threshold)
        kept.extend(int(rows[j]) for j in local)

    kept.sort(key=lambda idx: float(scores[idx]), reverse=True)
    if max_detections > 0:
        kept = kept[:max_detections]
    return kept
