"""Per-class precision / recall / F1, a confusion matrix that includes
``background`` and ``unknown``, the **false-class rate** (this phase's headline
number - a prediction that overlaps a real object but names it wrongly), and
AP@0.5 / mAP@0.5.

Every metric is reported next to its sample count so a number from three boxes is
not read like a number from three hundred (BLOCK 3.1.7).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from predictivesense.eval.matching import greedy_match

__all__ = ["ClassMetrics", "EvalMetrics", "evaluate", "BACKGROUND", "UNKNOWN"]

BACKGROUND = "background"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class ClassMetrics:
    name: str
    support: int          # ground-truth boxes of this class
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    ap50: float | None    # None when support == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.name,
            "support": self.support,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "ap50": None if self.ap50 is None else round(self.ap50, 4),
        }


@dataclass(frozen=True)
class EvalMetrics:
    iou_threshold: float
    class_names: list[str]
    per_class: list[ClassMetrics]
    n_images: int
    n_gt: int
    n_pred: int
    n_matched: int
    false_class_count: int
    false_class_rate: float
    unknown_on_object_count: int
    unknown_on_object_rate: float
    localization_recall: float
    background_fp_count: int
    background_fp_rate: float
    map50: float | None
    confusion_labels: list[str]
    confusion: list[list[int]]
    top_confusions: list[dict[str, Any]]

    def macro_f1(self) -> float:
        scored = [c.f1 for c in self.per_class if c.support > 0]
        return float(np.mean(scored)) if scored else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "iou_threshold": self.iou_threshold,
            "sample_counts": {
                "images": self.n_images,
                "ground_truth_boxes": self.n_gt,
                "predictions": self.n_pred,
                "matched_pairs": self.n_matched,
            },
            "headline": {
                "false_class_rate": round(self.false_class_rate, 4),
                "false_class_count": self.false_class_count,
                "matched_pairs": self.n_matched,
            },
            "unknown_on_object_rate": round(self.unknown_on_object_rate, 4),
            "unknown_on_object_count": self.unknown_on_object_count,
            "localization_recall": round(self.localization_recall, 4),
            "background_fp_rate": round(self.background_fp_rate, 4),
            "background_fp_count": self.background_fp_count,
            "map50": None if self.map50 is None else round(self.map50, 4),
            "macro_f1": round(self.macro_f1(), 4),
            "per_class": [c.to_dict() for c in self.per_class],
            "confusion_labels": self.confusion_labels,
            "confusion": self.confusion,
            "top_confusions": self.top_confusions,
        }


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def evaluate(
    samples: list[dict[str, Any]],
    class_names: list[str],
    *,
    iou_threshold: float = 0.5,
) -> EvalMetrics:
    """Compute every metric from a list of per-image ``{image_id, gt, pred}``.

    ``gt``  boxes: ``{"bbox": [x1,y1,x2,y2], "label": <class>}``.
    ``pred`` boxes: ``{"bbox": [x1,y1,x2,y2], "label": <class|'unknown'>, "score": float}``.
    Safe on empty input (all zeros, ``map50`` = ``None``).
    """

    names = list(class_names)
    name_set = set(names)
    conf_labels = [*names, UNKNOWN, BACKGROUND]
    idx = {n: i for i, n in enumerate(conf_labels)}
    confusion = [[0 for _ in conf_labels] for _ in conf_labels]

    tp = Counter[str]()
    fp = Counter[str]()
    fn = Counter[str]()
    support = Counter[str]()

    n_gt = n_pred = n_matched = 0
    false_class = 0
    unknown_on_object = 0
    background_fp = 0
    confusion_examples: dict[tuple[str, str], list[int]] = {}

    for s in samples:
        image_id = int(s.get("image_id", -1))
        gt = list(s.get("gt", []))
        pred = list(s.get("pred", []))
        n_gt += len(gt)
        n_pred += len(pred)
        for g in gt:
            support[str(g["label"])] += 1

        gt_boxes = np.array([g["bbox"] for g in gt], dtype=np.float64).reshape(-1, 4)
        pr_boxes = np.array([p["bbox"] for p in pred], dtype=np.float64).reshape(-1, 4)
        pr_scores = np.array([float(p.get("score", 0.0)) for p in pred], dtype=np.float64)

        m = greedy_match(gt_boxes, pr_boxes, pr_scores, iou_threshold=iou_threshold)

        matched_gt: set[int] = set()
        matched_pred: set[int] = set()
        for g_i, p_i, _iou in m.matches:
            matched_gt.add(g_i)
            matched_pred.add(p_i)
            n_matched += 1
            g_label = str(gt[g_i]["label"])
            p_label = str(pred[p_i]["label"])
            confusion[idx.get(g_label, idx[BACKGROUND])][idx.get(p_label, idx[UNKNOWN])] += 1
            if p_label == g_label:
                tp[g_label] += 1
            elif p_label == UNKNOWN:
                unknown_on_object += 1
                fn[g_label] += 1  # the real class was not recovered
            else:
                false_class += 1
                fp[p_label] += 1
                fn[g_label] += 1
                key = (g_label, p_label)
                confusion_examples.setdefault(key, [])
                if len(confusion_examples[key]) < 3 and image_id not in confusion_examples[key]:
                    confusion_examples[key].append(image_id)

        for p_i in m.unmatched_pred:
            p_label = str(pred[p_i]["label"])
            background_fp += 1
            confusion[idx[BACKGROUND]][idx.get(p_label, idx[UNKNOWN])] += 1
            if p_label in name_set:
                fp[p_label] += 1
            # an unmatched `unknown` prediction is neither a TP nor a class FP

        for g_i in m.unmatched_gt:
            g_label = str(gt[g_i]["label"])
            confusion[idx.get(g_label, idx[BACKGROUND])][idx[BACKGROUND]] += 1
            fn[g_label] += 1

    per_class: list[ClassMetrics] = []
    ap_values: list[float] = []
    for name in names:
        s = int(support[name])
        t, f_p, f_n = int(tp[name]), int(fp[name]), int(fn[name])
        precision = _safe_div(t, t + f_p)
        recall = _safe_div(t, t + f_n)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        ap = _average_precision(samples, name, iou_threshold) if s > 0 else None
        if ap is not None:
            ap_values.append(ap)
        per_class.append(
            ClassMetrics(
                name=name, support=s, tp=t, fp=f_p, fn=f_n,
                precision=precision, recall=recall, f1=f1, ap50=ap,
            )
        )

    top_confusions = [
        {
            "ground_truth": g,
            "predicted_as": p,
            "count": int(c),
            "example_image_ids": confusion_examples.get((g, p), []),
        }
        for (g, p), c in _pair_counts(samples, names, iou_threshold).most_common(10)
    ]

    return EvalMetrics(
        iou_threshold=iou_threshold,
        class_names=names,
        per_class=per_class,
        n_images=len(samples),
        n_gt=n_gt,
        n_pred=n_pred,
        n_matched=n_matched,
        false_class_count=false_class,
        false_class_rate=_safe_div(false_class, n_matched),
        unknown_on_object_count=unknown_on_object,
        unknown_on_object_rate=_safe_div(unknown_on_object, n_matched),
        localization_recall=_safe_div(n_matched, n_gt),
        background_fp_count=background_fp,
        background_fp_rate=_safe_div(background_fp, n_pred),
        map50=(float(np.mean(ap_values)) if ap_values else None),
        confusion_labels=conf_labels,
        confusion=confusion,
        top_confusions=top_confusions,
    )


def _pair_counts(
    samples: list[dict[str, Any]], names: list[str], iou_threshold: float
) -> Counter:
    """(gt_label, pred_label) counts for matched pairs whose labels differ and
    whose prediction is a real class (not ``unknown``)."""

    out = Counter()
    for s in samples:
        gt = list(s.get("gt", []))
        pred = list(s.get("pred", []))
        gt_boxes = np.array([g["bbox"] for g in gt], dtype=np.float64).reshape(-1, 4)
        pr_boxes = np.array([p["bbox"] for p in pred], dtype=np.float64).reshape(-1, 4)
        pr_scores = np.array([float(p.get("score", 0.0)) for p in pred], dtype=np.float64)
        m = greedy_match(gt_boxes, pr_boxes, pr_scores, iou_threshold=iou_threshold)
        for g_i, p_i, _iou in m.matches:
            g_label = str(gt[g_i]["label"])
            p_label = str(pred[p_i]["label"])
            if p_label != g_label and p_label != UNKNOWN:
                out[(g_label, p_label)] += 1
    return out


def _average_precision(
    samples: list[dict[str, Any]], cls: str, iou_threshold: float
) -> float:
    """All-point-interpolated AP@IoU for one class (per-image greedy matching
    within the class, predictions ranked globally by score)."""

    entries: list[tuple[float, int]] = []  # (score, is_tp)
    n_pos = 0
    for s in samples:
        gt = [g for g in s.get("gt", []) if str(g["label"]) == cls]
        pred = [p for p in s.get("pred", []) if str(p["label"]) == cls]
        n_pos += len(gt)
        if not pred:
            continue
        gt_boxes = np.array([g["bbox"] for g in gt], dtype=np.float64).reshape(-1, 4)
        pr_boxes = np.array([p["bbox"] for p in pred], dtype=np.float64).reshape(-1, 4)
        pr_scores = np.array([float(p.get("score", 0.0)) for p in pred], dtype=np.float64)
        m = greedy_match(gt_boxes, pr_boxes, pr_scores, iou_threshold=iou_threshold)
        tp_pred = {p_i for _g, p_i, _i in m.matches}
        for i, p in enumerate(pred):
            entries.append((float(p.get("score", 0.0)), 1 if i in tp_pred else 0))

    if n_pos == 0:
        return 0.0
    if not entries:
        return 0.0

    entries.sort(key=lambda e: -e[0])
    tp_cum = 0
    fp_cum = 0
    precisions: list[float] = []
    recalls: list[float] = []
    for _score, is_tp in entries:
        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        precisions.append(tp_cum / (tp_cum + fp_cum))
        recalls.append(tp_cum / n_pos)

    # all-point interpolation
    rec = np.array([0.0, *recalls, 1.0])
    prec = np.array([0.0, *precisions, 0.0])
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    idx = np.where(rec[1:] != rec[:-1])[0]
    return float(np.sum((rec[idx + 1] - rec[idx]) * prec[idx + 1]))
