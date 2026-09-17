"""Classifier evaluation - accuracy metrics on held-out positives, plus a
false-class rate on held-out negatives/hard-negatives (Phase 11 Part B
section 15). Latency is reported separately from accuracy, never blended into
one score, per section 15's explicit requirement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from predictivesense.training.crop_data import load_crop_tensor
from predictivesense.training.dataset import CropRecord

__all__ = [
    "ClassificationMetrics",
    "evaluate_predictor",
    "predictor_from_detector",
    "predictor_from_torch_model",
]

PredictFn = Callable[[CropRecord], tuple[str, float]]  # crop -> (predicted_class, latency_ms)


@dataclass(frozen=True)
class ClassificationMetrics:
    n_positive_examples: int
    accuracy: float
    precision_per_class: dict[str, float]
    recall_per_class: dict[str, float]
    f1_per_class: dict[str, float]
    confusion_matrix: dict[str, dict[str, int]]  # true -> {predicted: count}
    n_rejection_examples: int
    false_class_rate: float  # of rejection examples, fraction wrongly named as negative_for
    latency_ms_mean: float
    latency_ms_p95: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_positive_examples": self.n_positive_examples,
            "accuracy": round(self.accuracy, 4),
            "precision_per_class": {k: round(v, 4) for k, v in self.precision_per_class.items()},
            "recall_per_class": {k: round(v, 4) for k, v in self.recall_per_class.items()},
            "f1_per_class": {k: round(v, 4) for k, v in self.f1_per_class.items()},
            "confusion_matrix": self.confusion_matrix,
            "n_rejection_examples": self.n_rejection_examples,
            "false_class_rate": round(self.false_class_rate, 4),
            "latency_ms_mean": round(self.latency_ms_mean, 3),
            "latency_ms_p95": round(self.latency_ms_p95, 3),
        }


def predictor_from_torch_model(model: torch.nn.Module, class_map: dict[str, int]) -> PredictFn:
    """Wrap a trained ``CropClassifier`` (eval mode, CPU) as a ``PredictFn``."""

    index_to_class = {i: c for c, i in class_map.items()}
    model.eval()

    def predict(record: CropRecord) -> tuple[str, float]:
        tensor = load_crop_tensor(record).unsqueeze(0)
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(tensor)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        idx = int(torch.argmax(logits, dim=1).item())
        return index_to_class[idx], latency_ms

    return predict


def predictor_from_detector(detector: Any) -> PredictFn:
    """Baseline for section 15: what the EXISTING pretrained detector alone
    would call this crop - its own vocabulary (COCO-80) has no idea what a
    Studio-enrolled custom object is, so this is expected, honest, and almost
    always wrong on custom classes. That IS the baseline result (section 15:
    "if it is worse, that is the result, and it is publishable"), not a bug in
    this evaluator."""

    import time as _time

    import numpy as _np

    from predictivesense.core.types import Frame

    def predict(record: CropRecord) -> tuple[str, float]:
        import cv2

        image = cv2.imread(record.image_path, cv2.IMREAD_COLOR)
        if image is None:
            return "no_detection", 0.0
        frame = Frame(
            frame_id=0, capture_ts=0.0, image=_np.ascontiguousarray(image),
            width=image.shape[1], height=image.shape[0], source_id="baseline-eval", seq=0,
        )
        t0 = _time.perf_counter()
        detections = detector.infer(frame)
        latency_ms = (_time.perf_counter() - t0) * 1000.0
        if not detections:
            return "no_detection", latency_ms
        best = max(detections, key=lambda d: d.score)
        return best.class_name, latency_ms

    return predict


def evaluate_predictor(
    predict: PredictFn,
    positive_examples: list[CropRecord],
    rejection_examples: list[CropRecord],
    classes: list[str],
) -> ClassificationMetrics:
    """Run ``predict`` over held-out positives (accuracy/P/R/F1/confusion) and
    held-out negatives/hard-negatives (false-class rate) - never the same
    examples for both, and never blended into one number."""

    confusion: dict[str, dict[str, int]] = {c: {c2: 0 for c2 in classes} for c in classes}
    latencies: list[float] = []
    correct = 0
    for record in positive_examples:
        pred, latency_ms = predict(record)
        latencies.append(latency_ms)
        confusion.setdefault(record.object_id, {}).setdefault(pred, 0)
        confusion[record.object_id][pred] += 1
        if pred == record.object_id:
            correct += 1

    precision: dict[str, float] = {}
    recall: dict[str, float] = {}
    f1: dict[str, float] = {}
    for c in classes:
        tp = confusion.get(c, {}).get(c, 0)
        predicted_as_c = sum(confusion.get(t, {}).get(c, 0) for t in classes)
        actual_c = sum(confusion.get(c, {}).values())
        p = tp / predicted_as_c if predicted_as_c else 0.0
        r = tp / actual_c if actual_c else 0.0
        precision[c] = p
        recall[c] = r
        f1[c] = (2 * p * r / (p + r)) if (p + r) else 0.0

    false_class = 0
    for record in rejection_examples:
        pred, latency_ms = predict(record)
        latencies.append(latency_ms)
        if pred in record.negative_for:
            false_class += 1

    latencies_sorted = sorted(latencies)
    p95 = (
        latencies_sorted[min(len(latencies_sorted) - 1, int(round(0.95 * (len(latencies_sorted) - 1))))]
        if latencies_sorted else 0.0
    )
    return ClassificationMetrics(
        n_positive_examples=len(positive_examples),
        accuracy=(correct / len(positive_examples)) if positive_examples else 0.0,
        precision_per_class=precision,
        recall_per_class=recall,
        f1_per_class=f1,
        confusion_matrix=confusion,
        n_rejection_examples=len(rejection_examples),
        false_class_rate=(false_class / len(rejection_examples)) if rejection_examples else 0.0,
        latency_ms_mean=(sum(latencies) / len(latencies)) if latencies else 0.0,
        latency_ms_p95=p95,
    )
