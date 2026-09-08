"""Precision / recall / F1, confusion matrix, false-class rate on a hand case."""

from __future__ import annotations

import pytest

from predictivesense.eval.metrics import evaluate

pytestmark = pytest.mark.unit

_CLASSES = ["person", "cup", "bottle", "donut"]


def _samples() -> list[dict]:
    return [
        {
            "image_id": 1,
            "gt": [
                {"bbox": [0, 0, 10, 10], "label": "person"},
                {"bbox": [50, 50, 60, 60], "label": "cup"},
            ],
            "pred": [
                {"bbox": [0, 0, 10, 10], "label": "person", "score": 0.9},
                {"bbox": [50, 50, 60, 60], "label": "donut", "score": 0.7},  # wrong class
            ],
        },
        {
            "image_id": 2,
            "gt": [{"bbox": [0, 0, 20, 20], "label": "bottle"}],
            "pred": [{"bbox": [0, 0, 20, 20], "label": "unknown", "score": 0.5}],
        },
        {
            "image_id": 3,
            "gt": [{"bbox": [0, 0, 10, 10], "label": "person"}],
            "pred": [{"bbox": [200, 200, 210, 210], "label": "person", "score": 0.8}],
        },
    ]


def test_headline_and_sample_counts() -> None:
    m = evaluate(_samples(), _CLASSES, iou_threshold=0.5)
    assert (m.n_images, m.n_gt, m.n_pred, m.n_matched) == (3, 4, 4, 3)
    assert m.false_class_count == 1
    assert m.false_class_rate == pytest.approx(1 / 3)
    assert m.unknown_on_object_count == 1
    assert m.unknown_on_object_rate == pytest.approx(1 / 3)
    assert m.localization_recall == pytest.approx(3 / 4)
    assert m.background_fp_count == 1
    assert m.background_fp_rate == pytest.approx(0.25)


def test_per_class_values() -> None:
    m = {c.name: c for c in evaluate(_samples(), _CLASSES).per_class}
    assert (m["person"].tp, m["person"].fp, m["person"].fn) == (1, 1, 1)
    assert m["person"].precision == pytest.approx(0.5)
    assert m["person"].recall == pytest.approx(0.5)
    assert m["person"].f1 == pytest.approx(0.5)
    assert m["cup"].support == 1 and m["cup"].tp == 0 and m["cup"].fn == 1
    assert m["bottle"].support == 1 and m["bottle"].fn == 1
    assert m["donut"].support == 0 and m["donut"].fp == 1
    assert m["donut"].ap50 is None


def test_confusion_matrix_reconciles() -> None:
    m = evaluate(_samples(), _CLASSES)
    assert m.confusion_labels == ["person", "cup", "bottle", "donut", "unknown", "background"]
    idx = {n: i for i, n in enumerate(m.confusion_labels)}
    C = m.confusion
    assert C[idx["person"]][idx["person"]] == 1
    assert C[idx["cup"]][idx["donut"]] == 1
    assert C[idx["bottle"]][idx["unknown"]] == 1
    assert C[idx["background"]][idx["person"]] == 1
    assert C[idx["person"]][idx["background"]] == 1
    assert sum(sum(row) for row in C) == 5  # matched(3) + bg-fp(1) + miss(1)


def test_top_confusions_carries_example_ids() -> None:
    m = evaluate(_samples(), _CLASSES)
    assert m.top_confusions[0]["ground_truth"] == "cup"
    assert m.top_confusions[0]["predicted_as"] == "donut"
    assert m.top_confusions[0]["example_image_ids"] == [1]


def test_map50_matches_hand_computation() -> None:
    m = evaluate(_samples(), _CLASSES)
    assert m.map50 == pytest.approx(1 / 6, abs=1e-6)  # person 0.5, cup 0, bottle 0
    assert m.macro_f1() == pytest.approx(1 / 6, abs=1e-6)


def test_empty_input_is_safe() -> None:
    m = evaluate([], _CLASSES)
    assert m.n_matched == 0
    assert m.false_class_rate == 0.0
    assert m.map50 is None
    assert m.macro_f1() == 0.0
    assert sum(sum(r) for r in m.confusion) == 0
