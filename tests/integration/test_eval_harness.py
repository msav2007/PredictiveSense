"""The evaluation harness on a tiny fixture: expected metric values, both output
files written; plus a `dataset`-marked consistency check on the real set when it
exists."""

from __future__ import annotations

import json

import pytest

from predictivesense.eval import evaluate
from predictivesense.eval.report import run_metadata, write_reports

pytestmark = pytest.mark.integration

_CLASSES = ["person", "cup", "bottle"]

_FIXTURE = [
    {
        "image_id": 10,
        "gt": [{"bbox": [0, 0, 10, 10], "label": "person"},
               {"bbox": [40, 40, 50, 50], "label": "cup"}],
        "pred": [{"bbox": [0, 0, 10, 10], "label": "person", "score": 0.95},
                 {"bbox": [40, 40, 50, 50], "label": "cup", "score": 0.80}],
    },
    {
        "image_id": 11,
        "gt": [{"bbox": [0, 0, 20, 20], "label": "bottle"}],
        "pred": [{"bbox": [0, 0, 20, 20], "label": "unknown", "score": 0.5},
                 {"bbox": [100, 100, 110, 110], "label": "person", "score": 0.6}],
    },
]


def test_metric_values_on_the_fixture(tmp_path) -> None:
    m = evaluate(_FIXTURE, _CLASSES, iou_threshold=0.5)
    assert (m.n_images, m.n_gt, m.n_pred, m.n_matched) == (2, 3, 4, 3)
    assert m.false_class_count == 0                     # nothing named the wrong real class
    assert m.unknown_on_object_count == 1              # bottle -> unknown
    assert m.background_fp_count == 1                  # the far person prediction
    per = {c.name: c for c in m.per_class}
    assert per["person"].tp == 1 and per["person"].fp == 1
    assert per["cup"].tp == 1 and per["cup"].precision == pytest.approx(1.0)
    assert per["bottle"].recall == 0.0


def test_writes_both_output_files(tmp_path) -> None:
    m = evaluate(_FIXTURE, _CLASSES)
    meta = run_metadata(
        model_name="fixture", model_hash="deadbeef", policy={"enabled": True},
        split="val", split_hash="abc123", git_commit="0" * 40, seed=1,
    )
    jp, mp = write_reports(m, meta, json_path=tmp_path / "e.json", md_path=tmp_path / "e.md")
    assert jp.is_file() and mp.is_file()
    doc = json.loads(jp.read_text(encoding="utf-8"))
    assert doc["metadata"]["model_name"] == "fixture"
    assert doc["metrics"]["sample_counts"]["images"] == 2
    assert "false-class rate" in mp.read_text(encoding="utf-8").lower()
    assert "| bottle |" in mp.read_text(encoding="utf-8")


@pytest.mark.dataset
def test_real_eval_set_loads_and_splits_do_not_leak(require_eval_set) -> None:
    from predictivesense.config.settings import load_config
    from predictivesense.dataset.splits import assert_no_leakage, build_splits

    store = require_eval_set
    cfg = load_config("dev")
    splits = build_splits(
        store, val_fraction=cfg.eval.val_fraction,
        test_fraction=cfg.eval.test_fraction, seed=cfg.eval.split_seed,
    )
    assert_no_leakage(splits)
    counts = store.counts()
    assert counts["annotations"] >= 0
    # every labelled image is in exactly one split
    everywhere = sorted(splits.image_ids["val"] + splits.image_ids["test"])
    labelled = sorted(i for i in store.image_ids() if store.image(i).get("labelled"))
    assert everywhere == labelled
