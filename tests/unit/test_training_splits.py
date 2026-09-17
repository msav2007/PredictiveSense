"""Phase 11 Part B section 11.3 - session/object-disjoint crop splits."""

from __future__ import annotations

import pytest

from predictivesense.training.dataset import validate_dataset
from predictivesense.training.splits import (
    CropSplitLeakageError,
    CropSplits,
    assert_no_crop_leakage,
    build_crop_splits,
    crop_splits_content_hash,
    load_crop_splits,
)
from predictivesense.training.synthetic_fixture import build_synthetic_object_store

pytestmark = pytest.mark.unit


def _records(tmp_path, **kwargs):
    root = tmp_path / "objects"
    build_synthetic_object_store(root, **kwargs)
    records, _summary = validate_dataset(root, min_samples_per_class=4)
    return records


def test_every_class_appears_in_every_split_with_enough_sessions(tmp_path):
    # Regression test for a real bug found building this pipeline: splitting
    # over one GLOBAL session pool could, by chance, send every session of a
    # class to train, leaving it entirely absent from test (0% "accuracy" that
    # was actually 0 test examples, not a bad model). Per-class stratification
    # fixes it - assert it holds across several seeds.
    records = _records(tmp_path, n_per_class=15, n_sessions_per_class=3, seed=7)
    for seed in range(10):
        splits = build_crop_splits(
            records, train_fraction=0.6, val_fraction=0.2, test_fraction=0.2, seed=seed
        )
        by_id = {r.crop_id: r for r in records}
        for split_name in ("train", "val", "test"):
            classes_present = {by_id[cid].object_id for cid in splits.crop_ids[split_name]}
            assert classes_present == {"circle", "square", "triangle"}, (
                f"seed={seed} split={split_name} missing classes"
            )


def test_no_session_or_crop_id_spans_two_splits(tmp_path):
    records = _records(tmp_path, n_per_class=12, n_sessions_per_class=3, seed=1)
    splits = build_crop_splits(records, train_fraction=0.6, val_fraction=0.2, test_fraction=0.2, seed=0)
    assert_no_crop_leakage(splits)  # must not raise


def test_a_deliberately_leaky_split_fails():
    leaky = CropSplits(
        sessions={"train": ["s1", "s2"], "val": ["s2"], "test": []},
        crop_ids={"train": ["c1"], "val": ["c2"], "test": []},
        content_hash="deadbeef", seed=0, fractions={},
    )
    with pytest.raises(CropSplitLeakageError, match="sessions in both"):
        assert_no_crop_leakage(leaky)


def test_a_deliberately_leaky_crop_id_fails():
    leaky = CropSplits(
        sessions={"train": ["s1"], "val": ["s2"], "test": ["s3"]},
        crop_ids={"train": ["shared"], "val": ["shared"], "test": []},
        content_hash="deadbeef", seed=0, fractions={},
    )
    with pytest.raises(CropSplitLeakageError, match="crop ids in both"):
        assert_no_crop_leakage(leaky)


def test_content_hash_changes_when_the_dataset_changes(tmp_path):
    records = _records(tmp_path, n_per_class=8, n_sessions_per_class=2, seed=4)
    splits = build_crop_splits(records, train_fraction=0.6, val_fraction=0.2, test_fraction=0.2, seed=0)
    changed = crop_splits_content_hash(records[:-1], splits.sessions)
    assert changed != splits.content_hash


def test_load_crop_splits_detects_a_stale_split_file(tmp_path):
    records = _records(tmp_path, n_per_class=8, n_sessions_per_class=2, seed=9)
    splits = build_crop_splits(records, train_fraction=0.6, val_fraction=0.2, test_fraction=0.2, seed=0)
    path = tmp_path / "splits.json"
    splits.save(path)

    loaded = load_crop_splits(path, records)
    assert loaded.content_hash == splits.content_hash

    with pytest.raises(CropSplitLeakageError, match="stale"):
        load_crop_splits(path, records[:-1])
