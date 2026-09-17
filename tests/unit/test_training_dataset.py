"""Phase 11 Part B section 11 - Studio samples -> crop dataset. Pure/no torch."""

from __future__ import annotations

import pytest

from predictivesense.training.dataset import (
    DatasetError,
    class_map_for,
    positive_records,
    rejection_records,
    session_key_for,
    validate_dataset,
)
from predictivesense.training.synthetic_fixture import build_synthetic_object_store

pytestmark = pytest.mark.unit


def test_session_key_uses_batch_id_when_present():
    key_a = session_key_for("watch", {"batch_id": "b1", "device_label": "cam1", "captured_utc": "x"})
    key_b = session_key_for("watch", {"batch_id": "b1", "device_label": "cam2", "captured_utc": "y"})
    assert key_a == key_b  # same batch -> same session, regardless of device/time
    assert "watch" in key_a  # object-disjoint by construction


def test_session_key_buckets_camera_captures_by_time_window():
    near_a = session_key_for(
        "watch", {"device_label": "cam1", "captured_utc": "2026-01-01T10:00:00.000000Z"}
    )
    near_b = session_key_for(
        "watch", {"device_label": "cam1", "captured_utc": "2026-01-01T10:01:00.000000Z"}
    )
    far = session_key_for(
        "watch", {"device_label": "cam1", "captured_utc": "2026-01-01T11:00:00.000000Z"}
    )
    assert near_a == near_b  # 1 minute apart -> same session bucket
    assert near_a != far  # 1 hour apart -> a different session


def test_session_key_differs_by_object_even_with_identical_batch():
    # A hard-negative sample lives under a DIFFERENT object's folder than the
    # object it is a negative_for, but its own object_id still scopes the key.
    a = session_key_for("watch", {"batch_id": "shared"})
    b = session_key_for("clock", {"batch_id": "shared"})
    assert a != b


def test_validate_dataset_builds_a_correct_summary(tmp_path):
    root = tmp_path / "objects"
    build_synthetic_object_store(root, n_per_class=6, n_sessions_per_class=2, seed=3)
    records, summary = validate_dataset(root, min_samples_per_class=4)
    assert summary.classes == ("circle", "square", "triangle")
    assert summary.total_crops == len(records)
    assert summary.hard_negative_count == 3  # one per class, per the fixture
    assert summary.human_confirmed_fraction == 1.0  # fixture stamps it True throughout
    for oid in summary.classes:
        assert summary.per_class_counts[oid]["positive"] == 6


def test_validate_dataset_refuses_when_a_class_has_too_few_positives(tmp_path):
    root = tmp_path / "objects"
    build_synthetic_object_store(root, n_per_class=2, n_sessions_per_class=1, seed=1)
    with pytest.raises(DatasetError, match="too few POSITIVE samples"):
        validate_dataset(root, min_samples_per_class=4)


def test_validate_dataset_refuses_on_empty_root(tmp_path):
    root = tmp_path / "objects"
    root.mkdir()
    with pytest.raises(DatasetError, match="no committed samples"):
        validate_dataset(root)


def test_hard_negative_never_becomes_a_training_positive(tmp_path):
    # Section 11.2: getting roles wrong silently poisons the model - test it
    # explicitly. A hard_negative crop stored under "circle" must never be
    # handed to the classifier labelled "circle".
    root = tmp_path / "objects"
    build_synthetic_object_store(root, n_per_class=6, n_sessions_per_class=2, seed=2)
    records, _summary = validate_dataset(root)

    circle_hard_negatives = [r for r in records if r.object_id == "circle" and r.role == "hard_negative"]
    assert circle_hard_negatives  # the fixture creates exactly one

    positives = positive_records(records)
    rejections = rejection_records(records)
    assert set(r.crop_id for r in circle_hard_negatives).isdisjoint({p.crop_id for p in positives})
    assert set(r.crop_id for r in circle_hard_negatives).issubset({r.crop_id for r in rejections})
    # And every rejection example's negative_for correctly names its own object.
    for r in circle_hard_negatives:
        assert "circle" in r.negative_for


def test_class_map_is_sorted_and_deterministic(tmp_path):
    root = tmp_path / "objects"
    build_synthetic_object_store(root, n_per_class=5, n_sessions_per_class=1, seed=5)
    records, _ = validate_dataset(root, min_samples_per_class=4)
    cmap = class_map_for(records)
    assert cmap == {"circle": 0, "square": 1, "triangle": 2}


# -- Phase 12 section 9.1/9.2: external manifests combine with Studio data --

def _write_external_manifest(path, *, object_id: str, n: int, source: str = "open-images-v7") -> None:
    import json

    samples = [
        {
            "sample_id": f"ext_{object_id}_{i}",
            "object_id": object_id,
            "image_id": f"img{i}",
            "image_path": f"/fake/ext/{object_id}_{i}.jpg",
            "box": [1.0, 1.0, 10.0, 10.0],
            "role": "positive",
            "negative_for": [],
            "box_confirmed_by_human": True,
        }
        for i in range(n)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 1, "source": source, "object_id": object_id, "samples": samples}), encoding="utf-8")


def test_validate_dataset_combines_external_manifest_with_studio_samples(tmp_path):
    root = tmp_path / "objects"
    build_synthetic_object_store(root, n_per_class=6, n_sessions_per_class=2, seed=3)
    manifest = tmp_path / "external" / "watch" / "manifest.json"
    _write_external_manifest(manifest, object_id="watch", n=10)

    records, summary = validate_dataset(root, min_samples_per_class=4, external_manifests=[manifest])

    assert "watch" in summary.classes
    assert summary.per_class_counts["watch"]["positive"] == 10
    # Studio classes are untouched, never replaced.
    assert summary.per_class_counts["circle"]["positive"] == 6
    watch_records = [r for r in records if r.object_id == "watch"]
    assert len(watch_records) == 10
    assert all(r.source == "external:open-images-v7" for r in watch_records)


def test_dataset_summary_reports_composition_per_source(tmp_path):
    root = tmp_path / "objects"
    build_synthetic_object_store(root, n_per_class=6, n_sessions_per_class=2, seed=3)
    manifest = tmp_path / "external" / "watch" / "manifest.json"
    _write_external_manifest(manifest, object_id="watch", n=10)

    _records, summary = validate_dataset(root, min_samples_per_class=4, external_manifests=[manifest])

    assert summary.per_source_counts["external:open-images-v7"] == {"watch": 10}
    # 6 positive + 1 hard_negative from the synthetic fixture, same source.
    assert summary.per_source_counts["camera"]["circle"] == 7


def test_external_samples_are_image_disjoint_across_splits(tmp_path):
    from predictivesense.training.dataset import build_crop_records
    from predictivesense.training.splits import build_crop_splits

    manifest = tmp_path / "external" / "watch" / "manifest.json"
    # Two boxes on the SAME image (image_id "img0" repeated) - both must land
    # in the same split (section 7.3: source- and image-disjoint, never
    # crop-disjoint).
    import json

    samples = [
        {
            "sample_id": "ext_watch_0a", "object_id": "watch", "image_id": "img0",
            "image_path": "/fake/img0.jpg", "box": [0, 0, 5, 5], "role": "positive",
            "negative_for": [], "box_confirmed_by_human": True,
        },
        {
            "sample_id": "ext_watch_0b", "object_id": "watch", "image_id": "img0",
            "image_path": "/fake/img0.jpg", "box": [5, 5, 5, 5], "role": "positive",
            "negative_for": [], "box_confirmed_by_human": True,
        },
    ] + [
        {
            "sample_id": f"ext_watch_{i}", "object_id": "watch", "image_id": f"img{i}",
            "image_path": f"/fake/img{i}.jpg", "box": [0, 0, 5, 5], "role": "positive",
            "negative_for": [], "box_confirmed_by_human": True,
        }
        for i in range(1, 10)
    ]
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps({"version": 1, "source": "open-images-v7", "object_id": "watch", "samples": samples}),
        encoding="utf-8",
    )
    records = build_crop_records(tmp_path / "objects_empty", external_manifests=[manifest])
    splits = build_crop_splits(records, train_fraction=0.6, val_fraction=0.2, test_fraction=0.2, seed=0)

    split_of = {cid: name for name, ids in splits.crop_ids.items() for cid in ids}
    assert split_of["ext_watch_0a"] == split_of["ext_watch_0b"]
