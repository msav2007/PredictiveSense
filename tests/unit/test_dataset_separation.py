"""The object training set (data/objects) and the Phase 2.5 evaluation set
(data/eval) must never merge (P4 Block 4.6.22).

Asserts: no image path, no image content hash and no source identifier appears in
both stores, and that scripts/export_objects_coco.py refuses a violating export.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from predictivesense.objects.registry import ObjectRegistry
from predictivesense.objects.samples import SampleStore

from scripts import export_objects_coco as exporter

pytestmark = pytest.mark.unit


def _jpeg(fill: int) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", np.full((48, 64, 3), fill, np.uint8))
    assert ok
    return buf.tobytes()


def _make_object_store(root, *, image_bytes, original_filename=None) -> None:
    reg = ObjectRegistry(root)
    reg.create("mug")
    store = SampleStore(reg.object_dir("mug"), "mug")
    store.add(
        image_bytes=image_bytes,
        width=64,
        height=48,
        box=[8, 8, 20, 20],
        conditions={},
        role="positive",
        source="upload" if original_filename else "camera",
        original_filename=original_filename,
    )


def _make_eval_set(root, *, frame_bytes=None, source_clip=None) -> None:
    frames = root / "frames"
    frames.mkdir(parents=True)
    if frame_bytes is not None:
        (frames / "eval_0.jpg").write_bytes(frame_bytes)
    images = []
    if source_clip is not None:
        images.append(
            {
                "id": 1,
                "file_name": "eval_0.jpg",
                "width": 64,
                "height": 48,
                "ps_provenance": {"source_clip": source_clip},
            }
        )
    (root / "annotations.json").write_text(
        json.dumps({"images": images, "annotations": [], "categories": []}),
        encoding="utf-8",
    )


def test_disjoint_stores_have_no_violations(tmp_path) -> None:
    obj_root = tmp_path / "objects"
    eval_root = tmp_path / "eval"
    _make_object_store(obj_root, image_bytes=_jpeg(30))
    _make_eval_set(eval_root, frame_bytes=_jpeg(200))
    assert exporter.check_dataset_separation(obj_root, eval_root) == []


def test_shared_image_content_hash_is_a_violation(tmp_path) -> None:
    obj_root = tmp_path / "objects"
    eval_root = tmp_path / "eval"
    shared = _jpeg(123)
    _make_object_store(obj_root, image_bytes=shared)
    _make_eval_set(eval_root, frame_bytes=shared)
    violations = exporter.check_dataset_separation(obj_root, eval_root)
    assert violations and any("content hash" in v for v in violations)


def test_shared_source_filename_is_a_violation(tmp_path) -> None:
    obj_root = tmp_path / "objects"
    eval_root = tmp_path / "eval"
    _make_object_store(obj_root, image_bytes=_jpeg(30), original_filename="cabin_clip_07.webm")
    _make_eval_set(eval_root, frame_bytes=_jpeg(200), source_clip="/data/raw/x/cabin_clip_07.webm")
    violations = exporter.check_dataset_separation(obj_root, eval_root)
    assert violations and any("source clip" in v for v in violations)


def test_exporter_refuses_a_violating_export(tmp_path, monkeypatch) -> None:
    obj_root = tmp_path / "objects"
    eval_root = tmp_path / "eval"
    shared = _jpeg(77)
    _make_object_store(obj_root, image_bytes=shared)
    _make_eval_set(eval_root, frame_bytes=shared)

    from predictivesense.config.settings import load_config

    cfg = load_config("dev").model_copy(
        update={
            "objects": load_config("dev").objects.model_copy(update={"root": obj_root}),
            "dataset": load_config("dev").dataset.model_copy(update={"root": eval_root}),
        }
    )
    monkeypatch.setattr(exporter, "load_config", lambda *_a, **_k: cfg)
    rc = exporter.main(["--out", str(tmp_path / "coco_train.json")])
    assert rc == 2
    assert not (tmp_path / "coco_train.json").exists()


def test_split_of_is_deterministic_and_object_balanced() -> None:
    # deterministic per sample id, no wrapper randomness
    ids = [f"sample-{i:03d}" for i in range(400)]
    once = [exporter._split_of(i, 0.25) for i in ids]
    twice = [exporter._split_of(i, 0.25) for i in ids]
    assert once == twice
    val = sum(1 for s in once if s == "val")
    assert 0.20 * 400 < val < 0.30 * 400  # roughly the requested fraction


def test_exporter_builds_disjoint_train_val_split(tmp_path, monkeypatch) -> None:
    obj_root = tmp_path / "objects"
    eval_root = tmp_path / "eval"
    reg = ObjectRegistry(obj_root)
    reg.create("mug")
    reg.create("cup")
    store_m = SampleStore(reg.object_dir("mug"), "mug")
    store_c = SampleStore(reg.object_dir("cup"), "cup")
    for i in range(30):
        store_m.add(image_bytes=_jpeg(10 + i), width=64, height=48, box=[4, 4, 20, 20],
                    conditions={}, role="positive")
        store_c.add(image_bytes=_jpeg(40 + i), width=64, height=48, box=[4, 4, 20, 20],
                    conditions={}, role="positive")
    _make_eval_set(eval_root, frame_bytes=_jpeg(255))

    from predictivesense.config.settings import load_config

    base = load_config("dev")
    cfg = base.model_copy(
        update={
            "objects": base.objects.model_copy(update={"root": obj_root}),
            "dataset": base.dataset.model_copy(update={"root": eval_root}),
        }
    )
    monkeypatch.setattr(exporter, "load_config", lambda *_a, **_k: cfg)
    out = tmp_path / "coco_train.json"
    assert exporter.main(["--out", str(out), "--val-fraction", "0.25"]) == 0

    train = json.loads(out.read_text(encoding="utf-8"))
    val = json.loads((tmp_path / "coco_val.json").read_text(encoding="utf-8"))
    train_files = {im["file_name"] for im in train["images"]}
    val_files = {im["file_name"] for im in val["images"]}
    assert train_files
    assert train_files.isdisjoint(val_files)
    assert len(train_files) + len(val_files) == 60
    # 60 random-uuid samples at 0.25 -> a val split with overwhelming probability
    assert val_files, "expected a non-empty val split from 60 samples"
    # every object appears in the training split (>=1 guaranteed)
    assert {im["ps_object_id"] for im in train["images"]} == {"mug", "cup"}
    manifest = json.loads((tmp_path / "export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["train_boxes"] + manifest["counts"]["val_boxes"] == 60
