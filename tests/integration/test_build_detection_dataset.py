"""Stage 5 section 4/5: the end-to-end detection-dataset build, on synthetic
fixtures mirroring the real Open Images export layout (never the real
2.2GB download) plus a synthetic Studio object store.

Covers: a real external multi-class build succeeds and produces a store +
splits + audit; the own-capture minimum-image-count refusal fires with a
clear message naming the class and its count (section 3.5, "one comb image
must not become a training set by accident"); a train/val <-> eval collision
hard-fails the whole build (section 3.4); determinism (same seed -> same
split content hash).
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from predictivesense.objects.registry import ObjectRegistry
from predictivesense.objects.samples import SampleStore
from predictivesense.dataset.coco_store import CocoStore
from predictivesense.dataset.detection_splits import DetectionSplitError, EvalCollisionError
from scripts.build_detection_dataset import BuildAbortedError, run_build

pytestmark = pytest.mark.integration


def _write_class_map(path, entries):
    path.write_text(json.dumps({"entries": entries}), encoding="utf-8")


def _jpeg_bytes(seed: int, size=(64, 64)) -> bytes:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(*size, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return buf.tobytes()


def _build_oi_export(root, *, images: dict[str, int], boxes: list[tuple]) -> None:
    split_dir = root / "train"
    data_dir, labels_dir, metadata_dir = split_dir / "data", split_dir / "labels", split_dir / "metadata"
    for d in (data_dir, labels_dir, metadata_dir):
        d.mkdir(parents=True, exist_ok=True)
    for image_id, seed in images.items():
        (data_dir / f"{image_id}.jpg").write_bytes(_jpeg_bytes(seed))
    (metadata_dir / "classes.csv").write_text(
        "/m/0gjkl,Watch\n/m/02jvh9,Mug\n/m/02p5f1q,Coffee cup\n", encoding="utf-8"
    )
    rows = ["ImageID,LabelName,XMin,XMax,YMin,YMax\n"]
    for image_id, mid, xmin, xmax, ymin, ymax in boxes:
        rows.append(f"{image_id},{mid},{xmin},{xmax},{ymin},{ymax}\n")
    (labels_dir / "detections.csv").write_text("".join(rows), encoding="utf-8")
    lic_rows = ["ImageID,License,Author,OriginalURL,Title\n"]
    for image_id in images:
        lic_rows.append(f"{image_id},https://creativecommons.org/licenses/by/2.0/,Tester,http://example.com,t\n")
    (metadata_dir / "image_ids.csv").write_text("".join(lic_rows), encoding="utf-8")


_CLASS_MAP_ENTRIES = [
    {"ps_class": "watch", "status": "mapped", "source_classes": [{"dataset": "open-images-v7", "name": "Watch", "mid": "/m/0gjkl"}]},
    {"ps_class": "mug/cup", "status": "mapped", "source_classes": [
        {"dataset": "open-images-v7", "name": "Mug", "mid": "/m/02jvh9"},
        {"dataset": "open-images-v7", "name": "Coffee cup", "mid": "/m/02p5f1q"},
    ]},
    {"ps_class": "comb", "status": "unavailable", "source_classes": []},
]


def test_a_real_multi_class_build_succeeds_with_enough_images(tmp_path):
    class_map = tmp_path / "class_map.json"
    _write_class_map(class_map, _CLASS_MAP_ENTRIES)
    oi_root = tmp_path / "oi"
    # 60 watch-only images, 60 mug-only images, 1 image with BOTH (exercises exhaustive requery for real)
    images = {f"w{i}": i for i in range(60)}
    images.update({f"m{i}": 1000 + i for i in range(60)})
    images["dual0"] = 9999
    boxes = [(f"w{i}", "/m/0gjkl", 0.1, 0.3, 0.1, 0.3) for i in range(60)]
    boxes += [(f"m{i}", "/m/02jvh9", 0.1, 0.3, 0.1, 0.3) for i in range(60)]
    boxes += [("dual0", "/m/0gjkl", 0.05, 0.25, 0.05, 0.25), ("dual0", "/m/02p5f1q", 0.5, 0.9, 0.5, 0.9)]
    _build_oi_export(oi_root, images=images, boxes=boxes)

    out_dir = tmp_path / "out"
    summary = run_build(
        external_classes=["watch", "mug/cup"], own_capture=[],
        class_map_path=class_map, oi_root=oi_root, oi_split="train",
        objects_root=tmp_path / "objects", eval_root=tmp_path / "eval", out_dir=out_dir,
        train_fraction=0.7, val_fraction=0.15, test_fraction=0.15, seed=0, min_images_per_class=10,
    )
    assert summary["n_images"] == 121
    assert summary["per_class_totals"]["watch"]["images"] == 61  # 60 + dual0
    assert summary["per_class_totals"]["mug/cup"]["images"] == 61  # 60 + dual0
    store = CocoStore.load(out_dir / "coco_detection.json")
    dual_id = next(i for i in store.image_ids() if "dual0" in store.image(i)["file_name"])
    dual_classes = {store.category_name(a["category_id"]) for a in store.annotations_for(dual_id)}
    assert dual_classes == {"watch", "mug/cup"}, "exhaustive requery must keep both classes on the shared image"


def test_same_seed_gives_identical_split_content_hash(tmp_path):
    class_map = tmp_path / "class_map.json"
    _write_class_map(class_map, _CLASS_MAP_ENTRIES)
    oi_root = tmp_path / "oi"
    images = {f"w{i}": i for i in range(20)}
    boxes = [(f"w{i}", "/m/0gjkl", 0.1, 0.3, 0.1, 0.3) for i in range(20)]
    _build_oi_export(oi_root, images=images, boxes=boxes)

    kwargs = dict(
        external_classes=["watch"], own_capture=[], class_map_path=class_map, oi_root=oi_root, oi_split="train",
        objects_root=tmp_path / "objects", eval_root=tmp_path / "eval",
        train_fraction=0.7, val_fraction=0.15, test_fraction=0.15, seed=7, min_images_per_class=5,
    )
    s1 = run_build(out_dir=tmp_path / "out1", **kwargs)
    s2 = run_build(out_dir=tmp_path / "out2", **kwargs)
    assert s1["split_content_hash"] == s2["split_content_hash"]


def test_own_capture_with_one_image_refuses_a_training_split(tmp_path):
    class_map = tmp_path / "class_map.json"
    _write_class_map(class_map, _CLASS_MAP_ENTRIES)
    objects_root = tmp_path / "objects"
    reg = ObjectRegistry(objects_root)
    reg.create("comb")
    store = SampleStore(reg.object_dir("comb"), "comb")
    store.add(image_bytes=_jpeg_bytes(1), width=64, height=64, box=[4, 4, 20, 20], conditions={}, role="positive")

    with pytest.raises(DetectionSplitError, match=r"comb.*1 image"):
        run_build(
            external_classes=[], own_capture=[("comb", "comb")],
            class_map_path=class_map, oi_root=tmp_path / "unused", oi_split="train",
            objects_root=objects_root, eval_root=tmp_path / "eval", out_dir=tmp_path / "out",
            train_fraction=0.7, val_fraction=0.15, test_fraction=0.15, seed=0, min_images_per_class=50,
        )


def test_own_capture_images_in_different_sessions_can_land_in_different_splits(tmp_path):
    """Session-disjointness (stage5 section 3.4/3.5): every image within ONE
    session must land in the same split (proven directly against
    build_detection_splits' image_to_group below), and the own-capture path
    derives that session the identical way the crop-classifier path does
    (predictivesense.training.dataset.session_key_for, reused not
    reimplemented) - a same-session pair must never straddle a split."""

    class_map = tmp_path / "class_map.json"
    _write_class_map(class_map, _CLASS_MAP_ENTRIES + [
        {"ps_class": "watch2", "status": "mapped", "source_classes": []},
    ])
    objects_root = tmp_path / "objects"
    reg = ObjectRegistry(objects_root)
    reg.create("watch2")
    store = SampleStore(reg.object_dir("watch2"), "watch2")
    # two samples in the SAME bulk-upload batch (= one session), 60 more spread
    # across distinct batches so the per-class group count clears the minimum.
    store.add(image_bytes=_jpeg_bytes(1), width=64, height=64, box=[4, 4, 20, 20], conditions={}, role="positive", batch_id="batchA")
    store.add(image_bytes=_jpeg_bytes(2), width=64, height=64, box=[4, 4, 20, 20], conditions={}, role="positive", batch_id="batchA")
    for i in range(3, 63):
        store.add(image_bytes=_jpeg_bytes(i), width=64, height=64, box=[4, 4, 20, 20], conditions={}, role="positive", batch_id=f"batch{i}")

    summary = run_build(
        external_classes=[], own_capture=[("watch2", "watch2")],
        class_map_path=class_map, oi_root=tmp_path / "unused", oi_split="train",
        objects_root=objects_root, eval_root=tmp_path / "eval", out_dir=tmp_path / "out",
        train_fraction=0.7, val_fraction=0.15, test_fraction=0.15, seed=0, min_images_per_class=10,
    )
    from predictivesense.dataset.detection_splits import load_detection_splits
    splits = load_detection_splits(tmp_path / "out" / "splits.json")
    all_ids = splits.image_ids["train"] + splits.image_ids["val"] + splits.image_ids["test"]
    assert len(all_ids) == len(set(all_ids)) == 62
    store_obj = CocoStore.load(tmp_path / "out" / "coco_detection.json")
    batch_a_ids = [i for i in store_obj.image_ids() if store_obj.session_of(i) == "watch2:batch:batchA"]
    assert len(batch_a_ids) == 2
    # both batchA images must land in the SAME split
    in_split = {name: set(ids) for name, ids in splits.image_ids.items()}
    homes = [name for iid in batch_a_ids for name, ids in in_split.items() if iid in ids]
    assert len(set(homes)) == 1, f"batchA session split across splits: {homes}"


def test_a_training_image_matching_an_eval_frame_hard_fails_the_whole_build(tmp_path):
    class_map = tmp_path / "class_map.json"
    _write_class_map(class_map, _CLASS_MAP_ENTRIES)
    oi_root = tmp_path / "oi"
    shared_bytes_seed = 55
    images = {f"w{i}": i for i in range(1, 20)}
    images["dup_of_eval"] = shared_bytes_seed
    boxes = [(f"w{i}", "/m/0gjkl", 0.1, 0.3, 0.1, 0.3) for i in range(1, 20)]
    boxes.append(("dup_of_eval", "/m/0gjkl", 0.1, 0.3, 0.1, 0.3))
    _build_oi_export(oi_root, images=images, boxes=boxes)

    eval_root = tmp_path / "eval"
    (eval_root / "frames").mkdir(parents=True)
    (eval_root / "frames" / "eval_0.jpg").write_bytes(_jpeg_bytes(shared_bytes_seed))

    out_dir = tmp_path / "out"
    with pytest.raises(BuildAbortedError):
        run_build(
            external_classes=["watch"], own_capture=[],
            class_map_path=class_map, oi_root=oi_root, oi_split="train",
            objects_root=tmp_path / "objects", eval_root=eval_root, out_dir=out_dir,
            train_fraction=0.7, val_fraction=0.15, test_fraction=0.15, seed=0, min_images_per_class=5,
        )
    assert not (out_dir / "splits.json").exists()
