"""Phase 12 section 12: the external-dataset importer, on a synthetic fixture
mirroring the real FiftyOne export layout (never the real 2.2GB download).

Covers: the class mapping rejects an invalid/unavailable mapping; filtering
rejects corrupt/undersized/out-of-bounds/extreme-aspect boxes with counted
reasons; deduplication catches exact and near duplicates within and across
stores; an external<->eval collision fails the import (nothing written);
licence/attribution fields survive the import.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from predictivesense.objects.registry import ObjectRegistry
from predictivesense.objects.samples import SampleStore
from scripts.import_external_dataset import ImportAbortedError, import_class

pytestmark = pytest.mark.unit


def _jpeg_bytes(seed: int, size=(48, 64)) -> bytes:
    """A deterministic per-seed RANDOM-noise image, not a solid fill: a
    perceptual hash correctly calls two near-solid-colour images "duplicates"
    of each other, so distinct logical test images need real texture to be
    distinguishable, while passing the same seed still reproduces identical
    bytes for the intentional-duplicate tests."""

    import cv2

    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(*size, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return buf.tobytes()


def _write_jpeg(path, seed: int, size=(48, 64)) -> None:
    path.write_bytes(_jpeg_bytes(seed, size))


def _build_export(
    root,
    *,
    images: dict[str, int],  # image_id -> seed (byte-identical images share a seed)
    boxes: list[tuple[str, str, float, float, float, float]],  # (image_id, mid, xmin, xmax, ymin, ymax)
    licenses: dict[str, dict[str, str]] | None = None,
) -> None:
    split_dir = root / "train"
    data_dir = split_dir / "data"
    labels_dir = split_dir / "labels"
    metadata_dir = split_dir / "metadata"
    for d in (data_dir, labels_dir, metadata_dir):
        d.mkdir(parents=True, exist_ok=True)

    for image_id, fill in images.items():
        _write_jpeg(data_dir / f"{image_id}.jpg", fill)

    (metadata_dir / "classes.csv").write_text(
        "/m/0gjkl,Watch\n/m/050k8,Mobile phone\n", encoding="utf-8"
    )

    header = "ImageID,LabelName,XMin,XMax,YMin,YMax\n"
    rows = [header]
    for image_id, mid, xmin, xmax, ymin, ymax in boxes:
        rows.append(f"{image_id},{mid},{xmin},{xmax},{ymin},{ymax}\n")
    (labels_dir / "detections.csv").write_text("".join(rows), encoding="utf-8")

    lic_header = "ImageID,License,Author,OriginalURL,Title\n"
    lic_rows = [lic_header]
    for image_id, meta in (licenses or {}).items():
        lic_rows.append(
            f"{image_id},{meta.get('license', '')},{meta.get('author', '')},"
            f"{meta.get('source_url', '')},{meta.get('title', '')}\n"
        )
    (metadata_dir / "image_ids.csv").write_text("".join(lic_rows), encoding="utf-8")


def _write_mapping(path, *, mapped_classes: dict[str, str]) -> None:
    decisions = [
        {"ps_class": name, "status": "mapped", "oi_classes": [{"mid": mid, "name": name}]}
        for name, mid in mapped_classes.items()
    ]
    decisions.append({"ps_class": "comb", "status": "unavailable", "oi_classes": []})
    path.write_text(json.dumps({"decisions": decisions}), encoding="utf-8")


def test_rejects_import_of_an_unavailable_class(tmp_path):
    mapping = tmp_path / "mapping.json"
    _write_mapping(mapping, mapped_classes={"watch": "/m/0gjkl"})
    root = tmp_path / "oi"
    _build_export(root, images={}, boxes=[])
    with pytest.raises(ValueError, match="unavailable"):
        import_class(
            "comb", root=root, split="train", objects_root=tmp_path / "objects",
            eval_root=tmp_path / "eval", external_root=tmp_path / "external", mapping_path=mapping,
        )


def test_filters_bad_boxes_with_counted_reasons(tmp_path):
    mapping = tmp_path / "mapping.json"
    _write_mapping(mapping, mapped_classes={"watch": "/m/0gjkl"})
    root = tmp_path / "oi"
    _build_export(
        root,
        images={"good": 1, "tiny": 2, "oob": 3, "sliver": 4},
        boxes=[
            ("good", "/m/0gjkl", 0.1, 0.6, 0.1, 0.6),  # valid: 0.5x0.5 = 0.25 area
            ("tiny", "/m/0gjkl", 0.50, 0.51, 0.50, 0.51),  # 0.01x0.01 = 0.0001 area -> too small
            ("oob", "/m/0gjkl", 0.9, 1.05, 0.1, 0.5),  # XMax > 1.0 -> out of bounds
            ("sliver", "/m/0gjkl", 0.0, 0.5, 0.0, 0.03),  # 0.5x0.03=0.015 area, aspect 16.7 -> extreme
        ],
    )
    manifest = import_class(
        "watch", root=root, split="train", objects_root=tmp_path / "objects",
        eval_root=tmp_path / "eval", external_root=tmp_path / "external", mapping_path=mapping,
    )
    assert manifest["counts"]["positive"] == 1
    assert manifest["samples"][0]["image_id"] == "good"
    assert manifest["rejected_reasons"] == {
        "box_too_small": 1, "box_out_of_bounds": 1, "box_extreme_aspect_ratio": 1,
    }


def test_deduplicates_near_identical_images_within_external_set(tmp_path):
    mapping = tmp_path / "mapping.json"
    _write_mapping(mapping, mapped_classes={"watch": "/m/0gjkl"})
    root = tmp_path / "oi"
    _build_export(
        root,
        images={"first": 99, "duplicate": 99},  # byte-identical -> identical dHash
        boxes=[
            ("first", "/m/0gjkl", 0.1, 0.6, 0.1, 0.6),
            ("duplicate", "/m/0gjkl", 0.1, 0.6, 0.1, 0.6),
        ],
    )
    manifest = import_class(
        "watch", root=root, split="train", objects_root=tmp_path / "objects",
        eval_root=tmp_path / "eval", external_root=tmp_path / "external", mapping_path=mapping,
    )
    assert manifest["counts"]["positive"] == 1
    assert manifest["rejected_reasons"].get("near_duplicate_within_external_set") == 1


def test_excludes_image_that_duplicates_an_objects_store_sample(tmp_path):
    mapping = tmp_path / "mapping.json"
    _write_mapping(mapping, mapped_classes={"watch": "/m/0gjkl"})
    objects_root = tmp_path / "objects"
    reg = ObjectRegistry(objects_root)
    reg.create("watch")
    store = SampleStore(reg.object_dir("watch"), "watch")
    shared_bytes = _jpeg_bytes(55)
    store.add(image_bytes=shared_bytes, width=64, height=48, box=[4, 4, 20, 20], conditions={}, role="positive")

    root = tmp_path / "oi"
    _build_export(root, images={"dup_of_studio": 55}, boxes=[("dup_of_studio", "/m/0gjkl", 0.1, 0.6, 0.1, 0.6)])
    manifest = import_class(
        "watch", root=root, split="train", objects_root=objects_root,
        eval_root=tmp_path / "eval", external_root=tmp_path / "external", mapping_path=mapping,
    )
    assert manifest["counts"]["positive"] == 0
    assert manifest["rejected_reasons"].get("cross_store_duplicate_of_objects") == 1


def test_external_eval_collision_aborts_the_import_and_writes_nothing(tmp_path):
    mapping = tmp_path / "mapping.json"
    _write_mapping(mapping, mapped_classes={"watch": "/m/0gjkl"})
    eval_root = tmp_path / "eval"
    (eval_root / "frames").mkdir(parents=True)
    (eval_root / "frames" / "eval_0.jpg").write_bytes(_jpeg_bytes(77))

    root = tmp_path / "oi"
    _build_export(root, images={"dup_of_eval": 77}, boxes=[("dup_of_eval", "/m/0gjkl", 0.1, 0.6, 0.1, 0.6)])
    external_root = tmp_path / "external"
    with pytest.raises(ImportAbortedError):
        import_class(
            "watch", root=root, split="train", objects_root=tmp_path / "objects",
            eval_root=eval_root, external_root=external_root, mapping_path=mapping,
        )
    assert not (external_root / "open-images-v7" / "watch" / "manifest.json").exists()


def test_license_and_attribution_fields_survive_the_import(tmp_path):
    mapping = tmp_path / "mapping.json"
    _write_mapping(mapping, mapped_classes={"watch": "/m/0gjkl"})
    root = tmp_path / "oi"
    _build_export(
        root,
        images={"licensed": 5},
        boxes=[("licensed", "/m/0gjkl", 0.1, 0.6, 0.1, 0.6)],
        licenses={"licensed": {
            "license": "https://creativecommons.org/licenses/by/2.0/",
            "author": "Jane Photographer", "source_url": "https://example.com/photo.jpg",
            "title": "A watch",
        }},
    )
    manifest = import_class(
        "watch", root=root, split="train", objects_root=tmp_path / "objects",
        eval_root=tmp_path / "eval", external_root=tmp_path / "external", mapping_path=mapping,
    )
    sample = manifest["samples"][0]
    assert sample["license"] == "https://creativecommons.org/licenses/by/2.0/"
    assert sample["author"] == "Jane Photographer"
    assert sample["box_confirmed_by_human"] is True


def test_hard_negatives_exclude_images_contaminated_with_the_target_class(tmp_path):
    mapping = tmp_path / "mapping.json"
    _write_mapping(mapping, mapped_classes={"watch": "/m/0gjkl"})
    root = tmp_path / "oi"
    _build_export(
        root,
        images={"watch_img": 1, "phone_only": 2, "both": 3},
        boxes=[
            ("watch_img", "/m/0gjkl", 0.1, 0.6, 0.1, 0.6),
            ("phone_only", "/m/050k8", 0.1, 0.6, 0.1, 0.6),
            ("both", "/m/0gjkl", 0.1, 0.6, 0.1, 0.6),
            ("both", "/m/050k8", 0.2, 0.7, 0.2, 0.7),
        ],
    )
    manifest = import_class(
        "watch", root=root, split="train", objects_root=tmp_path / "objects",
        eval_root=tmp_path / "eval", external_root=tmp_path / "external", mapping_path=mapping,
        hard_negative_mids=["/m/050k8"], hard_negative_limit=10,
    )
    hn = [s for s in manifest["samples"] if s["role"] == "hard_negative"]
    assert {s["image_id"] for s in hn} == {"phone_only"}  # "both" is excluded - it has a watch box too
    assert all(s["negative_for"] == ["watch"] for s in hn)
