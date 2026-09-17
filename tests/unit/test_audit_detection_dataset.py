"""Stage 5 section 3.3: duplicate / blur / geometry / licence audit over a
small synthetic detection store."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from predictivesense.dataset.coco_store import CocoStore, domain_categories
from predictivesense.dataset.detection_convert import DetectionBox, DetectionImageRecord, add_detection_record
from scripts.audit_detection_dataset import audit_detection_store

pytestmark = pytest.mark.unit


def _write_image(path, seed: int, size=(64, 64)) -> None:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(*size, 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)


def _record(name, path, *, ps_class, box, seed_identical_to=None):
    return DetectionImageRecord(
        file_name=name, width=64, height=64, session_id=f"grp:{name}",
        image_path=str(path), image_sha256="unused-placeholder",
        source_dataset="test", source_version="v1", original_image_id=name,
        license="cc-by-2.0", author="tester", source_url="http://example.com",
        boxes=(DetectionBox(ps_class=ps_class, bbox=box, source_class=ps_class, box_confirmed_by_human=True),) if box else (),
    )


def test_audit_reports_per_class_totals_and_zero_box_images(tmp_path):
    store = CocoStore.create(domain_categories(["watch", "mug/cup"]))
    p1 = tmp_path / "a.jpg"
    p2 = tmp_path / "b.jpg"
    p3 = tmp_path / "c.jpg"
    _write_image(p1, 1)
    _write_image(p2, 2)
    _write_image(p3, 3)

    add_detection_record(store, _record("a", p1, ps_class="watch", box=(5.0, 5.0, 20.0, 20.0)))
    add_detection_record(store, _record("b", p2, ps_class="mug/cup", box=(5.0, 5.0, 10.0, 10.0)))
    add_detection_record(store, _record("c", p3, ps_class="watch", box=None))  # hard negative, zero boxes

    report = audit_detection_store(store)
    assert report["per_class_totals"]["watch"]["images"] == 1
    assert report["per_class_totals"]["watch"]["boxes"] == 1
    assert report["zero_box_images"] == 1
    assert report["license_breakdown"] == {"cc-by-2.0": 3}


def test_exact_byte_duplicate_images_are_grouped_not_dropped(tmp_path):
    store = CocoStore.create(domain_categories(["watch"]))
    p1 = tmp_path / "a.jpg"
    p2 = tmp_path / "a_copy.jpg"
    _write_image(p1, 42)
    import shutil
    shutil.copy(p1, p2)

    add_detection_record(store, _record("a", p1, ps_class="watch", box=(1.0, 1.0, 10.0, 10.0)))
    add_detection_record(store, _record("a_copy", p2, ps_class="watch", box=(1.0, 1.0, 10.0, 10.0)))

    report = audit_detection_store(store)
    assert len(store.image_ids()) == 2  # nothing deleted
    assert len(report["exact_byte_duplicate_groups"]) == 1
    (group,) = report["exact_byte_duplicate_groups"].values()
    assert len(group) == 2


def test_a_tiny_box_is_counted_below_min_area(tmp_path):
    store = CocoStore.create(domain_categories(["watch"]))
    p1 = tmp_path / "tiny.jpg"
    _write_image(p1, 7)
    add_detection_record(store, _record("tiny", p1, ps_class="watch", box=(1.0, 1.0, 1.0, 1.0)))  # 1x1 of 64x64
    report = audit_detection_store(store, min_box_area_frac=0.01)
    assert report["boxes_below_min_area_or_extreme_aspect"] == 1
