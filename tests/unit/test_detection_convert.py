"""Stage 5: image-oriented detection records -> the existing COCO store.
Zero-box images survive as zero-annotation records (section 2.2); own-capture
records carry session id / role / provenance / quality (section 3.2/3.5).
"""

from __future__ import annotations

from predictivesense.dataset.coco_store import CocoStore, domain_categories
from predictivesense.dataset.detection_convert import (
    DetectionBox,
    DetectionImageRecord,
    add_detection_record,
    own_capture_records,
)

import pytest

pytestmark = pytest.mark.unit


def _store() -> CocoStore:
    return CocoStore.create(domain_categories(["watch", "mug/cup"]))


def test_a_record_with_boxes_is_added_with_correct_category_and_bbox():
    store = _store()
    record = DetectionImageRecord(
        file_name="ext:oi:img1", width=100, height=100, session_id="grp1",
        image_path="/tmp/img1.jpg", image_sha256="abc123",
        source_dataset="open-images-v7", source_version="train",
        original_image_id="img1",
        boxes=(DetectionBox(ps_class="watch", bbox=(1.0, 2.0, 10.0, 10.0), source_class="Watch", box_confirmed_by_human=True),),
    )
    iid = add_detection_record(store, record)
    anns = store.annotations_for(iid)
    assert len(anns) == 1
    assert store.category_name(anns[0]["category_id"]) == "watch"
    assert anns[0]["bbox"] == [1.0, 2.0, 10.0, 10.0]
    assert anns[0]["source_class"] == "Watch"
    assert anns[0]["box_confirmed_by_human"] is True


def test_zero_box_image_survives_as_a_labelled_zero_annotation_record():
    store = _store()
    record = DetectionImageRecord(
        file_name="ext:oi:hardneg1", width=100, height=100, session_id="grp2",
        image_path="/tmp/hardneg1.jpg", image_sha256="def456",
        source_dataset="open-images-v7", source_version="train",
        original_image_id="hardneg1", boxes=(),
    )
    iid = add_detection_record(store, record)
    assert store.annotations_for(iid) == []
    assert store.image(iid)["labelled"] is True


def test_two_boxes_of_different_target_classes_both_survive_on_one_image():
    store = _store()
    record = DetectionImageRecord(
        file_name="ext:oi:multi1", width=200, height=200, session_id="grp3",
        image_path="/tmp/multi1.jpg", image_sha256="ghi789",
        source_dataset="open-images-v7", source_version="train",
        original_image_id="multi1",
        boxes=(
            DetectionBox(ps_class="watch", bbox=(1.0, 1.0, 5.0, 5.0), source_class="Watch", box_confirmed_by_human=True),
            DetectionBox(ps_class="mug/cup", bbox=(50.0, 50.0, 20.0, 20.0), source_class="Mug", box_confirmed_by_human=True),
        ),
    )
    iid = add_detection_record(store, record)
    names = sorted(store.category_name(a["category_id"]) for a in store.annotations_for(iid))
    assert names == ["mug/cup", "watch"]
    assert store.image(iid)["ps_provenance"]["target_classes_present"] == ["mug/cup", "watch"]


def test_re_adding_the_same_file_name_is_idempotent():
    store = _store()
    record = DetectionImageRecord(
        file_name="ext:oi:img1", width=100, height=100, session_id="grp1",
        image_path="/tmp/img1.jpg", image_sha256="abc123",
        source_dataset="open-images-v7", source_version="train",
        original_image_id="img1", boxes=(),
    )
    iid1 = add_detection_record(store, record)
    iid2 = add_detection_record(store, record)
    assert iid1 == iid2
    assert len(store.image_ids()) == 1


def _fake_session_key(object_id: str, sample: dict) -> str:
    return f"{object_id}:sess:{sample.get('batch_id', 'nobatch')}"


def test_own_capture_positive_sample_carries_session_role_provenance_quality(tmp_path):
    obj_dir = tmp_path / "comb"
    (obj_dir / "images").mkdir(parents=True)
    (obj_dir / "images" / "s1.jpg").write_bytes(b"fake-jpeg-bytes")
    samples = [{
        "sample_id": "s1", "path": "images/s1.jpg", "role": "positive",
        "box": [1.0, 2.0, 3.0, 4.0], "width": 640, "height": 480,
        "box_confirmed_by_human": None, "device_label": "cam1",
        "captured_utc": "2026-09-16T15:15:30.136363Z", "conditions": {"view": "back"},
        "consent_ack": True, "quality": {"blur_var": 100.0, "phash": "abcd"},
        "git_commit": "deadbeef", "batch_id": None,
    }]
    records = own_capture_records(
        "comb", "comb", samples, object_dir_str=str(obj_dir), session_key_fn=_fake_session_key,
    )
    assert len(records) == 1
    r = records[0]
    assert r.session_id == "comb:sess:None"
    assert r.own_capture_role == "positive"
    assert r.own_capture_provenance["device_label"] == "cam1"
    assert r.own_capture_quality == {"blur_var": 100.0, "phash": "abcd"}
    assert len(r.boxes) == 1
    assert r.boxes[0].bbox == (1.0, 2.0, 3.0, 4.0)
    assert r.image_sha256 != ""  # real sha256 of the on-disk bytes, not the phash


def test_own_capture_negative_sample_has_no_boxes():
    samples = [{
        "sample_id": "s2", "path": "images/s2.jpg", "role": "negative",
        "box": None, "width": 640, "height": 480, "batch_id": "b1",
    }]
    records = own_capture_records(
        "comb", "comb", samples, object_dir_str="/nonexistent", session_key_fn=_fake_session_key,
    )
    assert records[0].boxes == ()
    assert records[0].own_capture_role == "negative"
