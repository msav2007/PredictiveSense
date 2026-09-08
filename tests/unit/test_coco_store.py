"""COCO store: round-trip, schema validation, id allocation, seeded flag."""

from __future__ import annotations

import json

import pytest

from predictivesense.dataset.coco_store import CocoStore, CocoStoreError, domain_categories

pytestmark = pytest.mark.unit

_VOCAB = ("person", "cup", "bottle")


def _store() -> CocoStore:
    return CocoStore.create(domain_categories(_VOCAB))


def test_round_trip_preserves_everything(tmp_path) -> None:
    s = _store()
    iid = s.add_image(
        file_name="a.jpg", width=640, height=480, session_id="sess1",
        provenance={"source_clip": "c.webm", "timestamp_s": 1.5},
    )
    s.set_frame_boxes(
        iid, [{"category": "cup", "bbox": [10, 20, 30, 40]}], seeded=True
    )
    path = s.save(tmp_path / "ann.json")

    back = CocoStore.load(path)
    im = back.image(iid)
    assert im["seeded"] is True and im["labelled"] is True
    assert im["ps_session_id"] == "sess1"
    assert im["ps_provenance"]["source_clip"] == "c.webm"
    anns = back.annotations_for(iid)
    assert len(anns) == 1
    assert anns[0]["bbox"] == [10.0, 20.0, 30.0, 40.0]
    assert back.category_name(anns[0]["category_id"]) == "cup"
    assert back.content_hash() == s.content_hash()


def test_schema_validation_rejects_malformed(tmp_path) -> None:
    bad = tmp_path / "bad.json"

    bad.write_text(json.dumps({"images": [], "annotations": []}), encoding="utf-8")  # no categories
    with pytest.raises(CocoStoreError):
        CocoStore.load(bad)

    bad.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "a.jpg", "width": 10, "height": 10}],
                "annotations": [{"id": 1, "image_id": 99, "category_id": 1, "bbox": [0, 0, 1, 1]}],
                "categories": [{"id": 1, "name": "person"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(CocoStoreError):  # annotation references a missing image
        CocoStore.load(bad)

    bad.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "a.jpg", "width": 10, "height": 10}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, -5, 5]}],
                "categories": [{"id": 1, "name": "person"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(CocoStoreError):  # non-positive w/h
        CocoStore.load(bad)


def test_id_allocation_never_collides(tmp_path) -> None:
    s = _store()
    ids = [s.add_image(file_name=f"f{i}.jpg", width=8, height=8, session_id="s") for i in range(5)]
    assert len(set(ids)) == 5
    for i, iid in enumerate(ids):
        s.set_frame_boxes(iid, [{"category": "person", "bbox": [0, 0, 4, 4]}], seeded=False)
    # reload and add more - ids continue past the persisted maximum
    path = s.save(tmp_path / "a.json")
    s2 = CocoStore.load(path)
    new_id = s2.add_image(file_name="new.jpg", width=8, height=8, session_id="s")
    assert new_id not in ids
    all_ann_ids = [a["id"] for a in s2.to_doc()["annotations"]]
    assert len(all_ann_ids) == len(set(all_ann_ids))


def test_re_adding_same_file_name_is_idempotent() -> None:
    s = _store()
    a = s.add_image(file_name="x.jpg", width=8, height=8, session_id="s")
    b = s.add_image(file_name="x.jpg", width=8, height=8, session_id="s")
    assert a == b and len(s.image_ids()) == 1


def test_save_backs_up_before_overwrite(tmp_path) -> None:
    s = _store()
    s.add_image(file_name="a.jpg", width=8, height=8, session_id="s")
    p = s.save(tmp_path / "ann.json")
    assert not (tmp_path / "ann.json.bak").exists()
    s.add_image(file_name="b.jpg", width=8, height=8, session_id="s")
    s.save(p)
    assert (tmp_path / "ann.json.bak").is_file()


def test_seeded_flag_updates_on_relabel() -> None:
    s = _store()
    iid = s.add_image(file_name="a.jpg", width=8, height=8, session_id="s")
    s.set_frame_boxes(iid, [{"category": "cup", "bbox": [0, 0, 2, 2]}], seeded=True)
    assert s.image(iid)["seeded"] is True
    s.set_frame_boxes(iid, [{"category": "cup", "bbox": [0, 0, 2, 2]}], seeded=False)
    assert s.image(iid)["seeded"] is False
    assert len(s.annotations_for(iid)) == 1  # replaced, not appended


def test_counts_report_per_class_and_per_session() -> None:
    s = _store()
    a = s.add_image(file_name="a.jpg", width=8, height=8, session_id="s1")
    b = s.add_image(file_name="b.jpg", width=8, height=8, session_id="s2")
    s.set_frame_boxes(a, [{"category": "person", "bbox": [0, 0, 2, 2]},
                          {"category": "cup", "bbox": [1, 1, 2, 2]}], seeded=False)
    s.set_frame_boxes(b, [{"category": "person", "bbox": [0, 0, 2, 2]}], seeded=True)
    c = s.counts()
    assert c["labelled"] == 2 and c["seeded"] == 1 and c["unseeded"] == 1
    assert c["per_class"]["person"] == 2 and c["per_class"]["cup"] == 1
    assert set(c["per_session"]) == {"s1", "s2"}
    assert c["unseeded_fraction"] == 0.5
