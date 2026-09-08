"""ObjectRegistry + SampleStore: CRUD, slug collisions, counts, atomic writes,
soft delete, and corrupt-manifest rejection (Phase 4)."""

from __future__ import annotations

import json

import pytest

from predictivesense.objects.registry import ObjectRegistry, ObjectStoreError, slugify
from predictivesense.objects.samples import SampleStore

pytestmark = pytest.mark.unit


def _reg(tmp_path):
    return ObjectRegistry(tmp_path / "objects")


def test_create_read_update_and_slug(tmp_path) -> None:
    reg = _reg(tmp_path)
    a = reg.create("My Watch!", kind="class", category="wearable")
    assert a.object_id == "my-watch"
    assert a.kind == "class" and a.category == "wearable"
    assert a.created_utc and a.updated_utc

    # confusable_with is seeded from the observed failure modes
    assert "clock" in reg.create("watch").confusable_with

    got = reg.get("my-watch")
    assert got.name == "My Watch!"
    upd = reg.update("my-watch", name="Renamed", status="ready_for_training")
    assert upd.name == "Renamed" and upd.status == "ready_for_training"
    assert reg.get("my-watch").updated_utc >= a.updated_utc


def test_slug_collisions_are_disambiguated(tmp_path) -> None:
    reg = _reg(tmp_path)
    ids = [reg.create("cup").object_id for _ in range(3)]
    assert ids == ["cup", "cup-2", "cup-3"]
    assert slugify("   ") == "object"


def test_instance_requires_parent_class(tmp_path) -> None:
    reg = _reg(tmp_path)
    with pytest.raises(ObjectStoreError):
        reg.create("Dad's mug", kind="instance")
    ok = reg.create("Dad's mug", kind="instance", parent_class="mug")
    assert ok.kind == "instance" and ok.parent_class == "mug"


def test_atomic_write_no_tempfile_left_and_valid_json(tmp_path) -> None:
    reg = _reg(tmp_path)
    reg.create("bottle")
    assert reg.path.is_file()
    assert not reg.path.with_suffix(".json.tmp").exists()
    json.loads(reg.path.read_text(encoding="utf-8"))  # parses


def test_corrupt_registry_is_rejected(tmp_path) -> None:
    reg = _reg(tmp_path)
    reg.create("cup")
    reg.path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(ObjectStoreError):
        reg.list()
    reg.path.write_text('{"profiles": "not a list"}', encoding="utf-8")
    with pytest.raises(ObjectStoreError):
        reg.list()


def test_soft_delete_moves_folder_and_drops_from_registry(tmp_path) -> None:
    reg = _reg(tmp_path)
    reg.create("keyboard")
    (reg.images_dir("keyboard") / "x.jpg").write_bytes(b"jpeg")
    result = reg.soft_delete("keyboard")
    assert result["soft_deleted"] is True
    assert result["moved_to"] and (tmp_path / "objects" / "_deleted" / "keyboard").is_dir()
    assert not reg.exists("keyboard")
    with pytest.raises(ObjectStoreError):
        reg.get("keyboard")


def test_sample_store_counts_and_soft_delete(tmp_path) -> None:
    reg = _reg(tmp_path)
    reg.create("mug")
    store = SampleStore(reg.object_dir("mug"), "mug")

    jpg = b"\xff\xd8\xff" + b"fake-jpeg" * 8
    s1 = store.add(image_bytes=jpg, width=64, height=48, box=[4, 4, 20, 20],
                   conditions={"view": "front"}, role="positive")
    store.add(image_bytes=jpg, width=64, height=48, box=[1, 1, 10, 10],
              conditions={}, role="hard_negative")
    c = store.counts()
    assert c == {"sample_count": 2, "positive_count": 1, "negative_count": 1}
    assert (reg.object_dir("mug") / s1.path).is_file()
    assert s1.git_commit  # provenance stamped
    assert s1.conditions["view"] == "front" and s1.conditions["distance"] == "close"  # filled

    store.soft_delete(s1.sample_id)
    assert store.counts()["sample_count"] == 1
    assert not (reg.object_dir("mug") / s1.path).is_file()
    assert (reg.object_dir("mug") / "_deleted" / f"{s1.sample_id}.jpg").is_file()


def test_box_outside_bounds_is_rejected(tmp_path) -> None:
    reg = _reg(tmp_path)
    reg.create("cup")
    store = SampleStore(reg.object_dir("cup"), "cup")
    jpg = b"\xff\xd8\xff\xd9"
    with pytest.raises(ObjectStoreError):
        store.add(image_bytes=jpg, width=64, height=48, box=[50, 40, 40, 40],
                  conditions={}, role="positive")
    with pytest.raises(ObjectStoreError):
        store.add(image_bytes=jpg, width=64, height=48, box=[0, 0, 0, 10],
                  conditions={}, role="positive")


def test_corrupt_sample_manifest_is_rejected(tmp_path) -> None:
    reg = _reg(tmp_path)
    reg.create("cup")
    store = SampleStore(reg.object_dir("cup"), "cup")
    store._dir.mkdir(parents=True, exist_ok=True)
    (store._dir / "manifest.json").write_text("{bad", encoding="utf-8")
    with pytest.raises(ObjectStoreError):
        store.list()
