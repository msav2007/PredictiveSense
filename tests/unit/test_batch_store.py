"""Bulk-upload staging store (Phase 7, BLOCK 9.1).

Staging create / list / update / discard; manifest round-trip; TTL cleanup;
staged items excluded from sample counts and coverage.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from predictivesense.objects.batches import BatchItem, BatchStore, BatchStoreError
from predictivesense.objects.quality import CoverageTargets, coverage_summary
from predictivesense.objects.registry import ObjectRegistry
from predictivesense.objects.samples import SampleStore

pytestmark = pytest.mark.unit


def _jpeg(fill: int, w: int = 64, h: int = 48) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", np.full((h, w, 3), fill, np.uint8))
    assert ok
    return buf.tobytes()


@pytest.fixture()
def store(tmp_path):
    reg = ObjectRegistry(tmp_path / "objects")
    reg.create("watch")
    return BatchStore(reg.object_dir("watch"), "watch")


def _new_batch(store: BatchStore, n: int = 3, role: str = "positive") -> str:
    items = [BatchItem(item_id=f"it{i}", filename=f"img{i}.jpg", width=64, height=48) for i in range(n)]
    bid = "batch0001"
    store.create(bid, role=role, items=items)
    for i in range(n):
        store.add_staged_image(
            bid, f"it{i}", image_bytes=_jpeg(10 + i), thumb_bytes=_jpeg(200 + i, 24, 18),
            width=64, height=48,
        )
    return bid


def test_create_list_and_manifest_round_trip(store: BatchStore) -> None:
    bid = _new_batch(store)
    assert store.list_batches() == [bid]
    doc = store.load(bid)
    assert doc["object_id"] == "watch" and doc["total"] == 3 and doc["status"] == "processing"
    assert {it["item_id"] for it in doc["items"]} == {"it0", "it1", "it2"}
    for it in doc["items"]:
        assert it["staged_path"] and it["thumbnail_path"]
        assert store.staged_image_path(bid, it["item_id"]).is_file()
        assert store.thumbnail_image_path(bid, it["item_id"]).is_file()
    # BatchItem <-> dict round trip
    rebuilt = BatchItem.from_dict(doc["items"][0])
    assert rebuilt.item_id == "it0" and rebuilt.width == 64


def test_update_item_and_status(store: BatchStore) -> None:
    bid = _new_batch(store)
    store.update_item(bid, "it1", box=[1, 2, 10, 12], status="edited", box_confirmed_by_human=True)
    store.set_progress(bid, 2)
    store.set_status(bid, "ready")
    doc = store.load(bid)
    it1 = next(it for it in doc["items"] if it["item_id"] == "it1")
    assert it1["box"] == [1, 2, 10, 12] and it1["status"] == "edited"
    assert it1["box_confirmed_by_human"] is True
    assert doc["processed"] == 2 and doc["status"] == "ready"
    with pytest.raises(BatchStoreError):
        store.update_item(bid, "nope", box=[0, 0, 1, 1])
    with pytest.raises(BatchStoreError):
        store.update_item(bid, "it1", bogus_field=1)


def test_discard_removes_everything(store: BatchStore) -> None:
    bid = _new_batch(store)
    bdir = store.batch_dir(bid)
    assert bdir.is_dir()
    res = store.discard(bid)
    assert res["discarded"] is True and res["existed"] is True
    assert not bdir.exists()
    assert store.list_batches() == []
    with pytest.raises(BatchStoreError):
        store.load(bid)


def test_ttl_cleanup_drops_only_stale_batches(store: BatchStore) -> None:
    bid = _new_batch(store)
    # not stale yet
    assert store.cleanup_stale(ttl_hours=24) == []
    # backdate the manifest
    mpath = store.manifest_path(bid)
    doc = json.loads(mpath.read_text(encoding="utf-8"))
    doc["created_utc"] = "2000-01-01T00:00:00.000000Z"
    mpath.write_text(json.dumps(doc), encoding="utf-8")
    removed = store.cleanup_stale(ttl_hours=24)
    assert removed == [bid]
    assert not store.batch_dir(bid).exists()


def test_ttl_cleanup_falls_back_to_dir_mtime_when_manifest_time_missing(store: BatchStore) -> None:
    bid = _new_batch(store)
    mpath = store.manifest_path(bid)
    doc = json.loads(mpath.read_text(encoding="utf-8"))
    doc.pop("created_utc", None)
    mpath.write_text(json.dumps(doc), encoding="utf-8")
    # fresh mtime -> kept
    assert store.cleanup_stale(ttl_hours=1) == []


def test_unsafe_batch_id_is_rejected(store: BatchStore) -> None:
    for bad in ("../evil", "a/b", "", ".."):
        with pytest.raises(BatchStoreError):
            store.batch_dir(bad)


def test_staged_items_are_excluded_from_sample_counts_and_coverage(tmp_path) -> None:
    reg = ObjectRegistry(tmp_path / "objects")
    reg.create("mug")
    sample_store = SampleStore(reg.object_dir("mug"), "mug")
    sample_store.add(
        image_bytes=_jpeg(50), width=64, height=48, box=[4, 4, 20, 20],
        conditions={}, role="positive",
    )
    batch_store = BatchStore(reg.object_dir("mug"), "mug")
    _new_batch(batch_store, n=5)

    # counts see only the one committed sample, not the 5 staged images
    assert sample_store.counts()["sample_count"] == 1
    cov = coverage_summary(sample_store.list(), CoverageTargets())
    assert cov["total_positives"] == 1

    # the staging dir exists on disk but under _staging/, never manifest.json
    assert (reg.object_dir("mug") / "_staging").is_dir()
    manifest = json.loads((reg.object_dir("mug") / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["samples"]) == 1
