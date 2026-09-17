"""Stage 5 section 3.4/5: image-, near-duplicate-group-, and session-disjoint
detection splits. Determinism, leakage assertions, eval collision hard-fail,
and the minimum-image-count refusal (section 3.5).
"""

from __future__ import annotations

import pytest

from predictivesense.dataset.coco_store import CocoStore, domain_categories
from predictivesense.dataset.detection_convert import DetectionBox, DetectionImageRecord, add_detection_record
from predictivesense.dataset.detection_splits import (
    DetectionSplitError,
    EvalCollisionError,
    UnionFind,
    assert_no_eval_collision,
    assert_no_split_leakage,
    build_detection_splits,
    merge_duplicate_groups,
)

pytestmark = pytest.mark.unit


def _store_with_images(n: int, *, ps_class="watch", group_prefix="grp") -> tuple[CocoStore, dict[int, str]]:
    store = CocoStore.create(domain_categories(["watch", "mug/cup"]))
    image_to_group: dict[int, str] = {}
    for i in range(n):
        record = DetectionImageRecord(
            file_name=f"img{i}", width=100, height=100, session_id=f"{group_prefix}{i}",
            image_path=f"/tmp/img{i}.jpg", image_sha256=f"sha{i}",
            source_dataset="open-images-v7", source_version="train", original_image_id=str(i),
            boxes=(DetectionBox(ps_class=ps_class, bbox=(1.0, 1.0, 5.0, 5.0), source_class=ps_class, box_confirmed_by_human=True),),
        )
        iid = add_detection_record(store, record)
        image_to_group[iid] = f"{group_prefix}{i}"
    return store, image_to_group


# -- UnionFind / merge_duplicate_groups ------------------------------------


def test_union_find_merges_transitively():
    uf = UnionFind(["a", "b", "c", "d"])
    uf.union("a", "b")
    uf.union("b", "c")
    assert uf.find("a") == uf.find("c")
    assert uf.find("a") != uf.find("d")


def test_merge_duplicate_groups_unions_sessions_connected_by_a_dup_edge():
    base = {"img1": "sessA", "img2": "sessB", "img3": "sessC"}
    merged = merge_duplicate_groups(base, [("img1", "img2")])
    assert merged["img1"] == merged["img2"]
    assert merged["img3"] != merged["img1"]


# -- build_detection_splits: disjointness + determinism --------------------


def test_splits_are_image_and_group_disjoint():
    store, image_to_group = _store_with_images(60)
    splits = build_detection_splits(
        store, image_to_group, train_fraction=0.7, val_fraction=0.15, test_fraction=0.15,
        seed=0, min_images_per_class=10,
    )
    assert_no_split_leakage(splits)  # raises on failure
    all_ids = splits.image_ids["train"] + splits.image_ids["val"] + splits.image_ids["test"]
    assert len(all_ids) == len(set(all_ids)) == 60


def test_same_seed_gives_identical_content_hash():
    store1, g1 = _store_with_images(60)
    store2, g2 = _store_with_images(60)
    s1 = build_detection_splits(store1, g1, train_fraction=0.7, val_fraction=0.15, test_fraction=0.15, seed=42, min_images_per_class=10)
    s2 = build_detection_splits(store2, g2, train_fraction=0.7, val_fraction=0.15, test_fraction=0.15, seed=42, min_images_per_class=10)
    assert s1.content_hash == s2.content_hash


def test_a_near_duplicate_group_never_straddles_a_split():
    # 60 singleton-group images, then merge two of them into one group via a dup edge.
    store, image_to_group = _store_with_images(60)
    # find the two images belonging to groups grp0/grp1 and merge their groups
    id0 = next(iid for iid, g in image_to_group.items() if g == "grp0")
    id1 = next(iid for iid, g in image_to_group.items() if g == "grp1")
    merged = merge_duplicate_groups(image_to_group, [(id0, id1)])
    # NB merge_duplicate_groups keys on the *values already used as group ids* in this
    # helper's simplified test setup, so remap explicitly for clarity:
    image_to_group2 = dict(image_to_group)
    image_to_group2[id1] = image_to_group2[id0]
    splits = build_detection_splits(
        store, image_to_group2, train_fraction=0.7, val_fraction=0.15, test_fraction=0.15,
        seed=1, min_images_per_class=10,
    )
    # id0 and id1 (same merged group) must land in the same split
    for name in ("train", "val", "test"):
        assert (id0 in splits.image_ids[name]) == (id1 in splits.image_ids[name])


def test_realized_per_class_box_and_image_counts_are_reported_separately():
    store, image_to_group = _store_with_images(60)
    splits = build_detection_splits(store, image_to_group, train_fraction=0.7, val_fraction=0.15, test_fraction=0.15, seed=0, min_images_per_class=10)
    total_watch_boxes = sum(d.get("watch", 0) for d in splits.realized_per_class_box_counts.values())
    total_watch_images = sum(d.get("watch", 0) for d in splits.realized_per_class_image_counts.values())
    assert total_watch_boxes == 60  # one box per image in this fixture
    assert total_watch_images == 60


def test_every_image_lands_in_exactly_one_split():
    """Stage 5.1: the defect this guards against - 11/635 real zero-box
    images used to fall through every stratum and land in no split at all."""

    store, image_to_group = _store_with_images(60)
    splits = build_detection_splits(store, image_to_group, train_fraction=0.7, val_fraction=0.15, test_fraction=0.15, seed=0, min_images_per_class=10)
    all_split_ids = splits.image_ids["train"] + splits.image_ids["val"] + splits.image_ids["test"]
    assert len(all_split_ids) == len(store.image_ids())
    assert set(all_split_ids) == set(store.image_ids())


def test_zero_box_images_are_assigned_to_a_split_not_dropped():
    """The actual bug: a group whose every image has zero boxes has no
    primary class, so it used to be absent from every groups_by_class entry
    and therefore from `assign` entirely."""

    store = CocoStore.create(domain_categories(["watch"]))
    image_to_group: dict[int, str] = {}
    for i in range(10):  # 10 real watch images, enough to clear the min
        record = DetectionImageRecord(
            file_name=f"pos{i}", width=100, height=100, session_id=f"posgrp{i}",
            image_path=f"/tmp/pos{i}.jpg", image_sha256=f"possha{i}",
            source_dataset="test", source_version="v1", original_image_id=str(i),
            boxes=(DetectionBox(ps_class="watch", bbox=(1.0, 1.0, 5.0, 5.0), source_class="watch", box_confirmed_by_human=True),),
        )
        iid = add_detection_record(store, record)
        image_to_group[iid] = f"posgrp{i}"
    zero_box_ids = []
    for i in range(5):  # 5 zero-box (hard negative) images, own singleton groups
        record = DetectionImageRecord(
            file_name=f"neg{i}", width=100, height=100, session_id=f"neggrp{i}",
            image_path=f"/tmp/neg{i}.jpg", image_sha256=f"negsha{i}",
            source_dataset="test", source_version="v1", original_image_id=f"n{i}", boxes=(),
        )
        iid = add_detection_record(store, record)
        image_to_group[iid] = f"neggrp{i}"
        zero_box_ids.append(iid)

    splits = build_detection_splits(store, image_to_group, train_fraction=0.7, val_fraction=0.15, test_fraction=0.15, seed=0, min_images_per_class=5)
    all_split_ids = set(splits.image_ids["train"] + splits.image_ids["val"] + splits.image_ids["test"])
    assert all_split_ids == set(store.image_ids())
    for zid in zero_box_ids:
        assert zid in all_split_ids, f"zero-box image {zid} fell through every stratum"


# -- minimum-image-count refusal (section 3.5) ------------------------------


def test_a_class_with_one_image_refuses_a_training_split():
    store, image_to_group = _store_with_images(1, ps_class="watch", group_prefix="soloA")
    with pytest.raises(DetectionSplitError, match="1 image"):
        build_detection_splits(
            store, image_to_group, train_fraction=0.7, val_fraction=0.15, test_fraction=0.15,
            seed=0, min_images_per_class=50,
        )


def test_a_class_with_enough_images_is_not_refused():
    store, image_to_group = _store_with_images(50, ps_class="watch")
    splits = build_detection_splits(
        store, image_to_group, train_fraction=0.7, val_fraction=0.15, test_fraction=0.15,
        seed=0, min_images_per_class=10,
    )
    assert len(splits.image_ids["train"]) > 0


# -- eval collision hard-fail ------------------------------------------------


def test_train_val_image_matching_an_eval_sha_hard_fails():
    store, image_to_group = _store_with_images(5)
    train_ids = list(image_to_group.keys())
    colliding_sha = store.image(train_ids[0])["ps_provenance"]["image_sha256"]
    with pytest.raises(EvalCollisionError):
        assert_no_eval_collision(
            train_val_image_ids=train_ids, store=store, eval_image_shas={colliding_sha},
        )


def test_no_collision_passes_silently():
    store, image_to_group = _store_with_images(5)
    train_ids = list(image_to_group.keys())
    assert_no_eval_collision(train_val_image_ids=train_ids, store=store, eval_image_shas={"not-a-real-sha"})
