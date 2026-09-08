"""Splits are session-disjoint: no session and no image id in two splits, the
hash changes with content, and a hand-crafted leaky split is rejected."""

from __future__ import annotations

import pytest

from predictivesense.dataset.coco_store import CocoStore, domain_categories
from predictivesense.dataset.splits import (
    SplitLeakageError,
    Splits,
    assert_no_leakage,
    build_splits,
    splits_content_hash,
)

pytestmark = pytest.mark.unit


def _store(n_sessions: int, per_session: int = 4) -> CocoStore:
    s = CocoStore.create(domain_categories(("person", "cup")))
    for si in range(n_sessions):
        for fi in range(per_session):
            iid = s.add_image(
                file_name=f"s{si}_f{fi}.jpg", width=8, height=8, session_id=f"sess{si}"
            )
            s.set_frame_boxes(iid, [{"category": "person", "bbox": [0, 0, 4, 4]}], seeded=False)
    return s


def test_no_session_or_image_in_two_splits() -> None:
    s = _store(4)
    sp = build_splits(s, val_fraction=0.5, test_fraction=0.5, seed=1)
    assert set(sp.sessions["val"]) & set(sp.sessions["test"]) == set()
    assert set(sp.image_ids["val"]) & set(sp.image_ids["test"]) == set()
    # every labelled image landed in exactly one split
    assert sorted(sp.image_ids["val"] + sp.image_ids["test"]) == s.image_ids()
    assert sp.sessions["val"] and sp.sessions["test"]


def test_hash_changes_when_content_changes() -> None:
    s = _store(4)
    sp = build_splits(s, val_fraction=0.5, test_fraction=0.5, seed=1)
    h0 = sp.content_hash
    # relabel a frame -> store content changes -> hash for the same assignment moves
    first = s.image_ids()[0]
    s.set_frame_boxes(first, [{"category": "cup", "bbox": [1, 1, 3, 3]}], seeded=False)
    h1 = splits_content_hash(s, sp.sessions)
    assert h1 != h0


def test_deliberately_leaky_split_is_rejected() -> None:
    leaky = Splits(
        sessions={"val": ["sessA"], "test": ["sessA"]},
        image_ids={"val": [1, 2], "test": [2, 3]},
        content_hash="x",
        seed=0,
    )
    with pytest.raises(SplitLeakageError):
        assert_no_leakage(leaky)


def test_single_session_gives_empty_test() -> None:
    s = _store(1)
    sp = build_splits(s, val_fraction=0.5, test_fraction=0.5, seed=1)
    assert sp.sessions["val"] and not sp.sessions["test"]
    assert_no_leakage(sp)


def test_build_refuses_when_nothing_labelled() -> None:
    s = CocoStore.create(domain_categories(("person",)))
    s.add_image(file_name="a.jpg", width=8, height=8, session_id="s")  # not labelled
    with pytest.raises(SplitLeakageError):
        build_splits(s, val_fraction=0.5, test_fraction=0.5, seed=1)


def test_split_is_deterministic_for_a_seed() -> None:
    s = _store(6)
    a = build_splits(s, val_fraction=0.5, test_fraction=0.5, seed=7)
    b = build_splits(s, val_fraction=0.5, test_fraction=0.5, seed=7)
    assert a.sessions == b.sessions and a.content_hash == b.content_hash
