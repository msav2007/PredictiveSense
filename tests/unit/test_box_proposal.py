"""Class-agnostic box proposal from raw detector output (Phase 7, BLOCK 9.2)."""

from __future__ import annotations

import pytest

from predictivesense.objects.proposals import (
    BoxProposal,
    centred_default_box,
    propose_box,
)

pytestmark = pytest.mark.unit


def _det(bbox, score, cls="cup"):
    return {"bbox": bbox, "score": score, "raw_class_name": cls}


def test_highest_score_wins() -> None:
    dets = [
        _det((10, 10, 50, 50), 0.4, "bowl"),
        _det((60, 60, 120, 120), 0.9, "cup"),
    ]
    p = propose_box(dets, 200, 200)
    assert p.source == "detector" and p.status == "ready"
    assert p.raw_class == "cup" and p.score == pytest.approx(0.9)
    # xyxy (60,60,120,120) -> xywh
    assert p.box == [60.0, 60.0, 60.0, 60.0]


def test_tie_break_prefers_larger_then_more_central() -> None:
    # equal score: the larger box wins
    dets = [
        _det((0, 0, 20, 20), 0.5, "a"),  # area 400
        _det((80, 80, 140, 140), 0.5, "b"),  # area 3600, also nearer centre
    ]
    p = propose_box(dets, 200, 200)
    assert p.raw_class == "b"

    # equal score AND equal area: the more central box wins
    dets2 = [
        _det((0, 0, 40, 40), 0.5, "far"),
        _det((90, 90, 130, 130), 0.5, "near"),
    ]
    p2 = propose_box(dets2, 200, 200)
    assert p2.raw_class == "near"


def test_empty_detector_output_is_manual_required_with_centred_default() -> None:
    p = propose_box([], 400, 300)
    assert isinstance(p, BoxProposal)
    assert p.status == "manual_required"
    assert p.source == "default_centred"
    assert p.raw_class is None and p.score is None
    assert p.box == centred_default_box(400, 300) == [120.0, 90.0, 160.0, 120.0]


def test_below_min_score_is_treated_as_no_detection() -> None:
    p = propose_box([_det((10, 10, 50, 50), 0.05)], 200, 200, min_score=0.10)
    assert p.status == "manual_required" and p.source == "default_centred"


def test_raw_class_is_a_hint_only_never_a_sample_class() -> None:
    # The proposal object exposes the detector's guess as `raw_class`; it has no
    # field that could be mistaken for the sample's class.
    p = propose_box([_det((10, 10, 90, 90), 0.7, "donut")], 200, 200)
    assert p.raw_class == "donut"
    assert not hasattr(p, "class_name")
    assert "class_name" not in p.__dict__


def test_policy_is_bypassed_an_implausible_tier_raw_class_still_yields_a_box() -> None:
    # `donut` / `clock` are exactly what a watch gets proposed as, and what the
    # recognition policy's `implausible` tier would suppress. The proposer must
    # still return that rectangle.
    for cls in ("donut", "clock", "surfboard", "bicycle"):
        p = propose_box([_det((30, 30, 130, 110), 0.6, cls)], 240, 180)
        assert p.status == "ready" and p.source == "detector"
        assert p.box == [30.0, 30.0, 100.0, 80.0]
        assert p.raw_class == cls


def test_alternates_are_offered_cheaply_when_present() -> None:
    dets = [_det((0, 0, 10, 10), 0.9), _det((5, 5, 40, 40), 0.8), _det((1, 1, 8, 8), 0.7)]
    p = propose_box(dets, 100, 100)
    assert len(p.alternates) == 2
    assert p.alternates[0]["score"] == pytest.approx(0.8)


def test_box_is_clamped_into_the_image() -> None:
    p = propose_box([_det((-20, -10, 250, 400), 0.9)], 200, 300)
    x, y, w, h = p.box
    assert x >= 0 and y >= 0 and x + w <= 200 + 0.01 and y + h <= 300 + 0.01
