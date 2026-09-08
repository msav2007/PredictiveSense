"""Recognition policy: each rule in isolation, counts reconcile, disabled is a
pass-through (BLOCK 12.5). Phase 5: tier-based suppression, six states.

This is the *extended* Phase 2.5 reconciliation test - ``test_policy_states.py``
adds the per-state construction coverage."""

from __future__ import annotations

import pytest

from predictivesense.config.settings import PolicyConfig
from predictivesense.core.types import Detection
from predictivesense.perception.policy import RecognitionPolicy

pytestmark = pytest.mark.unit

_W, _H = 640, 480


def _det(name, score, *, bbox=(10, 10, 110, 110), runner=None) -> Detection:
    return Detection(
        bbox=bbox, class_id=0, class_name=name, score=score, frame_id=0,
        raw_class_name=name, runner_up=runner,
    )


def _policy(**overrides) -> RecognitionPolicy:
    # Uses the shipped three-tier vocabulary (a total, disjoint partition of
    # COCO-80). primary: cup/person/bottle/bowl. secondary: clock/wine glass.
    # implausible: donut/banana.
    cfg = PolicyConfig(
        default_threshold=0.35,
        margin_min=0.10,
        min_box_area_frac=0.01,  # 100x100 on 640x480 = 0.0326 -> above; 10x10 below
        **overrides,
    )
    return RecognitionPolicy(cfg)


def _apply(policy, dets):
    return policy.apply(dets, frame_width=_W, frame_height=_H)


def test_implausible_tier_is_suppressed_not_unknown() -> None:
    out = _apply(_policy(), [_det("donut", 0.9)])
    d = out.detections[0]
    assert d.policy_state == "suppressed_implausible"
    assert d.policy_state not in ("unknown_low_confidence", "unknown_margin")
    assert d.tier == "implausible"
    assert d.raw_class_name == "donut"
    assert out.counts.suppressed_implausible == 1


def test_secondary_tier_is_shown_de_emphasised() -> None:
    out = _apply(_policy(), [_det("clock", 0.92, runner=("vase", 0.03))])
    d = out.detections[0]
    assert d.policy_state == "accepted_secondary"
    assert d.class_name == "clock"  # kept, not "unknown"
    assert d.tier == "secondary"
    assert out.counts.accepted_secondary == 1


def test_below_threshold_becomes_unknown_low_confidence() -> None:
    out = _apply(_policy(), [_det("cup", 0.20)])
    d = out.detections[0]
    assert d.policy_state == "unknown_low_confidence"
    assert d.class_name == "unknown"
    assert out.counts.unknown_low_confidence == 1


def test_small_top2_margin_becomes_unknown_margin() -> None:
    out = _apply(_policy(), [_det("cup", 0.50, runner=("bowl", 0.45))])
    d = out.detections[0]
    assert d.policy_state == "unknown_margin"
    assert d.class_name == "unknown"
    assert out.counts.unknown_margin == 1


def test_tiny_box_is_rejected_size() -> None:
    out = _apply(_policy(), [_det("cup", 0.9, bbox=(10, 10, 20, 20))])  # 10x10
    d = out.detections[0]
    assert d.policy_state == "rejected_size"
    assert out.counts.rejected_size == 1


def test_aspect_ratio_bound_rejects_size() -> None:
    p = _policy(aspect_ratio_bounds={"cup": (0.5, 2.0)})
    out = _apply(p, [_det("cup", 0.9, bbox=(0, 0, 400, 40))])  # w/h = 10 -> outside
    assert out.detections[0].policy_state == "rejected_size"


def test_confident_primary_detection_is_accepted() -> None:
    out = _apply(_policy(), [_det("cup", 0.92, runner=("bowl", 0.05))])
    d = out.detections[0]
    assert d.policy_state == "accepted"
    assert d.class_name == "cup"
    assert d.tier == "primary"
    assert out.counts.accepted == 1


def test_counts_reconcile_exactly_across_all_six_states() -> None:
    dets = [
        _det("cup", 0.92, runner=("bowl", 0.02)),          # accepted
        _det("clock", 0.90, runner=("vase", 0.01)),        # accepted_secondary
        _det("cup", 0.2),                                   # unknown_low_confidence
        _det("cup", 0.5, runner=("bowl", 0.45)),            # unknown_margin
        _det("donut", 0.9),                                 # suppressed_implausible
        _det("cup", 0.9, bbox=(0, 0, 8, 8)),               # rejected_size
    ]
    out = _apply(_policy(), dets)
    c = out.counts
    assert c.input == 6
    assert c.accepted_all + c.unknown + c.suppressed + c.rejected == c.input
    assert c.reconciles
    assert len(out.detections) == len(dets)
    assert len(out.raw_to_decided) == len(dets)
    states = {d.policy_state for d in out.detections}
    assert states == set(
        (
            "accepted",
            "accepted_secondary",
            "unknown_low_confidence",
            "unknown_margin",
            "suppressed_implausible",
            "rejected_size",
        )
    )


def test_each_rule_is_independently_switchable() -> None:
    # suppression off -> an implausible class falls through (0.9 clears threshold,
    # no runner_up so margin cannot fire) and is accepted with its raw label
    out = _apply(_policy(domain_restriction=False), [_det("donut", 0.9)])
    assert out.detections[0].policy_state == "accepted"
    assert out.detections[0].class_name == "donut"
    # margin rule off -> a small-margin detection is accepted
    out = _apply(_policy(margin_rule=False), [_det("cup", 0.5, runner=("bowl", 0.49))])
    assert out.detections[0].policy_state == "accepted"
    # size rule off -> a tiny box is accepted
    out = _apply(_policy(size_rule=False), [_det("cup", 0.9, bbox=(0, 0, 4, 4))])
    assert out.detections[0].policy_state == "accepted"
    # threshold rule off -> a weak detection is accepted
    out = _apply(_policy(per_class_threshold_rule=False), [_det("cup", 0.05)])
    assert out.detections[0].policy_state == "accepted"


def test_disabled_policy_is_a_passthrough() -> None:
    dets = [_det("donut", 0.9), _det("cup", 0.1, bbox=(0, 0, 3, 3))]
    out = _apply(_policy(enabled=False), dets)
    for src, res in zip(dets, out.detections):
        assert res.bbox == src.bbox
        assert res.class_name == src.class_name
        assert res.score == src.score
        assert res.policy_state == "accepted"
        assert res.raw_class_name == src.class_name
    assert out.counts.accepted == len(dets)
    assert out.counts.reconciles


def test_per_class_threshold_overrides_default() -> None:
    p = _policy(per_class_thresholds={"cup": 0.8})
    out = _apply(p, [_det("cup", 0.6, runner=("bowl", 0.01))])
    assert out.detections[0].policy_state == "unknown_low_confidence"
