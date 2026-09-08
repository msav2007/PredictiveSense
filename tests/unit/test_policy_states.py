"""Phase 5 policy states (BLOCK 11.2).

Every one of the six ``policy_state`` values is produced by a constructed
detection; the counts reconcile exactly; ``suppressed_implausible`` is distinct
from ``unknown``; a disabled policy is a pass-through; a rule that raises is
caught and counted, never killing the loop.
"""

from __future__ import annotations

import pytest

from predictivesense.config.settings import PolicyConfig
from predictivesense.core.types import Detection
from predictivesense.perception.policy import POLICY_STATES, RecognitionPolicy

pytestmark = pytest.mark.unit

_W, _H = 640, 480


def _det(name, score, *, bbox=(20, 20, 220, 220), runner=None, cid=0) -> Detection:
    return Detection(
        bbox=bbox, class_id=cid, class_name=name, score=score, frame_id=0,
        raw_class_name=name, runner_up=runner,
    )


def _policy(**overrides) -> RecognitionPolicy:
    return RecognitionPolicy(
        PolicyConfig(default_threshold=0.35, margin_min=0.10, min_box_area_frac=0.01, **overrides)
    )


_CASES = {
    "accepted": _det("cup", 0.95, runner=("bowl", 0.01)),
    "accepted_secondary": _det("clock", 0.95, runner=("vase", 0.01)),
    "unknown_low_confidence": _det("cup", 0.20, runner=("bowl", 0.01)),
    "unknown_margin": _det("cup", 0.60, runner=("bowl", 0.55)),
    "suppressed_implausible": _det("donut", 0.95),
    "rejected_size": _det("cup", 0.95, bbox=(0, 0, 6, 6)),
}


@pytest.mark.parametrize("state,det", list(_CASES.items()))
def test_each_state_is_produced(state: str, det: Detection) -> None:
    assert state in POLICY_STATES
    out = _policy().apply([det], frame_width=_W, frame_height=_H)
    assert out.detections[0].policy_state == state
    assert out.counts.as_metrics("")[state] == 1.0


def test_counts_reconcile_exactly_across_all_states() -> None:
    dets = list(_CASES.values())
    out = _policy().apply(dets, frame_width=_W, frame_height=_H)
    c = out.counts
    assert c.input == len(dets) == 6
    assert c.accepted_all + c.unknown + c.suppressed + c.rejected == c.input
    assert c.reconciles
    # every emitted metric except input/errors sums back to input
    m = c.as_metrics("")
    tallied = sum(m[k] for k in POLICY_STATES)
    assert tallied == float(c.input)


def test_suppressed_implausible_is_not_unknown() -> None:
    out = _policy().apply([_CASES["suppressed_implausible"]], frame_width=_W, frame_height=_H)
    d = out.detections[0]
    assert d.policy_state == "suppressed_implausible"
    assert d.policy_state not in ("unknown_low_confidence", "unknown_margin")
    assert d.tier == "implausible"
    assert d.raw_class_name == "donut"
    # counts keep it out of the unknown bucket
    assert out.counts.unknown == 0
    assert out.counts.suppressed == 1


def test_disabled_policy_is_pass_through() -> None:
    dets = [_det("donut", 0.9), _det("cup", 0.05, bbox=(0, 0, 4, 4))]
    out = RecognitionPolicy(PolicyConfig(enabled=False)).apply(
        dets, frame_width=_W, frame_height=_H
    )
    assert [d.class_name for d in out.detections] == ["donut", "cup"]
    assert {d.policy_state for d in out.detections} == {"accepted"}
    assert out.counts.reconciles and out.counts.accepted == 2
    # tier is still annotated so Diagnostics can show it
    assert out.detections[0].tier == "implausible"


def test_a_rule_that_raises_is_caught_and_counted(monkeypatch) -> None:
    policy = _policy()

    def _boom(_name: str) -> float:
        raise RuntimeError("synthetic rule failure")

    monkeypatch.setattr(policy, "threshold_for", _boom)
    dets = [_det("cup", 0.9), _det("donut", 0.9)]
    out = policy.apply(dets, frame_width=_W, frame_height=_H)
    assert out.counts.errors == 1
    assert out.counts.input == 2
    # passed through unchanged, loop stays alive
    assert [d.class_name for d in out.detections] == ["cup", "donut"]
    assert {d.policy_state for d in out.detections} == {"accepted"}
    assert out.counts.reconciles
