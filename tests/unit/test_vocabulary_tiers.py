"""Phase 5 three-tier class vocabulary (BLOCK 11.1).

The partition must be total and disjoint over the detector's full class list; a
typo / duplicate / unassigned class fails loudly; tier lookup is correct; and
the shipped ``dev`` / ``eval`` profiles carry a valid partition.
"""

from __future__ import annotations

import pytest

from predictivesense.config.settings import PolicyVocabularyConfig, load_config
from predictivesense.perception.classes import COCO_CLASSES
from predictivesense.perception.vocabulary import (
    IMPLAUSIBLE_TIER,
    PRIMARY_TIER,
    SECONDARY_TIER,
    VocabularyError,
    VocabularyTiers,
    validate_partition,
)

pytestmark = pytest.mark.unit


def test_default_tiers_are_a_total_disjoint_partition_of_coco80() -> None:
    validate_partition(PRIMARY_TIER, SECONDARY_TIER, IMPLAUSIBLE_TIER)
    union = set(PRIMARY_TIER) | set(SECONDARY_TIER) | set(IMPLAUSIBLE_TIER)
    assert union == set(COCO_CLASSES)
    assert len(PRIMARY_TIER) + len(SECONDARY_TIER) + len(IMPLAUSIBLE_TIER) == len(COCO_CLASSES) == 80
    # pairwise disjoint
    assert not (set(PRIMARY_TIER) & set(SECONDARY_TIER))
    assert not (set(PRIMARY_TIER) & set(IMPLAUSIBLE_TIER))
    assert not (set(SECONDARY_TIER) & set(IMPLAUSIBLE_TIER))


def test_unknown_class_name_in_a_tier_fails_loudly() -> None:
    with pytest.raises(VocabularyError) as ei:
        validate_partition(("person", "not_a_coco_class"), SECONDARY_TIER, IMPLAUSIBLE_TIER)
    assert "cannot emit" in str(ei.value)


def test_missing_class_fails_loudly() -> None:
    primary = tuple(c for c in PRIMARY_TIER if c != "cup")  # drop one
    with pytest.raises(VocabularyError) as ei:
        validate_partition(primary, SECONDARY_TIER, IMPLAUSIBLE_TIER)
    assert "not assigned to any tier" in str(ei.value)
    assert "cup" in str(ei.value)


def test_overlap_between_tiers_fails_loudly() -> None:
    with pytest.raises(VocabularyError) as ei:
        validate_partition(PRIMARY_TIER, SECONDARY_TIER + ("cup",), IMPLAUSIBLE_TIER)
    assert "more than one tier" in str(ei.value)


def test_duplicate_within_a_tier_fails_loudly() -> None:
    with pytest.raises(VocabularyError) as ei:
        validate_partition(PRIMARY_TIER + ("cup",), SECONDARY_TIER, IMPLAUSIBLE_TIER)
    assert "duplicate" in str(ei.value)


def test_tier_lookup_is_correct() -> None:
    tiers = VocabularyTiers.default()
    assert tiers.tier_for("person") == "primary"
    assert tiers.tier_for("laptop") == "primary"
    assert tiers.tier_for("clock") == "secondary"
    assert tiers.tier_for("wine glass") == "secondary"
    assert tiers.tier_for("donut") == "implausible"
    assert tiers.tier_for("zebra") == "implausible"
    # a class the detector cannot emit (non-COCO model) -> unlisted, shown
    assert tiers.tier_for("charger") == "unlisted"
    assert tiers.shown_classes == tiers.primary | tiers.secondary


def test_shipped_profiles_carry_a_valid_partition() -> None:
    for profile in ("dev", "eval"):
        cfg = load_config(profile)
        v = cfg.policy.vocabulary
        validate_partition(v.primary, v.secondary, v.implausible)
        # back-compat alias: primary tier == the eval / threshold-fitting scope
        assert set(cfg.policy.domain_classes) == set(v.primary)
        # general-model baseline classes work with no enrolment
        for cls in ("person", "laptop", "keyboard", "chair", "bottle", "cup"):
            assert cls in v.primary
        assert "clock" in v.secondary and "tv" in v.secondary


def test_config_rejects_a_broken_partition() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PolicyVocabularyConfig(primary=("person",))  # secondary+implausible defaults -> gap
