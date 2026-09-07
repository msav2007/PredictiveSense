"""Class map, alias map and pose skeleton are internally consistent."""

from __future__ import annotations

import pytest

from predictivesense.perception.classes import (
    ALIAS_MAP,
    COCO_CLASSES,
    KEYPOINT_NAMES,
    REQUIRED_CLASSES,
    SKELETON_EDGES,
    alias_for,
    class_color,
)

pytestmark = pytest.mark.unit


def test_coco_has_80_unique_classes() -> None:
    assert len(COCO_CLASSES) == 80
    assert len(set(COCO_CLASSES)) == 80


def test_every_required_class_resolves_in_coco() -> None:
    for name in REQUIRED_CLASSES:
        assert name in COCO_CLASSES, f"required class {name!r} is not a COCO class"


def test_alias_map_targets_are_unique_and_source_labels_are_real() -> None:
    targets = list(ALIAS_MAP.values())
    assert len(targets) == len(set(targets)), "alias map has duplicate targets"
    for raw in ALIAS_MAP:
        assert raw in COCO_CLASSES, f"alias source {raw!r} is not a COCO class"


def test_alias_for_every_class_is_nonempty_and_distinct() -> None:
    aliases = [alias_for(c) for c in COCO_CLASSES]
    assert all(a and a[0].isupper() for a in aliases)
    assert len(set(aliases)) == len(aliases), "alias_for produced a collision"


def test_keypoints_and_skeleton_are_consistent() -> None:
    assert len(KEYPOINT_NAMES) == 17
    assert len(set(KEYPOINT_NAMES)) == 17
    for a, b in SKELETON_EDGES:
        assert 0 <= a < 17 and 0 <= b < 17, f"skeleton edge ({a},{b}) out of range"
        assert a != b


def test_class_color_is_deterministic_and_in_range() -> None:
    for cid in range(80):
        c1 = class_color(cid)
        c2 = class_color(cid)
        assert c1 == c2
        assert len(c1) == 3 and all(0 <= v <= 255 for v in c1)
    # not all the same colour
    assert len({class_color(i) for i in range(80)}) > 40
