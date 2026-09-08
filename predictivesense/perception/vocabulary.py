"""Three-tier class vocabulary for the recognition policy (Phase 5).

Replaces the Phase 2.5 binary ``domain_classes`` whitelist. Every class the
shipped detector can emit (COCO-80) belongs to exactly one tier:

* **primary** - classes the MVP safety scenarios depend on. Shown normally.
* **secondary** - other classes plausible in an indoor room (furniture,
  appliances, tableware). **Shown normally**, visually de-emphasised, and marked
  ``secondary`` so later phases can ignore them without the user losing them.
* **implausible** - classes that cannot occur in an indoor cabin scene and are
  almost always detector misfires (outdoor vehicles, wild animals, the food
  set, outdoor sports gear). Suppressed, with the reason recorded.

A class the detector emits that is in **no** tier (only possible with a
non-COCO model) resolves to ``unlisted`` and is treated as shown-normally, so
the general model keeps working out of the box.

The tier assignment is an **environment-specific judgement, not a measured
result** (``docs/decisions.md``). It is cheap to revise once labelled data
exists. ``validate_partition`` fails loudly (BLOCK 10) on a typo, a duplicate,
or a class the detector cannot emit, and on any COCO class left unassigned.

Stdlib only; imports only ``perception.classes`` (also stdlib-only).
"""

from __future__ import annotations

from dataclasses import dataclass

from predictivesense.perception.classes import COCO_CLASSES

__all__ = [
    "PRIMARY_TIER",
    "SECONDARY_TIER",
    "IMPLAUSIBLE_TIER",
    "TIERS",
    "VocabularyError",
    "validate_partition",
    "VocabularyTiers",
]

# -- default tiers (total + disjoint partition of COCO_CLASSES) -----------------

PRIMARY_TIER: tuple[str, ...] = (
    "person", "cup", "bottle", "laptop", "keyboard", "mouse", "chair", "book",
    "cell phone", "backpack", "handbag", "scissors", "bowl", "remote",
)

SECONDARY_TIER: tuple[str, ...] = (
    "clock", "tv", "bed", "couch", "dining table", "sink", "potted plant",
    "vase", "refrigerator", "microwave", "oven", "toaster", "toilet",
    "teddy bear", "hair drier", "toothbrush", "wine glass", "fork", "knife",
    "spoon", "bench", "umbrella", "tie", "suitcase", "cat", "dog",
)

IMPLAUSIBLE_TIER: tuple[str, ...] = (
    # outdoor vehicles + street furniture
    "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter",
    # animals (wild / farm)
    "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    # outdoor sports gear
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    # the food set
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake",
)

TIERS: tuple[str, ...] = ("primary", "secondary", "implausible", "unlisted")


class VocabularyError(ValueError):
    """The three tiers are not a total, disjoint partition of the universe."""


def validate_partition(
    primary: tuple[str, ...],
    secondary: tuple[str, ...],
    implausible: tuple[str, ...],
    *,
    universe: tuple[str, ...] = COCO_CLASSES,
) -> None:
    """Assert ``primary``/``secondary``/``implausible`` partition ``universe``.

    Raises :class:`VocabularyError` (BLOCK 10) when a tier repeats a class,
    when two tiers overlap, when a tier names a class outside ``universe`` (a
    typo, or a class the detector cannot emit), or when a ``universe`` class is
    left unassigned.
    """

    uni = set(universe)
    named: dict[str, str] = {}
    dupes: list[str] = []
    unknown: list[str] = []
    overlap: list[str] = []
    for tier_name, tier in (
        ("primary", primary),
        ("secondary", secondary),
        ("implausible", implausible),
    ):
        seen_in_tier: set[str] = set()
        for name in tier:
            if name in seen_in_tier:
                dupes.append(f"{name!r} (twice in {tier_name})")
                continue
            seen_in_tier.add(name)
            if name not in uni:
                unknown.append(f"{name!r} in {tier_name}")
            elif name in named:
                overlap.append(f"{name!r} in both {named[name]} and {tier_name}")
            else:
                named[name] = tier_name

    missing = sorted(uni - set(named))
    problems: list[str] = []
    if unknown:
        problems.append(
            "classes the detector cannot emit (not in the model vocabulary): "
            + ", ".join(sorted(unknown))
        )
    if dupes:
        problems.append("duplicate entries: " + ", ".join(sorted(dupes)))
    if overlap:
        problems.append("classes in more than one tier: " + ", ".join(sorted(overlap)))
    if missing:
        problems.append(
            f"{len(missing)} class(es) not assigned to any tier: " + ", ".join(missing)
        )
    if problems:
        raise VocabularyError(
            "policy.vocabulary must be a total, disjoint partition of the "
            f"{len(uni)}-class detector vocabulary. " + "; ".join(problems)
        )


@dataclass(frozen=True)
class VocabularyTiers:
    """Resolved tier lookup built from a validated partition."""

    primary: frozenset[str]
    secondary: frozenset[str]
    implausible: frozenset[str]

    @classmethod
    def from_lists(
        cls,
        primary: tuple[str, ...],
        secondary: tuple[str, ...],
        implausible: tuple[str, ...],
        *,
        universe: tuple[str, ...] = COCO_CLASSES,
    ) -> "VocabularyTiers":
        validate_partition(primary, secondary, implausible, universe=universe)
        return cls(
            primary=frozenset(primary),
            secondary=frozenset(secondary),
            implausible=frozenset(implausible),
        )

    @classmethod
    def from_config(cls, vocabulary_config: object) -> "VocabularyTiers":
        return cls.from_lists(
            tuple(getattr(vocabulary_config, "primary")),
            tuple(getattr(vocabulary_config, "secondary")),
            tuple(getattr(vocabulary_config, "implausible")),
        )

    @classmethod
    def default(cls) -> "VocabularyTiers":
        return cls.from_lists(PRIMARY_TIER, SECONDARY_TIER, IMPLAUSIBLE_TIER)

    def tier_for(self, class_name: str) -> str:
        """``"primary"`` | ``"secondary"`` | ``"implausible"`` | ``"unlisted"``."""

        if class_name in self.primary:
            return "primary"
        if class_name in self.secondary:
            return "secondary"
        if class_name in self.implausible:
            return "implausible"
        return "unlisted"

    @property
    def shown_classes(self) -> frozenset[str]:
        """Classes surfaced on the overlay (primary + secondary)."""

        return self.primary | self.secondary
