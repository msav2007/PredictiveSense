"""Fixed vocabularies for the Object Learning Studio.

Nothing about the *object* vocabulary is hard-coded (the developer adds objects
at runtime), but the *condition* dimensions and the sample ``role`` / object
``kind`` / ``status`` sets are fixed here so the stored data is uniform and the
coverage accounting is well-defined.

The ``confusable_with`` seed map is transcribed from the developer's Phase 2
failure-mode observations (``docs/phase-reports`` / the P4 prompt Block 4.5): the
classes each object is actually mistaken for. It seeds new profiles so the Studio
can prompt for hard negatives; the developer may edit it per object afterwards.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CONDITION_DIMENSIONS",
    "CONDITION_VOCAB",
    "DEFAULT_CONDITIONS",
    "ROLES",
    "KINDS",
    "STATUSES",
    "CONFUSABLE_SEEDS",
    "default_conditions",
    "validate_conditions",
    "validate_role",
    "validate_kind",
    "validate_status",
]

# Ordered so the UI and the coverage report iterate deterministically.
CONDITION_VOCAB: dict[str, tuple[str, ...]] = {
    "view": ("front", "back", "left", "right", "top", "tilted"),
    "distance": ("close", "medium", "far"),
    "lighting": ("bright", "normal", "dim"),
    "background": ("plain", "cluttered"),
    "occlusion": ("none", "partial"),
    "held": ("on-surface", "in-hand"),
    "frame_position": ("centre", "edge"),
}

CONDITION_DIMENSIONS: tuple[str, ...] = tuple(CONDITION_VOCAB)

# The first value of each dimension is the remembered default for a fresh session.
DEFAULT_CONDITIONS: dict[str, str] = {dim: values[0] for dim, values in CONDITION_VOCAB.items()}

ROLES: tuple[str, ...] = ("positive", "negative", "hard_negative")
KINDS: tuple[str, ...] = ("class", "instance")
STATUSES: tuple[str, ...] = ("collecting", "ready_for_training", "archived")

# object name -> classes it is actually mistaken for (developer observations).
CONFUSABLE_SEEDS: dict[str, list[str]] = {
    "watch": ["clock", "bracelet", "hand", "donut"],
    "spectacles": ["scissors"],
    "glasses": ["scissors"],
    "headphones": ["person"],
    "mug": ["cell phone", "bowl", "wine glass"],
    "cup": ["cell phone", "bowl", "wine glass"],
    "bottle": ["can", "cylinder"],
    "shaker": ["can", "cylinder", "bottle"],
    "keyboard": ["remote"],
    "can": ["cell phone", "bottle"],
    "charger": [],
}


def default_conditions() -> dict[str, str]:
    """A fresh, fully-populated condition dict (all defaults)."""

    return dict(DEFAULT_CONDITIONS)


def validate_conditions(raw: dict[str, Any] | None) -> dict[str, str]:
    """Return a complete, validated condition dict.

    Unknown keys or values raise :class:`ValueError`. Missing keys are filled
    with the dimension default so a partial submission from the UI is accepted.
    """

    raw = dict(raw or {})
    unknown_keys = set(raw) - set(CONDITION_VOCAB)
    if unknown_keys:
        raise ValueError(f"unknown condition dimension(s): {sorted(unknown_keys)}")
    out: dict[str, str] = {}
    for dim, allowed in CONDITION_VOCAB.items():
        value = raw.get(dim, DEFAULT_CONDITIONS[dim])
        if value not in allowed:
            raise ValueError(
                f"condition {dim!r}={value!r} is not one of {list(allowed)}"
            )
        out[dim] = value
    return out


def validate_role(role: str) -> str:
    if role not in ROLES:
        raise ValueError(f"role {role!r} is not one of {list(ROLES)}")
    return role


def validate_kind(kind: str) -> str:
    if kind not in KINDS:
        raise ValueError(f"kind {kind!r} is not one of {list(KINDS)}")
    return kind


def validate_status(status: str) -> str:
    if status not in STATUSES:
        raise ValueError(f"status {status!r} is not one of {list(STATUSES)}")
    return status
