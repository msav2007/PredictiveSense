"""Session-disjoint train/val/test splitting for the evaluation set.

Adjacent frames are near-duplicates, so a frame-level split leaks (BLOCK 3.1.5 /
BLOCK 17.5). Whole recording *sessions* are assigned to ``val`` or ``test``; a
session (and therefore every image id in it) appears in exactly one split. The
split file carries a content hash so a later run can detect that it no longer
matches the annotation store.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from random import Random
from typing import Any

from predictivesense.dataset.coco_store import CocoStore
from predictivesense.logging_setup import get_logger

__all__ = [
    "Splits",
    "SplitLeakageError",
    "build_splits",
    "assert_no_leakage",
    "splits_content_hash",
    "load_splits",
]

_LOG = get_logger(__name__)
_SPLIT_NAMES = ("val", "test")


class SplitLeakageError(RuntimeError):
    """A session id or image id appears in more than one split, or the split
    file's hash does not match its content."""


@dataclass(frozen=True)
class Splits:
    """Which sessions and image ids belong to each split."""

    sessions: dict[str, list[str]]
    image_ids: dict[str, list[int]]
    content_hash: str
    seed: int
    fractions: dict[str, float] = field(default_factory=dict)

    def to_doc(self) -> dict[str, Any]:
        return {
            "format": 1,
            "seed": self.seed,
            "fractions": self.fractions,
            "content_hash": self.content_hash,
            "sessions": {k: sorted(v) for k, v in self.sessions.items()},
            "image_ids": {k: sorted(v) for k, v in self.image_ids.items()},
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_doc(), indent=2) + "\n", encoding="utf-8")
        return p


def splits_content_hash(store: CocoStore, sessions: dict[str, list[str]]) -> str:
    """Hash of the store content plus the session->split assignment. Changes
    whenever an annotation, an image, or the assignment changes."""

    blob = json.dumps(
        {
            "store": store.content_hash(),
            "sessions": {k: sorted(v) for k, v in sorted(sessions.items())},
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(blob.encode("utf-8")).hexdigest()


def build_splits(
    store: CocoStore,
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    labelled_only: bool = True,
) -> Splits:
    """Assign whole sessions to ``val`` / ``test`` by a seeded shuffle.

    ``val_fraction`` / ``test_fraction`` are shares of the *session* count (they
    are renormalised if they do not sum to 1). At least one session goes to each
    split when two or more sessions exist; with a single session it goes to
    ``val`` and ``test`` stays empty (the harness then refuses to open ``test``).
    """

    by_session: dict[str, list[int]] = {}
    for iid in store.image_ids():
        im = store.image(iid)
        if labelled_only and not im.get("labelled"):
            continue
        by_session.setdefault(store.session_of(iid), []).append(iid)

    session_ids = sorted(by_session)
    if not session_ids:
        raise SplitLeakageError(
            "no labelled images to split - label frames at /label first"
        )

    rng = Random(seed)
    shuffled = session_ids[:]
    rng.shuffle(shuffled)

    total = val_fraction + test_fraction
    v_share = val_fraction / total if total > 0 else 0.5
    n = len(shuffled)
    if n == 1:
        assign = {shuffled[0]: "val"}
    else:
        n_val = max(1, min(n - 1, round(n * v_share)))
        assign = {s: ("val" if i < n_val else "test") for i, s in enumerate(shuffled)}

    sessions: dict[str, list[str]] = {name: [] for name in _SPLIT_NAMES}
    image_ids: dict[str, list[int]] = {name: [] for name in _SPLIT_NAMES}
    for sess, split in assign.items():
        sessions[split].append(sess)
        image_ids[split].extend(by_session[sess])

    chash = splits_content_hash(store, sessions)
    splits = Splits(
        sessions=sessions,
        image_ids=image_ids,
        content_hash=chash,
        seed=seed,
        fractions={"val": round(v_share, 4), "test": round(1.0 - v_share, 4)},
    )
    assert_no_leakage(splits)
    _LOG.info(
        "splits: val=%d sessions/%d images, test=%d sessions/%d images (seed %d)",
        len(sessions["val"]), len(image_ids["val"]),
        len(sessions["test"]), len(image_ids["test"]), seed,
    )
    return splits


def assert_no_leakage(splits: Splits) -> None:
    """Raise :class:`SplitLeakageError` if any session id or image id is shared
    between ``val`` and ``test``."""

    v_sess, t_sess = set(splits.sessions.get("val", [])), set(splits.sessions.get("test", []))
    shared_sessions = v_sess & t_sess
    if shared_sessions:
        raise SplitLeakageError(f"sessions in both val and test: {sorted(shared_sessions)}")

    v_ids, t_ids = set(splits.image_ids.get("val", [])), set(splits.image_ids.get("test", []))
    shared_ids = v_ids & t_ids
    if shared_ids:
        raise SplitLeakageError(f"image ids in both val and test: {sorted(shared_ids)[:20]}")


def load_splits(path: str | Path, store: CocoStore | None = None) -> Splits:
    """Load a split file. If ``store`` is given, verify the stored hash still
    matches the store content (BLOCK 10 - a stale split file fails loudly)."""

    p = Path(path)
    if not p.is_file():
        raise SplitLeakageError(f"split file not found: {p} - run scripts/build_splits.py")
    doc = json.loads(p.read_text(encoding="utf-8"))
    sessions = {k: list(v) for k, v in doc.get("sessions", {}).items()}
    image_ids = {k: [int(i) for i in v] for k, v in doc.get("image_ids", {}).items()}
    splits = Splits(
        sessions=sessions,
        image_ids=image_ids,
        content_hash=str(doc.get("content_hash", "")),
        seed=int(doc.get("seed", 0)),
        fractions=doc.get("fractions", {}),
    )
    assert_no_leakage(splits)
    if store is not None:
        now = splits_content_hash(store, sessions)
        if now != splits.content_hash:
            raise SplitLeakageError(
                f"split file {p} is stale: its content_hash {splits.content_hash[:12]}… "
                f"no longer matches the annotation store {now[:12]}…. "
                f"Re-run scripts/build_splits.py."
            )
    return splits
