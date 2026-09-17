"""Session- and object-disjoint train/val/test splits for the crop dataset
(Phase 11 Part B section 11.3). Mirrors the pattern already established by
``predictivesense/dataset/splits.py`` for the Phase 2.5 evaluation set (whole
*sessions* assigned to one split, a content hash, a leakage assertion) rather
than inventing a new one.

A crop's ``session_key`` (``training.dataset.session_key_for``) already
includes its ``object_id``, so assigning whole sessions to a split is
session- **and** object-disjoint by construction: no session, and therefore
no object's positives from one capture burst, is ever split across train/val/
test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from random import Random
from typing import Any

from predictivesense.training.dataset import CropRecord

__all__ = [
    "CropSplits",
    "CropSplitLeakageError",
    "build_crop_splits",
    "assert_no_crop_leakage",
    "crop_splits_content_hash",
    "load_crop_splits",
]

_SPLIT_NAMES = ("train", "val", "test")


class CropSplitLeakageError(RuntimeError):
    """A session id or crop id appears in more than one split, or the split
    file's hash no longer matches the dataset it was built from."""


@dataclass(frozen=True)
class CropSplits:
    sessions: dict[str, list[str]]
    crop_ids: dict[str, list[str]]
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
            "crop_ids": {k: sorted(v) for k, v in self.crop_ids.items()},
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_doc(), indent=2) + "\n", encoding="utf-8")
        return p


def crop_splits_content_hash(records: list[CropRecord], sessions: dict[str, list[str]]) -> str:
    """Hash of the crop set plus the session->split assignment. Changes
    whenever a crop is added/removed/reassigned or the split changes."""

    dataset_blob = sorted((r.crop_id, r.object_id, r.session_key, r.role) for r in records)
    blob = json.dumps(
        {
            "dataset": dataset_blob,
            "sessions": {k: sorted(v) for k, v in sorted(sessions.items())},
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(blob.encode("utf-8")).hexdigest()


def _assign_sessions(session_ids: list[str], *, t_share: float, v_share: float, rng: Random) -> dict[str, str]:
    """Seeded shuffle-and-cut of one group of sessions into train/val/test,
    favouring train first when there are too few sessions to give every
    split at least one."""

    shuffled = session_ids[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    assign: dict[str, str] = {}
    if n == 1:
        assign[shuffled[0]] = "train"
    elif n == 2:
        assign[shuffled[0]] = "train"
        assign[shuffled[1]] = "val"
    else:
        n_train = max(1, round(n * t_share))
        n_val = max(1, round(n * v_share))
        n_train = min(n_train, n - 2)  # leave >=1 each for val and test
        n_val = min(n_val, n - n_train - 1)
        for i, sess in enumerate(shuffled):
            if i < n_train:
                assign[sess] = "train"
            elif i < n_train + n_val:
                assign[sess] = "val"
            else:
                assign[sess] = "test"
    return assign


def build_crop_splits(
    records: list[CropRecord],
    *,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> CropSplits:
    """Assign whole sessions to train/val/test by a seeded shuffle, done
    PER CLASS (``object_id``) independently and then merged.

    Splitting per class, not over one global session pool, is what makes this
    genuinely object-disjoint-AND-stratified (section 11.3): with only a
    handful of sessions per class, one global shuffle can accidentally send
    every one of a class's sessions to train, leaving it entirely absent from
    val/test (measured while building this pipeline - see
    `docs/decisions.md` Phase 11). Splitting fractions are shares of each
    class's OWN session count.
    """

    by_session: dict[str, list[str]] = {}
    sessions_by_class: dict[str, set[str]] = {}
    for r in records:
        by_session.setdefault(r.session_key, []).append(r.crop_id)
        sessions_by_class.setdefault(r.object_id, set()).add(r.session_key)

    if not by_session:
        raise CropSplitLeakageError("no crops to split")

    total = train_fraction + val_fraction + test_fraction
    if total <= 0:
        raise ValueError("fractions must sum to a positive number")
    t_share = train_fraction / total
    v_share = val_fraction / total

    rng = Random(seed)
    assign: dict[str, str] = {}
    for object_id in sorted(sessions_by_class):
        class_sessions = sorted(sessions_by_class[object_id])
        assign.update(_assign_sessions(class_sessions, t_share=t_share, v_share=v_share, rng=rng))

    sessions: dict[str, list[str]] = {name: [] for name in _SPLIT_NAMES}
    crop_ids: dict[str, list[str]] = {name: [] for name in _SPLIT_NAMES}
    for sess, split in assign.items():
        sessions[split].append(sess)
        crop_ids[split].extend(by_session[sess])

    n_sessions = len(assign)
    chash = crop_splits_content_hash(records, sessions)
    splits = CropSplits(
        sessions=sessions,
        crop_ids=crop_ids,
        content_hash=chash,
        seed=seed,
        fractions={
            "train": round(len(sessions["train"]) / n_sessions, 4) if n_sessions else 0.0,
            "val": round(len(sessions["val"]) / n_sessions, 4) if n_sessions else 0.0,
            "test": round(len(sessions["test"]) / n_sessions, 4) if n_sessions else 0.0,
        },
    )
    assert_no_crop_leakage(splits)
    return splits


def assert_no_crop_leakage(splits: CropSplits) -> None:
    """Raise :class:`CropSplitLeakageError` if any session or crop id is
    shared between two splits."""

    sess_sets = {name: set(ids) for name, ids in splits.sessions.items()}
    names = list(sess_sets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = sess_sets[a] & sess_sets[b]
            if shared:
                raise CropSplitLeakageError(
                    f"sessions in both {a!r} and {b!r}: {sorted(shared)}"
                )

    crop_sets = {name: set(ids) for name, ids in splits.crop_ids.items()}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = crop_sets[a] & crop_sets[b]
            if shared:
                raise CropSplitLeakageError(
                    f"crop ids in both {a!r} and {b!r}: {sorted(shared)[:20]}"
                )


def load_crop_splits(path: str | Path, records: list[CropRecord] | None = None) -> CropSplits:
    p = Path(path)
    if not p.is_file():
        raise CropSplitLeakageError(f"split file not found: {p}")
    doc = json.loads(p.read_text(encoding="utf-8"))
    sessions = {k: list(v) for k, v in doc.get("sessions", {}).items()}
    crop_ids = {k: list(v) for k, v in doc.get("crop_ids", {}).items()}
    splits = CropSplits(
        sessions=sessions, crop_ids=crop_ids,
        content_hash=str(doc.get("content_hash", "")),
        seed=int(doc.get("seed", 0)), fractions=doc.get("fractions", {}),
    )
    assert_no_crop_leakage(splits)
    if records is not None:
        now = crop_splits_content_hash(records, sessions)
        if now != splits.content_hash:
            raise CropSplitLeakageError(
                f"split file {p} is stale: content_hash {splits.content_hash[:12]}… "
                f"no longer matches the dataset {now[:12]}…. Re-run the export."
            )
    return splits
