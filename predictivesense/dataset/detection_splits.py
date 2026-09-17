"""Stage 5 section 3.4: image-, near-duplicate-group-, and session-disjoint
train/val/test splits for the detection dataset.

Extends, not reimplements: the per-class-independent seeded shuffle-and-cut
(assign whole GROUPS to a split, one class at a time, then merge) is the
exact algorithm ``predictivesense/training/splits.py::build_crop_splits``
already established for the crop dataset (itself modelled on
``predictivesense/dataset/splits.py``'s Phase 2.5 session split). The
extension here is what counts as a "group": for own-capture data a GROUP is
still a capture session (``predictivesense.training.dataset.session_key_for``,
imported at the call site); for external data, images that are near-duplicates
of each other (Hamming distance <= threshold, supplied by the caller from
``predictivesense.objects.quality.dhash``/``hamming``) are merged into one
atomic group via union-find BEFORE the per-class shuffle runs, so a
near-duplicate group can never straddle a split boundary (stage5 section
3.4) - not even when the two images came from different original sources.

``data/eval`` is never a source for train or val: :func:`assert_no_eval_collision`
hard-fails the whole build on any id/sha256 collision - it is called by the
orchestration script BEFORE this module's splits are even attempted, and
again here as a second, cheap belt-and-braces check once the groups are
known.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from random import Random
from typing import Any, Iterable

from predictivesense.dataset.coco_store import CocoStore

__all__ = [
    "DetectionSplits",
    "DetectionSplitError",
    "EvalCollisionError",
    "UnionFind",
    "merge_duplicate_groups",
    "build_detection_splits",
    "assert_no_split_leakage",
    "assert_no_eval_collision",
    "detection_splits_content_hash",
    "load_detection_splits",
]

_SPLIT_NAMES = ("train", "val", "test")
_MIN_IMAGES_FOR_TRAINING_SPLIT_DEFAULT = 50
_ZERO_BOX_STRATUM = "__zero_box__"  # pseudo-class for hard-negative/background groups (stage5.1)


class DetectionSplitError(RuntimeError):
    """A group or image id appears in more than one split, or the split file
    no longer matches the store it was built from."""


class EvalCollisionError(RuntimeError):
    """A training-set image id or content hash collides with ``data/eval`` -
    hard failure, never a warning (stage5 section 3.4)."""


class UnionFind:
    """Minimal union-find for merging near-duplicate groups. Stdlib only."""

    def __init__(self, items: Iterable[str]) -> None:
        self._parent = {x: x for x in items}

    def find(self, x: str) -> str:
        self._parent.setdefault(x, x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


def merge_duplicate_groups(
    base_groups: dict[str, str], duplicate_pairs: Iterable[tuple[str, str]]
) -> dict[str, str]:
    """``base_groups``: ``image_id -> group_id`` (e.g. one group per own-capture
    session, or one singleton group per external image). ``duplicate_pairs``:
    ``(image_id_a, image_id_b)`` pairs flagged near-duplicate by the audit.
    Returns a new ``image_id -> merged_group_id`` where every image whose
    ORIGINAL group is connected (directly or transitively, and through
    duplicate edges) ends up sharing one group id - so a duplicate group can
    never straddle a split even when its images started in different
    sessions or came from different sources."""

    uf = UnionFind(base_groups.values())
    for a, b in duplicate_pairs:
        ga, gb = base_groups.get(a), base_groups.get(b)
        if ga is not None and gb is not None:
            uf.union(ga, gb)
    return {img: uf.find(grp) for img, grp in base_groups.items()}


@dataclass(frozen=True)
class DetectionSplits:
    groups: dict[str, list[str]]
    image_ids: dict[str, list[int]]
    content_hash: str
    seed: int
    fractions: dict[str, float] = field(default_factory=dict)
    # Stage 5.1: split into two unambiguously-named fields - the old single
    # `realized_per_class_counts` held BOX counts under a name that reads as
    # an image count (and was misread that way in the Stage 5 report: watch's
    # "160/28/32" sums to 220 boxes across only 187 images, not 220 images).
    realized_per_class_box_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    realized_per_class_image_counts: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_doc(self) -> dict[str, Any]:
        return {
            "format": 2,
            "seed": self.seed,
            "fractions": self.fractions,
            "content_hash": self.content_hash,
            "groups": {k: sorted(v) for k, v in self.groups.items()},
            "image_ids": {k: sorted(v) for k, v in self.image_ids.items()},
            "realized_per_class_box_counts": self.realized_per_class_box_counts,
            "realized_per_class_image_counts": self.realized_per_class_image_counts,
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_doc(), indent=2) + "\n", encoding="utf-8")
        return p


def detection_splits_content_hash(store: CocoStore, groups: dict[str, list[str]]) -> str:
    blob = json.dumps(
        {"store": store.content_hash(), "groups": {k: sorted(v) for k, v in sorted(groups.items())}},
        sort_keys=True, separators=(",", ":"),
    )
    return sha256(blob.encode("utf-8")).hexdigest()


def _primary_class(store: CocoStore, image_id: int) -> str | None:
    """The target class with the most boxes on this image (alphabetical
    tiebreak) - used only to stratify images that happen to carry more than
    one target class. Every image converted in this stage's actual run
    carries at most one (see docs/phase-reports/phase13-stage5-dataset.md);
    this exists so the machinery is correct if that ever changes."""

    counts: dict[str, int] = {}
    for ann in store.annotations_for(image_id):
        name = store.category_name(int(ann["category_id"]))
        counts[name] = counts.get(name, 0) + 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _assign_groups(group_ids: list[str], *, t_share: float, v_share: float, rng: Random) -> dict[str, str]:
    shuffled = group_ids[:]
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
        n_train = min(n_train, n - 2)
        n_val = min(n_val, n - n_train - 1)
        for i, grp in enumerate(shuffled):
            if i < n_train:
                assign[grp] = "train"
            elif i < n_train + n_val:
                assign[grp] = "val"
            else:
                assign[grp] = "test"
    return assign


def build_detection_splits(
    store: CocoStore,
    image_to_group: dict[int, str],
    *,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    min_images_per_class: int = _MIN_IMAGES_FOR_TRAINING_SPLIT_DEFAULT,
) -> DetectionSplits:
    """Assign whole GROUPS to train/val/test, per class independently (the
    ``build_crop_splits`` pattern), then merge. Refuses (raises
    :class:`DetectionSplitError`) to emit a ``train`` split for any class
    whose realized image count is below ``min_images_per_class`` - stage5
    section 3.5's "one comb image must not become a training set by
    accident" requirement, generalised to every class, not just comb.

    Stage 5.1 fix: a group whose every image is a ZERO-BOX record (hard
    negative / background) never has a primary class, so it used to be
    silently absent from every ``groups_by_class`` entry and therefore from
    every split - 11 of 635 real images in the Stage 5 run. Such groups now
    get their own pseudo-class stratum (:data:`_ZERO_BOX_STRATUM`), assigned
    by the identical seeded shuffle-and-cut, NOT subject to
    ``min_images_per_class`` (hard negatives are not a trainable positive
    class - refusing to split them would just re-lose them a different way).
    """

    by_group: dict[str, list[int]] = {}
    groups_by_class: dict[str, set[str]] = {}
    ungrouped_images: list[int] = []
    for iid in store.image_ids():
        grp = image_to_group.get(iid)
        if grp is None:
            ungrouped_images.append(iid)
            continue
        by_group.setdefault(grp, []).append(iid)
        cls = _primary_class(store, iid)
        if cls is not None:
            groups_by_class.setdefault(cls, set()).add(grp)
    if ungrouped_images:
        raise DetectionSplitError(f"images with no assigned split group: {ungrouped_images[:20]}")

    classified_groups = {g for groups in groups_by_class.values() for g in groups}
    zero_box_groups = set(by_group) - classified_groups
    if zero_box_groups:
        groups_by_class[_ZERO_BOX_STRATUM] = zero_box_groups

    total = train_fraction + val_fraction + test_fraction
    if total <= 0:
        raise ValueError("fractions must sum to a positive number")
    t_share, v_share = train_fraction / total, val_fraction / total

    rng = Random(seed)
    assign: dict[str, str] = {}
    for cls in sorted(groups_by_class):
        class_groups = sorted(groups_by_class[cls])
        class_assign = _assign_groups(class_groups, t_share=t_share, v_share=v_share, rng=rng)
        n_train_images = sum(
            len(by_group[g]) for g, split in class_assign.items() if split == "train"
        )
        if cls != _ZERO_BOX_STRATUM and n_train_images < min_images_per_class:
            raise DetectionSplitError(
                f"class {cls!r} has {n_train_images} image(s) available for a train split - below the "
                f"configured minimum of {min_images_per_class}. Refusing to emit a training split for it "
                f"(stage5 section 3.5) - collect more data or lower --min-images-per-class explicitly."
            )
        assign.update(class_assign)

    groups: dict[str, list[str]] = {name: [] for name in _SPLIT_NAMES}
    image_ids: dict[str, list[int]] = {name: [] for name in _SPLIT_NAMES}
    for grp, split in assign.items():
        groups[split].append(grp)
        image_ids[split].extend(by_group[grp])

    # Box counts, NOT image counts - a per-image count would need distinct
    # image ids per class, but these are annotation (box) tallies, and one
    # image commonly carries more than one box of its class (stage5.1 fix:
    # this was previously reported ambiguously as `realized_per_class_counts`
    # with no unit in the key, and misread as image counts in the Stage 5
    # report - e.g. watch's "160/28/32" sums to 220 boxes across 187 images).
    realized_boxes: dict[str, dict[str, int]] = {name: {} for name in _SPLIT_NAMES}
    realized_images: dict[str, dict[str, int]] = {name: {} for name in _SPLIT_NAMES}
    for split in _SPLIT_NAMES:
        for iid in image_ids[split]:
            anns = store.annotations_for(iid)
            classes_here: set[str] = set()
            for ann in anns:
                cname = store.category_name(int(ann["category_id"]))
                realized_boxes[split][cname] = realized_boxes[split].get(cname, 0) + 1
                classes_here.add(cname)
            for cname in classes_here:
                realized_images[split][cname] = realized_images[split].get(cname, 0) + 1
            if not anns:
                realized_images[split]["__zero_box__"] = realized_images[split].get("__zero_box__", 0) + 1

    n_groups = len(assign)
    chash = detection_splits_content_hash(store, groups)
    splits = DetectionSplits(
        groups=groups, image_ids=image_ids, content_hash=chash, seed=seed,
        fractions={
            name: round(len(groups[name]) / n_groups, 4) if n_groups else 0.0
            for name in _SPLIT_NAMES
        },
        realized_per_class_box_counts=realized_boxes,
        realized_per_class_image_counts=realized_images,
    )
    assert_no_split_leakage(splits)
    n_split_images = sum(len(ids) for ids in image_ids.values())
    n_store_images = len(store.image_ids())
    if n_split_images != n_store_images:
        raise DetectionSplitError(
            f"internal error: {n_split_images} images landed in a split but the store has "
            f"{n_store_images} - every image must land in exactly one split."
        )
    return splits


def assert_no_split_leakage(splits: DetectionSplits) -> None:
    names = list(splits.groups)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared_groups = set(splits.groups[a]) & set(splits.groups[b])
            if shared_groups:
                raise DetectionSplitError(f"groups in both {a!r} and {b!r}: {sorted(shared_groups)}")
            shared_images = set(splits.image_ids[a]) & set(splits.image_ids[b])
            if shared_images:
                raise DetectionSplitError(f"image ids in both {a!r} and {b!r}: {sorted(shared_images)[:20]}")


def assert_no_eval_collision(
    *,
    train_val_image_ids: Iterable[int],
    store: CocoStore,
    eval_image_shas: set[str],
    eval_phashes: list[tuple[str, str]] = (),
    duplicate_hamming_max: int = 6,
) -> None:
    """Hard-fail (:class:`EvalCollisionError`) if any train/val image's
    ``image_sha256`` (from its ``ps_provenance``) matches, or near-duplicate-
    matches, anything in ``data/eval``. ``test`` is intentionally excluded
    from this check's caller - only train/val ever touch model selection."""

    from predictivesense.objects.quality import hamming

    for iid in train_val_image_ids:
        prov = store.image(iid).get("ps_provenance") or {}
        sha = str(prov.get("image_sha256", ""))
        if sha and sha in eval_image_shas:
            raise EvalCollisionError(
                f"image id {iid} (sha256 {sha[:12]}...) content-hash-collides with data/eval - "
                "train/val <-> eval collision. Build aborted."
            )
        phash = str((prov.get("own_capture_quality") or {}).get("phash", ""))
        if phash and eval_phashes:
            for eval_id, other in eval_phashes:
                if hamming(phash, other) <= duplicate_hamming_max:
                    raise EvalCollisionError(
                        f"image id {iid} is a near-duplicate (Hamming <= {duplicate_hamming_max}) of eval "
                        f"frame {eval_id} - train/val <-> eval collision. Build aborted."
                    )


def load_detection_splits(path: str | Path, store: CocoStore | None = None) -> DetectionSplits:
    p = Path(path)
    if not p.is_file():
        raise DetectionSplitError(f"split file not found: {p}")
    doc = json.loads(p.read_text(encoding="utf-8"))
    groups = {k: list(v) for k, v in doc.get("groups", {}).items()}
    image_ids = {k: [int(i) for i in v] for k, v in doc.get("image_ids", {}).items()}
    splits = DetectionSplits(
        groups=groups, image_ids=image_ids,
        content_hash=str(doc.get("content_hash", "")), seed=int(doc.get("seed", 0)),
        fractions=doc.get("fractions", {}),
        realized_per_class_box_counts=doc.get("realized_per_class_box_counts", {}),
        realized_per_class_image_counts=doc.get("realized_per_class_image_counts", {}),
    )
    assert_no_split_leakage(splits)
    if store is not None:
        now = detection_splits_content_hash(store, groups)
        if now != splits.content_hash:
            raise DetectionSplitError(
                f"split file {p} is stale: content_hash {splits.content_hash[:12]}... "
                f"no longer matches the store {now[:12]}.... Re-run the build."
            )
    return splits
