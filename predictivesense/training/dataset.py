"""Studio samples -> crop-classification dataset (Phase 11 Part B section 11).

Reuses the existing Object Learning Studio storage untouched (section 11.1):
no second, incompatible dataset format is created. A :class:`CropRecord` is a
thin, additional VIEW over one already-stored :class:`~predictivesense.objects.samples.ObjectSample`
- the crop pixels are read from the sample's own image + box at train/eval
time, never duplicated to disk.

Session grouping (section 11.3): an ``ObjectSample`` carries no explicit
capture-session id, so one is derived here - documented, not invented
silently. A bulk-upload sample's ``batch_id`` IS its session (every sample in
one upload batch is a "session" by definition). A camera-captured sample has
no batch id, so its session is ``(object_id, device_label, a
capture_session_window_s-second bucket of captured_utc)`` - adjacent camera
captures a few seconds apart (near-duplicates, section 11.3's stated concern)
fall in the same bucket; captures more than the window apart are treated as
separate sessions. This already folds in the object id, so a split assignment
at the session level is session- **and** object-disjoint by construction (a
session belongs to exactly one object). See ``docs/decisions.md`` Phase 11.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from predictivesense.objects.registry import ObjectRegistry
from predictivesense.objects.samples import SampleStore

__all__ = [
    "CropRecord",
    "DatasetError",
    "DatasetSummary",
    "build_crop_records",
    "class_map_for",
    "positive_records",
    "rejection_records",
    "session_key_for",
    "validate_dataset",
]

_DEFAULT_SESSION_WINDOW_S = 900  # 15 minutes
_MIN_SAMPLES_PER_CLASS = 4  # too few to split into train/val/test meaningfully


class DatasetError(RuntimeError):
    """The Studio dataset cannot be turned into a trainable crop dataset."""


@dataclass(frozen=True)
class CropRecord:
    """One trainable crop - a view over one committed ``ObjectSample``."""

    crop_id: str  # == sample_id (stable, unique within an object)
    object_id: str  # the class label (Studio object profiles are the classes)
    role: str  # "positive" | "negative" | "hard_negative"
    negative_for: tuple[str, ...]
    session_key: str
    image_path: str  # absolute path to the ORIGINAL (not thumbnail) image
    box: tuple[float, float, float, float]  # x, y, w, h, pixels
    box_confirmed_by_human: bool | None
    source: str


def _parse_captured_utc(value: str) -> float | None:
    try:
        return (
            datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )
    except (ValueError, TypeError):
        return None


def session_key_for(
    object_id: str, sample: dict[str, Any], *, window_s: int = _DEFAULT_SESSION_WINDOW_S
) -> str:
    """Derive a capture-session key for one sample (see module docstring)."""

    batch_id = sample.get("batch_id")
    if batch_id:
        return f"{object_id}:batch:{batch_id}"
    device = sample.get("device_label") or "unknown-device"
    epoch = _parse_captured_utc(str(sample.get("captured_utc") or ""))
    bucket = "no-timestamp" if epoch is None else str(int(epoch // window_s))
    return f"{object_id}:cam:{device}:{bucket}"


def _iter_committed_samples(root: Path):
    """Yield ``(object_id, sample_dict, object_dir)`` for every committed
    (non-staging) sample across every object profile - mirrors
    ``scripts/export_objects_coco.py``'s ``_iter_object_samples`` exactly, so
    the two exporters never disagree about what "committed" means."""

    reg = ObjectRegistry(root)
    for prof in reg.list():
        object_dir = reg.object_dir(prof.object_id)
        store = SampleStore(object_dir, prof.object_id)
        for sample in store.list():
            rel = str(sample.get("path", ""))
            if "_staging" in Path(rel).parts:
                continue
            yield prof.object_id, sample, object_dir


def _load_external_crop_records(manifest_path: Path) -> list[CropRecord]:
    """Phase 12 section 9.1: an external-dataset manifest (produced by
    ``scripts/import_external_dataset.py``, never by this module) is a VIEW
    over already-filtered/deduplicated external samples, exactly like a
    Studio ``CropRecord`` is a view over an ``ObjectSample`` - the image
    pixels are referenced (``image_path``, section 5.3: never copied into the
    repo), not embedded here. Deliberately stdlib-only (``json`` only): the
    external-data dependency (pandas, used to build this manifest) is
    confined to the import script (section 9.4), never to this module."""

    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    object_id = str(doc["object_id"])
    source_name = str(doc.get("source", "external"))
    records: list[CropRecord] = []
    for s in doc.get("samples", []):
        box = s.get("box") or []
        if len(box) != 4:
            continue
        records.append(
            CropRecord(
                crop_id=str(s["sample_id"]),
                object_id=object_id,
                role=str(s.get("role", "positive")),
                negative_for=tuple(s.get("negative_for") or []),
                # Image-disjoint (section 7.3): every box drawn from the same
                # source image shares one session, so they can never straddle
                # a train/val/test split - reuses build_crop_splits() as-is.
                session_key=f"{object_id}:ext:{source_name}:{s['image_id']}",
                image_path=str(s["image_path"]),
                box=tuple(float(v) for v in box),  # type: ignore[assignment]
                box_confirmed_by_human=s.get("box_confirmed_by_human"),
                source=f"external:{source_name}",
            )
        )
    return records


def build_crop_records(
    root: str | Path, *, external_manifests: Sequence[str | Path] = ()
) -> list[CropRecord]:
    """Every committed Studio sample, plus every sample from any given
    external-dataset manifest (Phase 12 section 9.1 - the SAME pipeline,
    never a parallel one), as :class:`CropRecord`."""

    root = Path(root)
    records: list[CropRecord] = []
    for object_id, sample, object_dir in _iter_committed_samples(root):
        box = sample.get("box") or []
        if len(box) != 4:
            continue  # malformed record; validate_dataset() below counts it
        records.append(
            CropRecord(
                crop_id=str(sample["sample_id"]),
                object_id=object_id,
                role=str(sample.get("role", "positive")),
                negative_for=tuple(sample.get("negative_for") or []),
                session_key=session_key_for(object_id, sample),
                image_path=str((object_dir / sample["path"]).resolve()),
                box=tuple(float(v) for v in box),  # type: ignore[assignment]
                box_confirmed_by_human=sample.get("box_confirmed_by_human"),
                source=str(sample.get("source", "camera")),
            )
        )
    for manifest_path in external_manifests:
        records.extend(_load_external_crop_records(Path(manifest_path)))
    return records


def class_map_for(records: list[CropRecord]) -> dict[str, int]:
    """Deterministic ``object_id -> class index`` (sorted, so re-exports and
    re-runs never silently renumber a class)."""

    return {oid: i for i, oid in enumerate(sorted({r.object_id for r in records}))}


def positive_records(records: list[CropRecord]) -> list[CropRecord]:
    """Only ``role == "positive"`` crops - the closed-set softmax classifier's
    training/accuracy signal (section 11.2). A ``hard_negative`` crop stored
    under an object's own folder is, by definition, NOT a positive of that
    class - a crop of the confusable object instead - so it must never be
    handed to the classifier as a labelled example of ``object_id``, or the
    model silently learns to call the confusable object by the wrong name."""

    return [r for r in records if r.role == "positive"]


def rejection_records(records: list[CropRecord]) -> list[CropRecord]:
    """``negative`` / ``hard_negative`` crops - never trained on as a positive
    of any class; used only to measure whether the trained classifier wrongly
    predicts ``negative_for`` for them (section 15's false-class rate)."""

    return [r for r in records if r.role in ("negative", "hard_negative")]


@dataclass(frozen=True)
class DatasetSummary:
    classes: tuple[str, ...]
    per_class_counts: dict[str, dict[str, int]]  # object_id -> {role: count}
    total_crops: int
    invalid_samples: int
    hard_negative_count: int
    human_confirmed_fraction: float
    # Phase 12 section 9.2: a training run declares its composition - how many
    # samples came from each source, per class. ``source`` is CropRecord.source
    # verbatim (e.g. "camera", "upload", "external:open-images-v7").
    per_source_counts: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "classes": list(self.classes),
            "per_class_counts": self.per_class_counts,
            "total_crops": self.total_crops,
            "invalid_samples": self.invalid_samples,
            "hard_negative_count": self.hard_negative_count,
            "human_confirmed_fraction": round(self.human_confirmed_fraction, 4),
            "per_source_counts": self.per_source_counts,
        }


def validate_dataset(
    root: str | Path,
    *,
    min_samples_per_class: int = _MIN_SAMPLES_PER_CLASS,
    external_manifests: Sequence[str | Path] = (),
) -> tuple[list[CropRecord], DatasetSummary]:
    """Build + validate the crop dataset (section 11.4). Raises
    :class:`DatasetError` with a clear message when a class has too few
    positive samples to train meaningfully, or when nothing is collected.

    ``external_manifests`` (Phase 12 section 9.1/9.2) adds imported external
    samples to the SAME dataset/validation pass - Studio samples are never
    replaced, only combined; the returned summary's ``per_source_counts``
    records exactly how much each source contributed, per class."""

    root = Path(root)
    invalid = 0
    for object_id, sample, _dir in _iter_committed_samples(root):
        box = sample.get("box") or []
        if len(box) != 4 or any(not isinstance(v, (int, float)) for v in box):
            invalid += 1

    records = build_crop_records(root, external_manifests=external_manifests)
    if not records:
        raise DatasetError(
            f"no committed samples found under {root} (and no external manifest "
            "supplied) - collect samples in the Object Learning Studio first, or "
            "pass --external-manifest."
        )

    per_class: dict[str, dict[str, int]] = {}
    per_source: dict[str, dict[str, int]] = {}
    for r in records:
        bucket = per_class.setdefault(r.object_id, {"positive": 0, "negative": 0, "hard_negative": 0})
        bucket[r.role] = bucket.get(r.role, 0) + 1
        source_bucket = per_source.setdefault(r.source, {})
        source_bucket[r.object_id] = source_bucket.get(r.object_id, 0) + 1

    too_few = {
        oid: counts["positive"]
        for oid, counts in per_class.items()
        if counts["positive"] < min_samples_per_class
    }
    if too_few:
        raise DatasetError(
            "the following classes have too few POSITIVE samples to train "
            f"meaningfully (need >= {min_samples_per_class} each): {too_few} - "
            "collect more samples in the Studio before training."
        )

    confirmed = sum(1 for r in records if r.box_confirmed_by_human is True)
    total_with_known_provenance = sum(
        1 for r in records if r.box_confirmed_by_human is not None
    )
    summary = DatasetSummary(
        classes=tuple(sorted(per_class)),
        per_class_counts=per_class,
        total_crops=len(records),
        invalid_samples=invalid,
        hard_negative_count=sum(1 for r in records if r.role == "hard_negative"),
        human_confirmed_fraction=(
            confirmed / total_with_known_provenance if total_with_known_provenance else 0.0
        ),
        per_source_counts=per_source,
    )
    return records, summary
