"""Stage 5 section 3.2/3.5: image-oriented detection records -> the existing
COCO store, for both external-source and own-capture data through the
IDENTICAL path.

Stdlib only (mirrors ``predictivesense/dataset/``'s existing no-cv2/no-pandas
discipline). Raw image decoding, CSV reading and pixel-level quality/dedup
checks happen in ``scripts/`` (which may use cv2/pandas) and are handed to
this module as already-extracted, plain-Python records.

A :class:`DetectionImageRecord` with an empty ``boxes`` tuple is a
LEGITIMATE zero-annotation image (hard negative / background, stage5 section
2.2) - it is added to the store exactly like any other image, never dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from predictivesense.dataset.coco_store import CocoStore

__all__ = [
    "DetectionBox",
    "DetectionImageRecord",
    "add_detection_record",
    "own_capture_records",
]


@dataclass(frozen=True)
class DetectionBox:
    """One annotated box on a :class:`DetectionImageRecord`."""

    ps_class: str  # the TARGET class this box counts as (post class-map/merge)
    bbox: tuple[float, float, float, float]  # x, y, w, h, pixels
    source_class: str  # the SOURCE class name as the source dataset named it
    box_confirmed_by_human: bool | None


@dataclass(frozen=True)
class DetectionImageRecord:
    """One image entering the detection dataset, from any source."""

    file_name: str  # store dedupe key - stable across repeated conversion runs
    width: int
    height: int
    session_id: str  # split-grouping key: own-capture session, or an external per-image/dup-group id
    image_path: str
    image_sha256: str
    source_dataset: str  # "open-images-v7" | "data/objects" | ...
    source_version: str  # split name / audit ref / git_commit, whatever the source records
    original_image_id: str
    license: str = ""
    author: str = ""
    source_url: str = ""
    boxes: tuple[DetectionBox, ...] = ()
    filtering_notes: tuple[str, ...] = ()
    own_capture_role: str | None = None
    own_capture_provenance: dict[str, Any] | None = None
    own_capture_quality: dict[str, Any] | None = None


def add_detection_record(store: CocoStore, record: DetectionImageRecord) -> int:
    """Register ``record`` on ``store`` (idempotent on ``file_name``) and set
    its boxes, including the zero-box case. Returns the image id."""

    iid = store.add_image(
        file_name=record.file_name,
        width=record.width,
        height=record.height,
        session_id=record.session_id,
        provenance={
            "source_dataset": record.source_dataset,
            "source_version": record.source_version,
            "original_image_id": record.original_image_id,
            "image_sha256": record.image_sha256,
            "image_path": record.image_path,
            "license": record.license,
            "author": record.author,
            "source_url": record.source_url,
            "target_classes_present": sorted({b.ps_class for b in record.boxes}),
            "filtering_notes": list(record.filtering_notes),
            "own_capture_role": record.own_capture_role,
            "own_capture_provenance": record.own_capture_provenance,
            "own_capture_quality": record.own_capture_quality,
        },
    )
    boxes_payload = [
        {
            "category": b.ps_class,
            "bbox": list(b.bbox),
            "extra": {
                "source_class": b.source_class,
                "box_confirmed_by_human": b.box_confirmed_by_human,
            },
        }
        for b in record.boxes
    ]
    store.set_frame_boxes(iid, boxes_payload, seeded=False, labelled=True)
    return iid


def own_capture_records(
    object_id: str,
    ps_class: str,
    samples: list[dict[str, Any]],
    *,
    object_dir_str: str,
    session_key_fn: Any,
) -> list[DetectionImageRecord]:
    """One :class:`DetectionImageRecord` per committed own-capture sample of
    ``object_id`` mapped to ``ps_class``. ``session_key_fn`` is
    ``predictivesense.training.dataset.session_key_for`` passed in by the
    caller (stage5 section 3.5: identical session derivation as the
    crop-classifier path, reused rather than reimplemented; imported at the
    call site, not here, to keep this module free of a hard dependency on
    ``training/``)."""

    records: list[DetectionImageRecord] = []
    for sample in samples:
        box = sample.get("box") or []
        role = str(sample.get("role", "positive"))
        boxes: tuple[DetectionBox, ...] = ()
        if role == "positive" and len(box) == 4:
            boxes = (
                DetectionBox(
                    ps_class=ps_class,
                    bbox=tuple(float(v) for v in box),  # type: ignore[arg-type]
                    source_class=f"studio:{object_id}",
                    box_confirmed_by_human=sample.get("box_confirmed_by_human"),
                ),
            )
        rel_path = str(sample.get("path", ""))
        image_path = Path(object_dir_str) / rel_path
        image_sha = sha256(image_path.read_bytes()).hexdigest() if image_path.is_file() else ""
        records.append(
            DetectionImageRecord(
                file_name=f"objects:{object_id}:{sample['sample_id']}",
                width=int(sample.get("width", 0)),
                height=int(sample.get("height", 0)),
                session_id=session_key_fn(object_id, sample),
                image_path=str(image_path),
                image_sha256=image_sha,
                source_dataset="data/objects",
                source_version=str(sample.get("git_commit", "")),
                original_image_id=str(sample["sample_id"]),
                boxes=boxes,
                filtering_notes=(f"role:{role}",) if role != "positive" else (),
                own_capture_role=role,
                own_capture_provenance={
                    "device_label": sample.get("device_label"),
                    "captured_utc": sample.get("captured_utc"),
                    "conditions": sample.get("conditions"),
                    "consent_ack": sample.get("consent_ack"),
                },
                own_capture_quality=sample.get("quality"),
            )
        )
    return records
