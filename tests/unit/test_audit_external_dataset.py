"""Phase 12 section 12: the external-dataset audit script produces correct
per-class counts on a synthetic fixture that mimics a real FiftyOne export -
including the master (unfiltered) detections.csv quirk the real download has.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.audit_external_dataset import audit_external_dataset

pytestmark = pytest.mark.unit


def _jpeg_file(path, fill: int, size=(48, 64)) -> None:
    import cv2

    ok, buf = cv2.imencode(".jpg", np.full((*size, 3), fill, np.uint8))
    assert ok
    path.write_bytes(buf.tobytes())


def _build_fixture_export(root, *, image_ids: list[str]) -> None:
    """A tiny export mirroring the real one's layout: classes.csv, an
    UNFILTERED master detections.csv (rows for images NOT in this download,
    exactly like the real 2.2GB file), and only ``image_ids`` under data/."""

    split_dir = root / "train"
    data_dir = split_dir / "data"
    labels_dir = split_dir / "labels"
    metadata_dir = split_dir / "metadata"
    for d in (data_dir, labels_dir, metadata_dir):
        d.mkdir(parents=True, exist_ok=True)

    for i, image_id in enumerate(image_ids):
        _jpeg_file(data_dir / f"{image_id}.jpg", fill=10 + i)

    (metadata_dir / "classes.csv").write_text(
        "/m/0gjkl,Watch\n/m/04dr76w,Bottle\n/m/0fz0h,Honeycomb\n", encoding="utf-8"
    )

    header = "ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside\n"
    rows = [header]
    # A row for an image NOT in this download's data/ - the real detections.csv
    # is the full multi-split master file; the audit must filter these out.
    rows.append("not_in_download_0001,xclick,/m/0gjkl,1,0.1,0.2,0.1,0.2,0,0,0,0,0\n")
    for image_id in image_ids[:2]:
        rows.append(f"{image_id},xclick,/m/0gjkl,1,0.10,0.60,0.20,0.80,0,0,0,0,0\n")
    if len(image_ids) > 2:
        rows.append(f"{image_ids[2]},xclick,/m/04dr76w,1,0.05,0.15,0.05,0.15,0,0,0,0,0\n")
    (labels_dir / "detections.csv").write_text("".join(rows), encoding="utf-8")


def test_audit_filters_master_csv_to_downloaded_images_only(tmp_path):
    image_ids = ["imgaaa1", "imgaaa2", "imgaaa3"]
    _build_fixture_export(tmp_path, image_ids=image_ids)

    report = audit_external_dataset(
        tmp_path, "train", target_classes={"/m/0gjkl": "Watch", "/m/04dr76w": "Bottle"}
    )

    assert report["total_samples"] == 3
    # The "not_in_download" row must never be counted.
    assert report["total_annotations_in_subset"] == 3
    assert report["target_class_audit"]["Watch"]["boxes"] == 2
    assert report["target_class_audit"]["Watch"]["images"] == 2
    assert report["target_class_audit"]["Bottle"]["boxes"] == 1
    assert report["images_with_no_target_class_annotation"] == 0


def test_audit_reports_zero_matches_for_absent_classes(tmp_path):
    image_ids = ["imgbbb1"]
    _build_fixture_export(tmp_path, image_ids=image_ids)

    report = audit_external_dataset(
        tmp_path, "train", target_classes={"/m/does_not_exist": "Comb"}
    )
    assert report["target_classes_with_zero_matches"] == ["Comb"]
    assert report["images_with_no_target_class_annotation"] == 1
