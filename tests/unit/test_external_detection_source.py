"""Stage 5 section 2.1: exhaustive multi-class requery of a raw Open Images
export. This is the test the stage prompt explicitly demands - an image
known to contain two target classes must yield BOTH classes' boxes, proving
the converter does not silently teach the model that an unboxed second class
is background.
"""

from __future__ import annotations

import pytest

from scripts.external_detection_source import query_active_classes

pytestmark = pytest.mark.unit


def _build_export(root, *, boxes: list[tuple[str, str, float, float, float, float]]) -> None:
    split_dir = root / "train"
    labels_dir = split_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    header = "ImageID,LabelName,XMin,XMax,YMin,YMax\n"
    rows = [header] + [f"{iid},{mid},{xmin},{xmax},{ymin},{ymax}\n" for iid, mid, xmin, xmax, ymin, ymax in boxes]
    (labels_dir / "detections.csv").write_text("".join(rows), encoding="utf-8")


def test_an_image_with_two_target_classes_yields_both_classes_boxes(tmp_path):
    # "dual" was originally found by querying for watch, but ALSO contains an
    # unboxed (from a single-class importer's perspective) mug - the exact
    # landmine section 2.1 describes.
    _build_export(
        tmp_path,
        boxes=[
            ("dual", "/m/0gjkl", 0.1, 0.3, 0.1, 0.3),      # watch
            ("dual", "/m/02jvh9", 0.5, 0.9, 0.5, 0.9),     # mug
            ("watch_only", "/m/0gjkl", 0.2, 0.4, 0.2, 0.4),
            ("mug_only", "/m/02jvh9", 0.2, 0.4, 0.2, 0.4),
            ("irrelevant", "/m/050k8", 0.1, 0.2, 0.1, 0.2),  # a mobile phone - not an active class
        ],
    )
    labels_csv = tmp_path / "train" / "labels" / "detections.csv"
    active_mids = {"/m/0gjkl": "watch", "/m/02jvh9": "mug/cup"}
    hits = query_active_classes(
        labels_csv, image_ids={"dual", "watch_only", "mug_only", "irrelevant"}, active_mids=set(active_mids),
    )

    assert set(hits["dual"]) or True  # sanity: key exists
    dual_mids = {b.mid for b in hits["dual"]}
    assert dual_mids == {"/m/0gjkl", "/m/02jvh9"}, "both classes' boxes must survive on the shared image"

    assert {b.mid for b in hits["watch_only"]} == {"/m/0gjkl"}
    assert {b.mid for b in hits["mug_only"]} == {"/m/02jvh9"}
    assert "irrelevant" not in hits, "a box for a class outside the active set must not appear"


def test_an_image_found_via_one_class_query_is_not_pre_filtered_before_the_other_classes_run(tmp_path):
    """The landmine, made concrete: a naive per-class importer would query
    image_ids={"dual"} scoped to watch's own positive set, then separately
    scope to mug's own positive set - never noticing they're the SAME image
    unless the caller explicitly unions first. query_active_classes takes the
    image_id universe as a single argument precisely so that never has a
    chance to happen: there is no per-class image-set restriction anywhere
    in its signature."""

    _build_export(
        tmp_path,
        boxes=[
            ("dual", "/m/0gjkl", 0.1, 0.3, 0.1, 0.3),
            ("dual", "/m/02jvh9", 0.5, 0.9, 0.5, 0.9),
        ],
    )
    labels_csv = tmp_path / "train" / "labels" / "detections.csv"
    hits = query_active_classes(labels_csv, image_ids={"dual"}, active_mids={"/m/0gjkl", "/m/02jvh9"})
    assert len(hits["dual"]) == 2


def test_empty_active_mids_raises_rather_than_silently_dropping_everything(tmp_path):
    _build_export(tmp_path, boxes=[("x", "/m/0gjkl", 0.1, 0.3, 0.1, 0.3)])
    labels_csv = tmp_path / "train" / "labels" / "detections.csv"
    with pytest.raises(ValueError, match="active_mids"):
        query_active_classes(labels_csv, image_ids={"x"}, active_mids=set())


def test_two_boxes_of_the_same_class_on_one_image_both_survive(tmp_path):
    """The 187-image / 220-box shape of the real watch manifest comes from
    images with more than one watch in frame - the grouping must keep both
    boxes, not collapse or overwrite one."""

    _build_export(
        tmp_path,
        boxes=[
            ("two_watches", "/m/0gjkl", 0.05, 0.25, 0.05, 0.25),
            ("two_watches", "/m/0gjkl", 0.6, 0.8, 0.6, 0.8),
        ],
    )
    labels_csv = tmp_path / "train" / "labels" / "detections.csv"
    hits = query_active_classes(labels_csv, image_ids={"two_watches"}, active_mids={"/m/0gjkl"})
    assert len(hits["two_watches"]) == 2


def test_image_outside_the_id_scope_is_excluded_even_with_a_matching_box(tmp_path):
    _build_export(tmp_path, boxes=[("in_scope", "/m/0gjkl", 0.1, 0.3, 0.1, 0.3), ("out_of_scope", "/m/0gjkl", 0.1, 0.3, 0.1, 0.3)])
    labels_csv = tmp_path / "train" / "labels" / "detections.csv"
    hits = query_active_classes(labels_csv, image_ids={"in_scope"}, active_mids={"/m/0gjkl"})
    assert list(hits.keys()) == ["in_scope"]
