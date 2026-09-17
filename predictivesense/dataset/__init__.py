"""Phase 2.5 labelled evaluation set - COCO detection JSON store, session-disjoint
splits, and per-frame provenance / quality accounting.

Nothing here imports OpenCV, ONNX Runtime, or any HTTP client. The store is plain
JSON + stdlib; ``scripts/build_eval_frames.py`` (which does read video) lives
outside the package.
"""

from __future__ import annotations

from predictivesense.dataset.coco_store import (
    CocoStore,
    CocoStoreError,
    domain_categories,
)
from predictivesense.dataset.detection_convert import (
    DetectionBox,
    DetectionImageRecord,
    add_detection_record,
    own_capture_records,
)
from predictivesense.dataset.detection_splits import (
    DetectionSplitError,
    DetectionSplits,
    EvalCollisionError,
    UnionFind,
    assert_no_eval_collision,
    assert_no_split_leakage,
    build_detection_splits,
    detection_splits_content_hash,
    load_detection_splits,
    merge_duplicate_groups,
)
from predictivesense.dataset.quality import (
    FrameProvenance,
    ProgressSummary,
    unseeded_image_ids,
)
from predictivesense.dataset.splits import (
    SplitLeakageError,
    Splits,
    assert_no_leakage,
    build_splits,
    splits_content_hash,
)

__all__ = [
    "CocoStore",
    "CocoStoreError",
    "domain_categories",
    "DetectionBox",
    "DetectionImageRecord",
    "add_detection_record",
    "own_capture_records",
    "DetectionSplitError",
    "DetectionSplits",
    "EvalCollisionError",
    "UnionFind",
    "assert_no_eval_collision",
    "assert_no_split_leakage",
    "build_detection_splits",
    "detection_splits_content_hash",
    "load_detection_splits",
    "merge_duplicate_groups",
    "FrameProvenance",
    "ProgressSummary",
    "unseeded_image_ids",
    "Splits",
    "SplitLeakageError",
    "assert_no_leakage",
    "build_splits",
    "splits_content_hash",
]
