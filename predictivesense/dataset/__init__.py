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
    "FrameProvenance",
    "ProgressSummary",
    "unseeded_image_ids",
    "Splits",
    "SplitLeakageError",
    "assert_no_leakage",
    "build_splits",
    "splits_content_hash",
]
