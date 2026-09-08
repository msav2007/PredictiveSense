"""Phase 2.5 detection-evaluation harness: IoU matching, per-class metrics, a
confusion matrix that includes ``background`` and ``unknown``, the false-class
rate (the headline number), and json + markdown writers.

Pure Python + numpy. No ONNX Runtime here - ``scripts/eval_detection.py`` runs
the detector and feeds predictions in.
"""

from __future__ import annotations

from predictivesense.eval.matching import MatchResult, greedy_match, iou_matrix, iou_xyxy
from predictivesense.eval.metrics import (
    ClassMetrics,
    EvalMetrics,
    evaluate,
)
from predictivesense.eval.report import write_reports

__all__ = [
    "MatchResult",
    "greedy_match",
    "iou_matrix",
    "iou_xyxy",
    "ClassMetrics",
    "EvalMetrics",
    "evaluate",
    "write_reports",
]
