"""Per-frame perception: an ONNX object detector and an ONNX pose estimator.

Scope (Phase 2): per-frame detection and pose only. No tracking, no identity, no
association, no temporal state, no risk, no voice. The detector and pose wrappers
are model-agnostic - model path, input size, class list and thresholds all come
from ``config.perception`` (see ``docs/decisions.md``).

``onnxruntime`` is imported **only** under this package and
``predictivesense/camera/`` (amended ``tests/unit/test_no_forbidden_imports.py``).
"""

from __future__ import annotations

from predictivesense.perception.engine import PerceptionEngine, build_perception
from predictivesense.perception.types import PerceptionResult

__all__ = ["PerceptionEngine", "build_perception", "PerceptionResult"]
