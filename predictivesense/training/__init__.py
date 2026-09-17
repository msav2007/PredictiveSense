"""Phase 11 Part B - Studio-to-trained-model pipeline.

Everything under this package is optional: importing it requires the
``[train]`` extra (``torch``, ``torchvision``); the runtime app
(``predictivesense.pipeline``, ``predictivesense.api``) never imports it.
``predictivesense/perception/classifier.py`` is the one exception - it loads a
**trained ONNX artifact** via the existing ``onnxruntime`` runtime, exactly
like the detector and pose estimator, so serving a trained model never
requires torch.

Two-stage design (section 10.1, `docs/decisions.md` Phase 11): the existing
YOLO detector stays a class-agnostic region proposer; a small crop classifier
is trained on Object Learning Studio samples (each already exactly one box per
image - crop-classifier data with no conversion) and named via ONNX Runtime
inference at serve time. No detector fine-tuning (CPU-only, tens of images per
class, and the unresolved Ultralytics-weights licence note in
``docs/decisions.md``).
"""

from __future__ import annotations
