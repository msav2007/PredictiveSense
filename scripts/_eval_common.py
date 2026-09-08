"""Shared helpers for scripts/eval_detection.py and scripts/fit_thresholds.py.

Model registry, per-frame detection over a labelled split, and split-selection /
completeness checks. Kept in ``scripts/`` (not the package) because it needs
``cv2.imread`` and a live ONNX detector - both of which are scoped away from
``predictivesense/eval/`` by the import guard.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from predictivesense.config.settings import AppConfig, DetectorConfig
from predictivesense.core.types import Frame
from predictivesense.dataset.coco_store import CocoStore
from predictivesense.dataset.splits import Splits, load_splits
from predictivesense.logging_setup import get_logger
from predictivesense.perception.detector import ObjectDetector
from predictivesense.perception.policy import RecognitionPolicy

_LOG = get_logger("predictivesense.scripts.eval")


class EvalInputError(RuntimeError):
    """A split is empty, partially labelled, stale, or missing frame images."""


@dataclass(frozen=True)
class ModelSpec:
    name: str
    detector_config: DetectorConfig

    def model_hash(self) -> str:
        p = Path(self.detector_config.model_path)
        if not p.is_file():
            return "missing"
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()


def model_spec(name: str, config: AppConfig, *, models_dir: Path | None = None) -> ModelSpec:
    """Resolve a ``--model`` name to a detector config.

    ``yolo11n`` uses the profile's detector settings. ``yolox_tiny`` is the
    Phase 2.5 permissively-licensed comparison model (Apache-2.0): 416px input,
    the ``yolox`` decode variant, otherwise identical thresholds / NMS / policy.
    """

    md = models_dir or Path("models")
    base = config.perception.detector
    key = name.lower().replace("-", "_")
    if key in ("yolo11n", "yolo", "yolo11"):
        return ModelSpec(name="yolo11n", detector_config=base)
    if key in ("yolox_tiny", "yolox", "yoloxtiny"):
        return ModelSpec(
            name="yolox_tiny",
            detector_config=base.model_copy(
                update={
                    "model_path": md / "yolox_tiny.onnx",
                    "input_size": 416,
                    "decode": "yolox",
                }
            ),
        )
    raise EvalInputError(
        f"unknown --model {name!r}; known: yolo11n, yolox_tiny "
        f"(or pass an explicit --model-path / --decode / --input-size)"
    )


def select_split(
    store: CocoStore, splits: Splits, split: str
) -> list[int]:
    """Image ids for ``split``, asserting the split is present and fully
    labelled (BLOCK 3.11 - the harness refuses a partial split and says what is
    missing)."""

    if split not in ("val", "test"):
        raise EvalInputError(f"split must be 'val' or 'test', got {split!r}")
    ids = [iid for iid in splits.image_ids.get(split, []) if store.has_image(iid)]
    if not ids:
        raise EvalInputError(
            f"the {split!r} split has no images. "
            + (
                "Only one recording session is labelled, so test is empty - "
                "record and label more sessions."
                if split == "test"
                else "Label frames at /label and re-run scripts/build_splits.py."
            )
        )
    unlabelled = [iid for iid in ids if not store.image(iid).get("labelled")]
    if unlabelled:
        raise EvalInputError(
            f"{len(unlabelled)} of {len(ids)} images in the {split!r} split are "
            f"not labelled yet (ids {unlabelled[:10]}{'…' if len(unlabelled) > 10 else ''}). "
            f"Finish labelling before evaluating."
        )
    return ids


def load_store_and_splits(config: AppConfig) -> tuple[CocoStore, Splits]:
    from predictivesense.dataset.coco_store import domain_categories

    coco_path = Path(config.dataset.coco_path)
    if not coco_path.is_file():
        raise EvalInputError(
            f"no annotation store at {coco_path} - run scripts/build_eval_frames.py"
        )
    store = CocoStore.load_or_create(
        coco_path, domain_categories(config.policy.domain_classes)
    )
    splits_path = Path(config.dataset.splits_path)
    if not splits_path.is_file():
        raise EvalInputError(
            f"no split file at {splits_path} - run scripts/build_splits.py"
        )
    splits = load_splits(splits_path, store)  # raises if the hash is stale
    return store, splits


def _frame_from(path: Path, frame_id: int) -> Frame:
    img = cv2.imread(str(path))
    if img is None:
        raise EvalInputError(f"could not read frame image {path}")
    img = np.ascontiguousarray(img)
    return Frame(
        frame_id=frame_id, capture_ts=float(frame_id), image=img,
        width=int(img.shape[1]), height=int(img.shape[0]),
        source_id=str(path), seq=frame_id,
    )


def run_predictions(
    store: CocoStore,
    image_ids: list[int],
    frames_dir: Path,
    detector: ObjectDetector,
    *,
    policy: RecognitionPolicy | None,
) -> tuple[list[dict[str, Any]], set[str], dict[str, int]]:
    """Detector (+ optional policy) over each frame -> per-image
    ``{image_id, gt, pred}`` samples for ``predictivesense.eval.evaluate``.

    Returns ``(samples, predicted_label_set, policy_state_counts)``. GT boxes are
    converted from COCO xywh to xyxy. Predicted labels are ``class_name`` after
    the policy (``"unknown"`` for every non-accepted detection).
    """

    samples: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    state_counts: dict[str, int] = {}
    for i, iid in enumerate(image_ids):
        im = store.image(iid)
        path = frames_dir / im["file_name"]
        frame = _frame_from(path, i)
        raw = detector.infer(frame)
        if policy is not None:
            outcome = policy.apply(
                raw, frame_width=frame.width, frame_height=frame.height
            )
            dets = outcome.detections
            for d in dets:
                state_counts[d.policy_state] = state_counts.get(d.policy_state, 0) + 1
        else:
            dets = raw

        gt = []
        for an in store.annotations_for(iid):
            x, y, w, h = (float(v) for v in an["bbox"])
            gt.append(
                {"bbox": [x, y, x + w, y + h], "label": store.category_name(int(an["category_id"]))}
            )
        pred = []
        for d in dets:
            seen_labels.add(d.class_name)
            pred.append(
                {"bbox": list(d.bbox), "label": d.class_name, "score": float(d.score)}
            )
        samples.append({"image_id": iid, "gt": gt, "pred": pred})
    return samples, seen_labels, state_counts
