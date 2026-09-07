"""ONNX object detector wrapper (YOLO-family detect head).

``ObjectDetector.infer(frame) -> list[Detection]``: letterbox to the model input,
normalise, run the session once, decode the ``(1, 84, anchors)`` head, apply
per-class confidence thresholds, class-aware NMS, then map boxes back to original
pixel coordinates. Model-agnostic: the class-name list comes from the model's own
``names`` metadata when present, else the COCO default; every other knob is
``config.perception.detector`` (see ``docs/decisions.md``).
"""

from __future__ import annotations

import ast

import numpy as np

from predictivesense.config.settings import DetectorConfig
from predictivesense.core.types import Detection, Frame
from predictivesense.logging_setup import get_logger
from predictivesense.perception.classes import COCO_CLASSES
from predictivesense.perception.postprocess import class_aware_nms, xywh_to_xyxy
from predictivesense.perception.preprocess import letterbox, scale_boxes_to_original
from predictivesense.perception.runtime import SessionHandle, create_session

__all__ = ["ObjectDetector"]

_LOG = get_logger(__name__)
# Hard guard on how many over-threshold anchors enter NMS on a pathological frame.
_NMS_CANDIDATE_CAP = 3000


class ObjectDetector:
    """A reusable ONNX detector. One session, created at construction, reused."""

    def __init__(
        self,
        config: DetectorConfig,
        *,
        provider: str,
        warmup: bool = True,
        intra_op_threads: int = 0,
    ) -> None:
        self._config = config
        self._input_size = int(config.input_size)
        self._handle: SessionHandle = create_session(
            config.model_path,
            provider=provider,
            input_size=self._input_size,
            warmup=warmup,
            intra_op_threads=intra_op_threads,
        )
        self._class_names = _read_class_names(self._handle) or COCO_CLASSES
        self._conf_lut = np.array(
            [
                float(config.class_thresholds.get(name, config.default_conf))
                for name in self._class_names
            ],
            dtype=np.float64,
        )
        self._min_conf = float(self._conf_lut.min())
        _LOG.info(
            "detector ready: %d classes, input %d, default_conf %.2f, %d per-class overrides",
            len(self._class_names),
            self._input_size,
            config.default_conf,
            len(config.class_thresholds),
        )

    # -- introspection (for Diagnostics / the report) -----------------

    @property
    def warmup_ms(self) -> float:
        return self._handle.warmup_ms

    @property
    def provider(self) -> str:
        return self._handle.provider

    @property
    def model_name(self) -> str:
        return self._handle.model_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]

    @property
    def input_size(self) -> int:
        return self._handle.fixed_input_size or self._input_size

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(self._class_names)

    # -- inference ---------------------------------------------------

    def infer(self, frame: Frame) -> list[Detection]:
        """Detections for one frame, boxes in original-image pixels."""

        image = frame.image
        orig_h, orig_w = int(image.shape[0]), int(image.shape[1])
        lb = letterbox(image, self.input_size)

        out = self._handle.session.run(None, {self._handle.input_name: lb.blob})[0]
        # (1, 4 + num_classes, anchors) -> (anchors, 4 + num_classes)
        preds = np.asarray(out[0], dtype=np.float64).T
        num_classes = preds.shape[1] - 4
        boxes_xywh = preds[:, :4]
        cls_scores = preds[:, 4 : 4 + num_classes]

        class_ids = cls_scores.argmax(axis=1)
        confidences = cls_scores.max(axis=1)

        lut = self._conf_lut if num_classes == self._conf_lut.shape[0] else None
        thr = lut[class_ids] if lut is not None else self._config.default_conf
        keep_mask = confidences >= thr
        if not np.any(keep_mask):
            return []

        boxes_xywh = boxes_xywh[keep_mask]
        class_ids = class_ids[keep_mask]
        confidences = confidences[keep_mask]

        if confidences.shape[0] > _NMS_CANDIDATE_CAP:
            top = np.argpartition(confidences, -_NMS_CANDIDATE_CAP)[-_NMS_CANDIDATE_CAP:]
            boxes_xywh, class_ids, confidences = (
                boxes_xywh[top],
                class_ids[top],
                confidences[top],
            )

        boxes_xyxy = xywh_to_xyxy(boxes_xywh)
        kept = class_aware_nms(
            boxes_xyxy,
            confidences,
            class_ids,
            iou_threshold=self._config.nms_iou,
            max_detections=self._config.max_detections,
        )
        if not kept:
            return []

        mapped = scale_boxes_to_original(boxes_xyxy[kept], lb, orig_w, orig_h)
        kept_class_ids = class_ids[kept]
        kept_scores = confidences[kept]

        detections: list[Detection] = []
        for row, cid, score in zip(mapped, kept_class_ids, kept_scores):
            x1, y1, x2, y2 = (float(v) for v in row)
            if x2 <= x1 or y2 <= y1:
                continue
            cid_int = int(cid)
            name = (
                self._class_names[cid_int]
                if 0 <= cid_int < len(self._class_names)
                else str(cid_int)
            )
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    class_id=cid_int,
                    class_name=name,
                    score=float(score),
                    frame_id=frame.frame_id,
                )
            )
        return detections


def _read_class_names(handle: SessionHandle) -> tuple[str, ...] | None:
    """Class names from the model's ``names`` metadata, if it carries them."""

    try:
        meta = handle.session.get_modelmeta().custom_metadata_map
    except Exception:  # noqa: BLE001 - metadata is optional
        return None
    raw = meta.get("names")
    if not raw:
        return None
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None
    if isinstance(parsed, dict):
        return tuple(parsed[i] for i in range(len(parsed)))
    if isinstance(parsed, (list, tuple)):
        return tuple(str(x) for x in parsed)
    return None
