"""ONNX pose estimator wrapper (YOLO-family pose head).

``PoseEstimator.infer(frame) -> list[Pose]``: letterbox, run once, decode the
``(1, 4 + 1 + K*3, anchors)`` head into a person box, a score and ``K`` keypoints
of ``(x, y, visibility)``, suppress overlapping people, keep the top
``max_persons``, and map every coordinate back to original pixels. Keypoint names
and the skeleton live in ``classes.py`` and are shared with the overlay.
"""

from __future__ import annotations

import numpy as np

from predictivesense.config.settings import PoseConfig
from predictivesense.core.types import Frame, Pose
from predictivesense.logging_setup import get_logger
from predictivesense.perception.classes import KEYPOINT_NAMES
from predictivesense.perception.postprocess import nms, xywh_to_xyxy
from predictivesense.perception.preprocess import (
    letterbox,
    scale_boxes_to_original,
    scale_points_to_original,
)
from predictivesense.perception.runtime import SessionHandle, create_session

__all__ = ["PoseEstimator"]

_LOG = get_logger(__name__)
_POSE_NMS_IOU = 0.45
_NUM_KEYPOINTS = len(KEYPOINT_NAMES)


class PoseEstimator:
    """A reusable ONNX pose estimator. One session, created at construction."""

    def __init__(
        self,
        config: PoseConfig,
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
        _LOG.info(
            "pose ready: input %d, conf %.2f, max_persons %d",
            self.input_size,
            config.conf,
            config.max_persons,
        )

    @property
    def warmup_ms(self) -> float:
        return self._handle.warmup_ms

    @property
    def provider(self) -> str:
        return self._handle.provider

    @property
    def ep_name(self) -> str:
        """The ONNX Runtime EP name actually backing the session."""

        return self._handle.ep_name

    @property
    def model_name(self) -> str:
        return self._handle.model_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]

    @property
    def input_size(self) -> int:
        return self._handle.fixed_input_size or self._input_size

    def infer(self, frame: Frame) -> list[Pose]:
        image = frame.image
        orig_h, orig_w = int(image.shape[0]), int(image.shape[1])
        lb = letterbox(image, self.input_size)

        out = self._handle.session.run(None, {self._handle.input_name: lb.blob})[0]
        preds = np.asarray(out[0], dtype=np.float64).T  # (anchors, 56)
        if preds.shape[1] < 5 + _NUM_KEYPOINTS * 3:
            return []

        boxes_xywh = preds[:, :4]
        scores = preds[:, 4]
        kpts = preds[:, 5 : 5 + _NUM_KEYPOINTS * 3].reshape(-1, _NUM_KEYPOINTS, 3)

        mask = scores >= self._config.conf
        if not np.any(mask):
            return []
        boxes_xywh, scores, kpts = boxes_xywh[mask], scores[mask], kpts[mask]

        boxes_xyxy = xywh_to_xyxy(boxes_xywh)
        kept = nms(boxes_xyxy, scores, _POSE_NMS_IOU)
        kept = kept[: self._config.max_persons]
        if not kept:
            return []

        mapped_boxes = scale_boxes_to_original(boxes_xyxy[kept], lb, orig_w, orig_h)
        mapped_kpts_xy = scale_points_to_original(
            kpts[kept][:, :, :2], lb, orig_w, orig_h
        )
        kpt_vis = kpts[kept][:, :, 2]

        poses: list[Pose] = []
        for box, xy, vis, score in zip(
            mapped_boxes, mapped_kpts_xy, kpt_vis, scores[kept]
        ):
            x1, y1, x2, y2 = (float(v) for v in box)
            keypoints = tuple(
                (float(px), float(py), float(pv))
                for (px, py), pv in zip(xy, vis)
            )
            poses.append(
                Pose(
                    keypoints=keypoints,
                    bbox=(x1, y1, x2, y2),
                    score=float(score),
                    frame_id=frame.frame_id,
                )
            )
        return poses
