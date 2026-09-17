"""ONNX Runtime inference for a trained custom crop classifier (Phase 11 Part
B section 10.3).

Loads exactly like the detector and pose estimator (``perception/runtime.py``
- same provider resolution, same reused-session-for-life-of-process pattern),
never through torch: serving a trained model never requires the ``[train]``
extra. Runs on a crop of an image (a box the class-agnostic detector already
proposed), never a full frame, never re-detecting.
"""

from __future__ import annotations

import cv2
import numpy as np

from predictivesense.perception.runtime import SessionHandle, create_session

__all__ = ["CropClassifierModel"]


class CropClassifierModel:
    """A reusable ONNX crop classifier. One session, created at construction."""

    def __init__(
        self,
        model_path: str,
        *,
        class_map: dict[str, int],
        image_size: int,
        provider: str,
        warmup: bool = True,
        intra_op_threads: int = 0,
    ) -> None:
        self._handle: SessionHandle = create_session(
            model_path, provider=provider, input_size=image_size,
            warmup=warmup, intra_op_threads=intra_op_threads,
        )
        self._image_size = int(image_size)
        self._index_to_class = {i: c for c, i in class_map.items()}
        self._output_name = self._handle.session.get_outputs()[0].name

    @property
    def provider(self) -> str:
        return self._handle.provider

    @property
    def ep_name(self) -> str:
        return self._handle.ep_name

    @property
    def classes(self) -> tuple[str, ...]:
        return tuple(self._index_to_class[i] for i in sorted(self._index_to_class))

    def infer_crop(self, image_bgr: np.ndarray) -> tuple[str, float]:
        """Classify one already-cropped BGR image. Returns
        ``(class_name, softmax_confidence)`` - **real model inference only**,
        no class map substitution, no filename or image-specific rule (section
        16). ``class_name`` is always a member of the trained ``class_map``."""

        resized = cv2.resize(image_bgr, (self._image_size, self._image_size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        chw = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)
        batch = chw[np.newaxis, ...]
        (logits,) = self._handle.session.run([self._output_name], {self._handle.input_name: batch})
        logits = logits[0]
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        idx = int(np.argmax(probs))
        return self._index_to_class[idx], float(probs[idx])
