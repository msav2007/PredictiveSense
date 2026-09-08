"""Letterbox resize, normalisation, and coordinate mapping back to source pixels.

Pure NumPy, no ONNX Runtime, no model - so this is unit-tested without weights.
The letterbox is aspect-preserving: the source image is scaled by a single ratio
to fit the square model input and centred on a constant-colour pad. Undoing it
is exact up to the resize interpolation: subtract the pad, divide by the ratio.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

__all__ = [
    "LetterboxResult",
    "letterbox",
    "scale_boxes_to_original",
    "scale_points_to_original",
    "PAD_VALUE",
]

# Grey pad, matching the Ultralytics export convention.
PAD_VALUE = 114


@dataclass(frozen=True)
class LetterboxResult:
    """``blob`` is NCHW float32 (RGB in [0, 1] by default); ``ratio`` and the
    pads undo it."""

    blob: np.ndarray
    ratio: float
    pad_x: int
    pad_y: int
    input_size: int


def letterbox(
    image: np.ndarray,
    size: int,
    *,
    to_rgb: bool = True,
    scale: float = 1.0 / 255.0,
    center: bool = True,
) -> LetterboxResult:
    """Aspect-preserving resize of a BGR ``HxWx3`` uint8 image to ``size x size``.

    Defaults match the Ultralytics YOLO convention: BGR->RGB, ``/255`` to [0, 1],
    and the image centred on a grey pad. The Phase 2.5 YOLOX variant passes
    ``to_rgb=False, scale=1.0, center=False`` (raw BGR, no normalisation,
    top-left aligned) - the YOLO path is byte-identical to before.

    Returns a contiguous ``(1, 3, size, size)`` float32 blob plus the ratio and
    pad offsets needed to map model-space boxes back.
    """

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 image, got shape {image.shape!r}")
    if size <= 0:
        raise ValueError(f"letterbox size must be positive, got {size}")

    src_h, src_w = int(image.shape[0]), int(image.shape[1])
    ratio = min(size / src_h, size / src_w)
    new_w, new_h = round(src_w * ratio), round(src_h * ratio)
    if center:
        pad_x = (size - new_w) // 2
        pad_y = (size - new_h) // 2
    else:
        pad_x = 0
        pad_y = 0

    interp = cv2.INTER_AREA if ratio < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
    canvas = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    chw = canvas[:, :, ::-1] if to_rgb else canvas
    blob = chw.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
    if scale != 1.0:
        blob = blob * np.float32(scale)
    return LetterboxResult(
        blob=np.ascontiguousarray(blob),
        ratio=ratio,
        pad_x=pad_x,
        pad_y=pad_y,
        input_size=size,
    )


def scale_boxes_to_original(
    boxes_xyxy: np.ndarray,
    lb: LetterboxResult,
    orig_w: int,
    orig_h: int,
) -> np.ndarray:
    """Map ``(N, 4)`` xyxy boxes from letterbox space back to source pixels."""

    out = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4).copy()
    out[:, [0, 2]] -= lb.pad_x
    out[:, [1, 3]] -= lb.pad_y
    out /= lb.ratio
    out[:, [0, 2]] = out[:, [0, 2]].clip(0.0, orig_w)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0.0, orig_h)
    return out


def scale_points_to_original(
    points_xy: np.ndarray,
    lb: LetterboxResult,
    orig_w: int,
    orig_h: int,
) -> np.ndarray:
    """Map ``(..., 2)`` xy points from letterbox space back to source pixels."""

    out = np.asarray(points_xy, dtype=np.float64).copy()
    out[..., 0] = ((out[..., 0] - lb.pad_x) / lb.ratio).clip(0.0, orig_w)
    out[..., 1] = ((out[..., 1] - lb.pad_y) / lb.ratio).clip(0.0, orig_h)
    return out
