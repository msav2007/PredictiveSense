"""Shared OpenCV helpers for the camera package.

``cv2`` is imported here (allowed only under ``predictivesense/camera/``).
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import cv2
import numpy as np

__all__ = [
    "quiet_opencv_logging",
    "fourcc_to_str",
    "read_image_bgr",
    "decode_image_bgr",
    "encode_jpeg",
    "thumbnail_jpeg",
]

_LOG_LEVEL_ERROR = 2  # cv2 log levels: 0 SILENT, 1 FATAL, 2 ERROR, 3 WARNING, ...
_quieted = False
_lock = threading.Lock()


def quiet_opencv_logging() -> None:
    """Drop OpenCV's (and its FFmpeg wrapper's) log verbosity, once.

    The MSMF/DSHOW back ends emit a ``[ WARN ] ... can't be used to capture by
    index`` line for every probe of a non-existent camera index (enumeration
    sweeps 0..9); the FFmpeg reader prints ``EBML header parsing failed`` when
    the clip-duration probe is handed a partial/garbage file. Both are expected
    and noisy. Genuine errors are still shown.
    """

    global _quieted
    with _lock:
        if _quieted:
            return
        # AV_LOG level -8 = quiet; read lazily by OpenCV's FFmpeg wrapper.
        os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
        try:
            cv2.setLogLevel(_LOG_LEVEL_ERROR)
        except (AttributeError, cv2.error):  # pragma: no cover - build-dependent
            pass
        _quieted = True


def read_image_bgr(path: str | Path) -> np.ndarray | None:
    """Decode an image file to a contiguous ``HxWx3`` BGR ``uint8`` array, or
    ``None`` if it cannot be read. Used by the Phase 2.5 labelling seed path so
    ``cv2`` stays out of the API layer."""

    img = cv2.imread(str(path))
    if img is None or img.ndim != 3 or img.shape[2] != 3:
        return None
    return np.ascontiguousarray(img)


def decode_image_bgr(data: bytes) -> np.ndarray | None:
    """Decode encoded image bytes (JPEG/PNG/...) to contiguous ``HxWx3`` BGR
    ``uint8``, or ``None`` if they are not a readable 3-channel image. Used by the
    Phase 4 Object Learning Studio upload / capture path so ``cv2`` stays out of
    ``predictivesense/api/``."""

    try:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except cv2.error:
        return None
    if img is None or img.ndim != 3 or img.shape[2] != 3:
        return None
    return np.ascontiguousarray(img)


def encode_jpeg(image: np.ndarray, quality: int = 92) -> bytes:
    """Encode an ``HxWx3`` BGR array to JPEG bytes. Raises ``ValueError`` if the
    encode fails."""

    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("cv2.imencode('.jpg') failed")
    return buf.tobytes()


def thumbnail_jpeg(image: np.ndarray, max_px: int, quality: int = 82) -> bytes:
    """Downscale ``image`` so its longer side is at most ``max_px`` and return
    JPEG bytes. Never upscales."""

    h, w = int(image.shape[0]), int(image.shape[1])
    longest = max(h, w)
    if longest > max_px > 0:
        scale = max_px / float(longest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return encode_jpeg(image, quality=quality)


def fourcc_to_str(value: float | int) -> str:
    """Decode a packed ``CAP_PROP_FOURCC`` value into its 4-char code.

    MSMF reports an internal numeric format id rather than a real FOURCC; that
    is surfaced as ``"raw:<n>"`` so logs and benchmark output stay readable.
    """

    code = int(value)
    if code <= 0:
        return ""
    chars = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)).rstrip("\x00")
    if chars and all(32 <= ord(c) < 127 for c in chars):
        return chars
    return f"raw:{code}"
