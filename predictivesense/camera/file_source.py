"""``FileSource`` - sequential decode of a recorded video file (Mode B).

Decodes a file frame by frame with OpenCV. ``capture_ts`` derives from the
file's presentation timestamp (``CAP_PROP_POS_MSEC``), falling back to
``frame_index / fps`` when the container reports no usable timestamps. It is
never wall-clock time, so replaying the same file yields the same ``capture_ts``
sequence every time.

Two replay modes:

* ``asfast`` - no pacing; decode as fast as the CPU allows.
* ``realtime`` - sleep so wall time tracks the file's own timestamps.

Both yield an identical *frame* sequence (same frames, same ``capture_ts``,
same order); only wall timing differs. :meth:`restart` seeks back to frame 0
deterministically.

:class:`FileSource` implements the pull ``FrameSource`` protocol so it can drive
the live loop, but Mode B analysis runs through
:class:`~predictivesense.pipeline.recorded.RecordedDriver`, which does not use
the mailbox and drops nothing.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from predictivesense.core.enums import SourceKind
from predictivesense.core.types import Frame, SourceInfo
from predictivesense.logging_setup import get_logger

__all__ = ["FileSource", "video_duration_s"]

_LOG = get_logger(__name__)
_TS_DECIMALS = 6  # round pts seconds to avoid float-repr drift in JSONL output


def video_duration_s(path: str | Path) -> float | None:
    """Best-effort clip duration in seconds via OpenCV; ``None`` if unreadable."""

    from predictivesense.camera._opencv import quiet_opencv_logging

    quiet_opencv_logging()
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps > 0 and count > 0:
            return round(count / fps, 3)
    except cv2.error as exc:  # pragma: no cover - probe is optional metadata
        _LOG.debug("duration probe failed for %s: %r", path, exc)
    finally:
        cap.release()
    return None


class FileSource:
    """Implements the :class:`~predictivesense.camera.source.FrameSource` protocol."""

    def __init__(self, path: str | Path, *, replay_mode: str = "asfast") -> None:
        if replay_mode not in ("asfast", "realtime"):
            raise ValueError(f"replay_mode must be 'asfast' or 'realtime', got {replay_mode!r}")
        self._path = Path(path)
        if not self._path.is_file():
            raise FileNotFoundError(f"video file not found: {self._path}")
        self._replay_mode = replay_mode
        self._source_id = f"file:{self._path.name}"

        self._lock = threading.Lock()
        self._cap: cv2.VideoCapture | None = None
        self._running = False

        self._frame_index = -1
        self._seq = -1
        self._last_pts_s = 0.0
        self._pts_fallbacks = 0
        self._wall_start: float | None = None

        # Static metadata, filled on first open.
        self._fps = 0.0
        self._width = 0
        self._height = 0
        self._frame_count_hint = 0

    # -- lifecycle -------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._open_locked()
            self._running = True
            self._wall_start = time.monotonic()

    def stop(self) -> None:
        with self._lock:
            self._running = False
            self._release_locked()

    def restart(self) -> None:
        """Seek back to frame 0 and resume. The next :meth:`read` returns frame 0.

        Explicit re-run: reopens even if a prior :meth:`read` hit EOF and cleared
        ``running``. Call :meth:`stop` to end the source instead.
        """

        with self._lock:
            self._release_locked()
            self._frame_index = -1
            self._seq = -1
            self._last_pts_s = 0.0
            self._pts_fallbacks = 0
            self._open_locked()
            self._running = True
            self._wall_start = time.monotonic()

    # -- production ----------------------------------------------------

    def read(self) -> Frame | None:
        with self._lock:
            if not self._running or self._cap is None:
                return None
            pos_msec = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            ok, image = self._cap.read()
            if not ok or image is None:
                self._running = False
                self._release_locked()
                return None

            self._frame_index += 1
            self._seq += 1
            capture_ts = self._pts_for_locked(self._frame_index, pos_msec)
            frame_index, seq = self._frame_index, self._seq
            replay_mode = self._replay_mode
            wall_start = self._wall_start

        if replay_mode == "realtime" and wall_start is not None:
            target = wall_start + capture_ts
            wait = target - time.monotonic()
            if wait > 0:
                time.sleep(wait)

        height, width = int(image.shape[0]), int(image.shape[1])
        return Frame(
            frame_id=frame_index,
            capture_ts=capture_ts,
            image=np.ascontiguousarray(image),
            width=width,
            height=height,
            source_id=self._source_id,
            seq=seq,
        )

    @property
    def is_running(self) -> bool:
        return self._running

    def info(self) -> dict[str, Any]:
        with self._lock:
            return {
                "source_id": self._source_id,
                "kind": SourceKind.FILE.value,
                "path": str(self._path),
                "replay_mode": self._replay_mode,
                "fps": self._fps,
                "width": self._width,
                "height": self._height,
                "frame_count_hint": self._frame_count_hint,
                "frames_read": self._frame_index + 1,
                "pts_fallbacks": self._pts_fallbacks,
                "running": self._running,
            }

    def source_info(self) -> SourceInfo:
        with self._lock:
            return SourceInfo(
                kind=SourceKind.FILE,
                source_id=self._source_id,
                label=self._path.name,
                width=self._width,
                height=self._height,
                achieved_fps=self._fps or None,
                backend="opencv",
                extra={
                    "path": str(self._path),
                    "replay_mode": self._replay_mode,
                    "frame_count_hint": str(self._frame_count_hint),
                },
            )

    # -- context manager --------------------------------------------

    def __enter__(self) -> "FileSource":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    # -- internals ------------------------------------------------

    def _open_locked(self) -> None:
        from predictivesense.camera._opencv import quiet_opencv_logging

        quiet_opencv_logging()
        cap = cv2.VideoCapture(str(self._path))
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"OpenCV could not open video file {self._path}")
        self._cap = cap
        self._fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_count_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def _release_locked(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except cv2.error as exc:  # pragma: no cover
                _LOG.warning("file source release raised: %r", exc)
            self._cap = None

    def _pts_for_locked(self, frame_index: int, pos_msec: float) -> float:
        """Deterministic presentation timestamp in seconds. Caller holds the lock."""

        pts_s = pos_msec / 1000.0 if pos_msec and pos_msec > 0 else None
        if pts_s is None or (frame_index > 0 and pts_s <= self._last_pts_s):
            # Container gave no usable / non-increasing timestamp: fall back to
            # the nominal frame cadence, which is still file-derived and stable.
            if self._fps > 0:
                pts_s = frame_index / self._fps
                self._pts_fallbacks += 1
            else:
                pts_s = self._last_pts_s
        pts_s = round(pts_s, _TS_DECIMALS)
        self._last_pts_s = pts_s
        return pts_s
