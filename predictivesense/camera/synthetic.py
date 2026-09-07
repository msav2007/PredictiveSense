"""A deterministic synthetic frame source.

Given a seed, the pixel content of frame *n* is fully determined. ``capture_ts``
is a real ``time.monotonic()`` reading, nudged forward by one nanosecond if the
clock did not advance between reads so the sequence is *strictly* increasing.

Only the producer thread calls :meth:`SyntheticSource.read`; ``start``/``stop``
and ``info`` may be called from other threads and take a short lock.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from predictivesense.core.types import Frame
from predictivesense.logging_setup import get_logger

__all__ = ["SyntheticSource"]

_LOG = get_logger(__name__)
_TS_EPSILON = 1e-9


class SyntheticSource:
    """Implements the :class:`~predictivesense.camera.source.FrameSource` protocol."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        target_fps: float,
        seed: int,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        if target_fps <= 0:
            raise ValueError("target_fps must be positive")
        if seed < 0:
            raise ValueError("seed must be non-negative")

        self._width = int(width)
        self._height = int(height)
        self._target_fps = float(target_fps)
        self._seed = int(seed)
        self._period = 1.0 / self._target_fps
        self._source_id = f"synthetic-{self._seed}-{self._width}x{self._height}"

        # Deterministic base image; each frame is a deterministic transform of it.
        rng = np.random.default_rng(self._seed)
        self._base = rng.integers(
            0, 256, size=(self._height, self._width, 3), dtype=np.uint8
        )

        self._lock = threading.Lock()
        self._running = False
        self._frame_id = -1          # monotonic per source, never reset
        self._seq = -1               # per session, reset on start()
        self._session_index = -1
        self._last_ts = float("-inf")
        self._next_due = 0.0

    # -- lifecycle -------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._session_index += 1
            self._seq = -1
            self._next_due = time.monotonic()
        _LOG.info(
            "synthetic source started source_id=%s session=%d %dx%d @ %.2f fps seed=%d",
            self._source_id,
            max(self._session_index, 0),
            self._width,
            self._height,
            self._target_fps,
            self._seed,
        )

    def stop(self) -> None:
        with self._lock:
            was_running = self._running
            self._running = False
        if was_running:
            _LOG.info(
                "synthetic source stopped source_id=%s frames_produced=%d",
                self._source_id,
                self._frame_id + 1,
            )

    # -- production ----------------------------------------------------

    def read(self) -> Frame | None:
        if not self._running:
            return None

        self._pace()

        with self._lock:
            if not self._running:
                return None
            self._frame_id += 1
            self._seq += 1
            frame_id = self._frame_id
            seq = self._seq
            capture_ts = self._next_strict_ts()

        image = self._render(frame_id)
        return Frame(
            frame_id=frame_id,
            capture_ts=capture_ts,
            image=image,
            width=self._width,
            height=self._height,
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
                "kind": "synthetic",
                "width": self._width,
                "height": self._height,
                "target_fps": self._target_fps,
                "seed": self._seed,
                "running": self._running,
                "session_index": max(self._session_index, 0),
                "frames_produced": self._frame_id + 1,
            }

    # -- context manager --------------------------------------------

    def __enter__(self) -> "SyntheticSource":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    # -- internals ------------------------------------------------

    def _render(self, frame_id: int) -> np.ndarray:
        """Deterministic per-frame image: the base rolled by ``frame_id`` columns."""

        shift = frame_id % self._width
        return np.roll(self._base, shift=shift, axis=1)

    def _pace(self) -> None:
        """Sleep until the next frame is due. Called only by the producer thread."""

        now = time.monotonic()
        wait = self._next_due - now
        if wait > 0:
            time.sleep(wait)
        self._next_due = max(self._next_due + self._period, time.monotonic())

    def _next_strict_ts(self) -> float:
        """Strictly increasing monotonic timestamp. Called under ``self._lock``."""

        now = time.monotonic()
        if now <= self._last_ts:
            now = self._last_ts + _TS_EPSILON
        self._last_ts = now
        return now
