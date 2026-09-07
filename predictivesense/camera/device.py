"""``DeviceSource`` - a backend-owned OpenCV camera (Mode A, headless).

Used only when ``capture.owner == "backend"`` and by the transport benchmark.
Opens ``cv2.VideoCapture`` preferring MSMF, falling back to DSHOW if the first
frame does not arrive within ``open_timeout_s``; the backend that won is logged.
Requested width/height/fps come from config; :meth:`info` reports what the
device *actually* returned - the requested values are never echoed back as
achievements.

Read failures schedule a **non-blocking** bounded exponential-backoff reopen:
:meth:`read` never sleeps while holding the lock and never blocks longer than
~0.25 s, so :meth:`stop` (device switching, shutdown) stays responsive and the
producer thread does not spin. ``stop()`` sets an event that interrupts any wait
and the open-probe loop.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from predictivesense.camera._opencv import fourcc_to_str, quiet_opencv_logging
from predictivesense.camera.enumerate import load_backend_hint
from predictivesense.core.enums import SourceKind
from predictivesense.core.types import Frame, SourceInfo
from predictivesense.logging_setup import get_logger

__all__ = ["DeviceSource"]

_LOG = get_logger(__name__)

_BACKENDS = {
    "msmf": cv2.CAP_MSMF,
    "dshow": cv2.CAP_DSHOW,
}
# Short first-frame probe while recovering: a camera that is genuinely back
# delivers a frame in well under this; otherwise the next backoff tick retries.
# Keeps any single read() well bounded even in the down state.
_RECONNECT_PROBE_S = 0.4
_MAX_READ_WAIT_S = 0.25    # cap on any single wait inside read()


class DeviceSource:
    """Implements the :class:`~predictivesense.camera.source.FrameSource` protocol."""

    def __init__(
        self,
        index: int,
        *,
        backend: str = "auto",
        request_width: int = 1280,
        request_height: int = 720,
        request_fps: float = 30.0,
        fourcc: str = "auto",
        buffer_size: int = 1,
        warmup_frames: int = 0,
        open_timeout_s: float = 5.0,
        reconnect_initial_s: float = 0.5,
        reconnect_max_s: float = 8.0,
        backend_cache_dir: Path | str | None = None,
    ) -> None:
        if backend not in ("auto", "msmf", "dshow"):
            raise ValueError(f"unknown device backend {backend!r}")
        self._index = int(index)
        self._backend_pref = backend
        self._backend_cache_dir = backend_cache_dir
        self._req_w = int(request_width)
        self._req_h = int(request_height)
        self._req_fps = float(request_fps)
        self._fourcc = fourcc
        self._buffer_size = int(buffer_size)
        self._warmup_frames = int(warmup_frames)
        self._open_timeout_s = float(open_timeout_s)
        self._reconnect_initial_s = float(reconnect_initial_s)
        self._reconnect_max_s = float(reconnect_max_s)
        self._source_id = f"device:{self._index}"

        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._cap: cv2.VideoCapture | None = None
        self._running = False
        self._backend_name: str | None = None
        self._fourcc_used: str | None = None
        self._achieved_w = 0
        self._achieved_h = 0
        self._achieved_fps: float | None = None

        self._frame_id = -1          # monotonic per source
        self._seq = -1               # per session, reset on reconnect
        self._read_failures = 0
        self._reconnects = 0             # successful recoveries after a failure
        self._reconnect_attempts = 0     # reopen tries (success or fail)
        self._reconnect_seconds_total = 0.0
        self._down_since: float | None = None   # monotonic when cap went None
        self._retry_at = 0.0
        self._backoff_s = 0.0
        self._time_to_first_frame_s: float | None = None
        self._start_mono = 0.0

    # -- lifecycle -------------------------------------------------------

    def start(self) -> None:
        quiet_opencv_logging()
        with self._lock:
            if self._running:
                return
            self._stop_evt.clear()
            self._start_mono = time.monotonic()
            self._backoff_s = self._reconnect_initial_s
            self._open_locked(self._open_timeout_s)
            self._running = True

    def stop(self) -> None:
        self._stop_evt.set()  # interrupt any wait / open-probe in read()
        with self._lock:
            self._running = False
            self._release_locked()

    # -- production ----------------------------------------------------

    def read(self) -> Frame | None:
        if not self._running or self._stop_evt.is_set():
            return None

        wait_s = 0.0
        with self._lock:
            if self._cap is not None:
                ok, image = self._cap.read()
                if ok and image is not None:
                    return self._frame_from_locked(image)
                self._read_failures += 1
                self._release_locked()
                self._down_since = time.monotonic()
                self._retry_at = 0.0  # first retry is immediate

            now = time.monotonic()
            if now >= self._retry_at:
                if self._reopen_locked():
                    return None  # next read() gets the first frame
                self._retry_at = time.monotonic() + self._backoff_s
                self._backoff_s = min(self._backoff_s * 2.0, self._reconnect_max_s)
            wait_s = min(_MAX_READ_WAIT_S, max(0.0, self._retry_at - time.monotonic()))

        # Absorb the backoff outside the lock so stop() is never blocked and the
        # producer thread does not busy-spin on None.
        if wait_s > 0:
            self._stop_evt.wait(wait_s)
        return None

    @property
    def is_running(self) -> bool:
        return self._running

    def info(self) -> dict[str, Any]:
        with self._lock:
            down_for = (
                round(time.monotonic() - self._down_since, 3)
                if self._down_since is not None
                else 0.0
            )
            return {
                "source_id": self._source_id,
                "kind": SourceKind.DEVICE.value,
                "index": self._index,
                "backend": self._backend_name,
                "fourcc": self._fourcc_used,
                "requested_width": self._req_w,
                "requested_height": self._req_h,
                "requested_fps": self._req_fps,
                "achieved_width": self._achieved_w,
                "achieved_height": self._achieved_h,
                "achieved_fps": self._achieved_fps,
                "running": self._running,
                "connected": self._cap is not None,
                "frames_read": self._frame_id + 1,
                "read_failures": self._read_failures,
                "reconnects": self._reconnects,
                "reconnect_attempts": self._reconnect_attempts,
                "reconnect_seconds_total": round(self._reconnect_seconds_total, 3),
                "currently_down_s": down_for,
                "time_to_first_frame_s": self._time_to_first_frame_s,
            }

    def source_info(self) -> SourceInfo:
        with self._lock:
            return SourceInfo(
                kind=SourceKind.DEVICE,
                source_id=self._source_id,
                label=f"Camera {self._index}",
                width=self._achieved_w,
                height=self._achieved_h,
                achieved_fps=self._achieved_fps,
                backend=self._backend_name,
                extra={
                    "index": str(self._index),
                    "fourcc": self._fourcc_used or "auto",
                    "requested_fps": str(self._req_fps),
                    "reconnects": str(self._reconnects),
                },
            )

    # -- context manager --------------------------------------------

    def __enter__(self) -> "DeviceSource":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    # -- internals ------------------------------------------------

    def _frame_from_locked(self, image: np.ndarray) -> Frame:
        """Build a Frame from a good read. Caller holds the lock. Resets backoff."""

        if self._time_to_first_frame_s is None:
            self._time_to_first_frame_s = time.monotonic() - self._start_mono
        if self._down_since is not None:
            self._reconnect_seconds_total += time.monotonic() - self._down_since
            self._down_since = None
        self._backoff_s = self._reconnect_initial_s
        self._frame_id += 1
        self._seq += 1
        return Frame(
            frame_id=self._frame_id,
            capture_ts=time.monotonic(),
            image=np.ascontiguousarray(image),
            width=int(image.shape[1]),
            height=int(image.shape[0]),
            source_id=self._source_id,
            seq=self._seq,
        )

    def _reopen_locked(self) -> bool:
        """One reopen attempt. Caller holds the lock. Returns True on success."""

        self._reconnect_attempts += 1
        try:
            self._open_locked(_RECONNECT_PROBE_S)
        except RuntimeError as exc:
            _LOG.warning(
                "device %d reopen attempt %d failed: %s",
                self._index,
                self._reconnect_attempts,
                exc,
            )
            return False
        self._reconnects += 1
        self._seq = -1  # sequence resets on reconnect
        if self._down_since is not None:
            self._reconnect_seconds_total += time.monotonic() - self._down_since
            self._down_since = None
        _LOG.info(
            "device %d reconnected on attempt %d (backend=%s)",
            self._index,
            self._reconnect_attempts,
            self._backend_name,
        )
        return True

    def _candidate_backends(self) -> list[tuple[str, int]]:
        if self._backend_pref != "auto":
            return [(self._backend_pref, _BACKENDS[self._backend_pref])]
        order = ["msmf", "dshow"]
        if self._backend_cache_dir is not None:
            hint = load_backend_hint(self._index, self._backend_cache_dir)
            if hint in order:
                order.remove(hint)
                order.insert(0, hint)
                _LOG.info(
                    "device %d: backend cache prefers %s (from %s)",
                    self._index,
                    hint,
                    self._backend_cache_dir,
                )
        return [(name, _BACKENDS[name]) for name in order]

    def _open_locked(self, probe_timeout_s: float) -> None:
        """Open the device, trying each candidate backend until one yields a frame."""

        last_err = "no backend attempted"
        for name, flag in self._candidate_backends():
            if self._stop_evt.is_set():
                raise RuntimeError("stop requested during open")
            cap = cv2.VideoCapture(self._index, flag)
            if self._fourcc != "auto":
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self._fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._req_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._req_h)
            cap.set(cv2.CAP_PROP_FPS, self._req_fps)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, self._buffer_size)
            except cv2.error:  # pragma: no cover - not all backends accept it
                pass

            if not cap.isOpened():
                cap.release()
                last_err = f"{name}: VideoCapture did not open"
                continue

            deadline = time.monotonic() + probe_timeout_s
            image = None
            while time.monotonic() < deadline and not self._stop_evt.is_set():
                ok, image = cap.read()
                if ok and image is not None:
                    break
                image = None
                time.sleep(0.03)

            if image is None:
                cap.release()
                last_err = f"{name}: no frame within {probe_timeout_s:.1f}s"
                continue

            for _ in range(self._warmup_frames):
                if self._stop_evt.is_set():
                    break
                cap.read()

            self._cap = cap
            self._backend_name = name
            self._fourcc_used = fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC)) or None
            self._achieved_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or int(image.shape[1])
            self._achieved_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or int(image.shape[0])
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            self._achieved_fps = fps if fps > 0 else None
            _LOG.info(
                "device %d opened via %s: %dx%d @ %s fps fourcc=%s (requested %dx%d @ %.1f)",
                self._index,
                name,
                self._achieved_w,
                self._achieved_h,
                f"{self._achieved_fps:.2f}" if self._achieved_fps else "unknown",
                self._fourcc_used or "auto",
                self._req_w,
                self._req_h,
                self._req_fps,
            )
            return

        raise RuntimeError(
            f"could not open camera index {self._index} "
            f"(backend={self._backend_pref}): {last_err}"
        )

    def _release_locked(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except cv2.error as exc:  # pragma: no cover - release rarely fails
                _LOG.warning("device %d release raised: %r", self._index, exc)
            self._cap = None
