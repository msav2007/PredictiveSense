"""``DeviceSource`` - a backend-owned OpenCV camera (Mode A, headless).

Used only when ``capture.owner == "backend"`` and by the transport benchmark.
Opens ``cv2.VideoCapture`` preferring MSMF, falling back to DSHOW if the first
frame does not arrive within ``open_timeout_s``; the backend that won is logged.
Requested width/height/fps come from config; :meth:`info` reports what the
device *actually* returned - the requested values are never echoed back as
achievements.

Read failures trigger a bounded exponential-backoff reconnect. The capture loop
runs on the analysis loop's producer thread and hands frames to the mailbox,
which never blocks; a reconnect backoff pauses only this source, never a
consumer.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import cv2
import numpy as np

from predictivesense.core.enums import SourceKind
from predictivesense.core.types import Frame, SourceInfo
from predictivesense.logging_setup import get_logger

__all__ = ["DeviceSource"]

_LOG = get_logger(__name__)

_BACKENDS = {
    "msmf": cv2.CAP_MSMF,
    "dshow": cv2.CAP_DSHOW,
}


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
        open_timeout_s: float = 5.0,
        reconnect_initial_s: float = 0.5,
        reconnect_max_s: float = 8.0,
    ) -> None:
        if backend not in ("auto", "msmf", "dshow"):
            raise ValueError(f"unknown device backend {backend!r}")
        self._index = int(index)
        self._backend_pref = backend
        self._req_w = int(request_width)
        self._req_h = int(request_height)
        self._req_fps = float(request_fps)
        self._open_timeout_s = float(open_timeout_s)
        self._reconnect_initial_s = float(reconnect_initial_s)
        self._reconnect_max_s = float(reconnect_max_s)
        self._source_id = f"device:{self._index}"

        self._lock = threading.Lock()
        self._cap: cv2.VideoCapture | None = None
        self._running = False
        self._backend_name: str | None = None
        self._achieved_w = 0
        self._achieved_h = 0
        self._achieved_fps: float | None = None

        self._frame_id = -1          # monotonic per source
        self._seq = -1               # per session, reset on reconnect
        self._read_failures = 0
        self._reconnects = 0
        self._reconnect_seconds_total = 0.0
        self._time_to_first_frame_s: float | None = None
        self._start_mono = 0.0

    # -- lifecycle -------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._start_mono = time.monotonic()
            self._open_locked()
            self._running = True

    def stop(self) -> None:
        with self._lock:
            self._running = False
            self._release_locked()

    # -- production ----------------------------------------------------

    def read(self) -> Frame | None:
        if not self._running:
            return None
        with self._lock:
            cap = self._cap
            if cap is None:
                self._reconnect_locked()
                cap = self._cap
                if cap is None:
                    return None
            ok, image = cap.read()
            if not ok or image is None:
                self._read_failures += 1
                self._reconnect_locked()
                return None

            if self._time_to_first_frame_s is None:
                self._time_to_first_frame_s = time.monotonic() - self._start_mono
            self._frame_id += 1
            self._seq += 1
            frame_id, seq = self._frame_id, self._seq

        height, width = int(image.shape[0]), int(image.shape[1])
        return Frame(
            frame_id=frame_id,
            capture_ts=time.monotonic(),
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
                "kind": SourceKind.DEVICE.value,
                "index": self._index,
                "backend": self._backend_name,
                "requested_width": self._req_w,
                "requested_height": self._req_h,
                "requested_fps": self._req_fps,
                "achieved_width": self._achieved_w,
                "achieved_height": self._achieved_h,
                "achieved_fps": self._achieved_fps,
                "running": self._running,
                "frames_read": self._frame_id + 1,
                "read_failures": self._read_failures,
                "reconnects": self._reconnects,
                "reconnect_seconds_total": round(self._reconnect_seconds_total, 3),
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

    def _candidate_backends(self) -> list[tuple[str, int]]:
        if self._backend_pref == "auto":
            return [("msmf", _BACKENDS["msmf"]), ("dshow", _BACKENDS["dshow"])]
        return [(self._backend_pref, _BACKENDS[self._backend_pref])]

    def _open_locked(self) -> None:
        """Open the device, trying each candidate backend until one yields a frame."""

        last_err = "no backend attempted"
        for name, flag in self._candidate_backends():
            cap = cv2.VideoCapture(self._index, flag)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._req_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._req_h)
            cap.set(cv2.CAP_PROP_FPS, self._req_fps)

            if not cap.isOpened():
                cap.release()
                last_err = f"{name}: VideoCapture did not open"
                continue

            deadline = time.monotonic() + self._open_timeout_s
            got_frame = False
            while time.monotonic() < deadline:
                ok, image = cap.read()
                if ok and image is not None:
                    got_frame = True
                    break
                time.sleep(0.05)

            if not got_frame:
                cap.release()
                last_err = f"{name}: no frame within {self._open_timeout_s:.1f}s"
                continue

            self._cap = cap
            self._backend_name = name
            self._achieved_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or int(image.shape[1])
            self._achieved_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or int(image.shape[0])
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            self._achieved_fps = fps if fps > 0 else None
            _LOG.info(
                "device %d opened via %s: %dx%d @ %s fps (requested %dx%d @ %.1f)",
                self._index,
                name,
                self._achieved_w,
                self._achieved_h,
                f"{self._achieved_fps:.2f}" if self._achieved_fps else "unknown",
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

    def _reconnect_locked(self) -> None:
        """Bounded exponential-backoff reopen. Caller holds ``self._lock``."""

        self._release_locked()
        if not self._running:
            return
        self._reconnects += 1
        backoff = self._reconnect_initial_s
        attempt = 0
        t0 = time.monotonic()
        while self._running:
            attempt += 1
            time.sleep(backoff)
            try:
                self._open_locked()
                elapsed = time.monotonic() - t0
                self._reconnect_seconds_total += elapsed
                _LOG.info(
                    "device %d reconnected after %d attempt(s) / %.2fs",
                    self._index,
                    attempt,
                    elapsed,
                )
                return
            except RuntimeError as exc:
                _LOG.warning("device %d reconnect attempt %d failed: %s", self._index, attempt, exc)
                backoff = min(backoff * 2.0, self._reconnect_max_s)
        self._reconnect_seconds_total += time.monotonic() - t0
