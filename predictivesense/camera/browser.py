"""``BrowserSource`` - a :class:`FrameSource` fed by the ``WS /ws/ingest`` socket.

The browser's analysis Web Worker pushes downscaled JPEG frames over the ingest
socket. The socket handler frames them and calls :meth:`BrowserSource.submit`,
which decodes the JPEG and stores the newest :class:`Frame` in a single-slot
buffer. The analysis loop's producer thread pulls that buffer with
:meth:`read`. Both buffers (this one and the downstream mailbox) are
single-slot, newest-wins, with exact drop accounting - nothing grows.

Decoding happens in :meth:`submit`, called off the event loop by the handler, so
a slow decode never stalls the socket. A disconnected client simply stops
calling :meth:`submit`; :meth:`read` then returns ``None`` and snapshots go
stale. Ingest errors are counted here and never propagate.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

import cv2
import numpy as np

from predictivesense.camera.framing import ClockOffset, capture_ts_seconds
from predictivesense.core.enums import SourceKind
from predictivesense.core.types import Frame, FrameTrace, IngestHeader, SourceInfo
from predictivesense.logging_setup import get_logger

__all__ = ["BrowserSource"]

_LOG = get_logger(__name__)
_BYTE_WINDOW_S = 5.0
_BYTE_WINDOW_MAXLEN = 2048  # (ts, nbytes) samples; bounded, ~200 s at 10 fps


class BrowserSource:
    """Implements the :class:`~predictivesense.camera.source.FrameSource` protocol."""

    def __init__(
        self,
        *,
        analysis_width: int,
        analysis_height: int,
        read_timeout_s: float = 0.1,
    ) -> None:
        self._expected_w = int(analysis_width)
        self._expected_h = int(analysis_height)
        self._read_timeout_s = float(read_timeout_s)
        self._source_id = "browser-ingest"

        self._lock = threading.Lock()
        self._slot: Frame | None = None
        self._frame_ready = threading.Event()
        self._running = False

        self._frame_id = -1          # monotonic per source, never reset
        self._seq = -1               # per client session, reset on new handshake
        self._submitted = 0
        self._dropped = 0            # newest-wins drops in this buffer
        self._decode_failures = 0
        self._malformed = 0          # framing rejects, reported by the handler
        self._last_decode_ms = 0.0
        self._byte_samples: deque[tuple[float, int]] = deque(maxlen=_BYTE_WINDOW_MAXLEN)
        self._clock_offset: ClockOffset | None = None
        self._clients_seen = 0

    # -- ingest side (called from the WS handler / a worker thread) ------

    def begin_client_session(self, offset: ClockOffset) -> None:
        """Record a fresh client's clock offset and reset the per-session seq."""

        with self._lock:
            self._clock_offset = offset
            self._seq = -1
            self._clients_seen += 1
        _LOG.info(
            "ingest client %d: clock offset=%.6fs rtt=%.2fms",
            self._clients_seen,
            offset.offset_s,
            offset.rtt_ms,
        )

    def note_malformed(self) -> None:
        """Count a framing-rejected message. Never fatal."""

        with self._lock:
            self._malformed += 1

    def submit(
        self, header: IngestHeader, jpeg: bytes, *, recv_ts: float | None = None
    ) -> None:
        """Decode one framed JPEG and store it newest-wins. Never raises.

        ``recv_ts`` is the ``time.monotonic()`` reading taken by the ingest
        handler the instant the WebSocket message arrived (Phase 8 stage
        attribution); ``None`` from any other caller.
        """

        t0 = time.perf_counter()
        try:
            arr = np.frombuffer(jpeg, dtype=np.uint8)
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except cv2.error as exc:
            with self._lock:
                self._decode_failures += 1
            _LOG.debug("ingest jpeg decode raised: %r", exc)
            return
        decode_ms = (time.perf_counter() - t0) * 1000.0

        if image is None or image.ndim != 3 or image.shape[2] != 3:
            with self._lock:
                self._decode_failures += 1
            return

        height, width = int(image.shape[0]), int(image.shape[1])
        now = time.monotonic()
        with self._lock:
            offset_s = self._clock_offset.offset_s if self._clock_offset else 0.0
            capture_ts = capture_ts_seconds(header.client_ts_ms, offset_s)
            self._frame_id += 1
            self._seq += 1
            trace = FrameTrace()
            trace.send_client_ms = header.client_ts_ms
            trace.send_ts = capture_ts
            trace.capture_client_ms = (
                header.cap_ts_ms if header.cap_ts_ms is not None else header.client_ts_ms
            )
            trace.capture_ts = capture_ts_seconds(trace.capture_client_ms, offset_s)
            trace.worker_encode_ms = header.enc_ms
            trace.recv_ts = recv_ts
            trace.decode_ms = decode_ms
            trace.src_enqueue_ts = now
            frame = Frame(
                frame_id=self._frame_id,
                capture_ts=capture_ts,
                image=image,
                width=width,
                height=height,
                source_id=self._source_id,
                seq=self._seq,
                trace=trace,
            )
            if self._slot is not None:
                self._dropped += 1
            self._slot = frame
            self._submitted += 1
            self._last_decode_ms = decode_ms
            self._byte_samples.append((now, len(jpeg)))
            self._frame_ready.set()

    # -- FrameSource protocol -----------------------------------------

    def start(self) -> None:
        with self._lock:
            self._running = True

    def stop(self) -> None:
        with self._lock:
            self._running = False
        self._frame_ready.set()  # release a blocked read()

    def read(self) -> Frame | None:
        if not self._running:
            return None
        if not self._frame_ready.wait(self._read_timeout_s):
            return None
        with self._lock:
            frame = self._slot
            self._slot = None
            self._frame_ready.clear()
            if not self._running:
                return None
        if frame is not None and frame.trace is not None:
            frame.trace.src_dequeue_ts = time.monotonic()
        return frame

    @property
    def is_running(self) -> bool:
        return self._running

    def info(self) -> dict[str, Any]:
        with self._lock:
            rtt = self._clock_offset.rtt_ms if self._clock_offset else None
            offset = self._clock_offset.offset_s if self._clock_offset else None
            return {
                "source_id": self._source_id,
                "kind": SourceKind.BROWSER.value,
                "expected_width": self._expected_w,
                "expected_height": self._expected_h,
                "running": self._running,
                "frames_submitted": self._submitted,
                "buffer_dropped": self._dropped,
                "decode_failures": self._decode_failures,
                "malformed": self._malformed,
                "decode_ms_last": self._last_decode_ms,
                "ingest_bytes_per_s": self._ingest_bytes_per_s_locked(),
                "clock_offset_rtt_ms": rtt,
                "clock_offset_s": offset,
                "clients_seen": self._clients_seen,
            }

    def source_info(self) -> SourceInfo:
        with self._lock:
            return SourceInfo(
                kind=SourceKind.BROWSER,
                source_id=self._source_id,
                label="Browser ingest (Web Worker)",
                width=self._expected_w,
                height=self._expected_h,
                achieved_fps=None,
                backend="ws:/ws/ingest",
                extra={"clients_seen": str(self._clients_seen)},
            )

    # -- context manager --------------------------------------------

    def __enter__(self) -> "BrowserSource":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    # -- internals ------------------------------------------------

    def _ingest_bytes_per_s_locked(self) -> float:
        """Bytes/second over the trailing window. Caller holds ``self._lock``."""

        cutoff = time.monotonic() - _BYTE_WINDOW_S
        while self._byte_samples and self._byte_samples[0][0] < cutoff:
            self._byte_samples.popleft()
        if len(self._byte_samples) < 2:
            return 0.0
        total = sum(n for _, n in self._byte_samples)
        span = self._byte_samples[-1][0] - self._byte_samples[0][0]
        return total / span if span > 0 else 0.0
