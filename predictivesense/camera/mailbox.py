"""A single-slot, overwrite-on-write frame mailbox.

``put`` never blocks and never grows. ``get`` returns the newest frame or
``None``. Depth is 0 or 1 at all times. ``dropped`` and ``consumed`` are exact.
"""

from __future__ import annotations

import threading

from predictivesense.core.types import Frame, MailboxStats

__all__ = ["LatestFrameMailbox"]


class LatestFrameMailbox:
    """Newest-wins handoff between one producer and one consumer."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._slot: Frame | None = None
        self._consumed = 0
        self._dropped = 0

    def put(self, frame: Frame) -> None:
        """Store ``frame``. If a frame is already waiting, it is dropped."""

        with self._lock:
            if self._slot is not None:
                self._dropped += 1
            self._slot = frame

    def get(self) -> Frame | None:
        """Return and clear the waiting frame, or ``None`` if the slot is empty."""

        with self._lock:
            frame = self._slot
            self._slot = None
            if frame is not None:
                self._consumed += 1
            return frame

    def stats(self) -> MailboxStats:
        """Exact counters plus current depth (0 or 1)."""

        with self._lock:
            return MailboxStats(
                consumed=self._consumed,
                dropped=self._dropped,
                depth=1 if self._slot is not None else 0,
            )
