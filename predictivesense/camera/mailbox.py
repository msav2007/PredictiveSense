"""A single-slot, overwrite-on-write frame mailbox.

``put`` never blocks and never grows. ``get`` returns the newest frame or
``None``. Depth is 0 or 1 at all times. ``dropped`` and ``consumed`` are exact.

Phase 8: ``get(block=True, timeout=...)`` lets a **completion-driven** consumer
sleep on a ``threading.Condition`` until the producer ``put``s a frame, instead
of waking on a fixed timer and finding the slot empty or stale. The default
``get()`` is byte-for-byte the old non-blocking behaviour, so the synthetic
loop, the recorded driver and every existing test are unchanged. Depth is still
0 or 1 and the newest-wins drop accounting is untouched.
"""

from __future__ import annotations

import threading

from predictivesense.core.types import Frame, MailboxStats

__all__ = ["LatestFrameMailbox"]


class LatestFrameMailbox:
    """Newest-wins handoff between one producer and one consumer."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._slot: Frame | None = None
        self._consumed = 0
        self._dropped = 0

    def put(self, frame: Frame) -> None:
        """Store ``frame``. If a frame is already waiting, it is dropped."""

        with self._cond:
            if self._slot is not None:
                self._dropped += 1
            self._slot = frame
            self._cond.notify()

    def get(self, *, block: bool = False, timeout: float | None = None) -> Frame | None:
        """Return and clear the waiting frame.

        ``block=False`` (default): return the frame or ``None`` immediately.
        ``block=True``: wait up to ``timeout`` seconds for a frame; still returns
        ``None`` if none arrives in time. Either way the newest frame wins and
        depth never exceeds 1.
        """

        with self._cond:
            if block and self._slot is None:
                self._cond.wait(timeout)
            frame = self._slot
            self._slot = None
            if frame is not None:
                self._consumed += 1
            return frame

    def wake(self) -> None:
        """Release a blocked :meth:`get` without delivering a frame (shutdown)."""

        with self._cond:
            self._cond.notify_all()

    def stats(self) -> MailboxStats:
        """Exact counters plus current depth (0 or 1)."""

        with self._cond:
            return MailboxStats(
                consumed=self._consumed,
                dropped=self._dropped,
                depth=1 if self._slot is not None else 0,
            )
