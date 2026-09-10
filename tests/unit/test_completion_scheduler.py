"""Phase 8 section 19 unit test 1 - completion-driven scheduling as a pure
scheduler test, with no AnalysisLoop and no real perception.

With inference slower than the rate ceiling the loop must:
  * take the newest available frame each iteration (never an old one),
  * never wait on a fixed tick (the idle wait is event-based - it blocks in
    the mailbox Condition, released the instant a frame is put), and
  * not busy-wait (no repeated zero-timeout polling).
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from predictivesense.camera.mailbox import LatestFrameMailbox
from predictivesense.core.types import Frame
from predictivesense.pipeline.scheduler import run_completion_consumer

pytestmark = pytest.mark.unit


def _frame(i: int) -> Frame:
    return Frame(
        frame_id=i, capture_ts=float(i), image=np.zeros((2, 2, 3), np.uint8),
        width=2, height=2, source_id="t", seq=i,
    )


class _SpyMailbox(LatestFrameMailbox):
    """Records how `get` was called so the test can assert 'event-based wait'."""

    def __init__(self) -> None:
        super().__init__()
        self.get_calls: list[tuple[bool, float | None]] = []
        self.blocking_waits = 0

    def get(self, *, block: bool = False, timeout: float | None = None):
        self.get_calls.append((block, timeout))
        empty_before = self.stats().depth == 0
        frame = super().get(block=block, timeout=timeout)
        if block and empty_before and frame is not None:
            # we were released from a real Condition.wait by a put()
            self.blocking_waits += 1
        return frame


def test_completion_loop_takes_newest_frame_and_waits_on_the_event() -> None:
    mbox = _SpyMailbox()
    stop = threading.Event()

    processed: list[int] = []
    inference_s = 0.05  # slower than the 1 kHz ceiling below

    def iterate_once(frame: Frame | None) -> None:
        if frame is not None:
            processed.append(frame.frame_id)
        time.sleep(inference_s)  # simulate slow perception

    t = threading.Thread(
        target=run_completion_consumer,
        kwargs=dict(
            mailbox=mbox,
            iterate_once=iterate_once,
            should_stop=stop.is_set,
            stop_wait=stop.wait,
            is_paused=lambda: False,
            stall_until=lambda: 0.0,
            min_iter_period_s=0.001,      # 1 kHz ceiling - never the limiter here
            idle_poll_timeout_s=0.5,
        ),
        daemon=True,
    )
    t.start()

    # Produce faster than inference: every 10 ms for ~0.5 s. Because inference is
    # 50 ms, most produced frames are overwritten (newest-wins) before the
    # consumer can take them - the consumer must always take the *latest*.
    produced_last = 0
    for i in range(1, 51):
        mbox.put(_frame(i))
        produced_last = i
        time.sleep(0.01)
    time.sleep(0.15)
    stop.set()
    mbox.wake()
    t.join(timeout=2.0)
    assert not t.is_alive()

    # ~0.65 s of wall time / 50 ms per iteration -> ~10-14 iterations, NOT ~50
    # (which is what a 10 ms tick would have produced) and NOT 1.
    assert 5 <= len(processed) <= 20, processed
    # every processed frame id is one that was actually produced, and the
    # sequence is monotonic (newest-wins never hands back an older frame)
    assert processed == sorted(processed)
    assert processed[-1] >= produced_last - 5  # kept up with the newest
    # newest-wins: frames produced while the consumer was busy were overwritten,
    # so the processed ids have gaps - it did not drain a queue in order.
    max_gap = max(b - a for a, b in zip(processed, processed[1:]))
    assert max_gap > 1, processed

    # the idle wait was delegated to the mailbox as a *blocking* get, and at
    # least one of those blocks was released by a put (Condition.wait, not spin)
    assert all(call[0] is True for call in mbox.get_calls), mbox.get_calls
    assert mbox.blocking_waits >= 1


def test_completion_loop_respects_the_rate_ceiling_when_everything_is_cheap() -> None:
    """Frames and inference both instant -> the ceiling, not a spin, paces it."""

    mbox = LatestFrameMailbox()
    stop = threading.Event()
    starts: list[float] = []

    def iterate_once(frame: Frame | None) -> None:
        starts.append(time.monotonic())  # returns immediately

    # keep the mailbox always full
    filler_stop = threading.Event()

    def filler() -> None:
        i = 0
        while not filler_stop.is_set():
            i += 1
            mbox.put(_frame(i))
            time.sleep(0.001)

    ft = threading.Thread(target=filler, daemon=True)
    ft.start()

    t = threading.Thread(
        target=run_completion_consumer,
        kwargs=dict(
            mailbox=mbox,
            iterate_once=iterate_once,
            should_stop=stop.is_set,
            stop_wait=stop.wait,
            is_paused=lambda: False,
            stall_until=lambda: 0.0,
            min_iter_period_s=0.02,  # 50 Hz ceiling
            idle_poll_timeout_s=0.2,
        ),
        daemon=True,
    )
    t.start()
    time.sleep(0.6)
    stop.set()
    filler_stop.set()
    mbox.wake()
    t.join(timeout=2.0)
    ft.join(timeout=1.0)

    # 0.6 s at a 50 Hz ceiling -> ~25-35 iterations, never hundreds.
    assert 10 <= len(starts) <= 45, len(starts)
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    # median gap is near the 20 ms ceiling, not ~0 (a busy spin)
    gaps.sort()
    assert gaps[len(gaps) // 2] >= 0.012
