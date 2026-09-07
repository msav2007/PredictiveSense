"""LatestFrameMailbox: newest-wins, exact counters, depth never above 1."""

from __future__ import annotations

import random
import threading

import pytest

from predictivesense.camera.mailbox import LatestFrameMailbox

pytestmark = pytest.mark.unit


def test_get_on_empty_returns_none_without_blocking() -> None:
    mb = LatestFrameMailbox()
    assert mb.get() is None
    assert mb.stats().depth == 0


def test_fast_producer_slow_consumer_counts_are_exact(make_frame) -> None:
    mb = LatestFrameMailbox()
    produced = 200
    consumed_frames = []

    for i in range(produced):
        mb.put(make_frame(i))
        if i % 5 == 0:  # consumer lags well behind the producer
            got = mb.get()
            if got is not None:
                consumed_frames.append(got)

    # drain the final frame so the slot is empty at the assertion point
    tail = mb.get()
    if tail is not None:
        consumed_frames.append(tail)

    stats = mb.stats()
    assert stats.depth == 0
    assert stats.consumed == len(consumed_frames)
    assert stats.consumed + stats.dropped == produced
    assert stats.dropped == produced - stats.consumed
    # consumer only ever sees the newest frame available at get() time
    assert consumed_frames[-1].frame_id == produced - 1


def test_depth_never_exceeds_one_under_randomised_sequence(make_frame) -> None:
    mb = LatestFrameMailbox()
    rng = random.Random(2024)
    fid = 0
    for _ in range(5000):
        if rng.random() < 0.6:
            mb.put(make_frame(fid))
            fid += 1
        else:
            mb.get()
        assert mb.stats().depth in (0, 1)


def test_counters_exact_under_concurrent_producer(make_frame) -> None:
    mb = LatestFrameMailbox()
    total = 3000
    stop = threading.Event()

    def produce() -> None:
        for i in range(total):
            mb.put(make_frame(i))
        stop.set()

    producer = threading.Thread(target=produce, name="test-producer")
    producer.start()

    consumed = 0
    while not stop.is_set() or mb.stats().depth == 1:
        if mb.get() is not None:
            consumed += 1
    producer.join(timeout=5)
    assert not producer.is_alive()

    stats = mb.stats()
    assert stats.depth == 0
    assert stats.consumed == consumed
    assert stats.consumed + stats.dropped == total
