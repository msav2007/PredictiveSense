"""DeviceSource reconnect/stop responsiveness (fake cv2.VideoCapture, no camera).

These exercise real threads and backoff timing, so they are ``integration`` and
``slow``. The guarantee under test: a down camera never blocks ``read()`` for a
whole backoff and never blocks ``stop()`` at all.
"""

from __future__ import annotations

import threading
import time

import cv2
import pytest

from predictivesense.camera.device import DeviceSource
from tests.fixtures.camfakes import factory

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_reads_then_reconnects_without_blocking(monkeypatch: pytest.MonkeyPatch) -> None:
    # First cap yields a handful of frames (one eaten by the open probe) then
    # fails forever; later caps yield nothing so reconnect keeps failing.
    monkeypatch.setattr(cv2, "VideoCapture", factory([8, 0]))
    src = DeviceSource(
        0, backend="msmf", open_timeout_s=0.5,
        reconnect_initial_s=0.05, reconnect_max_s=0.2,
    )
    src.start()
    try:
        frames = [f for f in (src.read() for _ in range(20)) if f is not None]
        assert len(frames) >= 3
        assert [f.frame_id for f in frames] == list(range(len(frames)))

        for _ in range(6):
            t0 = time.monotonic()
            assert src.read() is None
            assert time.monotonic() - t0 < 0.9  # bounded by one reopen probe

        info = src.info()
        assert info["read_failures"] >= 1
        assert info["reconnect_attempts"] >= 1
        assert info["connected"] is False
        assert info["currently_down_s"] > 0.0
    finally:
        t0 = time.monotonic()
        src.stop()
        assert time.monotonic() - t0 < 1.0


def test_recovers_when_the_camera_comes_back(monkeypatch: pytest.MonkeyPatch) -> None:
    # cap #0: 3 frames then dies. cap #1 (first reopen): dies immediately.
    # cap #2 onward: healthy again.
    monkeypatch.setattr(cv2, "VideoCapture", factory([4, 0, 50]))
    src = DeviceSource(
        0, backend="msmf", open_timeout_s=0.5,
        reconnect_initial_s=0.05, reconnect_max_s=0.2,
    )
    src.start()
    try:
        deadline = time.monotonic() + 5.0
        seen_after_recovery = 0
        recovered = False
        while time.monotonic() < deadline:
            f = src.read()
            if f is None:
                continue
            if src.info()["reconnects"] >= 1:
                recovered = True
                seen_after_recovery += 1
                if seen_after_recovery >= 5:
                    break
        assert recovered, "source never reconnected"
        assert seen_after_recovery >= 5
        assert src.info()["reconnect_seconds_total"] > 0.0
    finally:
        src.stop()


def test_stop_interrupts_a_reader_thread_mid_reconnect(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cv2, "VideoCapture", factory([1, 0]))
    src = DeviceSource(
        0, backend="msmf", open_timeout_s=0.3,
        reconnect_initial_s=0.1, reconnect_max_s=0.2,
    )
    src.start()
    stopped = threading.Event()

    def pump() -> None:
        while src.is_running:
            src.read()
        stopped.set()

    t = threading.Thread(target=pump, name="pump")
    t.start()
    time.sleep(0.3)
    t0 = time.monotonic()
    src.stop()
    t.join(timeout=2.0)
    assert stopped.is_set()
    assert time.monotonic() - t0 < 1.5
