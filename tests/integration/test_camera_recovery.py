"""Camera loss mid-run degrades gracefully and recovers with no restart.

Requirement 3.4 #18: when the camera becomes unavailable the backend keeps
running and snapshots go ``stale=true``; when the device returns the pipeline
recovers automatically - no page reload, no loop restart. Covered here with a
fake source that fails then recovers, and with ``DeviceSource`` over a fake
``cv2.VideoCapture``.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import cv2
import numpy as np
import pytest

from predictivesense.camera.device import DeviceSource
from predictivesense.camera.mailbox import LatestFrameMailbox
from predictivesense.core.types import Frame
from predictivesense.pipeline.loop import AnalysisLoop
from tests.fixtures.camfakes import factory

pytestmark = pytest.mark.integration


class FlakyFakeSource:
    """A ``FrameSource`` whose availability is flipped from the test thread."""

    def __init__(self) -> None:
        self._alive = True
        self._running = False
        self._frame_id = -1
        self._lock = threading.Lock()

    def set_alive(self, value: bool) -> None:
        with self._lock:
            self._alive = value

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def read(self) -> Frame | None:
        time.sleep(0.005)  # ~200 fps ceiling; keeps the producer from spinning
        with self._lock:
            if not self._running or not self._alive:
                return None
            self._frame_id += 1
            fid = self._frame_id
        return Frame(
            frame_id=fid,
            capture_ts=time.monotonic(),
            image=np.zeros((4, 4, 3), dtype=np.uint8),
            width=4,
            height=4,
            source_id="flaky-fake",
            seq=fid,
        )

    @property
    def is_running(self) -> bool:
        return self._running

    def info(self) -> dict[str, Any]:
        return {"kind": "flaky-fake", "alive": self._alive}


def _wait_until(predicate, timeout: float, poll: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll)
    return False


def test_fake_source_loss_and_recovery_keeps_backend_alive(dev_config) -> None:
    src = FlakyFakeSource()
    loop = AnalysisLoop(dev_config, src, LatestFrameMailbox())
    loop.start()
    try:
        assert _wait_until(
            lambda: loop.latest is not None and not loop.latest.stale, 3.0
        ), "never reached a fresh (non-stale) snapshot"
        threads_before = (loop._producer, loop._consumer)  # identity, not restart
        first_snap_id = loop.latest.snapshot_id

        # --- device lost -------------------------------------------------
        src.set_alive(False)
        assert _wait_until(lambda: loop.latest.stale, 2.0), "snapshot never went stale"
        # backend keeps running through the outage
        assert loop.error is None
        assert loop.threads_alive() == {"producer": True, "consumer": True}
        assert loop.latest.snapshot_id > first_snap_id  # still emitting snapshots

        # --- device back ----------------------------------------------
        t_request = time.monotonic()
        src.set_alive(True)
        assert _wait_until(lambda: not loop.latest.stale, 3.0), "did not recover"
        recovery_s = time.monotonic() - t_request

        assert recovery_s < 2.0, f"recovery took {recovery_s:.2f}s"
        assert (loop._producer, loop._consumer) == threads_before  # no restart
        assert loop.error is None
    finally:
        loop.stop()
    assert loop.error is None


@pytest.mark.slow
def test_device_source_reconnect_surfaces_as_stale_then_recovers(
    dev_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    # cap #0: a burst of frames then dies; caps #1-#3: reopen fails (0.4 s probe
    # each) so the outage clearly exceeds stale_after_ms; cap #4+: healthy again.
    monkeypatch.setattr(cv2, "VideoCapture", factory([12, 0, 0, 0, 400]))
    src = DeviceSource(
        0,
        backend="msmf",
        open_timeout_s=0.4,
        reconnect_initial_s=0.05,
        reconnect_max_s=0.2,
    )
    loop = AnalysisLoop(dev_config, src, LatestFrameMailbox())
    loop.start()
    try:
        assert _wait_until(
            lambda: loop.latest is not None and not loop.latest.stale, 3.0
        )
        assert _wait_until(lambda: loop.latest.stale, 3.0), "loss never surfaced as stale"
        assert loop.error is None
        assert loop.threads_alive()["producer"] is True

        assert _wait_until(lambda: not loop.latest.stale, 6.0), "device never recovered"
        assert src.info()["reconnects"] >= 1
        assert loop.error is None
    finally:
        loop.stop()
    assert loop.error is None
