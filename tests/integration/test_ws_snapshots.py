"""WebSocket state stream: increasing snapshots, and a slow client cannot slow the loop."""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient

from predictivesense.api.app import create_app
from predictivesense.api.broadcast import Broadcaster, serve_state_client
from predictivesense.pipeline.loop import build_loop

pytestmark = pytest.mark.integration

# The loop rate is allowed to move by this fraction of the configured sample
# rate while a stalled client is being served. The point is "not collapsed",
# not "unchanged": normal scheduling jitter on a shared machine is wider than
# a few percent.
LOOP_RATE_TOLERANCE_FRAC = 0.5

MIN_SNAPSHOTS = 5


def test_ws_delivers_strictly_increasing_snapshots(dev_config) -> None:
    app = create_app(dev_config)
    with TestClient(app) as client, client.websocket_connect("/ws/state") as ws:
        received = [json.loads(ws.receive_text()) for _ in range(MIN_SNAPSHOTS)]

    ids = [s["snapshot_id"] for s in received]
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)
    for snap in received:
        assert snap["mode"] == dev_config.mode.value
        assert snap["detections"] == []
        assert snap["poses"] == []
        assert snap["tracks"] == []
        assert snap["risk"] is None
        assert "loop_rate_hz" in snap["metrics"]


def _measure_snapshot_rate(loop, window_s: float) -> float:
    start_n = loop.summary()["snapshots"]
    start_t = time.monotonic()
    time.sleep(window_s)
    return (loop.summary()["snapshots"] - start_n) / (time.monotonic() - start_t)


class _StalledWebSocket:
    """A client whose sends never complete within the timeout."""

    def __init__(self) -> None:
        self.closed = False

    async def send_text(self, data: str) -> None:
        await asyncio.sleep(3600)

    async def close(self, code: int = 1000) -> None:
        self.closed = True


def test_slow_client_is_dropped_without_slowing_the_loop(dev_config) -> None:
    loop = build_loop(dev_config)
    broadcaster = Broadcaster()
    loop.add_snapshot_listener(broadcaster.publish)
    loop.start()
    try:
        rate_before = _measure_snapshot_rate(loop, 1.0)

        stalled = _StalledWebSocket()
        dropped = asyncio.run(
            serve_state_client(
                stalled,
                broadcaster,
                rate_hz=dev_config.broadcast.rate_hz,
                send_timeout_s=0.3,
            )
        )

        rate_after = _measure_snapshot_rate(loop, 1.0)
    finally:
        loop.stop()

    assert dropped is True
    assert stalled.closed is True
    assert broadcaster.dropped_clients == 1
    assert loop.error is None

    target = dev_config.consumer.sample_rate_hz
    band = target * LOOP_RATE_TOLERANCE_FRAC
    assert rate_before == pytest.approx(target, abs=band)
    assert rate_after == pytest.approx(target, abs=band)
    # the loop did not lose a large fraction of its throughput
    assert rate_after >= rate_before * (1.0 - LOOP_RATE_TOLERANCE_FRAC)
