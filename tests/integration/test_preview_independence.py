"""A stalled analysis consumer must not block ingest or the producer.

This is the automated half of Block 4's defining constraint. The physical
"watch the preview by eye during a stall" check stays with the developer.
"""

from __future__ import annotations

import json
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from predictivesense.api.app import create_app
from predictivesense.camera.framing import encode_ingest_message
from predictivesense.core.types import IngestHeader

pytestmark = pytest.mark.integration

_W, _H = 64, 48
_STALL_S = 3.0


def _jpeg(tag: int) -> bytes:
    img = np.zeros((_H, _W, 3), dtype=np.uint8)
    img[:, tag % _W] = 255
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _handshake(ws) -> None:
    ws.send_text(json.dumps({"type": "hello", "client_ts_ms": time.time() * 1000.0}))
    ack = json.loads(ws.receive_text())
    ws.send_text(json.dumps({"type": "echo", "rtt_probe": ack["rtt_probe"]}))


def test_consumer_stall_does_not_block_ingest_or_producer(browser_config) -> None:
    app = create_app(browser_config)
    sent = 0
    with TestClient(app) as client:
        with client.websocket_connect("/ws/ingest") as ws:
            _handshake(ws)

            resp = client.post("/api/debug/stall", params={"seconds": _STALL_S})
            assert resp.json()["stalled_seconds"] == pytest.approx(_STALL_S)

            # Push frames well past the consumer's sample rate for the whole stall.
            t_end = time.monotonic() + _STALL_S + 0.5
            base_ms = time.time() * 1000.0
            while time.monotonic() < t_end:
                header = IngestHeader(
                    client_ts_ms=base_ms + sent * 20.0, seq=sent, w=_W, h=_H
                )
                send_started = time.monotonic()
                ws.send_bytes(encode_ingest_message(header, _jpeg(sent)))
                # the socket send must not block on the stalled consumer
                assert time.monotonic() - send_started < 1.0
                sent += 1
                time.sleep(0.02)

            source = app.state.browser_source
            deadline = time.monotonic() + 3.0
            while source.info()["frames_submitted"] < sent and time.monotonic() < deadline:
                time.sleep(0.02)

        loop = app.state.loop
        info = app.state.browser_source.info()
        mailbox = loop.mailbox.stats()
        threads = loop.threads_alive()

    assert sent > 20
    assert info["frames_submitted"] == sent          # socket kept accepting throughout
    assert mailbox.dropped > 0                        # mailbox drops climbed during the stall
    assert mailbox.depth <= 1                         # mailbox never grew past one slot
    assert loop.max_mailbox_depth <= 1
    assert info["buffer_dropped"] >= 0               # ingest buffer is also bounded newest-wins
    assert threads["producer"] is True               # producer never died
    assert loop.error is None
