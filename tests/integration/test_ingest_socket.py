"""WS /ws/ingest: handshake, framed frames reach the mailbox, frame age is sane."""

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


def _jpeg(tag: int) -> bytes:
    img = np.zeros((_H, _W, 3), dtype=np.uint8)
    img[:, tag % _W] = 255
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _handshake(ws) -> None:
    ws.send_text(json.dumps({"type": "hello", "client_ts_ms": time.time() * 1000.0}))
    ack = json.loads(ws.receive_text())
    assert ack["type"] == "hello_ack"
    assert "server_mono" in ack and "rtt_probe" in ack
    ws.send_text(json.dumps({"type": "echo", "rtt_probe": ack["rtt_probe"]}))


def test_handshake_then_frames_reach_the_mailbox(browser_config) -> None:
    app = create_app(browser_config)
    n = 40
    live_frame_ids: list[int] = []
    live_ages: list[float] = []
    with TestClient(app) as client:
        with client.websocket_connect("/ws/ingest") as ws:
            _handshake(ws)
            base_ms = time.time() * 1000.0
            for i in range(n):
                header = IngestHeader(client_ts_ms=base_ms + i * 40.0, seq=i, w=_W, h=_H)
                ws.send_bytes(encode_ingest_message(header, _jpeg(i)))
                time.sleep(0.04)  # ~25 fps: faster than the 20 Hz consumer
                snap = app.state.loop.latest
                if snap is not None and snap.frame_id is not None:
                    live_frame_ids.append(snap.frame_id)
                    if snap.frame_age_ms is not None:
                        live_ages.append(snap.frame_age_ms)

        # Let the producer drain the ingest buffer and the consumer drain the
        # mailbox before sampling the accounting identity.
        loop = app.state.loop
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            info = app.state.browser_source.info()
            stats = loop.mailbox.stats()
            if stats.consumed + stats.dropped + info["buffer_dropped"] == n:
                break
            time.sleep(0.05)
        info = app.state.browser_source.info()
        stats = loop.mailbox.stats()

    assert info["frames_submitted"] == n
    assert info["decode_failures"] == 0
    assert info["malformed"] == 0
    assert info["clock_offset_rtt_ms"] is not None and info["clock_offset_rtt_ms"] >= 0.0

    # Every submitted frame is accounted for: consumed, or newest-wins dropped in
    # the mailbox, or newest-wins dropped in the ingest buffer. Nothing vanishes.
    assert stats.consumed + stats.dropped + info["buffer_dropped"] == n
    assert stats.consumed >= 1
    assert loop.max_mailbox_depth <= 1

    assert loop.error is None
    assert live_frame_ids, "no snapshot ever described an ingested frame"
    assert live_frame_ids == sorted(live_frame_ids)      # frame_id increases
    assert max(live_frame_ids) >= 1                       # ids advanced past the first
    # frame_age_ms is finite and positive (a small clock-jitter margin is allowed).
    assert live_ages
    assert all(-50.0 < age < 60_000.0 for age in live_ages)
    assert max(live_ages) > 0.0


def test_malformed_messages_are_counted_not_fatal(browser_config) -> None:
    app = create_app(browser_config)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/ingest") as ws:
            _handshake(ws)
            ws.send_bytes(b"\x00\x00\x00\x05not-a-real-header-or-jpeg")
            ws.send_bytes(b"\xff" * 32)
            # a good frame still gets through afterwards
            good = encode_ingest_message(
                IngestHeader(client_ts_ms=time.time() * 1000.0, seq=0, w=_W, h=_H),
                _jpeg(1),
            )
            ws.send_bytes(good)
            deadline = time.monotonic() + 3.0
            source = app.state.browser_source
            while source.info()["frames_submitted"] < 1 and time.monotonic() < deadline:
                time.sleep(0.02)

        info = app.state.browser_source.info()
        loop = app.state.loop

    assert info["malformed"] >= 2
    assert info["frames_submitted"] == 1
    assert loop.error is None


def test_ingest_refused_when_backend_owns_camera(dev_config) -> None:
    cfg = dev_config.model_copy(
        update={"capture": dev_config.capture.model_copy(update={"owner": "backend"})}
    )
    app = create_app(cfg, start_loop=False)
    with TestClient(app) as client:
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/ingest"):
                pass
