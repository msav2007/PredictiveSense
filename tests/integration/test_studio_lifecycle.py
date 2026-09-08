"""Object Learning Studio lifecycle (Phase 4, P4 Block 4.1 / Block 12.7).

Entering the Studio provably stops monitoring: zero perception invocations and
zero ingest frames over a bounded interval. Leaving restores the prior state
exactly.
"""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from predictivesense.api.app import create_app
from predictivesense.perception.policy import RecognitionPolicy
from predictivesense.perception.types import PerceptionResult
from predictivesense.pipeline.loop import build_loop

pytestmark = pytest.mark.integration


class _SpyPerception:
    """Counts every per-frame inference call the loop makes."""

    def __init__(self) -> None:
        self.calls = 0

    def info(self) -> dict:
        return {}

    def infer(self, frame, **_kw) -> PerceptionResult:
        self.calls += 1
        return PerceptionResult(detections=(), poses=(), detector_ms=0.2, pose_ms=None)


def test_enter_stops_perception_and_leave_restores(dev_config) -> None:
    spy = _SpyPerception()
    loop = build_loop(dev_config, perception=spy, policy=RecognitionPolicy(dev_config.policy))
    app = create_app(dev_config, loop=loop)

    with TestClient(app) as client:
        # monitoring is live: perception is being called
        time.sleep(0.5)
        assert spy.calls > 0
        assert client.get("/api/studio/status").json()["active"] is False

        r = client.post("/api/studio/enter")
        assert r.status_code == 200
        token = r.json()["token"]
        assert token
        assert loop.paused is True
        assert client.get("/api/studio/status").json()["active"] is True

        # zero perception invocations over a bounded interval while the Studio is open
        baseline = spy.calls
        time.sleep(1.0)
        assert spy.calls == baseline, f"perception ran {spy.calls - baseline} time(s) with the Studio open"

        # re-entering hands back the same token (page reload), still paused
        assert client.post("/api/studio/enter").json()["token"] == token

        # a stale token is refused
        assert client.post("/api/studio/leave", json={"token": "wrong"}).status_code == 409

        r = client.post("/api/studio/leave", json={"token": token})
        assert r.status_code == 200 and r.json()["restored"] is True
        assert loop.paused is False
        assert client.get("/api/studio/status").json()["active"] is False

        # monitoring resumed: perception is being called again
        resumed_from = spy.calls
        time.sleep(0.5)
        assert spy.calls > resumed_from


def _handshake(ws) -> None:
    ws.send_text(json.dumps({"type": "hello", "client_ts_ms": time.time() * 1000.0}))
    ack = json.loads(ws.receive_text())
    ws.send_text(json.dumps({"type": "echo", "rtt_probe": ack["rtt_probe"]}))


def test_enter_closes_the_ingest_socket_to_new_frames(browser_config) -> None:
    app = create_app(browser_config)
    with TestClient(app) as client:
        # baseline: ingest works before the Studio
        with client.websocket_connect("/ws/ingest") as ws:
            _handshake(ws)

        assert client.post("/api/studio/enter").status_code == 200

        # no frames may reach /ws/ingest while the Studio is open
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws/ingest") as ws:
                _handshake(ws)

        assert client.post("/api/studio/leave").json()["restored"] is True

        # ingest is accepted again after leaving
        with client.websocket_connect("/ws/ingest") as ws:
            _handshake(ws)

        # preview-independence still holds after returning from the Studio:
        # a stalled consumer does not block the producer, mailbox stays bounded.
        assert client.post("/api/debug/stall", params={"seconds": 1.0}).json()["stalled_seconds"] == pytest.approx(1.0)
        time.sleep(1.3)
        loop = app.state.loop
        assert loop.threads_alive()["producer"] is True
        assert loop.max_mailbox_depth <= 1
        assert loop.error is None
        assert loop.paused is False


def test_models_registry_route(dev_config) -> None:
    app = create_app(dev_config, start_loop=False)
    with TestClient(app) as client:
        body = client.get("/api/models/registry").json()
        assert body["available"] is True
        assert body["active"] == "v1"
        assert any(m["version_id"] == "v1" for m in body["models"])
        v1 = next(m for m in body["models"] if m["version_id"] == "v1")
        assert {f["role"] for f in v1["files"]} == {"detector", "pose"}


def test_studio_page_served(dev_config) -> None:
    app = create_app(dev_config, start_loop=False)
    with TestClient(app) as client:
        r = client.get("/studio")
        assert r.status_code == 200 and "text/html" in r.headers["content-type"]
        assert "Object Learning Studio" in r.text
