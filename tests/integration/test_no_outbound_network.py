"""Guardrail: nothing in the runtime opens an outbound (non-loopback) connection."""

from __future__ import annotations

import socket
import time

import pytest
from fastapi.testclient import TestClient

from predictivesense.api.app import create_app
from predictivesense.pipeline.loop import build_loop

pytestmark = pytest.mark.integration

_LOOPBACK = {"127.0.0.1", "::1", "localhost", "0.0.0.0"}


class OutboundConnectionError(AssertionError):
    """Raised if runtime code attempts to connect to a non-loopback address."""


@pytest.fixture()
def forbid_outbound(monkeypatch: pytest.MonkeyPatch):
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    seen: list[object] = []

    def _host_of(address: object) -> str | None:
        if isinstance(address, tuple) and address:
            return str(address[0])
        return None

    def guard_connect(self, address):  # noqa: ANN001
        host = _host_of(address)
        if host is not None and host not in _LOOPBACK:
            seen.append(address)
            raise OutboundConnectionError(f"outbound connect attempted to {address!r}")
        return real_connect(self, address)

    def guard_connect_ex(self, address):  # noqa: ANN001
        host = _host_of(address)
        if host is not None and host not in _LOOPBACK:
            seen.append(address)
            raise OutboundConnectionError(f"outbound connect_ex attempted to {address!r}")
        return real_connect_ex(self, address)

    monkeypatch.setattr(socket.socket, "connect", guard_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guard_connect_ex)
    return seen


def test_pipeline_and_api_make_no_outbound_connections(forbid_outbound, dev_config) -> None:
    # Pipeline for a few seconds.
    loop = build_loop(dev_config)
    loop.run_for(2.0)
    assert loop.error is None

    # API surface, including a websocket round trip.
    app = create_app(dev_config)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/api/config").status_code == 200
        with client.websocket_connect("/ws/state") as ws:
            ws.receive_text()
        time.sleep(0.5)

    assert forbid_outbound == [], f"outbound connections attempted: {forbid_outbound}"
