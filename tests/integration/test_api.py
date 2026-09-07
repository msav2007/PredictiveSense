"""HTTP endpoints return the documented shapes."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from predictivesense import __version__
from predictivesense.api.app import create_app

pytestmark = pytest.mark.integration


@pytest.fixture()
def client(dev_config):
    # start_loop=False: /health and /api/config need no threads.
    app = create_app(dev_config, start_loop=False)
    with TestClient(app) as c:
        yield c


def test_health_shape(client) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert isinstance(body["uptime_s"], (int, float)) and body["uptime_s"] >= 0
    assert body["mode"] in {"realtime", "recorded"}
    assert body["version"] == __version__


def test_api_config_is_resolved_config(client, dev_config) -> None:
    resp = client.get("/api/config")
    assert resp.status_code == 200
    assert resp.json() == dev_config.as_json_dict()


def test_index_page_served(client) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "state snapshot" in resp.text
    assert "/ws/state" in resp.text
