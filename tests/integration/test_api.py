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
    # Phase 1 rewrote the dashboard (Block 6). It loads the page logic as a
    # static module and never polls a <video> element for the preview.
    assert "PredictiveSense" in resp.text
    assert "/static/app.js" in resp.text


def test_static_assets_served(client) -> None:
    for asset, needle in (
        ("/static/app.js", "analysis-worker"),
        ("/static/analysis-worker.js", "/ws/ingest"),
        ("/static/app.css", "--accent"),
    ):
        r = client.get(asset)
        assert r.status_code == 200, asset
        assert needle in r.text


def test_cameras_endpoint_shape(client, monkeypatch) -> None:
    from predictivesense.api import app as app_mod

    monkeypatch.setattr(
        app_mod,
        "enumerate_devices",
        lambda **_: [],
    )
    monkeypatch.setattr(app_mod, "as_api_rows", lambda infos: [
        {"index": 0, "name": "Fake Cam", "available": True, "backend": "msmf"}
    ])
    resp = client.get("/api/cameras")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list) and body[0]["name"] == "Fake Cam"
    assert set(body[0]) == {"index", "name", "available", "backend"}
