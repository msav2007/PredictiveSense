"""Phase 8 section 13.3 / integration test 11 - the active provider and the
machine fingerprint are reported by ``GET /api/runtime`` and written into the
session manifest (not just the config value)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from predictivesense.api.app import create_app
from predictivesense.core.enums import SourceKind

pytestmark = pytest.mark.integration


def test_api_runtime_reports_the_live_provider_and_a_machine_fingerprint(browser_config):
    app = create_app(browser_config, start_loop=False)
    with TestClient(app) as client:
        r = client.get("/api/runtime")
        assert r.status_code == 200
        d = r.json()

    assert d["requested_provider"] == browser_config.perception.provider  # "auto"
    # perception is off in browser_config, so no live EP - but the shape is fixed
    assert "active_provider" in d and "available_providers" in d
    assert "scheduler" in d and d["scheduler"] in ("timer", "completion")
    assert "max_frame_age_ms" in d
    assert d["pose_cadence"] == browser_config.perception.pose_cadence
    fp = d["machine"]
    for key in ("cpu", "logical_cores", "ram_gb", "os", "python", "onnxruntime"):
        assert key in fp, key
    assert isinstance(fp["logical_cores"], int) and fp["logical_cores"] >= 1


@pytest.mark.models
def test_api_runtime_reports_a_concrete_ep_when_perception_is_on(
    require_models, perception_dev_config
):
    cfg = perception_dev_config.model_copy(update={
        "source": perception_dev_config.source.model_copy(update={"kind": SourceKind.BROWSER}),
        "perception": perception_dev_config.perception.model_copy(update={"provider": "auto"}),
    })
    app = create_app(cfg, start_loop=False)
    with TestClient(app) as client:
        d = client.get("/api/runtime").json()
    assert d["requested_provider"] == "auto"
    assert d["provider_resolved"] in ("cpu", "cuda", "directml")
    assert str(d["active_provider"]).endswith("ExecutionProvider")
    assert d["active_provider"] != "auto"
    assert d["model_version"]  # from models/registry.json


@pytest.mark.models
def test_session_manifest_carries_active_provider_and_fingerprint(
    require_models, perception_dev_config, tmp_path
):
    cfg = perception_dev_config.model_copy(update={
        "source": perception_dev_config.source.model_copy(update={"kind": SourceKind.BROWSER}),
        "results_dir": tmp_path,
    })
    app = create_app(cfg, start_loop=True, write_manifest=True)
    with TestClient(app):
        pass  # lifespan start + shutdown writes the manifest

    manifests = list(tmp_path.glob("session_*.json"))
    assert manifests, "no session manifest written"
    m = json.loads(manifests[0].read_text(encoding="utf-8"))
    perc = m["extra"]["perception"]
    assert perc["requested_provider"] == "auto"
    assert str(perc["active_provider"]).endswith("ExecutionProvider")
    assert perc["model_version"]
    fp = m["extra"]["machine"]
    assert fp["logical_cores"] >= 1 and fp["onnxruntime"] != "absent"
    assert m["extra"]["available_providers"]
