"""Phase 12 section 10.4: Diagnostics (``GET /api/objects/training_status``)
must report the REAL active classifier from the registry - config-driven and
test-isolated (``training.classifier_registry_path``), never the production
``models/classifier_registry.json``.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from predictivesense.api.app import create_app
from predictivesense.training.classifier_registry import ClassifierRegistry, ClassifierVersion

pytestmark = pytest.mark.integration


def _fake_version(version_id: str, *, validated: bool, active: bool) -> ClassifierVersion:
    return ClassifierVersion(
        version_id=version_id, created_utc="2026-01-01T00:00:00.000000Z",
        class_map={"watch": 0}, dataset_content_hash="x", split_content_hash="x",
        architecture="CropClassifier-3conv-gap", image_size=64, epochs=1, batch_size=1,
        learning_rate=1e-3, seed=0, augmentation="none", metrics={"accuracy": 1.0},
        artifact_path="models/custom/fake/model.onnx", artifact_sha256="0" * 64,
        wall_seconds=0.1, machine_fingerprint="test", provider="cpu",
        dataset_human_confirmed_fraction=1.0, validated=validated, active=active,
    )


@pytest.fixture()
def client(dev_config, tmp_path):
    cfg = dev_config.model_copy(update={
        "objects": dev_config.objects.model_copy(update={"root": tmp_path / "objects"}),
        "training": dev_config.training.model_copy(
            update={"classifier_registry_path": tmp_path / "classifier_registry.json"}
        ),
    })
    app = create_app(cfg, start_loop=False)
    with TestClient(app) as c:
        c._registry_path = tmp_path / "classifier_registry.json"  # type: ignore[attr-defined]
        yield c


def test_training_status_reports_no_active_classifier_by_default(client):
    body = client.get("/api/objects/training_status").json()
    assert body["active_classifier"] is None
    assert body["trained_versions"] == []


def test_training_status_reports_the_real_active_classifier(client):
    reg = ClassifierRegistry(path=client._registry_path)
    reg.register(_fake_version("v1", validated=False, active=False))
    reg.validate_version("v1")
    reg.activate("v1")

    body = client.get("/api/objects/training_status").json()
    assert body["active_classifier"]["version_id"] == "v1"
    assert body["active_classifier"]["classes"] == ["watch"]
    assert len(body["trained_versions"]) == 1
    assert body["trained_versions"][0]["active"] is True
