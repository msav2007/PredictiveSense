"""Model version registry (Phase 4): schema validation, exactly one active
version, and rejection of a non-SHA-256 / manifest-mismatched hash. No training."""

from __future__ import annotations

import json

import pytest

from predictivesense.models.registry import (
    ModelRegistry,
    ModelRegistryError,
    default_registry_path,
)

pytestmark = pytest.mark.unit

_GOOD_SHA = "0" * 64


def _doc(**over):
    base = {
        "version": 1,
        "models": [
            {
                "version_id": "v1",
                "name": "baseline",
                "task": "detect",
                "files": [{"role": "detector", "path": "models/x.onnx", "sha256": _GOOD_SHA, "size_bytes": 1}],
                "source": "test",
                "created_utc": "2026-01-01T00:00:00Z",
                "license": "AGPL-3.0-only",
                "metrics_ref": "results/eval_x.json",
                "active": True,
            }
        ],
    }
    base.update(over)
    return base


def _write(tmp_path, doc):
    p = tmp_path / "registry.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def test_shipped_registry_loads_and_has_one_active() -> None:
    reg = ModelRegistry.load(default_registry_path())
    assert reg.active().version_id == "v1"
    files = reg.resolve_active_files()
    assert "detector" in files and "pose" in files
    assert reg.active().license.startswith("AGPL")


def test_valid_two_version_registry(tmp_path) -> None:
    doc = _doc()
    doc["models"].append(
        {
            "version_id": "v2",
            "name": "candidate",
            "files": [{"role": "detector", "path": "models/y.onnx", "sha256": "a" * 64}],
            "source": "test",
            "created_utc": "2026-02-01T00:00:00Z",
            "license": "Apache-2.0",
            "metrics_ref": "results/eval_y.json",
            "active": False,
        }
    )
    reg = ModelRegistry.load(_write(tmp_path, doc), repo_root=tmp_path)
    assert [v.version_id for v in reg.list()] == ["v1", "v2"]
    assert reg.active().version_id == "v1"


def test_zero_or_multiple_active_is_rejected(tmp_path) -> None:
    doc = _doc()
    doc["models"][0]["active"] = False
    with pytest.raises(ModelRegistryError):
        ModelRegistry.load(_write(tmp_path, doc), repo_root=tmp_path)

    doc = _doc()
    doc["models"].append({**doc["models"][0], "version_id": "v1b"})
    with pytest.raises(ModelRegistryError):
        ModelRegistry.load(_write(tmp_path, doc), repo_root=tmp_path)


def test_bad_hash_is_rejected(tmp_path) -> None:
    doc = _doc()
    doc["models"][0]["files"][0]["sha256"] = "not-a-real-hash"
    with pytest.raises(ModelRegistryError):
        ModelRegistry.load(_write(tmp_path, doc), repo_root=tmp_path)


def test_hash_mismatch_against_manifest_is_rejected(tmp_path) -> None:
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "manifest.json").write_text(
        json.dumps({"models": {"d": {"filename": "real.onnx", "sha256": "b" * 64}}}),
        encoding="utf-8",
    )
    doc = _doc()
    doc["models"][0]["files"] = [{"role": "detector", "path": "models/real.onnx", "sha256": "c" * 64}]
    with pytest.raises(ModelRegistryError):
        ModelRegistry.load(_write(tmp_path, doc), repo_root=tmp_path)


def test_missing_registry_reports_not_crash(tmp_path) -> None:
    with pytest.raises(ModelRegistryError):
        ModelRegistry.load(tmp_path / "nope.json")
