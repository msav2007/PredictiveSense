"""Clip recorder: a small blob uploads with a full manifest; oversize is rejected."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from predictivesense.api.app import create_app
from predictivesense.core.types import ClipManifest

pytestmark = pytest.mark.integration

_MANIFEST_KEYS = set(ClipManifest.model_fields)
# A minimal but structurally valid MJPG-in-AVI-ish blob is unnecessary here: the
# recorder stores bytes verbatim and treats duration as optional metadata.
_FAKE_WEBM = b"\x1aE\xdf\xa3" + b"predictivesense-fake-clip" * 64


def _cfg(base, *, max_mb=5.0):
    from predictivesense.config.settings import load_config

    cfg = load_config("dev")
    return cfg.model_copy(
        update={
            "recorder": cfg.recorder.model_copy(
                update={"output_dir": Path(base), "max_clip_mb": max_mb, "enabled": True}
            )
        }
    )


def _meta():
    return {
        "scenario_tag": "cabin-cup-near-edge",
        "device_label": "Integrated Camera",
        "width": "1280",
        "height": "720",
        "nominal_fps": "30",
        "notes": "unit-test clip",
        "consent_ack": "true",
    }


def test_upload_stores_clip_and_full_manifest(tmp_path) -> None:
    app = create_app(_cfg(tmp_path), start_loop=False)
    with TestClient(app) as client:
        resp = client.post(
            "/api/record/upload",
            files={"file": ("clip.webm", _FAKE_WEBM, "video/webm")},
            data=_meta(),
        )
        assert resp.status_code == 200, resp.text
        manifest = resp.json()
        assert set(manifest) == _MANIFEST_KEYS
        assert manifest["scenario_tag"] == "cabin-cup-near-edge"
        assert manifest["consent_ack"] is True
        assert manifest["size_bytes"] == len(_FAKE_WEBM)
        assert manifest["config_profile"] == "dev"
        assert manifest["git_commit"]

        clip_path = Path(manifest["path"])
        assert clip_path.is_file()
        assert clip_path.read_bytes() == _FAKE_WEBM
        sidecar = clip_path.with_suffix(".json")
        assert sidecar.is_file()
        assert json.loads(sidecar.read_text(encoding="utf-8"))["clip_id"] == manifest["clip_id"]
        # data/raw containment
        assert Path(tmp_path) in clip_path.parents

        listing = client.get("/api/clips").json()
        assert any(c["clip_id"] == manifest["clip_id"] for c in listing)


def test_oversize_upload_is_rejected(tmp_path) -> None:
    app = create_app(_cfg(tmp_path, max_mb=0.01), start_loop=False)  # 10 KiB cap
    with TestClient(app) as client:
        big = b"x" * (20 * 1024)
        resp = client.post(
            "/api/record/upload",
            files={"file": ("big.webm", big, "video/webm")},
            data=_meta(),
        )
    assert resp.status_code == 413
    assert not list(Path(tmp_path).rglob("*.webm"))


def test_missing_required_field_is_a_422(tmp_path) -> None:
    app = create_app(_cfg(tmp_path), start_loop=False)
    with TestClient(app) as client:
        data = _meta()
        del data["scenario_tag"]
        resp = client.post(
            "/api/record/upload",
            files={"file": ("clip.webm", _FAKE_WEBM, "video/webm")},
            data=data,
        )
    assert resp.status_code == 422


def test_clips_listing_empty_when_dir_absent(tmp_path) -> None:
    app = create_app(_cfg(tmp_path / "nope"), start_loop=False)
    with TestClient(app) as client:
        assert client.get("/api/clips").json() == []
