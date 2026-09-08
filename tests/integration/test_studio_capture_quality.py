"""Studio capture quality (Phase 5, BLOCK 11.5).

A captured sample stores a full-resolution original with the achieved resolution
recorded (authoritative - taken from the decoded image, never the client's
claim), plus a *separate* thumbnail; requested and achieved are both recorded;
the original is not the thumbnail.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from predictivesense.api.app import create_app

pytestmark = pytest.mark.integration


def _jpeg(w: int, h: int, q: int = 95) -> bytes:
    import cv2

    rng = np.random.default_rng(1234)
    img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)  # high entropy
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    assert ok
    return buf.tobytes()


@pytest.fixture()
def client(dev_config, tmp_path):
    cfg = dev_config.model_copy(
        update={"objects": dev_config.objects.model_copy(update={"root": tmp_path / "objects"})}
    )
    app = create_app(cfg, start_loop=False)
    with TestClient(app) as c:
        c._objects_root = tmp_path / "objects"  # type: ignore[attr-defined]
        yield c


def test_capture_stores_full_res_original_and_a_separate_thumbnail(client) -> None:
    oid = client.post("/api/objects", json={"name": "kettle"}).json()["object_id"]

    W, H = 1280, 720
    r = client.post(
        f"/api/objects/{oid}/samples",
        files={"image": ("sample.jpg", _jpeg(W, H), "image/jpeg")},
        data={
            "box": json.dumps([100, 80, 400, 300]),
            "conditions": json.dumps({}),
            "role": "positive",
            "source": "camera",
            "capture_path": "imagebitmap",
            "requested_resolution": "1920x1080",
            "encoded_quality": "0.95",
        },
    )
    assert r.status_code == 201, r.text
    s = r.json()["sample"]

    # provenance recorded
    assert s["capture_path"] == "imagebitmap"
    assert s["requested_resolution"] == "1920x1080"
    assert s["achieved_resolution"] == f"{W}x{H}"  # authoritative, from the decoded image
    assert s["encoded_quality"] >= 0.92
    assert s["original_bytes"] > 0
    assert s["width"] == W and s["height"] == H

    # a separate thumbnail exists and is not the original
    assert s["thumb_path"] and s["thumb_path"] != s["path"]
    obj_dir = client._objects_root / oid
    orig = (obj_dir / s["path"]).read_bytes()
    thumb = (obj_dir / s["thumb_path"]).read_bytes()
    assert orig != thumb
    assert len(thumb) < len(orig)

    # the stored original decodes back at the full achieved resolution
    import cv2

    dec = cv2.imdecode(np.frombuffer(orig, np.uint8), cv2.IMREAD_COLOR)
    assert dec.shape[1] == W and dec.shape[0] == H
    # the thumbnail is genuinely downscaled (longer side <= configured thumbnail_px)
    tdec = cv2.imdecode(np.frombuffer(thumb, np.uint8), cv2.IMREAD_COLOR)
    assert max(tdec.shape[:2]) <= client.app.state.config.objects.thumbnail_px


def test_jpeg_upload_is_stored_without_recompression(client) -> None:
    oid = client.post("/api/objects", json={"name": "lamp"}).json()["object_id"]
    payload = _jpeg(800, 600, q=91)
    r = client.post(
        f"/api/objects/{oid}/samples",
        files={"image": ("photo.jpg", payload, "image/jpeg")},
        data={"box": json.dumps([10, 10, 100, 100]), "conditions": json.dumps({}), "source": "upload"},
    )
    assert r.status_code == 201, r.text
    s = r.json()["sample"]
    obj_dir = client._objects_root / oid
    stored = (obj_dir / s["path"]).read_bytes()
    # a JPEG upload is kept verbatim - no generation loss (BLOCK 3.23)
    assert stored == payload
    assert s["capture_path"] == "upload"
    assert s["achieved_resolution"] == "800x600"
