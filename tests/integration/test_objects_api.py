"""Object Learning Studio API lifecycle (Phase 4).

Create an object, post a sample with a box and tags, list, patch the box,
coverage reflects it, delete is soft; oversize upload and missing box rejected.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from predictivesense.api.app import create_app

pytestmark = pytest.mark.integration


def _jpeg(fill: int = 120, w: int = 96, h: int = 72) -> bytes:
    import cv2

    img = np.full((h, w, 3), fill, np.uint8)
    img[10:40, 20:60] = 255  # some structure so blur variance is non-trivial
    ok, buf = cv2.imencode(".jpg", img)
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


def test_full_object_and_sample_lifecycle(client) -> None:
    # vocab is served and not hard-coded in the client
    vocab = client.get("/api/objects/vocab").json()
    assert "front" in vocab["conditions"]["view"] and "hard_negative" in vocab["roles"]

    # create
    r = client.post("/api/objects", json={"name": "Watch", "kind": "class", "category": "wearable"})
    assert r.status_code == 201, r.text
    obj = r.json()
    oid = obj["object_id"]
    assert oid == "watch"
    assert "clock" in obj["confusable_with"]  # seeded from observed failures

    # post a sample: image + box + conditions + role
    r = client.post(
        f"/api/objects/{oid}/samples",
        files={"image": ("frame.jpg", _jpeg(), "image/jpeg")},
        data={
            "box": json.dumps([20, 10, 40, 30]),
            "conditions": json.dumps({"view": "left", "distance": "far", "lighting": "dim"}),
            "role": "positive",
            "source": "camera",
            "consent_ack": "true",
        },
    )
    assert r.status_code == 201, r.text
    body = r.json()
    sample = body["sample"]
    assert sample["box"] == [20.0, 10.0, 40.0, 30.0]
    assert sample["conditions"]["view"] == "left" and sample["conditions"]["background"] == "plain"
    assert sample["source"] == "camera" and sample["git_commit"]
    assert "phash" in sample["quality"] and "box_area_frac" in sample["quality"]
    assert body["coverage"]["total_positives"] == 1

    # upload path produces the same record shape
    r = client.post(
        f"/api/objects/{oid}/samples",
        files={"image": ("my photo.png", _jpeg(60), "image/png")},
        data={"box": json.dumps([5, 5, 30, 30]), "conditions": json.dumps({}),
              "role": "hard_negative", "source": "upload"},
    )
    assert r.status_code == 201, r.text
    hn = r.json()["sample"]
    assert hn["source"] == "upload" and hn["original_filename"] == "my photo.png"
    assert hn["role"] == "hard_negative" and hn["negative_for"] == [oid]

    # list
    samples = client.get(f"/api/objects/{oid}/samples").json()["samples"]
    assert len(samples) == 2
    sid = sample["sample_id"]

    # patch the box + role
    r = client.patch(
        f"/api/objects/{oid}/samples/{sid}",
        json={"box": [21, 11, 42, 33], "role": "positive"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["sample"]["box"] == [21.0, 11.0, 42.0, 33.0]

    # coverage reflects the samples + guidance strings
    cov = client.get(f"/api/objects/{oid}/coverage").json()
    assert cov["total_positives"] == 1 and cov["total_hard_negatives"] == 1
    assert cov["distances_present"] == ["far"]
    assert any("far-distance" not in g for g in cov["guidance"])  # 'far' now present
    assert isinstance(cov["guidance"], list) and cov["guidance"]

    # object listing carries counts + coverage summary
    listing = client.get("/api/objects").json()
    row = next(o for o in listing["objects"] if o["object_id"] == oid)
    assert row["sample_count"] == 2 and row["positive_count"] == 1 and row["negative_count"] == 1

    # the stored image is served
    img = client.get(f"/api/objects/{oid}/samples/{sid}/image")
    assert img.status_code == 200 and img.headers["content-type"].startswith("image/")
    assert client.get(f"/api/objects/{oid}/samples/{sid}/image?thumb=1").status_code == 200

    # delete a sample -> soft (files move to _deleted/), counts update
    r = client.delete(f"/api/objects/{oid}/samples/{sid}")
    assert r.status_code == 200 and r.json()["soft_deleted"] is True
    assert len(client.get(f"/api/objects/{oid}/samples").json()["samples"]) == 1
    deleted_dir = client._objects_root / oid / "_deleted"
    assert deleted_dir.is_dir() and any(deleted_dir.iterdir())

    # delete the object -> soft (folder moves), gone from the registry
    r = client.delete(f"/api/objects/{oid}")
    assert r.status_code == 200 and r.json()["soft_deleted"] is True
    assert r.json()["moved_to"]
    assert client.get(f"/api/objects/{oid}").status_code == 404


def test_missing_box_is_rejected(client) -> None:
    oid = client.post("/api/objects", json={"name": "cup"}).json()["object_id"]
    r = client.post(
        f"/api/objects/{oid}/samples",
        files={"image": ("f.jpg", _jpeg(), "image/jpeg")},
        data={"conditions": json.dumps({})},  # no box
    )
    assert r.status_code == 422  # required form field


def test_box_outside_image_is_rejected(client) -> None:
    oid = client.post("/api/objects", json={"name": "cup"}).json()["object_id"]
    r = client.post(
        f"/api/objects/{oid}/samples",
        files={"image": ("f.jpg", _jpeg(w=96, h=72), "image/jpeg")},
        data={"box": json.dumps([90, 60, 40, 40]), "conditions": json.dumps({})},
    )
    assert r.status_code == 400
    assert "bounds" in r.text.lower()


def test_oversize_upload_is_rejected(dev_config, tmp_path) -> None:
    cfg = dev_config.model_copy(
        update={
            "objects": dev_config.objects.model_copy(
                update={"root": tmp_path / "objects", "max_image_mb": 0.01}  # 10 KiB
            )
        }
    )
    app = create_app(cfg, start_loop=False)
    with TestClient(app) as c:
        oid = c.post("/api/objects", json={"name": "cup"}).json()["object_id"]
        big = b"\xff\xd8\xff" + b"x" * (20 * 1024)
        r = c.post(
            f"/api/objects/{oid}/samples",
            files={"image": ("big.jpg", big, "image/jpeg")},
            data={"box": json.dumps([1, 1, 5, 5]), "conditions": json.dumps({})},
        )
        assert r.status_code == 413
        assert "objects.max_image_mb" in r.text


def test_instance_kind_and_soft_delete_folder(client) -> None:
    r = client.post(
        "/api/objects",
        json={"name": "Dad's mug", "kind": "instance", "parent_class": "mug"},
    )
    assert r.status_code == 201
    assert r.json()["kind"] == "instance" and r.json()["parent_class"] == "mug"

    r = client.post("/api/objects", json={"name": "x", "kind": "instance"})  # no parent
    assert r.status_code == 400
