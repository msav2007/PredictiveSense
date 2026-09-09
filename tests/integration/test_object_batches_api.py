"""Bulk image upload API (Phase 7, BLOCK 9.3 / 9.4 / 9.5).

Upload many images to the selected object; roles + negative_for; corrupt /
unsupported / too-small / oversized files rejected with reasons while the rest
proceed; PATCH updates box/role/conditions; save commits valid items, leaves
invalid ones staged, reports counts; discard removes staging; duplicate
detection; provenance; dataset separation excludes staging.
"""

from __future__ import annotations

import io
import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from predictivesense.api.app import create_app

pytestmark = pytest.mark.integration


def _jpeg(fill: int = 120, w: int = 96, h: int = 72, *, noise: bool = False) -> bytes:
    import cv2

    if noise:
        img = np.random.default_rng(fill).integers(0, 255, (h, w, 3), dtype=np.uint8)
    else:
        img = np.full((h, w, 3), fill, np.uint8)
        img[8:40, 12:60] = 255
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


def _make_object(client, name="Watch") -> str:
    return client.post("/api/objects", json={"name": name}).json()["object_id"]


def _files(specs):
    return [("files", (name, data, "image/jpeg")) for name, data in specs]


def _poll(client, oid, bid, *, want="ready", tries=50):
    for _ in range(tries):
        body = client.get(f"/api/objects/{oid}/batches/{bid}").json()
        if body["status"] == want:
            return body
        time.sleep(0.05)
    raise AssertionError(f"batch never reached {want!r}: {body}")


# --------------------------------------------------------------------------


def test_upload_associates_all_images_with_the_selected_object(client) -> None:
    oid = _make_object(client)
    r = client.post(
        f"/api/objects/{oid}/batches",
        files=_files([(f"p{i}.jpg", _jpeg(30 + i * 20, noise=True)) for i in range(4)]),
        data={"role": "positive"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    bid = body["batch_id"]
    assert len(body["accepted"]) == 4 and body["rejected"] == []
    assert body["status"] == "processing"

    got = _poll(client, oid, bid)
    assert got["total"] == 4 and got["processed"] == 4
    # perception is off in dev_config -> every item needs a manual box, seeded centred
    for it in got["items"]:
        assert it["role"] == "positive"
        assert it["box"] is not None and len(it["box"]) == 4
        assert it["status"] in ("manual_required", "flagged", "ready")
        assert it["proposal_source"] in ("default_centred", "detector")
        assert it["box_confirmed_by_human"] is False
    # the staged images are NOT samples yet
    assert client.get(f"/api/objects/{oid}").json()["sample_count"] == 0


def test_negative_and_hard_negative_batches_carry_role_and_negative_for(client) -> None:
    oid = _make_object(client)
    for role in ("negative", "hard_negative"):
        r = client.post(
            f"/api/objects/{oid}/batches",
            files=_files([("a.jpg", _jpeg(90, noise=True)), ("b.jpg", _jpeg(140, noise=True))]),
            data={"role": role},
        )
        bid = r.json()["batch_id"]
        got = _poll(client, oid, bid)
        # give each a real box then save
        for it in got["items"]:
            client.patch(
                f"/api/objects/{oid}/batches/{bid}/items/{it['item_id']}",
                json={"box": [5, 5, 40, 40]},
            )
        saved = client.post(f"/api/objects/{oid}/batches/{bid}/save").json()
        assert saved["saved"] == 2
    samples = client.get(f"/api/objects/{oid}/samples").json()["samples"]
    assert samples and all(s["source"] == "upload_batch" for s in samples)
    assert all(s["negative_for"] == [oid] for s in samples)
    assert {s["role"] for s in samples} == {"negative", "hard_negative"}


def test_bad_files_are_rejected_with_reasons_while_the_rest_proceed(client) -> None:
    oid = _make_object(client)
    specs = [
        ("good1.jpg", _jpeg(40, noise=True)),
        ("corrupt.jpg", b"\xff\xd8\xff not really a jpeg"),
        ("text.txt", b"hello world" * 50),
        ("tiny.jpg", _jpeg(10, w=8, h=8)),
        ("good2.jpg", _jpeg(200, noise=True)),
    ]
    r = client.post(f"/api/objects/{oid}/batches", files=_files(specs), data={"role": "positive"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert sorted(body["accepted"]) == ["good1.jpg", "good2.jpg"]
    reasons = {d["filename"]: d["reason"] for d in body["rejected"]}
    assert set(reasons) == {"corrupt.jpg", "text.txt", "tiny.jpg"}
    assert "short side" in reasons["tiny.jpg"]


def test_batch_over_max_images_is_a_4xx(client, dev_config, tmp_path) -> None:
    cfg = dev_config.model_copy(
        update={
            "objects": dev_config.objects.model_copy(
                update={
                    "root": tmp_path / "o2",
                    "batch": dev_config.objects.batch.model_copy(update={"max_images": 3}),
                }
            )
        }
    )
    app = create_app(cfg, start_loop=False)
    with TestClient(app) as c:
        oid = c.post("/api/objects", json={"name": "cup"}).json()["object_id"]
        r = c.post(
            f"/api/objects/{oid}/batches",
            files=_files([(f"p{i}.jpg", _jpeg(i * 10, noise=True)) for i in range(5)]),
            data={"role": "positive"},
        )
        assert r.status_code == 400 and "max_images" in r.text


def test_oversize_total_batch_is_a_4xx(client, dev_config, tmp_path) -> None:
    cfg = dev_config.model_copy(
        update={
            "objects": dev_config.objects.model_copy(
                update={
                    "root": tmp_path / "o3",
                    "batch": dev_config.objects.batch.model_copy(update={"max_total_mb": 0.02}),
                }
            )
        }
    )
    app = create_app(cfg, start_loop=False)
    with TestClient(app) as c:
        oid = c.post("/api/objects", json={"name": "cup"}).json()["object_id"]
        big = _jpeg(0, w=400, h=400, noise=True)
        r = c.post(
            f"/api/objects/{oid}/batches",
            files=_files([("a.jpg", big), ("b.jpg", big)]),
            data={"role": "positive"},
        )
        assert r.status_code == 400 and "max_total_mb" in r.text


def test_patch_updates_box_role_conditions_and_marks_human_confirmed(client) -> None:
    oid = _make_object(client)
    bid = client.post(
        f"/api/objects/{oid}/batches",
        files=_files([("a.jpg", _jpeg(50, noise=True))]),
        data={"role": "positive"},
    ).json()["batch_id"]
    got = _poll(client, oid, bid)
    iid = got["items"][0]["item_id"]

    r = client.patch(
        f"/api/objects/{oid}/batches/{bid}/items/{iid}",
        json={"box": [3, 4, 30, 24], "role": "hard_negative",
              "conditions": {"view": "left", "lighting": "dim"}},
    )
    assert r.status_code == 200, r.text
    it = r.json()["item"]
    assert it["box"] == [3.0, 4.0, 30.0, 24.0]
    assert it["role"] == "hard_negative" and it["negative_for"] == [oid]
    assert it["conditions"]["view"] == "left" and it["conditions"]["background"] == "plain"
    assert it["box_confirmed_by_human"] is True and it["status"] == "edited"

    # a zero-area / out-of-bounds box is refused
    bad = client.patch(
        f"/api/objects/{oid}/batches/{bid}/items/{iid}", json={"box": [3, 4, 0, 10]}
    )
    assert bad.status_code == 400
    huge = client.patch(
        f"/api/objects/{oid}/batches/{bid}/items/{iid}", json={"box": [3, 4, 500, 10]}
    )
    assert huge.status_code == 400


def test_save_commits_valid_items_leaves_invalid_staged_and_reports_counts(client) -> None:
    oid = _make_object(client)
    bid = client.post(
        f"/api/objects/{oid}/batches",
        files=_files([(f"p{i}.jpg", _jpeg(20 + i * 30, noise=True)) for i in range(3)]),
        data={"role": "positive"},
    ).json()["batch_id"]
    got = _poll(client, oid, bid)
    ids = [it["item_id"] for it in got["items"]]
    # confirm boxes on 2 of 3, wipe the third
    client.patch(f"/api/objects/{oid}/batches/{bid}/items/{ids[0]}", json={"box": [2, 2, 40, 30]})
    client.patch(f"/api/objects/{oid}/batches/{bid}/items/{ids[1]}", json={"box": [4, 4, 30, 30]})
    client.patch(f"/api/objects/{oid}/batches/{bid}/items/{ids[2]}", json={"box": [1, 1, 20, 20],
                 "box_confirmed_by_human": True})
    # make item 3 invalid again by clearing it via the store is not exposed; instead
    # upload a fresh batch item that stays box-less by leaving perception off keeps
    # a centred default -> valid. So instead: PATCH item2 to an invalid box is 400.
    # Simulate "still needs a box" by discarding its box through a direct manifest edit.
    from predictivesense.objects.batches import BatchStore

    bs = BatchStore(client._objects_root / oid, oid)
    bs.update_item(bid, ids[2], box=None, status="manual_required")

    saved = client.post(f"/api/objects/{oid}/batches/{bid}/save").json()
    assert saved["saved"] == 2
    assert saved["remaining"] == 1
    assert saved["skipped"][0]["item_id"] == ids[2]
    assert saved["batch_cleared"] is False

    # the object gained exactly 2 samples; the batch still holds the unresolved one
    assert client.get(f"/api/objects/{oid}").json()["sample_count"] == 2
    left = client.get(f"/api/objects/{oid}/batches/{bid}").json()
    assert left["total"] == 1 and left["items"][0]["item_id"] == ids[2]

    # give it a box and save again -> batch clears
    client.patch(f"/api/objects/{oid}/batches/{bid}/items/{ids[2]}", json={"box": [2, 2, 20, 20]})
    saved2 = client.post(f"/api/objects/{oid}/batches/{bid}/save").json()
    assert saved2["saved"] == 1 and saved2["batch_cleared"] is True
    assert client.get(f"/api/objects/{oid}/batches/{bid}").status_code == 404
    assert client.get(f"/api/objects/{oid}").json()["sample_count"] == 3


def test_saved_samples_carry_full_bulk_upload_provenance(client) -> None:
    oid = _make_object(client)
    bid = client.post(
        f"/api/objects/{oid}/batches",
        files=_files([("photo one.jpg", _jpeg(77, noise=True))]),
        data={"role": "positive"},
    ).json()["batch_id"]
    got = _poll(client, oid, bid)
    iid = got["items"][0]["item_id"]
    client.patch(f"/api/objects/{oid}/batches/{bid}/items/{iid}", json={"box": [5, 5, 40, 30]})
    client.post(f"/api/objects/{oid}/batches/{bid}/save")

    s = client.get(f"/api/objects/{oid}/samples").json()["samples"][0]
    assert s["source"] == "upload_batch"
    assert s["batch_id"] == bid
    assert s["original_filename"] == "photo one.jpg"
    assert s["proposal_source"] in ("detector", "manual", "default_centred")
    assert s["box_confirmed_by_human"] is True
    # every field the camera path already writes is still present
    for key in ("sample_id", "path", "box", "width", "height", "conditions", "quality",
                "captured_utc", "git_commit", "thumb_path", "achieved_resolution"):
        assert key in s, key


def test_discard_removes_staging(client) -> None:
    oid = _make_object(client)
    bid = client.post(
        f"/api/objects/{oid}/batches",
        files=_files([("a.jpg", _jpeg(30, noise=True))]),
        data={"role": "positive"},
    ).json()["batch_id"]
    _poll(client, oid, bid)
    assert (client._objects_root / oid / "_staging" / bid).is_dir()
    r = client.delete(f"/api/objects/{oid}/batches/{bid}")
    assert r.status_code == 200 and r.json()["discarded"] is True
    assert not (client._objects_root / oid / "_staging" / bid).exists()
    assert client.delete(f"/api/objects/{oid}/batches/{bid}").status_code == 404


def test_duplicate_detection_flags_within_batch_and_against_existing(client) -> None:
    oid = _make_object(client)
    dup = _jpeg(123, noise=True)  # identical bytes twice within the batch
    bid = client.post(
        f"/api/objects/{oid}/batches",
        files=_files([("d1.jpg", dup), ("d2.jpg", dup), ("other.jpg", _jpeg(9, noise=True))]),
        data={"role": "positive"},
    ).json()["batch_id"]
    got = _poll(client, oid, bid)
    flagged = [it for it in got["items"] if "near_duplicate" in (it["quality"] or {}).get("flags", [])]
    assert flagged, "identical images in one batch must be flagged, not dropped"
    # nothing was auto-deleted
    assert got["total"] == 3

    # now save one, then a fresh batch with the same image -> flagged vs existing
    keep = got["items"][0]["item_id"]
    client.patch(f"/api/objects/{oid}/batches/{bid}/items/{keep}", json={"box": [5, 5, 40, 30]})
    client.post(f"/api/objects/{oid}/batches/{bid}/save")
    bid2 = client.post(
        f"/api/objects/{oid}/batches",
        files=_files([("again.jpg", dup)]),
        data={"role": "positive"},
    ).json()["batch_id"]
    got2 = _poll(client, oid, bid2)
    assert "near_duplicate" in (got2["items"][0]["quality"] or {}).get("flags", [])


def test_unknown_object_and_missing_batch_are_4xx(client) -> None:
    assert client.post(
        "/api/objects/nope/batches",
        files=_files([("a.jpg", _jpeg(1, noise=True))]),
        data={"role": "positive"},
    ).status_code == 404
    oid = _make_object(client)
    assert client.get(f"/api/objects/{oid}/batches/deadbeef").status_code == 404
    assert client.post(f"/api/objects/{oid}/batches/deadbeef/save").status_code == 404


@pytest.mark.models
def test_real_detector_proposes_a_box_bypassing_the_policy(
    require_models, perception_dev_config, tmp_path
) -> None:
    """With the real ONNX detector the proposal uses its RAW output - a class the
    recognition policy would suppress still yields a rectangle."""

    cfg = perception_dev_config.model_copy(
        update={"objects": perception_dev_config.objects.model_copy(update={"root": tmp_path / "o"})}
    )
    from pathlib import Path

    app = create_app(cfg, start_loop=False)
    fixtures = sorted((Path(__file__).resolve().parents[1] / "fixtures" / "perception").glob("*.jpg"))
    if not fixtures:
        pytest.skip("no perception fixtures")
    with TestClient(app) as c:
        oid = c.post("/api/objects", json={"name": "watch"}).json()["object_id"]
        r = c.post(
            f"/api/objects/{oid}/batches",
            files=[("files", (p.name, p.read_bytes(), "image/jpeg")) for p in fixtures[:3]],
            data={"role": "positive"},
        )
        bid = r.json()["batch_id"]
        got = _poll(c, oid, bid)
        assert got["total"] == len(fixtures[:3])
        for it in got["items"]:
            assert it["box"] and len(it["box"]) == 4
            # detector-proposed items keep the raw class as a HINT only
            if it["proposal_source"] == "detector":
                assert it["proposal_raw_class"]


def test_staging_is_excluded_from_the_coco_export(client, tmp_path, monkeypatch) -> None:
    from scripts import export_objects_coco as exporter

    oid = _make_object(client, name="mug")
    # one committed sample via the camera path
    client.post(
        f"/api/objects/{oid}/samples",
        files={"image": ("f.jpg", _jpeg(50, noise=True), "image/jpeg")},
        data={"box": json.dumps([4, 4, 30, 24]), "conditions": json.dumps({})},
    )
    # a staged batch that is never saved
    client.post(
        f"/api/objects/{oid}/batches",
        files=_files([("s1.jpg", _jpeg(60, noise=True)), ("s2.jpg", _jpeg(80, noise=True))]),
        data={"role": "positive"},
    )

    base = client.app.state.config
    monkeypatch.setattr(exporter, "load_config", lambda *_a, **_k: base)
    out = tmp_path / "coco_train.json"
    assert exporter.main(["--out", str(out)]) == 0
    train = json.loads(out.read_text(encoding="utf-8"))
    val = json.loads((tmp_path / "coco_val.json").read_text(encoding="utf-8"))
    files = {im["file_name"] for im in train["images"]} | {im["file_name"] for im in val["images"]}
    assert len(files) == 1  # only the committed camera sample
    assert not any("_staging" in f for f in files)
