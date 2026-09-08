"""Label API: post boxes, read them back, progress counts update, provenance
preserved."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from predictivesense.api.app import create_app
from predictivesense.dataset.coco_store import CocoStore, domain_categories

pytestmark = pytest.mark.integration


@pytest.fixture()
def label_app(dev_config, tmp_path):
    import cv2

    frames = tmp_path / "frames"
    frames.mkdir(parents=True)
    coco_path = tmp_path / "annotations.json"

    store = CocoStore.create(domain_categories(dev_config.policy.domain_classes))
    for si in range(2):
        for fi in range(3):
            name = f"s{si}_f{fi}.jpg"
            cv2.imwrite(str(frames / name), np.full((48, 64, 3), 40 * (fi + 1), np.uint8))
            store.add_image(
                file_name=name, width=64, height=48, session_id=f"sess{si}",
                provenance={"source_clip": f"clip{si}.webm", "timestamp_s": float(fi),
                            "camera_device": "cam-A", "condition_tags": ["cabin"]},
            )
    store.save(coco_path)

    cfg = dev_config.model_copy(
        update={
            "dataset": dev_config.dataset.model_copy(
                update={"root": tmp_path, "coco_path": coco_path,
                        "splits_path": tmp_path / "splits.json", "frames_dirname": "frames"}
            ),
            "eval": dev_config.eval.model_copy(update={"results_dir": tmp_path / "results"}),
        }
    )
    app = create_app(cfg, start_loop=False)
    with TestClient(app) as client:
        yield client


def test_frames_list_and_progress_start_empty(label_app) -> None:
    r = label_app.get("/api/labels/frames")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 6 and body["labelled"] == 0
    assert {f["session_id"] for f in body["frames"]} == {"sess0", "sess1"}

    p = label_app.get("/api/labels/progress").json()
    assert p["images"] == 6 and p["labelled"] == 0 and p["sessions"] == 2
    assert p["min_unseeded_fraction"] == 0.30


def test_post_boxes_round_trip_and_progress_updates(label_app) -> None:
    first = label_app.get("/api/labels/frames").json()["frames"][0]
    iid = first["image_id"]

    save = label_app.post(
        f"/api/labels/frame/{iid}",
        json={"boxes": [{"category": "person", "bbox": [1, 2, 20, 30]},
                        {"category": "cup", "bbox": [5, 5, 8, 8]}],
              "seeded": True, "done": True},
    )
    assert save.status_code == 200, save.text
    saved = save.json()
    assert saved["labelled"] is True and saved["seeded"] is True
    assert len(saved["boxes"]) == 2
    assert saved["provenance"]["source_clip"].startswith("clip")

    again = label_app.get(f"/api/labels/frame/{iid}").json()
    labels = sorted(b["category"] for b in again["boxes"])
    assert labels == ["cup", "person"]
    assert again["boxes"][0]["bbox"] == [1.0, 2.0, 20.0, 30.0]

    p = label_app.get("/api/labels/progress").json()
    assert p["labelled"] == 1 and p["seeded"] == 1 and p["unseeded"] == 0
    assert p["per_class"]["person"] == 1 and p["per_class"]["cup"] == 1
    assert p["meets_unseeded_target"] is False  # the only labelled frame is seeded


def test_relabel_replaces_boxes_and_seeded_flag(label_app) -> None:
    iid = label_app.get("/api/labels/frames").json()["frames"][1]["image_id"]
    label_app.post(f"/api/labels/frame/{iid}",
                   json={"boxes": [{"category": "person", "bbox": [0, 0, 10, 10]}],
                         "seeded": True, "done": True})
    label_app.post(f"/api/labels/frame/{iid}",
                   json={"boxes": [{"category": "bottle", "bbox": [1, 1, 5, 5]}],
                         "seeded": False, "done": True})
    frame = label_app.get(f"/api/labels/frame/{iid}").json()
    assert [b["category"] for b in frame["boxes"]] == ["bottle"]
    assert frame["seeded"] is False


def test_image_bytes_and_bad_category_and_404(label_app) -> None:
    iid = label_app.get("/api/labels/frames").json()["frames"][0]["image_id"]
    img = label_app.get(f"/api/labels/image/{iid}")
    assert img.status_code == 200 and img.headers["content-type"].startswith("image/")

    bad = label_app.post(f"/api/labels/frame/{iid}",
                         json={"boxes": [{"category": "unicorn", "bbox": [0, 0, 5, 5]}],
                               "seeded": False, "done": True})
    assert bad.status_code == 400

    assert label_app.get("/api/labels/frame/999999").status_code == 404


def test_eval_summary_reports_not_run(label_app) -> None:
    r = label_app.get("/api/labels/eval-summary").json()
    assert r["available"] is False and "eval" in r["reason"].lower()
