"""RecordedDriver: every frame processed, zero drops, byte-deterministic JSONL."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from predictivesense.api.app import create_app
from predictivesense.config.settings import load_config
from predictivesense.pipeline.recorded import RecordedDriver

pytestmark = pytest.mark.integration


def test_processes_every_frame_with_no_drops(fixture_video, fixture_video_frames, tmp_path) -> None:
    driver = RecordedDriver(results_dir=tmp_path)
    result = driver.run(fixture_video, replay_mode="asfast", run_id="run-a")

    assert result["frames"] == fixture_video_frames
    lines = Path(result["jsonl_path"]).read_text(encoding="utf-8").splitlines()
    assert len(lines) == fixture_video_frames  # one line per decoded frame, nothing dropped

    rows = [json.loads(line) for line in lines]
    assert [r["frame_id"] for r in rows] == list(range(fixture_video_frames))
    for r in rows:
        assert set(r) == {"frame_id", "capture_ts", "pts_s", "detections", "poses", "tracks"}
        assert r["detections"] == [] and r["poses"] == [] and r["tracks"] == []
        assert r["capture_ts"] == r["pts_s"]
    caps = [r["capture_ts"] for r in rows]
    assert caps == sorted(caps)


def test_two_runs_produce_byte_identical_jsonl(fixture_video, tmp_path) -> None:
    d1 = RecordedDriver(results_dir=tmp_path / "a")
    d2 = RecordedDriver(results_dir=tmp_path / "b")
    r1 = d1.run(fixture_video, replay_mode="asfast", run_id="x")
    r2 = d2.run(fixture_video, replay_mode="asfast", run_id="x")

    assert Path(r1["jsonl_path"]).read_bytes() == Path(r2["jsonl_path"]).read_bytes()


def test_realtime_and_asfast_jsonl_match(fixture_video, tmp_path) -> None:
    fast = RecordedDriver(results_dir=tmp_path / "fast").run(
        fixture_video, replay_mode="asfast", run_id="r"
    )
    rt = RecordedDriver(results_dir=tmp_path / "rt").run(
        fixture_video, replay_mode="realtime", run_id="r"
    )
    assert Path(fast["jsonl_path"]).read_bytes() == Path(rt["jsonl_path"]).read_bytes()


def test_manifest_written_with_frame_count(fixture_video, tmp_path) -> None:
    result = RecordedDriver(results_dir=tmp_path).run(fixture_video, run_id="m")
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["frames"] == result["frames"]
    assert manifest["kind"] == "recorded"
    assert manifest["replay_mode"] == "asfast"
    assert "git_commit" in manifest


def test_api_videos_and_analyze_roundtrip(fixture_video, fixture_video_frames, tmp_path) -> None:
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    shutil.copy(fixture_video, videos_dir / "clip.avi")

    cfg = load_config("eval")
    cfg = cfg.model_copy(
        update={
            "results_dir": tmp_path / "results",
            "video": cfg.video.model_copy(update={"input_dir": videos_dir}),
        }
    )
    app = create_app(cfg, start_loop=False)
    with TestClient(app) as client:
        listing = client.get("/api/videos").json()
        assert [row["path"] for row in listing] == ["clip.avi"]

        resp = client.post(
            "/api/analyze", json={"path": "clip.avi", "replay_mode": "asfast"}
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["frames"] == fixture_video_frames
        assert Path(body["jsonl_path"]).is_file()

        # path traversal is refused
        bad = client.post("/api/analyze", json={"path": "../../etc/passwd"})
        assert bad.status_code == 400
