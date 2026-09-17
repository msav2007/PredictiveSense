"""RecordedDriver stays byte-deterministic once the model is in the loop.

Two runs over the same clip with the same weights + config must produce
byte-identical JSONL, detections included. ``models``-marked: skips cleanly when
``models/`` is empty.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from predictivesense.perception.engine import build_perception
from predictivesense.perception.policy import RecognitionPolicy
from predictivesense.pipeline.recorded import RecordedDriver
from predictivesense.tracking import Tracker

pytestmark = [pytest.mark.integration, pytest.mark.models]


@pytest.fixture()
def engine(require_models, perception_dev_config):
    return build_perception(perception_dev_config, strict=True, warmup=False)


def test_two_runs_with_policy_are_byte_identical(engine, perception_dev_config, fixture_video, tmp_path) -> None:
    """BLOCK 12.8: perception + policy is deterministic run-to-run."""

    policy = RecognitionPolicy(perception_dev_config.policy)
    r1 = RecordedDriver(results_dir=tmp_path / "a").run(
        fixture_video, replay_mode="asfast", run_id="p", perception=engine, policy=policy
    )
    r2 = RecordedDriver(results_dir=tmp_path / "b").run(
        fixture_video, replay_mode="asfast", run_id="p", perception=engine, policy=policy
    )
    assert Path(r1["jsonl_path"]).read_bytes() == Path(r2["jsonl_path"]).read_bytes()
    # policy-annotated lines carry the additive fields; a no-policy run does not
    body = Path(r1["jsonl_path"]).read_text(encoding="utf-8")
    if '"detections":[{' in body.replace(" ", ""):
        assert "policy_state" in body


def test_two_runs_with_tracker_are_byte_identical(
    engine, perception_dev_config, fixture_video, tmp_path
) -> None:
    """Phase 9 section 13.2: one tracker implementation, fed the file's own PTS
    (never wall time) - two runs of the same clip/config/model/tracker produce
    identical track ids and states, byte for byte."""

    policy = RecognitionPolicy(perception_dev_config.policy)
    r1 = RecordedDriver(results_dir=tmp_path / "a").run(
        fixture_video, replay_mode="asfast", run_id="t",
        perception=engine, policy=policy, tracker=Tracker(perception_dev_config.tracking),
    )
    r2 = RecordedDriver(results_dir=tmp_path / "b").run(
        fixture_video, replay_mode="asfast", run_id="t",
        perception=engine, policy=policy, tracker=Tracker(perception_dev_config.tracking),
    )
    assert Path(r1["jsonl_path"]).read_bytes() == Path(r2["jsonl_path"]).read_bytes()


def test_tracks_are_shaped_and_bounded(
    engine, perception_dev_config, fixture_video, tmp_path
) -> None:
    policy = RecognitionPolicy(perception_dev_config.policy)
    result = RecordedDriver(results_dir=tmp_path).run(
        fixture_video, replay_mode="asfast", run_id="t",
        perception=engine, policy=policy, tracker=Tracker(perception_dev_config.tracking),
    )
    lines = Path(result["jsonl_path"]).read_text(encoding="utf-8").splitlines()
    expected_keys = {
        "track_id", "status", "class_name", "bbox", "observed_class", "track_class",
        "class_votes", "velocity", "fresh", "age_frames", "age_ms", "hits",
        "consecutive_misses", "last_detector_confidence",
        "last_detector_confidence_age_ms", "policy_state", "tier",
        # Phase 10 section 5: pose bound to the track, additive.
        "pose_keypoints", "pose_frame_id", "pose_age_ms", "pose_fresh",
    }
    for line in lines:
        row = json.loads(line)
        for tr in row["tracks"]:
            assert set(tr) == expected_keys
            assert tr["status"] in ("tentative", "confirmed", "coasting")
            assert len(tr["bbox"]) == 4
            conf = tr["last_detector_confidence"]
            assert conf is None or 0.0 <= conf <= 1.0
            assert tr["pose_age_ms"] >= 0.0
            for kp in tr["pose_keypoints"]:
                assert len(kp) == 3


def test_two_runs_with_models_are_byte_identical(engine, fixture_video, tmp_path) -> None:
    r1 = RecordedDriver(results_dir=tmp_path / "a").run(
        fixture_video, replay_mode="asfast", run_id="m", perception=engine
    )
    r2 = RecordedDriver(results_dir=tmp_path / "b").run(
        fixture_video, replay_mode="asfast", run_id="m", perception=engine
    )
    assert Path(r1["jsonl_path"]).read_bytes() == Path(r2["jsonl_path"]).read_bytes()


def test_detections_are_shaped_and_bounded(engine, fixture_video, fixture_video_frames, tmp_path) -> None:
    result = RecordedDriver(results_dir=tmp_path).run(
        fixture_video, replay_mode="asfast", run_id="m", perception=engine
    )
    lines = Path(result["jsonl_path"]).read_text(encoding="utf-8").splitlines()
    assert len(lines) == fixture_video_frames
    for line in lines:
        row = json.loads(line)
        assert set(row) == {"frame_id", "capture_ts", "pts_s", "detections", "poses", "tracks"}
        for det in row["detections"]:
            assert set(det) == {"bbox", "class_id", "class_name", "score"}
            assert 0.0 <= det["score"] <= 1.0
            assert len(det["bbox"]) == 4
        for pose in row["poses"]:
            assert set(pose) == {"bbox", "score", "keypoints"}
            assert len(pose["keypoints"]) == 17


def test_person_detected_on_the_real_clip(require_models, perception_dev_config, tmp_path) -> None:
    """The developer's recorded clip under data/raw/ has a person in frame."""

    raw = Path(__file__).resolve().parents[1].parent / "data" / "raw"
    clips = sorted(raw.rglob("*.webm")) + sorted(raw.rglob("*.mp4"))
    if not clips:
        pytest.skip("no recorded clip under data/raw/")

    engine = build_perception(perception_dev_config, strict=True, warmup=False)
    result = RecordedDriver(results_dir=tmp_path).run(
        clips[0], replay_mode="asfast", run_id="real", perception=engine
    )
    names = set()
    for line in Path(result["jsonl_path"]).read_text(encoding="utf-8").splitlines():
        for det in json.loads(line)["detections"]:
            names.add(det["class_name"])
    assert "person" in names
