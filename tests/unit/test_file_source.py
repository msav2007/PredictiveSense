"""FileSource: frame count/order, pts-derived capture_ts, replay-mode parity."""

from __future__ import annotations

import pytest

from predictivesense.camera.file_source import FileSource

pytestmark = pytest.mark.unit


def _drain(src: FileSource) -> list:
    frames = []
    while True:
        frame = src.read()
        if frame is None:
            break
        frames.append(frame)
    return frames


def test_replays_every_frame_in_order(fixture_video, fixture_video_frames) -> None:
    with FileSource(fixture_video, replay_mode="asfast") as src:
        frames = _drain(src)

    assert len(frames) == fixture_video_frames
    assert [f.frame_id for f in frames] == list(range(fixture_video_frames))
    assert [f.seq for f in frames] == list(range(fixture_video_frames))


def test_capture_ts_derives_from_pts_and_is_non_decreasing(fixture_video) -> None:
    with FileSource(fixture_video, replay_mode="asfast") as src:
        caps = [f.capture_ts for f in _drain(src)]

    assert caps[0] == pytest.approx(0.0, abs=1e-6)
    assert caps == sorted(caps)
    assert not any(c < 0 for c in caps)
    # Not wall-clock: every value is a small file-relative offset.
    assert max(caps) < 10.0


def test_asfast_and_realtime_yield_identical_frame_sequences(fixture_video) -> None:
    with FileSource(fixture_video, replay_mode="asfast") as fast:
        fast_caps = [f.capture_ts for f in _drain(fast)]
    with FileSource(fixture_video, replay_mode="realtime") as rt:
        rt_caps = [f.capture_ts for f in _drain(rt)]

    assert fast_caps == rt_caps


def test_restart_is_deterministic(fixture_video) -> None:
    src = FileSource(fixture_video, replay_mode="asfast")
    src.start()
    first = [f.capture_ts for f in _drain(src)]
    src.restart()
    second = [f.capture_ts for f in _drain(src)]
    src.stop()

    assert first == second
    assert len(first) > 0


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        FileSource("tests/fixtures/does-not-exist.avi")


def test_bad_replay_mode_raises(fixture_video) -> None:
    with pytest.raises(ValueError):
        FileSource(fixture_video, replay_mode="turbo")
