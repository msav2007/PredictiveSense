"""SyntheticSource: deterministic, strictly ordered, well-shaped, idempotent stop."""

from __future__ import annotations

import numpy as np
import pytest

from predictivesense.camera.source import create_frame_source
from predictivesense.camera.synthetic import SyntheticSource
from predictivesense.config.settings import SourceConfig
from predictivesense.core.enums import SourceKind

pytestmark = pytest.mark.unit

_FAST_FPS = 5000.0


def _source(seed: int = 42, w: int = 32, h: int = 24) -> SyntheticSource:
    return SyntheticSource(width=w, height=h, target_fps=_FAST_FPS, seed=seed)


def _read_many(src: SyntheticSource, n: int):
    return [src.read() for _ in range(n)]


def test_read_before_start_returns_none() -> None:
    src = _source()
    assert src.read() is None


def test_frame_id_strictly_increasing_without_gaps() -> None:
    with _source() as src:
        frames = _read_many(src, 50)
    assert [f.frame_id for f in frames] == list(range(50))
    assert [f.seq for f in frames] == list(range(50))


def test_capture_ts_strictly_increasing() -> None:
    with _source() as src:
        frames = _read_many(src, 80)
    ts = [f.capture_ts for f in frames]
    assert all(b > a for a, b in zip(ts, ts[1:]))


def test_image_shape_and_dtype() -> None:
    with _source(w=40, h=30) as src:
        frame = src.read()
    assert frame.image.shape == (30, 40, 3)
    assert frame.image.dtype == np.uint8
    assert frame.width == 40 and frame.height == 30


def test_same_seed_yields_identical_frames() -> None:
    with _source(seed=123) as a, _source(seed=123) as b:
        fa = _read_many(a, 25)
        fb = _read_many(b, 25)
    for x, y in zip(fa, fb):
        assert x.frame_id == y.frame_id
        assert x.source_id == y.source_id
        assert np.array_equal(x.image, y.image)


def test_different_seed_differs() -> None:
    with _source(seed=1) as a, _source(seed=2) as b:
        assert not np.array_equal(a.read().image, b.read().image)


def test_stop_is_idempotent_and_halts_reads() -> None:
    src = _source()
    src.start()
    assert src.read() is not None
    src.stop()
    src.stop()  # must not raise
    assert src.is_running is False
    assert src.read() is None


def test_factory_builds_synthetic_and_blocks_other_kinds() -> None:
    cfg = SourceConfig(kind=SourceKind.SYNTHETIC, width=16, height=16, target_fps=100.0, seed=0)
    assert isinstance(create_frame_source(cfg), SyntheticSource)

    for kind in (SourceKind.DEVICE, SourceKind.FILE, SourceKind.WEBRTC):
        bad = SourceConfig(kind=kind, width=16, height=16, target_fps=100.0, seed=0)
        with pytest.raises(NotImplementedError):
            create_frame_source(bad)


def test_info_reports_counters() -> None:
    with _source(seed=9) as src:
        _read_many(src, 5)
        info = src.info()
    assert info["kind"] == "synthetic"
    assert info["seed"] == 9
    assert info["frames_produced"] == 5
