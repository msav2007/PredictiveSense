"""CaptureConfig: Phase 1 / 1.5 camera-tuning fields validate as intended."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from predictivesense.config.settings import CaptureConfig, load_config

pytestmark = pytest.mark.unit

_COMBOS_FILE = (
    Path(__file__).resolve().parents[2] / "config" / "benchmarked_camera_combos.json"
)


def test_defaults_are_the_measured_choices() -> None:
    cap = CaptureConfig()
    assert cap.fourcc == "auto"        # forcing MJPG gave no measured benefit
    assert cap.buffer_size == 1        # newest-frame-only
    assert cap.warmup_frames == 0
    assert cap.device_backend == "auto"
    assert cap.max_ws_buffered_bytes == 1_000_000  # worker backpressure ceiling


@pytest.mark.parametrize("value", [0, -1])
def test_max_ws_buffered_bytes_must_be_positive(value: int) -> None:
    with pytest.raises(ValidationError):
        CaptureConfig(max_ws_buffered_bytes=value)


@pytest.mark.parametrize("value", ["auto", "MJPG", "YUY2", "H264"])
def test_valid_fourcc_accepted(value: str) -> None:
    assert CaptureConfig(fourcc=value).fourcc == value


@pytest.mark.parametrize("value", ["MJP", "MOTIONJPEG", ""])
def test_invalid_fourcc_rejected(value: str) -> None:
    with pytest.raises(ValidationError):
        CaptureConfig(fourcc=value)


@pytest.mark.parametrize("value", [0, -1])
def test_buffer_size_must_be_at_least_one(value: int) -> None:
    with pytest.raises(ValidationError):
        CaptureConfig(buffer_size=value)


def test_negative_warmup_rejected() -> None:
    with pytest.raises(ValidationError):
        CaptureConfig(warmup_frames=-1)


def test_shipped_profiles_carry_the_new_keys() -> None:
    for name in ("dev", "eval"):
        cap = load_config(name).capture
        assert cap.fourcc == "auto"
        assert cap.buffer_size == 1
        assert cap.warmup_frames == 0
        assert cap.max_ws_buffered_bytes == 1_000_000


def test_shipped_defaults_are_benchmarked() -> None:
    """Every shipped capture default must be a point the Phase 1.5 sweeps covered.

    Guards Requirement 3.3 #17: a future edit that moves a default off the
    measured grid (config/benchmarked_camera_combos.json) fails here.
    """

    combos = json.loads(_COMBOS_FILE.read_text(encoding="utf-8"))
    cam = combos["camera_matrix"]
    ap = combos["analysis_path"]

    for name in ("dev", "eval"):
        cap = load_config(name).capture
        assert [cap.request_width, cap.request_height] in cam["resolutions"], name
        assert cap.request_fps in cam["request_fps"], name
        assert cap.device_backend in cam["device_backend"], name
        assert cap.fourcc in cam["fourcc"], name

        assert cap.analysis_fps in ap["analysis_fps"], name
        assert [cap.analysis_width, cap.analysis_height] in ap["analysis_sizes"], name
        assert cap.analysis_jpeg_quality in ap["analysis_jpeg_quality"], name
