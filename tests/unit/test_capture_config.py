"""CaptureConfig: Phase 1 camera-tuning fields validate as intended."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from predictivesense.config.settings import CaptureConfig, load_config

pytestmark = pytest.mark.unit


def test_defaults_are_the_measured_choices() -> None:
    cap = CaptureConfig()
    assert cap.fourcc == "auto"        # forcing MJPG gave no measured benefit
    assert cap.buffer_size == 1        # newest-frame-only
    assert cap.warmup_frames == 0
    assert cap.device_backend == "auto"


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
