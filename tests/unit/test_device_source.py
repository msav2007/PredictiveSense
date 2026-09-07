"""DeviceSource fast unit checks with a fake cv2.VideoCapture (no real camera).

Timing / threaded reconnect behaviour lives in
``tests/integration/test_device_reconnect.py``; the hardware-backed test lives in
``tests/hardware/test_device_source.py``.
"""

from __future__ import annotations

import cv2
import pytest

from predictivesense.camera.device import DeviceSource
from tests.fixtures.camfakes import FakeCap, factory

pytestmark = pytest.mark.unit


def test_bad_backend_rejected() -> None:
    with pytest.raises(ValueError):
        DeviceSource(0, backend="v4l2")


def test_stop_before_start_is_safe() -> None:
    src = DeviceSource(0)
    src.stop()
    src.stop()
    assert src.is_running is False
    assert src.read() is None


def test_start_raises_when_no_first_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cv2, "VideoCapture", factory([0, 0]))  # opens, never a frame
    src = DeviceSource(0, backend="msmf", open_timeout_s=0.2)
    with pytest.raises(RuntimeError):
        src.start()
    assert src.is_running is False


def test_info_reports_new_tuning_keys() -> None:
    info = DeviceSource(0, fourcc="auto", buffer_size=1, warmup_frames=2).info()
    for key in (
        "fourcc", "connected", "reconnect_attempts", "currently_down_s",
        "reconnects", "read_failures", "achieved_fps",
    ):
        assert key in info


def test_fourcc_auto_leaves_fourcc_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    made: list[FakeCap] = []
    monkeypatch.setattr(cv2, "VideoCapture", factory([5], sink=made))
    src = DeviceSource(0, backend="msmf", fourcc="auto", open_timeout_s=0.5)
    src.start()
    src.stop()
    assert made and cv2.CAP_PROP_FOURCC not in made[0].props


def test_fourcc_explicit_is_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    made: list[FakeCap] = []
    monkeypatch.setattr(cv2, "VideoCapture", factory([5], sink=made))
    src = DeviceSource(0, backend="dshow", fourcc="MJPG", open_timeout_s=0.5)
    src.start()
    src.stop()
    assert made and cv2.CAP_PROP_FOURCC in made[0].props
    assert made[0].props[cv2.CAP_PROP_BUFFERSIZE] == 1
