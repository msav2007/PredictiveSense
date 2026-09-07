"""Hardware: open a real camera and read frames. Skipped unless --run-hardware."""

from __future__ import annotations

import pytest

from predictivesense.camera.device import DeviceSource
from predictivesense.config.settings import load_config

pytestmark = pytest.mark.hardware


def test_device_source_reads_thirty_frames() -> None:
    cap = load_config("dev").capture
    source = DeviceSource(
        cap.device_index,
        backend=cap.device_backend,
        request_width=cap.request_width,
        request_height=cap.request_height,
        request_fps=cap.request_fps,
        open_timeout_s=cap.open_timeout_s,
    )
    source.start()
    try:
        frames = 0
        for _ in range(200):  # allow for a few dropped reads
            frame = source.read()
            if frame is not None:
                frames += 1
            if frames >= 30:
                break
        assert frames >= 30
    finally:
        info = source.info()
        source.stop()

    assert info["backend"] in ("msmf", "dshow")
    assert info["achieved_width"] > 0 and info["achieved_height"] > 0
    print(  # noqa: T201 - hardware report is the point of this test
        f"\n[hardware] backend={info['backend']} "
        f"{info['achieved_width']}x{info['achieved_height']} "
        f"achieved_fps={info['achieved_fps']} read_failures={info['read_failures']}"
    )
