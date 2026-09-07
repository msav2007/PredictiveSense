"""A fake ``cv2.VideoCapture`` for DeviceSource tests (no real camera)."""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["FakeCap", "factory"]


class FakeCap:
    """Minimal stand-in for ``cv2.VideoCapture``."""

    def __init__(self, frames: int, *, opens: bool = True) -> None:
        self._opens = opens
        self._frames_left = frames
        self.props: dict[int, float] = {}

    def isOpened(self) -> bool:  # noqa: N802 - cv2 API name
        return self._opens

    def set(self, prop: int, value: float) -> bool:
        self.props[prop] = value
        return True

    def get(self, prop: int) -> float:
        return {
            cv2.CAP_PROP_FRAME_WIDTH: 64.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 48.0,
            cv2.CAP_PROP_FPS: 30.0,
        }.get(prop, self.props.get(prop, 0.0))

    def read(self):
        if self._frames_left > 0:
            self._frames_left -= 1
            return True, np.zeros((48, 64, 3), dtype=np.uint8)
        return False, None

    def release(self) -> None:
        self._opens = False


def factory(frame_plan: list[int], *, opens: bool = True, sink: list | None = None):
    """VideoCapture stand-in factory; the nth instance yields ``frame_plan[n]``.

    Instances past the end of the plan repeat its last entry. Pass ``sink`` to
    collect every created :class:`FakeCap`.
    """

    calls = {"n": 0}

    def make(index, backend=None):  # noqa: ANN001 - cv2 signature
        i = calls["n"]
        calls["n"] += 1
        frames = frame_plan[i] if i < len(frame_plan) else (frame_plan[-1] if frame_plan else 0)
        cap = FakeCap(frames, opens=opens)
        if sink is not None:
            sink.append(cap)
        return cap

    make.calls = calls
    return make
