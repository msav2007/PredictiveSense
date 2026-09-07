"""Generate a tiny, deterministic test clip.

A moving white rectangle on a black field: 12 frames, 160x120, 10 fps (1.2 s).
The pixel content of frame *n* is fully determined, and the file is written with
a stateless per-frame codec (MJPG in AVI) so regenerating it is reproducible.

Not committed: ``tests/fixtures/*.avi`` is git-ignored. ``conftest`` regenerates
it on demand.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

__all__ = ["FIXTURE_NAME", "SPEC", "make_fixture_video"]

FIXTURE_NAME = "fixture_clip.avi"


class _Spec:
    frames = 12
    width = 160
    height = 120
    fps = 10.0
    rect = 24  # rectangle side in pixels


SPEC = _Spec()


def _frame(index: int) -> np.ndarray:
    img = np.zeros((SPEC.height, SPEC.width, 3), dtype=np.uint8)
    step = (SPEC.width - SPEC.rect) // max(SPEC.frames - 1, 1)
    x = index * step
    y = SPEC.height // 2 - SPEC.rect // 2
    img[y : y + SPEC.rect, x : x + SPEC.rect] = 255
    return img


def make_fixture_video(path: str | Path) -> Path:
    """Write the fixture clip to ``path`` (overwriting). Returns the path."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out),
        cv2.VideoWriter_fourcc(*"MJPG"),
        SPEC.fps,
        (SPEC.width, SPEC.height),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open a VideoWriter for the fixture clip")
    try:
        for i in range(SPEC.frames):
            writer.write(_frame(i))
    finally:
        writer.release()
    if not out.is_file() or out.stat().st_size == 0:
        raise RuntimeError(f"fixture clip not written: {out}")
    return out


if __name__ == "__main__":  # manual regeneration
    dest = Path(__file__).with_name(FIXTURE_NAME)
    print(make_fixture_video(dest))  # noqa: T201 - CLI helper, not library code
