"""Letterbox is aspect-preserving and its coordinate mapping round-trips."""

from __future__ import annotations

import numpy as np
import pytest

from predictivesense.perception.preprocess import (
    PAD_VALUE,
    LetterboxResult,
    letterbox,
    scale_boxes_to_original,
    scale_points_to_original,
)

pytestmark = pytest.mark.unit

_ASPECTS = [(640, 480), (480, 640), (1920, 1080), (100, 400), (333, 777), (16, 9)]


@pytest.mark.parametrize("w,h", _ASPECTS)
@pytest.mark.parametrize("size", [320, 480, 640])
def test_letterbox_blob_shape_and_range(w: int, h: int, size: int) -> None:
    img = np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8)
    lb = letterbox(img, size)
    assert isinstance(lb, LetterboxResult)
    assert lb.blob.shape == (1, 3, size, size)
    assert lb.blob.dtype == np.float32
    assert lb.blob.flags["C_CONTIGUOUS"]
    assert 0.0 <= float(lb.blob.min()) and float(lb.blob.max()) <= 1.0


@pytest.mark.parametrize("w,h", _ASPECTS)
def test_letterbox_preserves_aspect_and_centres(w: int, h: int) -> None:
    size = 640
    img = np.full((h, w, 3), 200, dtype=np.uint8)
    lb = letterbox(img, size)

    ratio = min(size / h, size / w)
    new_w, new_h = round(w * ratio), round(h * ratio)
    # one axis fills the input, the other is padded symmetrically (+-1 for parity)
    assert new_w == size or new_h == size
    assert abs((size - new_w) - 2 * lb.pad_x) <= 1
    assert abs((size - new_h) - 2 * lb.pad_y) <= 1

    # the pad regions carry the constant pad colour
    canvas = (lb.blob[0].transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    if lb.pad_y > 0:
        assert np.all(canvas[: lb.pad_y] == PAD_VALUE)
    if lb.pad_x > 0:
        assert np.all(canvas[:, : lb.pad_x] == PAD_VALUE)


@pytest.mark.parametrize("w,h", _ASPECTS)
def test_box_corner_round_trip_within_tolerance(w: int, h: int) -> None:
    size = 640
    img = np.zeros((h, w, 3), dtype=np.uint8)
    lb = letterbox(img, size)

    # boxes in original pixels -> forward to letterbox space -> back
    rng = np.random.default_rng(h * 1000 + w)
    boxes = np.array(
        [
            [10.0, 10.0, min(w - 1.0, 100.0), min(h - 1.0, 80.0)],
            [w * 0.25, h * 0.25, w * 0.75, h * 0.9],
            [0.0, 0.0, float(w), float(h)],
        ]
    )
    forward = boxes.copy()
    forward[:, [0, 2]] = forward[:, [0, 2]] * lb.ratio + lb.pad_x
    forward[:, [1, 3]] = forward[:, [1, 3]] * lb.ratio + lb.pad_y

    back = scale_boxes_to_original(forward, lb, w, h)
    assert np.allclose(back, boxes, atol=1.0)


def test_points_round_trip() -> None:
    w, h, size = 800, 600, 640
    lb = letterbox(np.zeros((h, w, 3), np.uint8), size)
    pts = np.array([[[0.0, 0.0], [400.0, 300.0], [799.0, 599.0]]])
    fwd = pts.copy()
    fwd[..., 0] = fwd[..., 0] * lb.ratio + lb.pad_x
    fwd[..., 1] = fwd[..., 1] * lb.ratio + lb.pad_y
    back = scale_points_to_original(fwd, lb, w, h)
    assert np.allclose(back, pts, atol=1.0)


def test_letterbox_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        letterbox(np.zeros((10, 10), np.uint8), 640)
    with pytest.raises(ValueError):
        letterbox(np.zeros((10, 10, 3), np.uint8), 0)
