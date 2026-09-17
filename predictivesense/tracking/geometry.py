"""Pure bbox geometry helpers for the tracker. No I/O, no state, no imports
beyond the standard library - reused by association and lifecycle code so the
same definitions of "overlap", "centre" and "at the border" are used
everywhere (Phase 9 section 5.1: pure, dependency-free)."""

from __future__ import annotations

from predictivesense.core.types import BBox

__all__ = [
    "area",
    "center",
    "iou",
    "diagonal",
    "center_distance_score",
    "size_ratio_score",
    "shift",
    "is_off_frame",
]


def area(box: BBox) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def center(box: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def diagonal(box: BBox) -> float:
    x1, y1, x2, y2 = box
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return (w * w + h * h) ** 0.5


def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    union = area(a) + area(b) - inter
    return inter / union if union > 0.0 else 0.0


def center_distance_score(a: BBox, b: BBox) -> float:
    """1.0 when centres coincide, falling to 0.0 at one box diagonal apart
    (normalised by the larger of the two diagonals, so a tiny and a huge box
    are compared fairly)."""

    ax, ay = center(a)
    bx, by = center(b)
    dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    norm = max(diagonal(a), diagonal(b), 1e-6)
    return max(0.0, 1.0 - dist / norm)


def size_ratio_score(a: BBox, b: BBox) -> float:
    aa, ab = area(a), area(b)
    if aa <= 0.0 or ab <= 0.0:
        return 0.0
    return min(aa, ab) / max(aa, ab)


def shift(box: BBox, dx: float, dy: float) -> BBox:
    x1, y1, x2, y2 = box
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


def is_off_frame(box: BBox, *, frame_width: float, frame_height: float) -> bool:
    """True once the box centre has left the visible frame - used to
    distinguish a border exit from a mid-frame disappearance (section 6.3)."""

    cx, cy = center(box)
    return cx < 0.0 or cy < 0.0 or cx > frame_width or cy > frame_height
