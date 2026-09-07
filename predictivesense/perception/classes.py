"""Class-id -> name map, the user-facing alias map, and the pose skeleton.

Defined once and reused by the detector, the pose estimator and the overlay.
The COCO 80-class list is the default for a YOLO-family detector; a different
model supplies its own list through ``config.perception.detector`` (Phase 2 keeps
the default and does not wire a per-model override file - see
``docs/decisions.md``). ``ALIAS_MAP`` maps the raw model label to the string the
UI shows; every alias target is distinct.
"""

from __future__ import annotations

__all__ = [
    "COCO_CLASSES",
    "REQUIRED_CLASSES",
    "ALIAS_MAP",
    "alias_for",
    "KEYPOINT_NAMES",
    "SKELETON_EDGES",
    "class_color",
]

# Standard COCO 2017 detection classes, index == class id.
COCO_CLASSES: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)

# The classes the class-coverage audit reports on (blueprint addendum v3 §C).
REQUIRED_CLASSES: tuple[str, ...] = (
    "person", "cup", "bottle", "laptop", "chair", "backpack", "handbag",
    "book", "cell phone", "keyboard", "mouse", "scissors",
)

# Raw model label -> user-facing label. Only entries that differ from a plain
# title-case are listed; every value here is unique (asserted in test_classes).
ALIAS_MAP: dict[str, str] = {
    "cell phone": "Phone",
    "tv": "TV / monitor",
    "couch": "Sofa",
    "potted plant": "Plant",
    "dining table": "Table",
    "wine glass": "Wine glass",
    "sports ball": "Ball",
    "hair drier": "Hair dryer",
    "remote": "Remote control",
    "mouse": "Computer mouse",
}


def alias_for(class_name: str) -> str:
    """User-facing label for a raw model class name."""

    if class_name in ALIAS_MAP:
        return ALIAS_MAP[class_name]
    return class_name[:1].upper() + class_name[1:]


# COCO 17-keypoint pose, index == keypoint id.
KEYPOINT_NAMES: tuple[str, ...] = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)

# Skeleton edges as (keypoint_index, keypoint_index) pairs (Ultralytics order).
SKELETON_EDGES: tuple[tuple[int, int], ...] = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6),
)


def class_color(class_id: int) -> tuple[int, int, int]:
    """Deterministic, well-spread RGB for a class id (golden-ratio hue walk)."""

    hue = (class_id * 0.61803398875) % 1.0
    return _hsv_to_rgb(hue, 0.65, 0.95)


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    r, g, b = (
        (v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)
    )[i % 6]
    return round(r * 255), round(g * 255), round(b * 255)
