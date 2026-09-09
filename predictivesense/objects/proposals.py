"""Class-agnostic bounding-box proposal from raw detector output (Phase 7).

The bulk-upload workflow needs *a rectangle* per uploaded image, not a
classification. This module turns the detector's **raw** detections (taken
before the recognition policy - the policy exists to decide what to show during
monitoring and would suppress exactly the box we want; the bypass is deliberate
and recorded in ``docs/decisions.md``) into one proposed box:

* The sample's class is always the selected object. The detector's predicted
  label is kept only as a hint (``raw_class`` / ``score``) and is **never** used
  as the sample's class.
* Selection rule: highest score above ``min_score``; ties broken by larger area,
  then by the box centre closest to the image centre.
* When the detector returns nothing usable the image is marked
  ``manual_required`` and seeded with the same centred default box the camera
  path uses, for the developer to drag.

Stdlib + numpy only (matches the rest of ``predictivesense/objects/``); this
module never imports the detector, onnxruntime or cv2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

__all__ = [
    "BoxProposal",
    "centred_default_box",
    "propose_box",
]

# Matches static/studio/box-editor.js centre(): a 40% box in the middle.
_DEFAULT_FRAC_XY = 0.3
_DEFAULT_FRAC_WH = 0.4


def centred_default_box(img_w: int, img_h: int) -> list[float]:
    """The centred default ``[x, y, w, h]`` the camera path also seeds."""

    w = max(1.0, float(img_w))
    h = max(1.0, float(img_h))
    return [
        round(w * _DEFAULT_FRAC_XY, 2),
        round(h * _DEFAULT_FRAC_XY, 2),
        round(w * _DEFAULT_FRAC_WH, 2),
        round(h * _DEFAULT_FRAC_WH, 2),
    ]


@dataclass(frozen=True)
class BoxProposal:
    """One proposed box for a staged image.

    ``box`` is ``[x, y, w, h]`` in pixels, always populated (a centred default
    when nothing was detected). ``source`` is ``"detector"`` |
    ``"default_centred"``. ``raw_class`` / ``score`` are the detector's hint and
    are never the sample's class. ``status`` is ``"ready"`` |
    ``"manual_required"``.
    """

    box: list[float]
    source: str
    status: str
    raw_class: str | None = None
    score: float | None = None
    alternates: list[dict[str, Any]] = field(default_factory=list)


def _as_candidate(det: Any) -> tuple[tuple[float, float, float, float], str | None, float] | None:
    """Normalise a raw detection (a ``Detection`` model or a plain dict) to
    ``(xyxy, raw_class, score)``. Returns ``None`` if it has no usable box."""

    if isinstance(det, dict):
        bbox = det.get("bbox") or det.get("box")
        raw_class = det.get("raw_class_name") or det.get("class_name") or det.get("raw_class")
        score = det.get("score")
    else:
        bbox = getattr(det, "bbox", None)
        raw_class = getattr(det, "raw_class_name", None) or getattr(det, "class_name", None)
        score = getattr(det, "score", None)
    if bbox is None or len(tuple(bbox)) != 4 or score is None:
        return None
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2), (str(raw_class) if raw_class else None), float(score)


def _xyxy_to_xywh(xyxy: tuple[float, float, float, float], img_w: int, img_h: int) -> list[float]:
    x1, y1, x2, y2 = xyxy
    x = max(0.0, min(x1, float(img_w)))
    y = max(0.0, min(y1, float(img_h)))
    w = max(1.0, min(x2, float(img_w)) - x)
    h = max(1.0, min(y2, float(img_h)) - y)
    return [round(x, 2), round(y, 2), round(w, 2), round(h, 2)]


def propose_box(
    raw_detections: Iterable[Any],
    img_w: int,
    img_h: int,
    *,
    min_score: float = 0.10,
    max_alternates: int = 3,
) -> BoxProposal:
    """Pick one box for an image from the detector's raw detections.

    Class-agnostic: any raw class (including an ``implausible``-tier one such as
    ``donut`` for a watch) yields a box - the recognition policy is not
    consulted here.
    """

    cx, cy = img_w / 2.0, img_h / 2.0
    scored: list[tuple[float, float, float, tuple, str | None, float]] = []
    for det in raw_detections or ():
        cand = _as_candidate(det)
        if cand is None:
            continue
        xyxy, raw_class, score = cand
        if score < min_score:
            continue
        area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
        bcx = (xyxy[0] + xyxy[2]) / 2.0
        bcy = (xyxy[1] + xyxy[3]) / 2.0
        centre_dist = ((bcx - cx) ** 2 + (bcy - cy) ** 2) ** 0.5
        # sort key: score desc, area desc, centre distance asc
        scored.append((-score, -area, centre_dist, xyxy, raw_class, score))

    if not scored:
        return BoxProposal(
            box=centred_default_box(img_w, img_h),
            source="default_centred",
            status="manual_required",
        )

    scored.sort(key=lambda t: (t[0], t[1], t[2]))
    _, _, _, best_xyxy, best_class, best_score = scored[0]
    alternates = [
        {
            "box": _xyxy_to_xywh(row[3], img_w, img_h),
            "raw_class": row[4],
            "score": round(row[5], 4),
        }
        for row in scored[1 : 1 + max_alternates]
    ]
    return BoxProposal(
        box=_xyxy_to_xywh(best_xyxy, img_w, img_h),
        source="detector",
        status="ready",
        raw_class=best_class,
        score=round(best_score, 4),
        alternates=alternates,
    )
