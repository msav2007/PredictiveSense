"""Association cost function and greedy matcher (Phase 9 section 5.3).

A candidate (track, detection) pair is scored on five signals - IoU, centre
distance, box-size ratio, motion consistency with the constant-velocity
prediction, and class agreement - each weighted and summed to one score in
``[0, 1]``. Class agreement is deliberately the smallest-weighted signal and
is gated out entirely for a geometrically implausible pair (section 5.4): two
same-class boxes on opposite sides of the frame must never match on class
alone.

The matcher itself is a simple greedy assignment (highest score first,
skipping any track/detection already claimed) rather than an exact Hungarian
solve - deterministic given a stable sort key, `O(n*m log(n*m))`, and adds no
dependency (no scipy/`lap`) per section 5.2 / section 11.5.
"""

from __future__ import annotations

from dataclasses import dataclass

from predictivesense.config.settings import TrackingConfig
from predictivesense.core.types import BBox

from predictivesense.tracking import geometry

__all__ = ["AssociationCandidate", "score_pair", "greedy_match"]


@dataclass(frozen=True)
class AssociationCandidate:
    """One trackable side of an association: the box used for geometry (a
    track's *predicted* box, or a ghost's extrapolated box), its last-known
    (pre-prediction) box for the centre-distance signal, and its class for
    the class-agreement signal."""

    key: int  # track_id (or ghost id, same namespace)
    predicted_bbox: BBox
    last_bbox: BBox
    track_class: str


def score_pair(
    cand: AssociationCandidate,
    det_bbox: BBox,
    det_class: str,
    cfg: TrackingConfig,
) -> float | None:
    """Combined association score in ``[0, 1]``, or ``None`` if the pair is
    geometrically ineligible (section 5.4 - class agreement cannot rescue a
    geometrically implausible pair)."""

    iou_score = geometry.iou(cand.predicted_bbox, det_bbox)
    center_score = geometry.center_distance_score(cand.last_bbox, det_bbox)
    if iou_score <= 0.0 and center_score <= 0.5:
        return None

    size_score = geometry.size_ratio_score(cand.last_bbox, det_bbox)
    motion_score = geometry.center_distance_score(cand.predicted_bbox, det_bbox)
    class_score = 1.0 if det_class and det_class == cand.track_class else 0.0

    weights = (
        cfg.assoc_iou_weight,
        cfg.assoc_center_distance_weight,
        cfg.assoc_size_ratio_weight,
        cfg.assoc_motion_weight,
        cfg.assoc_class_weight,
    )
    values = (iou_score, center_score, size_score, motion_score, class_score)
    total_weight = sum(weights)
    if total_weight <= 0.0:
        return 0.0
    return sum(w * v for w, v in zip(weights, values)) / total_weight


def greedy_match(
    candidates: list[AssociationCandidate],
    det_boxes: list[BBox],
    det_classes: list[str],
    cfg: TrackingConfig,
    *,
    min_score: float,
) -> tuple[dict[int, int], set[int]]:
    """Greedy highest-score-first assignment.

    Returns ``(track_key -> detection_index, matched_detection_indices)``.
    Ties broken by ``track.key`` then detection index for determinism.
    """

    scored: list[tuple[float, int, int]] = []
    for cand in candidates:
        for di, (box, cls) in enumerate(zip(det_boxes, det_classes)):
            s = score_pair(cand, box, cls, cfg)
            if s is not None and s >= min_score:
                scored.append((s, cand.key, di))

    scored.sort(key=lambda t: (-t[0], t[1], t[2]))

    matched_tracks: dict[int, int] = {}
    matched_dets: set[int] = set()
    used_tracks: set[int] = set()
    for score, key, di in scored:
        if key in used_tracks or di in matched_dets:
            continue
        used_tracks.add(key)
        matched_dets.add(di)
        matched_tracks[key] = di
    return matched_tracks, matched_dets
