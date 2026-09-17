"""Internal mutable per-track bookkeeping and its conversion to the immutable,
public :class:`~predictivesense.core.types.Track` contract. Kept separate from
:mod:`tracker` so the (larger) lifecycle/association orchestration file stays
under the file-size guideline."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

from predictivesense.core.enums import TrackStatus
from predictivesense.core.types import BBox, Track

from predictivesense.tracking import geometry

__all__ = ["TrackState", "majority_class", "class_votes_tuple"]


def majority_class(history: deque) -> str:
    """Majority vote over a bounded class-name history. Ties broken by
    whichever tied class occurs *latest* in the history (section 5.4: the
    displayed label should track the most recent trend, not an arbitrary
    tie-break)."""

    if not history:
        return ""
    counts = Counter(history)
    best = max(counts.values())
    for cls in reversed(history):
        if counts[cls] == best:
            return cls
    return next(iter(history))  # unreachable, keeps type-checkers happy


def class_votes_tuple(history: deque) -> tuple[tuple[str, int], ...]:
    counts = Counter(history)
    return tuple(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


@dataclass
class TrackState:
    """Mutable working state for one track between :meth:`Tracker.update`
    calls. Converted to a frozen :class:`Track` for output every cycle."""

    track_id: int
    status: TrackStatus
    bbox: BBox
    velocity: tuple[float, float] = (0.0, 0.0)
    class_history: deque = field(default_factory=lambda: deque(maxlen=20))
    observed_class: str = ""
    hits: int = 0
    consecutive_misses: int = 0
    exited_border: bool = False
    created_ts: float = 0.0
    last_update_ts: float = 0.0
    first_seen_frame_id: int = 0
    last_detection_frame_id: int = 0
    last_detection_capture_ts: float = 0.0
    last_detector_confidence: float | None = None
    policy_state: str = ""
    tier: str = ""
    # Phase 10 section 5: the most recent pose bound to this track (empty
    # until a pose is ever bound). Persists across frames the tracker has no
    # pose input for - only cleared by aging out in to_track(), never wiped
    # on an empty-poses cycle (that is the point of binding it here instead
    # of reading straight off the per-frame poses list).
    pose_keypoints: tuple[tuple[float, float, float], ...] = ()
    pose_source_frame_id: int | None = None
    pose_capture_ts: float | None = None
    pose_is_fresh_binding: bool = False
    # Phase 11 Part B section 16: carried forward exactly like
    # last_detector_confidence - set only in record_hit(), never decayed,
    # never invented while coasting.
    custom_class_name: str | None = None
    custom_class_confidence: float | None = None

    def predicted_bbox(self, now_ts: float) -> BBox:
        dt = max(0.0, now_ts - self.last_update_ts)
        vx, vy = self.velocity
        return geometry.shift(self.bbox, vx * dt, vy * dt)

    def record_hit(
        self,
        *,
        bbox: BBox,
        raw_class: str,
        score: float,
        policy_state: str,
        tier: str,
        frame_id: int,
        capture_ts: float,
        custom_class_name: str | None = None,
        custom_class_confidence: float | None = None,
    ) -> None:
        dt = max(1e-6, capture_ts - self.last_update_ts)
        old_cx, old_cy = geometry.center(self.bbox)
        new_cx, new_cy = geometry.center(bbox)
        self.velocity = ((new_cx - old_cx) / dt, (new_cy - old_cy) / dt)
        self.bbox = bbox
        self.class_history.append(raw_class)
        self.observed_class = raw_class
        self.hits += 1
        self.consecutive_misses = 0
        self.exited_border = False
        self.last_update_ts = capture_ts
        self.last_detection_frame_id = frame_id
        self.last_detection_capture_ts = capture_ts
        self.last_detector_confidence = score
        self.policy_state = policy_state
        self.tier = tier
        self.custom_class_name = custom_class_name
        self.custom_class_confidence = custom_class_confidence

    def record_miss(self, *, predicted_bbox: BBox, now_ts: float) -> None:
        self.bbox = predicted_bbox
        self.consecutive_misses += 1
        self.last_update_ts = now_ts

    def record_pose(
        self,
        *,
        keypoints: tuple[tuple[float, float, float], ...],
        frame_id: int,
        capture_ts: float,
        fresh: bool,
    ) -> None:
        """Bind a pose observation to this track (section 5.3). Replaces
        whatever pose was previously bound; never mutates ``keypoints`` in
        place (the caller passes an immutable tuple straight from a frozen
        ``Pose``)."""

        self.pose_keypoints = keypoints
        self.pose_source_frame_id = frame_id
        self.pose_capture_ts = capture_ts
        self.pose_is_fresh_binding = fresh

    def to_track(self, *, now_ts: float, fresh: bool, pose_max_age_ms: float = 900.0) -> Track:
        track_class = majority_class(self.class_history)
        age_ms = max(0.0, (now_ts - self.created_ts) * 1000.0)
        conf_age_ms = (
            max(0.0, (now_ts - self.last_detection_capture_ts) * 1000.0)
            if self.last_detector_confidence is not None
            else 0.0
        )

        pose_keypoints: tuple[tuple[float, float, float], ...] = ()
        pose_frame_id: int | None = None
        pose_capture_ts: float | None = None
        pose_age_ms = 0.0
        pose_fresh = False
        if self.pose_capture_ts is not None:
            candidate_age_ms = max(0.0, (now_ts - self.pose_capture_ts) * 1000.0)
            if candidate_age_ms <= pose_max_age_ms:
                # Section 5.2: bounded by an explicit maximum pose age - an
                # aged-out pose is dropped entirely, never shown, rather than
                # silently frozen forever.
                pose_keypoints = self.pose_keypoints
                pose_frame_id = self.pose_source_frame_id
                pose_capture_ts = self.pose_capture_ts
                pose_age_ms = candidate_age_ms
                pose_fresh = self.pose_is_fresh_binding and candidate_age_ms <= 1e-6

        return Track(
            track_id=self.track_id,
            status=self.status,
            class_name=track_class or self.observed_class,
            bbox=self.bbox,
            last_seen_frame_id=self.last_detection_frame_id,
            observed_class=self.observed_class,
            track_class=track_class,
            class_votes=class_votes_tuple(self.class_history),
            velocity=self.velocity,
            fresh=fresh,
            age_frames=self.hits + self.consecutive_misses,
            age_ms=age_ms,
            hits=self.hits,
            consecutive_misses=self.consecutive_misses,
            first_seen_frame_id=self.first_seen_frame_id,
            last_detection_frame_id=self.last_detection_frame_id,
            last_detection_capture_ts=self.last_detection_capture_ts,
            last_detector_confidence=self.last_detector_confidence,
            last_detector_confidence_age_ms=conf_age_ms,
            policy_state=self.policy_state,
            tier=self.tier,
            pose_keypoints=pose_keypoints,
            pose_frame_id=pose_frame_id,
            pose_capture_ts=pose_capture_ts,
            pose_age_ms=pose_age_ms,
            pose_fresh=pose_fresh,
            custom_class_name=self.custom_class_name,
            custom_class_confidence=self.custom_class_confidence,
        )
