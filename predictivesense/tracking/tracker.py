"""Multi-object tracker orchestration (Phase 9 sections 5-8).

:class:`Tracker` is the one entry point: construct it once per analysis run
(real-time loop or recorded driver - the same class serves both, section
13.1), then call :meth:`Tracker.update` once per analysed frame with that
frame's already policy-annotated detections. It performs no inference, no
I/O, and holds only its own in-memory state.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

from predictivesense.config.settings import TrackingConfig
from predictivesense.core.enums import TrackStatus
from predictivesense.core.types import BBox, Detection, Pose, Track

from predictivesense.tracking import geometry
from predictivesense.tracking.association import AssociationCandidate, greedy_match, score_pair
from predictivesense.tracking.track_state import TrackState, majority_class

# Person-class tracks are the only pose-binding candidates (section 5.3: a
# pose belongs to the person track its box best overlaps). Matches the
# detector's own class name for "person" - not configurable, mirrors the
# hard-coded skeleton/keypoint assumption already baked into the pose model.
_POSE_TRACK_CLASS = "person"

__all__ = ["Tracker", "TrackerStats", "TRACK_ELIGIBLE_STATES"]

# States a tracker may receive a box for - excludes rejected_size (policy's
# own judgement: too small / bad aspect) and suppressed_implausible
# (confidently an out-of-domain class). Neither is a candidate object; feeding
# them to the tracker would waste bounded track slots on noise the policy has
# already flagged (Phase 9 root-cause table: results/grey_state_frequency.md).
TRACK_ELIGIBLE_STATES = frozenset(
    {"accepted", "accepted_secondary", "unknown_low_confidence", "unknown_margin"}
)


@dataclass(frozen=True)
class TrackerStats:
    """Cumulative, reconciling counters for one :class:`Tracker` instance's
    lifetime - the tracking-layer analogue of
    :class:`~predictivesense.perception.policy.PolicyCounts`."""

    created: int = 0
    confirmed: int = 0
    expired: int = 0
    expired_border: int = 0
    reassociated: int = 0
    dropped_max_tracks: int = 0


@dataclass
class _Ghost:
    """A recently-expired, non-border track kept for a bounded window in case
    it returns (section 6.5). Carries the full :class:`TrackState` so a
    reclaim resumes with unbroken hit/class-vote history."""

    track_id: int
    expired_at_frame_id: int
    bbox: BBox
    velocity: tuple[float, float]
    last_update_ts: float
    state: TrackState


class Tracker:
    def __init__(self, config: TrackingConfig) -> None:
        self._cfg = config
        self._tracks: dict[int, TrackState] = {}
        self._ghosts: list[_Ghost] = []
        self._next_id = 1
        self._counts: dict[str, int] = dict(
            created=0, confirmed=0, expired=0, expired_border=0,
            reassociated=0, dropped_max_tracks=0,
        )

    @property
    def stats(self) -> TrackerStats:
        return TrackerStats(**self._counts)

    @property
    def active_track_count(self) -> int:
        return len(self._tracks)

    def update(
        self,
        detections: Sequence[Detection],
        *,
        frame_id: int,
        capture_ts: float,
        frame_width: float,
        frame_height: float,
        poses: Sequence[Pose] = (),
        pose_fresh: bool = True,
    ) -> list[Track]:
        """One tracking cycle. ``capture_ts`` is seconds (matches
        ``Frame.capture_ts`` elsewhere in the pipeline); velocities and
        ``max_age_ms`` are derived/compared in milliseconds internally.

        ``poses`` (Phase 10 section 5) are this frame's perception-engine pose
        output - empty when no pose ran/reused this cycle, in which case
        existing pose bindings are left untouched (they age via
        ``TrackState.to_track``, they are not wiped) rather than vanishing
        the instant the engine's own cadence bookkeeping comes up empty.
        ``pose_fresh`` says whether ``poses`` is a brand-new engine
        observation (``True``) or an engine-level reuse (``False``); ignored
        when ``poses`` is empty.
        """

        cfg = self._cfg
        eligible = [d for d in detections if d.policy_state in TRACK_ELIGIBLE_STATES]
        high = [d for d in eligible if d.score >= cfg.high_score_split]
        low = [d for d in eligible if d.score < cfg.high_score_split]

        matched_track_ids, stage1_dets = self._match_and_apply_hits(high, low, capture_ts, frame_id)
        self._handle_misses(matched_track_ids, capture_ts, frame_width, frame_height)
        new_hit_ids = self._reassociate_or_spawn(high, stage1_dets, frame_id, capture_ts)
        fresh_ids = matched_track_ids | new_hit_ids

        if poses:
            self._bind_poses(poses, capture_ts=capture_ts, fresh=pose_fresh)

        return sorted(
            (
                ts.to_track(
                    now_ts=capture_ts, fresh=(tid in fresh_ids),
                    pose_max_age_ms=cfg.pose_max_age_ms,
                )
                for tid, ts in self._tracks.items()
            ),
            key=lambda t: t.track_id,
        )

    # -- pose binding (section 5) -------------------------------------------

    def _bind_poses(self, poses: Sequence[Pose], *, capture_ts: float, fresh: bool) -> None:
        """Bind each pose to the person track whose box it overlaps best
        (highest IoU, at or above ``pose_bind_min_iou``). A pose with no
        qualifying track, or whose top two candidate tracks are tied, gets no
        assignment (section 5.3) - never a guess.

        Phase 11 section 4/5: the binding is stamped with the POSE's own
        ``capture_ts`` (the true measurement instant, set once at inference -
        see ``core/types.Pose``), not this cycle's ``capture_ts`` (``now``).
        The engine hands back the SAME frozen ``Pose`` object, unmodified,
        across every cadence-due reuse cycle, so using ``pose.capture_ts``
        here means a track's reported pose age grows honestly across reuse
        instead of resetting to ~0 on every rebind - the measured root cause
        of the ~1-2s "continuous but not current" symptom
        (``results/track_pose_freshness_before.md``: before this fix,
        ``displayed_pose_age_ms`` never exceeded ``pose_max_reuse_ms`` because
        every successful bind - fresh or reused - reset the clock). Falls back
        to this cycle's ``capture_ts`` only for a ``Pose`` built without one
        (back-compat / test fixtures)."""

        person_tracks = [
            (tid, ts) for tid, ts in self._tracks.items()
            if majority_class(ts.class_history) == _POSE_TRACK_CLASS
        ]
        if not person_tracks:
            return
        cfg = self._cfg
        for pose in poses:
            scored = sorted(
                (
                    (geometry.iou(pose.bbox, ts.bbox), tid, ts)
                    for tid, ts in person_tracks
                ),
                key=lambda row: row[0],
                reverse=True,
            )
            if not scored or scored[0][0] < cfg.pose_bind_min_iou:
                continue
            if len(scored) > 1 and scored[0][0] == scored[1][0]:
                continue  # ambiguous tie - no assignment (section 5.3)
            _score, _tid, ts = scored[0]
            pose_ts = pose.capture_ts if pose.capture_ts is not None else capture_ts
            ts.record_pose(
                keypoints=pose.keypoints, frame_id=pose.frame_id,
                capture_ts=pose_ts, fresh=fresh,
            )

    # -- association -----------------------------------------------------

    def _match_and_apply_hits(
        self, high: list[Detection], low: list[Detection], capture_ts: float, frame_id: int
    ) -> tuple[set[int], set[int]]:
        cfg = self._cfg
        candidates = [
            AssociationCandidate(
                key=tid,
                predicted_bbox=ts.predicted_bbox(capture_ts),
                last_bbox=ts.bbox,
                track_class=majority_class(ts.class_history),
            )
            for tid, ts in self._tracks.items()
        ]
        high_boxes = [d.bbox for d in high]
        high_classes = [d.raw_class_name or d.class_name for d in high]
        stage1_tracks, stage1_dets = greedy_match(
            candidates, high_boxes, high_classes, cfg, min_score=cfg.assoc_min_score
        )

        matched: set[int] = set(stage1_tracks.keys())
        for tid, di in stage1_tracks.items():
            self._apply_hit(tid, high[di], frame_id, capture_ts)

        if low:
            eligible_track_ids = [
                tid for tid, ts in self._tracks.items()
                if tid not in matched and ts.status != TrackStatus.TENTATIVE
            ]
            low_boxes = [d.bbox for d in low]
            used_low: set[int] = set()
            for tid in eligible_track_ids:
                pred = self._tracks[tid].predicted_bbox(capture_ts)
                best_score, best_di = -1.0, -1
                for di, box in enumerate(low_boxes):
                    if di in used_low:
                        continue
                    s = geometry.iou(pred, box)
                    if s >= cfg.stage2_iou_threshold and s > best_score:
                        best_score, best_di = s, di
                if best_di >= 0:
                    used_low.add(best_di)
                    matched.add(tid)
                    self._apply_hit(tid, low[best_di], frame_id, capture_ts)
        return matched, stage1_dets

    def _apply_hit(self, tid: int, d: Detection, frame_id: int, capture_ts: float) -> None:
        ts = self._tracks[tid]
        ts.record_hit(
            bbox=d.bbox, raw_class=d.raw_class_name or d.class_name, score=d.score,
            policy_state=d.policy_state, tier=d.tier, frame_id=frame_id, capture_ts=capture_ts,
            custom_class_name=d.custom_class_name, custom_class_confidence=d.custom_class_confidence,
        )
        if ts.status == TrackStatus.COASTING:
            ts.status = TrackStatus.CONFIRMED
        elif ts.status == TrackStatus.TENTATIVE and ts.hits >= self._cfg.n_init:
            ts.status = TrackStatus.CONFIRMED
            self._counts["confirmed"] += 1

    # -- lifecycle ---------------------------------------------------------

    def _handle_misses(
        self, matched_track_ids: set[int], capture_ts: float, frame_width: float, frame_height: float
    ) -> None:
        cfg = self._cfg
        for tid in list(self._tracks.keys()):
            if tid in matched_track_ids:
                continue
            ts = self._tracks[tid]
            pred = ts.predicted_bbox(capture_ts)
            off_frame = geometry.is_off_frame(pred, frame_width=frame_width, frame_height=frame_height)

            if ts.status == TrackStatus.TENTATIVE:
                # A miss breaks the consecutive-hit streak n_init requires;
                # reviving it later would silently promote an interrupted
                # streak, so it is deleted rather than coasted (section 6.5:
                # an incorrectly reused id is worse than a new one).
                self._expire(tid, border=off_frame, ghost=False)
                continue

            ts.exited_border = ts.exited_border or off_frame
            budget_frames = cfg.border_exit_max_age_frames if ts.exited_border else cfg.max_age_frames
            new_misses = ts.consecutive_misses + 1
            ms_since_hit = max(0.0, (capture_ts - ts.last_detection_capture_ts) * 1000.0)
            over_ms_budget = (not ts.exited_border) and ms_since_hit > cfg.max_age_ms
            if new_misses > budget_frames or over_ms_budget:
                self._expire(tid, border=ts.exited_border, ghost=not ts.exited_border)
                continue

            ts.record_miss(predicted_bbox=pred, now_ts=capture_ts)
            ts.status = TrackStatus.COASTING

    def _expire(self, tid: int, *, border: bool, ghost: bool) -> None:
        ts = self._tracks.pop(tid)
        self._counts["expired"] += 1
        if border:
            self._counts["expired_border"] += 1
        if ghost:
            self._ghosts.append(_Ghost(
                track_id=tid, expired_at_frame_id=ts.last_detection_frame_id,
                bbox=ts.bbox, velocity=ts.velocity, last_update_ts=ts.last_update_ts,
                state=ts,
            ))

    # -- new tracks / re-association ---------------------------------------

    def _reassociate_or_spawn(
        self, high: list[Detection], stage1_dets: set[int], frame_id: int, capture_ts: float
    ) -> set[int]:
        cfg = self._cfg
        self._prune_ghosts(frame_id)
        new_hit_ids: set[int] = set()
        for di, d in enumerate(high):
            if di in stage1_dets:
                continue
            reclaimed = self._try_reassociate(d, capture_ts)
            if reclaimed is not None:
                new_hit_ids.add(reclaimed)
                continue
            if len(self._tracks) < cfg.max_tracks:
                new_hit_ids.add(self._spawn(d, frame_id=frame_id, capture_ts=capture_ts))
            else:
                self._counts["dropped_max_tracks"] += 1
        return new_hit_ids

    def _spawn(self, d: Detection, *, frame_id: int, capture_ts: float) -> int:
        tid = self._next_id
        self._next_id += 1
        ts = TrackState(
            track_id=tid, status=TrackStatus.TENTATIVE, bbox=d.bbox,
            class_history=self._new_history(), created_ts=capture_ts,
            last_update_ts=capture_ts, first_seen_frame_id=frame_id,
        )
        ts.record_hit(
            bbox=d.bbox, raw_class=d.raw_class_name or d.class_name, score=d.score,
            policy_state=d.policy_state, tier=d.tier, frame_id=frame_id, capture_ts=capture_ts,
            custom_class_name=d.custom_class_name, custom_class_confidence=d.custom_class_confidence,
        )
        self._tracks[tid] = ts
        self._counts["created"] += 1
        if self._cfg.n_init <= 1:
            ts.status = TrackStatus.CONFIRMED
            self._counts["confirmed"] += 1
        return tid

    def _new_history(self) -> deque:
        return deque(maxlen=self._cfg.class_vote_history)

    def _prune_ghosts(self, frame_id: int) -> None:
        window = self._cfg.reassoc_window_frames
        self._ghosts = [g for g in self._ghosts if frame_id - g.expired_at_frame_id <= window]

    def _try_reassociate(self, d: Detection, capture_ts: float) -> int | None:
        if not self._ghosts:
            return None
        cfg = self._cfg
        det_class = d.raw_class_name or d.class_name
        best_score, best_ghost = -1.0, None
        for g in self._ghosts:
            dt = max(0.0, capture_ts - g.last_update_ts)
            vx, vy = g.velocity
            predicted = geometry.shift(g.bbox, vx * dt, vy * dt)
            cand = AssociationCandidate(
                key=g.track_id, predicted_bbox=predicted, last_bbox=g.bbox,
                track_class=majority_class(g.state.class_history),
            )
            s = score_pair(cand, d.bbox, det_class, cfg)
            if s is not None and s >= cfg.reassoc_min_score and s > best_score:
                best_score, best_ghost = s, g
        if best_ghost is None:
            return None
        self._ghosts = [g for g in self._ghosts if g.track_id != best_ghost.track_id]
        ts = best_ghost.state
        ts.status = TrackStatus.CONFIRMED
        ts.record_hit(
            bbox=d.bbox, raw_class=det_class, score=d.score,
            policy_state=d.policy_state, tier=d.tier,
            frame_id=d.frame_id, capture_ts=capture_ts,
            custom_class_name=d.custom_class_name, custom_class_confidence=d.custom_class_confidence,
        )
        self._tracks[ts.track_id] = ts
        self._counts["reassociated"] += 1
        return ts.track_id
