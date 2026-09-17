"""Phase 9 section 14, unit tests 1-11 - pure tracker behaviour, no camera, no
model, scripted detection sequences. Test 12 (state/visual channel mapping) is
a rendering-layer concern that lands with the state/visual-mapping step, not
the pure module - see docs/phase-reports/phase9.md.
"""

from __future__ import annotations

import pytest

from predictivesense.config.settings import TrackingConfig
from predictivesense.core.enums import TrackStatus
from predictivesense.core.types import Detection, Pose
from predictivesense.tracking import Tracker
from predictivesense.tracking.track_state import TrackState

pytestmark = pytest.mark.unit

_W, _H = 640, 480


def _det(bbox, *, class_name="cup", score=0.9, frame_id=0, policy_state="accepted",
         tier="primary", raw=None) -> Detection:
    return Detection(
        bbox=bbox, class_id=0, class_name=class_name, score=score, frame_id=frame_id,
        raw_class_name=raw if raw is not None else class_name,
        policy_state=policy_state, tier=tier,
    )


def _box_at(cx, cy, w=40, h=40):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _tracker(**overrides) -> Tracker:
    cfg = TrackingConfig(**overrides)
    return Tracker(cfg)


# -- 1. Track creation: n_init consecutive hits promotes; fewer does not ----

def test_n_init_consecutive_hits_promotes_tentative_to_confirmed():
    t = _tracker(n_init=3)
    ts = 0.0
    box = _box_at(100, 100)
    for i in range(2):
        out = t.update([_det(box, frame_id=i)], frame_id=i, capture_ts=ts,
                        frame_width=_W, frame_height=_H)
        assert out[0].status == TrackStatus.TENTATIVE
        ts += 0.05
    out = t.update([_det(box, frame_id=2)], frame_id=2, capture_ts=ts,
                    frame_width=_W, frame_height=_H)
    assert out[0].status == TrackStatus.CONFIRMED
    assert out[0].hits == 3


def test_a_miss_during_tentative_does_not_later_promote_the_same_id():
    # 2 hits, then a miss (breaks the streak - tentative tracks are deleted on
    # any miss, section 6.5), then a "3rd" hit at the same place: this must be
    # a NEW id, tentative again, not a promotion of the interrupted streak.
    t = _tracker(n_init=3)
    box = _box_at(100, 100)
    out = t.update([_det(box, frame_id=0)], frame_id=0, capture_ts=0.0, frame_width=_W, frame_height=_H)
    first_id = out[0].track_id
    out = t.update([_det(box, frame_id=1)], frame_id=1, capture_ts=0.05, frame_width=_W, frame_height=_H)
    assert out[0].track_id == first_id and out[0].status == TrackStatus.TENTATIVE
    out = t.update([], frame_id=2, capture_ts=0.10, frame_width=_W, frame_height=_H)
    assert out == []  # deleted, not coasted
    out = t.update([_det(box, frame_id=3)], frame_id=3, capture_ts=0.15, frame_width=_W, frame_height=_H)
    assert out[0].track_id != first_id
    assert out[0].status == TrackStatus.TENTATIVE
    assert out[0].hits == 1


# -- 2. Track continuation: one id across frames; ids never reused ----------

def test_moving_box_keeps_one_id_and_ids_are_never_reused():
    t = _tracker(n_init=1)
    ts = 0.0
    ids_seen = []
    for i in range(10):
        box = _box_at(50 + i * 15, 100)
        out = t.update([_det(box, frame_id=i)], frame_id=i, capture_ts=ts,
                        frame_width=_W, frame_height=_H)
        ids_seen.append(out[0].track_id)
        ts += 0.05
    assert len(set(ids_seen)) == 1

    # A second, unrelated object elsewhere spawns a strictly higher id.
    out = t.update([_det(_box_at(500, 400), frame_id=10)], frame_id=10, capture_ts=ts,
                    frame_width=_W, frame_height=_H)
    new_ids = {tr.track_id for tr in out} - set(ids_seen)
    assert new_ids and min(new_ids) > max(ids_seen)


# -- 3. Brief missed detection: boundary exact both directions --------------

def test_track_survives_exactly_max_age_misses_and_expires_at_max_age_plus_one():
    t = _tracker(n_init=1, max_age_frames=3, max_age_ms=10_000.0)
    box = _box_at(200, 200)
    ts = 0.0
    out = t.update([_det(box, frame_id=0)], frame_id=0, capture_ts=ts, frame_width=_W, frame_height=_H)
    tid = out[0].track_id
    assert out[0].status == TrackStatus.CONFIRMED

    for miss_n in range(1, 4):  # 1, 2, 3 consecutive misses: still alive
        ts += 0.05
        out = t.update([], frame_id=miss_n, capture_ts=ts, frame_width=_W, frame_height=_H)
        assert len(out) == 1 and out[0].track_id == tid
        assert out[0].status == TrackStatus.COASTING
        assert out[0].consecutive_misses == miss_n

    ts += 0.05  # the 4th consecutive miss: max_age + 1 -> expires
    out = t.update([], frame_id=4, capture_ts=ts, frame_width=_W, frame_height=_H)
    assert out == []


# -- 4. Two-object association: crossing paths do not swap ids --------------

def test_crossing_objects_do_not_swap_ids():
    t = _tracker(n_init=1)
    ts = 0.0
    # A moves left->right, B moves right->left; they overlap heavily at step 5.
    a_id = b_id = None
    a_path = [(50 + i * 20, 150) for i in range(11)]
    b_path = [(250 - i * 20, 150) for i in range(11)]
    for i, (a_pos, b_pos) in enumerate(zip(a_path, b_path)):
        dets = [
            _det(_box_at(*a_pos), class_name="cup", frame_id=i),
            _det(_box_at(*b_pos), class_name="bottle", frame_id=i),
        ]
        out = t.update(dets, frame_id=i, capture_ts=ts, frame_width=_W, frame_height=_H)
        by_class = {tr.observed_class: tr.track_id for tr in out}
        if a_id is None:
            a_id, b_id = by_class["cup"], by_class["bottle"]
        else:
            assert by_class.get("cup") == a_id
            assert by_class.get("bottle") == b_id
        ts += 0.05
    assert a_id != b_id


# -- 5. Similar objects: two same-class boxes stay distinct -----------------

def test_two_same_class_boxes_side_by_side_stay_distinct():
    t = _tracker(n_init=1)
    ts = 0.0
    ids_left, ids_right = set(), set()
    for i in range(5):
        dets = [
            _det(_box_at(100, 100), class_name="cup", frame_id=i),
            _det(_box_at(300, 100), class_name="cup", frame_id=i),
        ]
        out = t.update(dets, frame_id=i, capture_ts=ts, frame_width=_W, frame_height=_H)
        assert len(out) == 2
        left = min(out, key=lambda tr: tr.bbox[0])
        right = max(out, key=lambda tr: tr.bbox[0])
        ids_left.add(left.track_id)
        ids_right.add(right.track_id)
        ts += 0.05
    assert len(ids_left) == 1 and len(ids_right) == 1
    assert ids_left != ids_right


# -- 6. Many simultaneous tracks: bounded, memory flat -----------------------

def test_bounded_max_tracks_drops_excess_new_detections():
    t = _tracker(n_init=1, max_tracks=5)
    dets = [_det(_box_at(50 + i * 60, 100), frame_id=0) for i in range(10)]
    out = t.update(dets, frame_id=0, capture_ts=0.0, frame_width=_W, frame_height=_H)
    assert len(out) == 5
    assert t.active_track_count == 5
    assert t.stats.dropped_max_tracks == 5

    # A second frame with more new objects stays bounded too - memory is flat.
    more = [_det(_box_at(700 + i * 5, 400), frame_id=1) for i in range(4)]
    out = t.update(more, frame_id=1, capture_ts=0.05, frame_width=_W, frame_height=_H)
    assert t.active_track_count <= 5


def test_state_stays_bounded_over_a_long_run_of_churn():
    # 500 cycles, a brand-new never-matched object every cycle: a 12-cell grid
    # (cells >= 50px apart, box width 40 -> never overlapping, so no cell ever
    # matches another) cycled with period 12, well beyond max_age(3) +
    # reassoc_window(5) = 8 cycles, so a cell's previous occupant is always
    # fully expired and pruned before the cell is reused - guaranteed churn
    # (spawn -> coast -> expire -> ghost -> pruned), never a reclaim. Internal
    # collections must never grow past their configured bounds - this is the
    # "memory flat" claim (sections 5.6 / 11.4), asserted structurally rather
    # than by measuring process RSS in a unit test.
    t = _tracker(n_init=2, max_age_frames=3, max_age_ms=10_000.0,
                  max_tracks=20, reassoc_window_frames=5)
    ts = 0.0
    for i in range(500):
        box = _box_at(20 + (i % 12) * 50, 300)
        t.update([_det(box, frame_id=i)], frame_id=i, capture_ts=ts, frame_width=_W, frame_height=_H)
        assert t.active_track_count <= 20
        assert len(t._ghosts) <= 5 + 1  # pruned to the window each cycle
        ts += 0.05
    stats = t.stats
    assert stats.created > 100  # real churn happened
    assert stats.expired > 100
    assert stats.created - stats.expired == t.active_track_count


# -- 7. Class change keeps one id; track_class by vote; observed_class fresh -

def test_class_flip_keeps_one_id_and_resolves_track_class_by_vote():
    t = _tracker(n_init=1)
    box = _box_at(150, 150)
    sequence = ["cup", "phone", "cup", "cup", "phone"]
    ts = 0.0
    tid = None
    last = None
    for i, cls in enumerate(sequence):
        out = t.update([_det(box, class_name=cls, frame_id=i)], frame_id=i, capture_ts=ts,
                        frame_width=_W, frame_height=_H)
        assert len(out) == 1
        if tid is None:
            tid = out[0].track_id
        assert out[0].track_id == tid
        assert out[0].observed_class == cls
        last = out[0]
        ts += 0.05
    assert last.track_class == "cup"  # 3 of 5 votes
    assert dict(last.class_votes) == {"cup": 3, "phone": 2}


# -- 8. Re-association: strong match reclaims id; weak match creates new ----

def test_reassociation_reclaims_id_on_strong_match_within_window():
    t = _tracker(n_init=1, max_age_frames=2, max_age_ms=10_000.0, reassoc_window_frames=10)
    box = _box_at(200, 200)
    ts = 0.0
    out = t.update([_det(box, frame_id=0)], frame_id=0, capture_ts=ts, frame_width=_W, frame_height=_H)
    original_id = out[0].track_id

    for miss_n in range(1, 3):  # expires on the 3rd consecutive miss
        ts += 0.05
        out = t.update([], frame_id=miss_n, capture_ts=ts, frame_width=_W, frame_height=_H)
    ts += 0.05
    out = t.update([], frame_id=3, capture_ts=ts, frame_width=_W, frame_height=_H)
    assert out == []  # expired, now a ghost

    # Returns nearby, same size, within the window: reclaimed.
    ts += 0.05
    out = t.update([_det(box, frame_id=4)], frame_id=4, capture_ts=ts, frame_width=_W, frame_height=_H)
    assert len(out) == 1
    assert out[0].track_id == original_id
    assert t.stats.reassociated == 1


def test_reassociation_creates_new_id_on_weak_match():
    # n_init=2 (not 1): a genuinely new spawn stays TENTATIVE, so status is an
    # extra discriminator confirming this is a fresh id, not a reclaim (a
    # reclaim goes straight to CONFIRMED - see the strong-match test above).
    # The original must actually reach CONFIRMED before it expires, or its
    # miss would take the tentative-immediate-delete path (no ghost at all).
    t = _tracker(n_init=2, max_age_frames=2, max_age_ms=10_000.0, reassoc_window_frames=10)
    box = _box_at(200, 200)
    ts = 0.0
    out = t.update([_det(box, frame_id=0)], frame_id=0, capture_ts=ts, frame_width=_W, frame_height=_H)
    original_id = out[0].track_id
    ts += 0.05
    out = t.update([_det(box, frame_id=1)], frame_id=1, capture_ts=ts, frame_width=_W, frame_height=_H)
    assert out[0].status == TrackStatus.CONFIRMED

    for miss_n in range(2, 5):  # max_age_frames=2: expires on the 3rd consecutive miss
        ts += 0.05
        out = t.update([], frame_id=miss_n, capture_ts=ts, frame_width=_W, frame_height=_H)
    assert out == []

    # A detection far away and a very different size: not the same object.
    ts += 0.05
    far_box = (550, 400, 600, 450)
    out = t.update([_det(far_box, frame_id=5)], frame_id=5, capture_ts=ts,
                    frame_width=_W, frame_height=_H)
    assert len(out) == 1
    assert out[0].track_id != original_id
    assert out[0].status == TrackStatus.TENTATIVE
    assert t.stats.reassociated == 0


# -- 9. Border exit vs mid-frame disappearance expire on different paths ----

def test_border_exit_expires_fast_and_is_not_reassociated():
    t = _tracker(n_init=1, max_age_frames=10, max_age_ms=10_000.0,
                  border_exit_max_age_frames=1, reassoc_window_frames=10)
    ts = 0.0
    # Moving steadily toward and past the right edge (width=640). Step size
    # (15px) stays well inside the 40px box so IoU-based continuity holds at
    # every step - the point under test is the border-exit budget, not
    # whether large jumps break association (that's covered elsewhere).
    positions = [(590, 100), (605, 100), (620, 100), (635, 100), (650, 100)]
    out = None
    for i, pos in enumerate(positions):
        out = t.update([_det(_box_at(*pos), frame_id=i)], frame_id=i, capture_ts=ts,
                        frame_width=_W, frame_height=_H)
        ts += 0.05
    original_id = out[0].track_id
    assert out[0].status == TrackStatus.CONFIRMED
    assert out[0].bbox[0] + (out[0].bbox[2] - out[0].bbox[0]) / 2 > _W  # centre is off-frame

    # One miss with the predicted position off-frame -> border exit flagged,
    # tolerance is border_exit_max_age_frames=1, so it survives this one miss...
    ts += 0.05
    out = t.update([], frame_id=len(positions), capture_ts=ts, frame_width=_W, frame_height=_H)
    assert len(out) == 1 and out[0].track_id == original_id

    # ...and expires on the next.
    ts += 0.05
    out = t.update([], frame_id=len(positions) + 1, capture_ts=ts, frame_width=_W, frame_height=_H)
    assert out == []
    assert t.stats.expired_border >= 1

    # Even within the reassociation window, a detection back at frame centre
    # must NOT reclaim the border-exited id (it left the ghost buffer by design).
    ts += 0.05
    out = t.update([_det(_box_at(600, 100), frame_id=len(positions) + 2)],
                    frame_id=len(positions) + 2, capture_ts=ts, frame_width=_W, frame_height=_H)
    assert len(out) == 1
    assert out[0].track_id != original_id


def test_mid_frame_disappearance_survives_up_to_ordinary_max_age():
    t = _tracker(n_init=1, max_age_frames=5, max_age_ms=10_000.0)
    box = _box_at(320, 240)  # dead centre - never near a border
    out = t.update([_det(box, frame_id=0)], frame_id=0, capture_ts=0.0, frame_width=_W, frame_height=_H)
    tid = out[0].track_id
    ts = 0.0
    for miss_n in range(1, 5):
        ts += 0.05
        out = t.update([], frame_id=miss_n, capture_ts=ts, frame_width=_W, frame_height=_H)
        assert len(out) == 1 and out[0].track_id == tid
    assert t.stats.expired_border == 0


# -- 10. Static object: zero velocity never shortens life --------------------

def test_static_object_survives_many_frames_with_zero_velocity():
    t = _tracker(n_init=1)
    box = _box_at(320, 240)
    ts = 0.0
    out = None
    for i in range(50):
        out = t.update([_det(box, frame_id=i)], frame_id=i, capture_ts=ts, frame_width=_W, frame_height=_H)
        ts += 0.05
    assert len(out) == 1
    assert out[0].hits == 50
    assert out[0].consecutive_misses == 0
    assert out[0].velocity == (0.0, 0.0)
    assert out[0].status == TrackStatus.CONFIRMED


# -- 11. Confidence semantics: carried forward, aged, never fabricated ------

def test_coasting_track_carries_previous_confidence_with_correct_age():
    t = _tracker(n_init=1, max_age_frames=5, max_age_ms=10_000.0)
    box = _box_at(320, 240)
    out = t.update([_det(box, score=0.71, frame_id=0)], frame_id=0, capture_ts=0.0,
                    frame_width=_W, frame_height=_H)
    assert out[0].last_detector_confidence == pytest.approx(0.71)
    assert out[0].last_detector_confidence_age_ms == pytest.approx(0.0, abs=1e-6)

    out = t.update([], frame_id=1, capture_ts=0.30, frame_width=_W, frame_height=_H)
    assert out[0].last_detector_confidence == pytest.approx(0.71)  # unchanged, not decayed
    assert out[0].last_detector_confidence_age_ms == pytest.approx(300.0, abs=1.0)

    out = t.update([], frame_id=2, capture_ts=0.55, frame_width=_W, frame_height=_H)
    assert out[0].last_detector_confidence == pytest.approx(0.71)
    assert out[0].last_detector_confidence_age_ms == pytest.approx(550.0, abs=1.0)


def test_track_with_no_observation_ever_has_no_confidence():
    # Defensive/schema check at the TrackState -> Track boundary: a track with
    # no recorded hit reports no confidence at all, never a fabricated 0 or
    # placeholder (section 8.4). The Tracker itself never constructs a track
    # this way (every track is born from a hit) - this exercises the
    # conversion function directly.
    ts = TrackState(track_id=1, status=TrackStatus.TENTATIVE, bbox=(0.0, 0.0, 10.0, 10.0))
    track = ts.to_track(now_ts=5.0, fresh=False)
    assert track.last_detector_confidence is None
    assert track.last_detector_confidence_age_ms == 0.0


# -- Phase 10 section 5: pose-to-track binding ------------------------------

def _pose(bbox, *, frame_id=0, n_kp=17, capture_ts=None) -> Pose:
    kp = tuple((float(bbox[0]) + 1.0, float(bbox[1]) + 1.0, 0.9) for _ in range(n_kp))
    return Pose(keypoints=kp, bbox=bbox, score=0.9, frame_id=frame_id, capture_ts=capture_ts)


def test_pose_binds_to_the_best_overlapping_person_track():
    t = _tracker()
    person_box = _box_at(100, 100, w=60, h=60)
    other_box = _box_at(400, 400, w=60, h=60)
    out = t.update(
        [_det(person_box, class_name="person", raw="person", frame_id=0),
         _det(other_box, class_name="cup", raw="cup", frame_id=0)],
        frame_id=0, capture_ts=0.0, frame_width=_W, frame_height=_H,
        poses=[_pose(person_box, frame_id=0)], pose_fresh=True,
    )
    person = next(tr for tr in out if tr.class_name == "person")
    cup = next(tr for tr in out if tr.class_name == "cup")
    assert len(person.pose_keypoints) == 17
    assert person.pose_fresh is True
    assert person.pose_age_ms == pytest.approx(0.0, abs=1.0)
    assert cup.pose_keypoints == ()  # never bound to a non-person track


def test_ambiguous_pose_gets_no_assignment():
    # Two equally-good person tracks (identical boxes) - the pose must not be
    # guessed onto either one (section 5.3).
    t = _tracker(max_tracks=8)
    box_a = _box_at(100, 100, w=60, h=60)
    box_b = _box_at(101, 101, w=60, h=60)  # near-duplicate, same IoU vs the pose box
    out = t.update(
        [_det(box_a, class_name="person", raw="person", frame_id=0),
         _det(box_b, class_name="person", raw="person", frame_id=0)],
        frame_id=0, capture_ts=0.0, frame_width=_W, frame_height=_H,
    )
    assert len(out) == 2
    pose_box = _box_at(100.5, 100.5, w=60, h=60)  # equidistant from both
    out2 = t.update(
        [_det(box_a, class_name="person", raw="person", frame_id=1),
         _det(box_b, class_name="person", raw="person", frame_id=1)],
        frame_id=1, capture_ts=0.05, frame_width=_W, frame_height=_H,
        poses=[_pose(pose_box, frame_id=1)], pose_fresh=True,
    )
    assert all(tr.pose_keypoints == () for tr in out2)


def test_pose_persists_through_a_frame_with_no_pose_input():
    # The measured Phase 10 root cause (results/pose_continuity_raw.md): the
    # engine's own per-frame cadence bookkeeping can come up empty. The
    # track-level binding must NOT be wiped by an empty `poses` cycle - it
    # ages instead (fixed by binding at the tracker layer, not reading
    # snap.poses per frame).
    t = _tracker()
    box = _box_at(100, 100, w=60, h=60)
    out = t.update(
        [_det(box, class_name="person", raw="person", frame_id=0)],
        frame_id=0, capture_ts=0.0, frame_width=_W, frame_height=_H,
        poses=[_pose(box, frame_id=0)], pose_fresh=True,
    )
    assert out[0].pose_keypoints != ()
    # Next cycle: person re-detected (keeps the track alive) but the pose
    # engine produced nothing this cycle (poses=()).
    out = t.update(
        [_det(box, class_name="person", raw="person", frame_id=1)],
        frame_id=1, capture_ts=0.20, frame_width=_W, frame_height=_H,
    )
    assert out[0].pose_keypoints != ()  # still there, just aged
    assert out[0].pose_fresh is False
    assert out[0].pose_age_ms == pytest.approx(200.0, abs=1.0)


def test_pose_beyond_max_age_is_dropped_not_shown():
    t = _tracker(pose_max_age_ms=500.0)
    box = _box_at(100, 100, w=60, h=60)
    t.update(
        [_det(box, class_name="person", raw="person", frame_id=0)],
        frame_id=0, capture_ts=0.0, frame_width=_W, frame_height=_H,
        poses=[_pose(box, frame_id=0)], pose_fresh=True,
    )
    out = t.update(
        [_det(box, class_name="person", raw="person", frame_id=1)],
        frame_id=1, capture_ts=0.6, frame_width=_W, frame_height=_H,
    )
    assert out[0].pose_keypoints == ()
    assert out[0].pose_frame_id is None
    assert out[0].pose_age_ms == 0.0


def test_reused_pose_objects_are_never_mutated_in_place():
    # PerceptionEngine hands the SAME frozen Pose object to every cadence-due
    # cycle it reuses across; binding it to a track must never mutate it.
    t = _tracker()
    box = _box_at(100, 100, w=60, h=60)
    pose = _pose(box, frame_id=0)
    original_keypoints = pose.keypoints
    t.update(
        [_det(box, class_name="person", raw="person", frame_id=0)],
        frame_id=0, capture_ts=0.0, frame_width=_W, frame_height=_H,
        poses=[pose], pose_fresh=True,
    )
    t.update(
        [_det(box, class_name="person", raw="person", frame_id=1)],
        frame_id=1, capture_ts=0.05, frame_width=_W, frame_height=_H,
        poses=[pose], pose_fresh=False,
    )
    assert pose.keypoints is original_keypoints  # frozen Pose - unchanged


# -- Phase 11 section 4/5: pose age must reflect the pose's OWN measurement
# instant, not the current cycle's capture_ts - a reused pose binding must not
# reset its reported age on every rebind (the measured root cause of the
# "continuous but not current" symptom - results/track_pose_freshness_before.md).

def test_reused_pose_age_grows_from_its_own_capture_ts_not_the_current_cycle():
    t = _tracker(pose_max_age_ms=2000.0)
    box = _box_at(100, 100, w=60, h=60)
    pose = _pose(box, frame_id=0, capture_ts=0.0)  # true measurement instant: t=0.0

    out = t.update(
        [_det(box, class_name="person", raw="person", frame_id=0)],
        frame_id=0, capture_ts=0.0, frame_width=_W, frame_height=_H,
        poses=[pose], pose_fresh=True,
    )
    assert out[0].pose_age_ms == pytest.approx(0.0, abs=1.0)

    # Cycle 1: the SAME frozen pose is handed back as an engine-level reuse,
    # 0.20s later. Before the fix this rebind would reset the track's own
    # pose_capture_ts to "now" (0.20), reporting age ~0 again; after the fix
    # it must report the pose's true age (0.20s), because the pose object
    # itself still carries capture_ts=0.0.
    out = t.update(
        [_det(box, class_name="person", raw="person", frame_id=1)],
        frame_id=1, capture_ts=0.20, frame_width=_W, frame_height=_H,
        poses=[pose], pose_fresh=False,
    )
    assert out[0].pose_age_ms == pytest.approx(200.0, abs=1.0)

    # Cycle 2: reused again at 0.45s - age keeps growing from the SAME true
    # origin (0.0), not resetting each time a bind happens.
    out = t.update(
        [_det(box, class_name="person", raw="person", frame_id=2)],
        frame_id=2, capture_ts=0.45, frame_width=_W, frame_height=_H,
        poses=[pose], pose_fresh=False,
    )
    assert out[0].pose_age_ms == pytest.approx(450.0, abs=1.0)


def test_pose_without_capture_ts_falls_back_to_the_current_cycle_time():
    # Back-compat: a Pose built without capture_ts (e.g. an older caller, or a
    # test fixture) must not crash or misbehave - the binding falls back to
    # exactly the pre-fix behaviour (this cycle's capture_ts).
    t = _tracker()
    box = _box_at(100, 100, w=60, h=60)
    out = t.update(
        [_det(box, class_name="person", raw="person", frame_id=0)],
        frame_id=0, capture_ts=1.0, frame_width=_W, frame_height=_H,
        poses=[_pose(box, frame_id=0, capture_ts=None)], pose_fresh=True,
    )
    assert out[0].pose_age_ms == pytest.approx(0.0, abs=1.0)
