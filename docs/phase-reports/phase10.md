# Phase 10 — perception regression repair + track & pose continuity

Three problems from physical testing after Phase 9 had to be separated before
anything was fixed: (A) recognition appearing worse than before, (B) the
green pose skeleton restarting roughly every second, (C) tracks/states
flickering. Scope actually delivered: the attrition harness and its published
table (section 3), the green-line root cause traced and measured (section 4),
pose bound to the person track so it persists through the measured gap
(section 5), the render predicate read from the shipped code and deliberately
kept (section 6), and the `input_size` 640 vs 480 re-comparison by per-class
recall (section 3.1/10). **No custom model training, no risk prediction, no
relationship engine, no voice, no Scene Snapshot Studio, no UI redesign
beyond what the pose-binding fix required.**

State semantics (section 7), confidence honesty (section 8), and person/
object track coexistence (section 12/13) were **not reworked this phase** —
Phase 9 already built the five-channel mapping, the never-fabricated
confidence rule, and the class-agnostic association, none of which this
phase's changes touch or regress (reverified: the Phase 9 browser tests for
all of it, `tests/browser/test_tracking_overlay.py`, still pass unchanged).

---

## PART 1 — MEASURED

### 1. The central diagnostic (section 3) — published before any fix

`scripts/pipeline_attrition.py` runs the real detector → policy → tracker
classes (identical code to the live loop / `RecordedDriver`) over a clip and
tabulates, per required class, how many raw detections existed and how many
survived each layer. Run on the only clip in `data/raw` (laptop integrated
camera — `results/pipeline_attrition_integrated_camera.md`; no OnePlus clip
exists yet, so that comparison could not be run this phase):

| class | raw | accepted | accepted_secondary | unknown | suppressed | rejected_size | tracker input | tracker output (tentative/confirmed/coasting) | rendered |
|---|---|---|---|---|---|---|---|---|---|
| person | 504 | 504 | 0 | 0 | 0 | 0 | 504 | 2/499/0 | 501 |
| cell phone | 22 | 22 | 0 | 0 | 0 | 0 | 22 | 6/5/3 | 14 |
| cup | 35 | 35 | 0 | 0 | 0 | 0 | 35 | 3/23/28 | 54 |
| bottle/laptop/keyboard/mouse/chair/book/backpack | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0/0/0 | 0 |

**Findings, numbers not reasoning:**

- Seven of the ten required classes had **zero raw detections** across all
  501 frames — not a pipeline bug to attribute attrition to; either the
  object never appears in this footage, or it is beyond the model's
  capability at any input size this phase tested (section 3.1 below narrows
  this for one of the two candidates).
- `cell phone`: 22 raw → 14 rendered. The attrition happens at the
  tracker-input→output boundary, not policy or render: a low-score detection
  (`score < tracking.high_score_split`) with no already-open, non-tentative
  track to extend is dropped — it can neither spawn a new track nor match a
  still-tentative one (`Tracker._match_and_apply_hits`). This is the concrete
  "detected but not shown" mechanism.
- `cup`: 35 raw but **54** rendered — tracking's coasting behaviour is
  working as intended here: a track keeps rendering (dimmed, per Phase 9's
  freshness channel) on frames the raw detector missed, so continuity
  actually *adds* visible frames relative to raw detections, not just loses
  them.
- The entire `accepted_secondary` bucket on this clip (350/350 detections) is
  one systematic misclassification of something in the room as `umbrella` —
  a `bed → chair`-class model error, flagged and measured, **not fixed** this
  phase (out of scope, same precedent as `bed → chair`).
- **Section 3.1 hypothesis 2 ("the overlay may now render only confirmed
  tracks") is REFUTED.** Reading the shipped `overlay.js` (`draw()` /
  `drawTrack()`, before any Phase 10 change) shows no status-based filter —
  every entry of `snap.tracks` is drawn, tentative included. The attrition
  table's `tracker output` and `rendered` columns are identical by
  construction and confirmed numerically (e.g. person: 2 tentative + 499
  confirmed + 0 coasting = 501 rendered).

### 2. `input_size` 640 vs 480 re-comparison, by per-class recall (section 3.1)

`scripts/input_size_recall_compare.py`, same clip, 501 frames
(`results/input_size_recall_integrated_camera.md`):

| class | raw @480 | raw @640 | delta |
|---|---|---|---|
| person | 504 | 505 | +1 (a wash) |
| cell phone | 22 | 32 | **+10 (+45% at 640)** |
| cup | 35 | 17 | −18 (480 detected MORE here — opposite direction, not investigated further) |
| bottle/laptop/keyboard/mouse/chair/book/backpack | 0 | 0 | 0 (undetermined either way on this footage) |

The Phase 8 sweep that originally justified 480 found identical primary-tier
totals (197/197/197) at 320/480/640 — which only proved that sweep's clip had
nothing small enough to discriminate, not that recall holds generally. This
measurement is the missing evidence, and it is real and mixed: `cell phone`
shows a material recall loss at 480; `cup` shows the opposite.

**Decision (recorded in `docs/decisions.md`): `input_size` stays 480,
unchanged.** The Phase 8 latency saving is real and load-bearing (~46ms vs
~73ms detector p50) and section 19 requires preserving it; the evidence here
is single-clip and directionally mixed, not a clean case for either value.
`input_size: 480` remains explicitly **provisional**, now with a concrete
counter-example on record rather than an untested assumption — re-run this
script on footage with a phone/watch at typical desk distance before treating
480 as settled.

### 3. Green-line restart root cause (section 4)

`scripts/pose_continuity_audit.py` — a real live session (real `/ws/ingest` →
`/ws/state`, real Phase 8 mailbox/staleness timing; a batch decode has no
ingest gaps to measure staleness from), 40s, same clip
(`results/pose_continuity_raw.md`):

| measure | value |
|---|---|
| non-stale snapshots | 544 |
| **non-stale snapshots with NO pose (visible gap)** | **198 (36.4%)** |
| non-stale snapshots showing a reused (engine-level stale) pose | 177 (32.5%) |
| fresh pose refresh interval p50 / p95 / max | 203.0 / 406.0 / 1016.0 ms |
| empty-pose-gap runs | 159 (p95 length 2 snapshots, max 9) |

**Hypothesis 1 (cadence/reuse-window mismatch) is not the primary cause**:
the measured fresh-pose refresh interval (p95 406ms) stays under
`pose_max_reuse_ms` (500ms) in steady state. **Empty-result gaps ARE the
material cause**: `PerceptionEngine.infer` (`perception/engine.py`)
overwrites `_last_poses` to an empty tuple whenever a cadence-due pose
inference returns `[]` (person score momentarily under `pose.conf`=0.4) — the
reuse branch's own guard (`elif ... and self._last_poses ...`) then finds
nothing to reuse until the next successful due-cycle. Pose had no identity of
its own pre-Phase-10 (drawn straight off `StateSnapshot.poses` every frame),
so this empty-result gap was directly, visibly the reported ~1s restart.

### 4. Pose bound to the person track (section 5) — the fix

`Tracker._bind_poses()`: each pose is matched to the person-class track its
own box overlaps best (highest IoU, at/above `tracking.pose_bind_min_iou` =
0.1); a tie between the top two candidates gets no assignment (never a
guess). The binding lives on the mutable `TrackState`
(`pose_keypoints`/`pose_source_frame_id`/`pose_capture_ts`), so a cycle where
the *engine* produces no pose at all (the measured 36.4% case) leaves the
existing binding untouched — it ages instead of vanishing, bounded by the new
`tracking.pose_max_age_ms` (900ms, derived from this session's measured
worst-case empty-gap run of ~580ms, with margin).

`Track` gained five additive fields (`pose_keypoints`, `pose_frame_id`,
`pose_capture_ts`, `pose_age_ms` — continuous, mirrors
`last_detector_confidence_age_ms`, deliberately not a second binary stale
flag — `pose_fresh`). `overlay.js` now draws each track's own bound pose with
a continuous age fraction (fading emphasis smoothly, switching to a dashed
edge as it ages) instead of looping the frame-level `snap.poses`.
`tests/browser/test_phase8_responsiveness.py::test_overlay_de_emphasises_an_aged_pose`
(renamed/rewritten from the Phase 8 stale-pose test) and five new unit tests
in `tests/unit/test_tracking.py` (binding to the best-overlapping track,
ambiguous-tie no-assignment, persistence across an empty-poses cycle, aging
out past `pose_max_age_ms`, and that a reused `Pose` object is never mutated
in place) cover it.

### 5. Latency floor — before/after, one session (section 11)

`scripts/benchmark_latency.py --only-end-to-end`, same clip, same session (to
control for this machine's current background load, which was elevated in
both runs relative to the Phase 8/9 baseline — `process_cpu_percent` ~920–960%
in both conditions):

| condition | capture→snapshot p50 | tracker_update_ms (incl. pose-binding) |
|---|---|---|
| tracking + pose-binding **on** | 159.22 ms | p50 0.21 / p95 0.4 ms |
| tracking **off** (control, same session) | 165.43 ms | — |

Tracking+pose-binding is not slower than the no-tracking control measured in
the same noisy conditions; `tracker_update_ms` (now including
`_bind_poses`) is statistically unchanged from Phase 9's measured 0.19/0.3ms
— pose-binding itself adds no measurable cost. The elevated absolute p50 vs
Phase 9's reported 112.63ms reflects this machine's current load, not a
regression from this change. A floor-clean re-measurement (idle machine) is
recommended before this number is treated as a final Phase 10 figure.

### 6. Full test suite

`pytest -q` (all markers): **442 passed, 4 skipped, 0 failed** — up from
Phase 9's 437/4/0 by the 5 new pose-binding unit tests. `-m models`: 20
passed, 2 skipped (weight-dependent, unchanged). `-m browser`: 26 passed (25
pre-existing/renamed + the rewritten aged-pose test). No new flake observed.

---

## PART 2 — PHYSICALLY OBSERVED BY THE DEVELOPER (pending, section 15)

Not yet performed — this requires a camera, which this session does not have
access to. Unchanged from Phase 9's own Part 2 list, now specifically
including: whether the skeleton visibly stops restarting during continuous
motion on both the laptop integrated camera and (once a clip/live session
exists) the OnePlus Nord 4; whether an aged pose is visibly distinguishable
from a fresh one without looking jarring; whether the `cell phone`
recall gap measured at `input_size: 480` is noticeable in practice at desk
distance.

---

## PART 3 — NOT VERIFIED / LIMITATIONS

- **No model trained, fine-tuned, or activated.** `bed → chair` and this
  clip's `umbrella` misclassification are model errors this phase measured
  and recorded, not fixed.
- **`results/pipeline_attrition_integrated_camera.md`,
  `results/pose_continuity_raw.md`, and
  `results/input_size_recall_integrated_camera.md` are each one clip / one
  session** on the laptop integrated camera only. `data/raw` has no OnePlus
  Nord 4 clip, so the camera comparison (section 10) could not be run this
  phase — re-derive all three once more/different footage exists, per each
  script's own stated caveat.
- **State semantics (section 7), the never-fabricated-confidence rule
  (section 8), and person/object track coexistence (section 12/13) were not
  reworked this phase** — Phase 9 already satisfies them and this phase's
  changes are additive and do not touch that logic (reverified: the relevant
  Phase 9 browser/unit tests are unchanged and still pass).
- **The `cell phone` @480-vs-@640 recall gap and the `cup` reverse-direction
  result are each a single-clip measurement**, not a validated conclusion —
  `input_size` stays provisional, not settled, by design.
- **Latency figures in this report were measured under elevated background
  load** (this machine, this session) — the before/after comparison (tracking
  on vs off, same session) is valid, but the absolute p50 numbers are not
  directly comparable to Phase 8/9's quieter-session baselines without a
  floor-clean re-run.
- **No tracking quality metric against labelled ground truth** (`MOTA`/
  `IDF1`) is claimed — unchanged from Phase 9, that footage does not exist.
- Physical camera verification (section 15) is entirely pending — Part 2.

---

## Acceptance criteria (Phase 10 prompt, section 17)

- [x] Attrition table published before any fix, per layer, per class — one
      camera (the only one available); guilty layer named with numbers.
- [x] Green-line restart root-caused in code with measured intervals, not
      inferred.
- [x] Person representation continuous: pose bound to the track, ages rather
      than vanishing; aged pose visibly distinct (browser test asserts it).
- [x] Render predicate deliberate and recorded (kept as-is — no
      confirmed-only gate existed or was added).
- [x] 640 vs 480 re-compared on this footage by per-class recall; default
      kept on the recorded evidence (still provisional).
- [x] Five states distinct, no two render identically — unchanged from Phase
      9, reverified (not reworked this phase, not regressed).
- [x] `Unknown` distinct from stale/coasting/missed; confidence never
      fabricated — unchanged from Phase 9, reverified. Pose confidence
      (`pose_age_ms`) follows the identical carried-forward-with-age pattern,
      never decayed into a fake value.
- [x] Track ids stable; person and object tracks coexist unchanged (no
      class special-casing added to association — pose-binding is a
      strictly additive post-update step).
- [x] Phase 8 latency floor preserved relative to a same-session no-tracking
      control; pose-binding cost measured and bounded (p50 0.21/p95 0.4ms).
- [x] Recorded mode deterministic (reverified, `-m models` green); one
      tracker/pose-binding implementation serves both modes.
- [x] Full suite green including browser and models markers.
- [x] `docs/phase-reports/phase10.md` complete (this file).
- [ ] Physical verification (section 15) — pending, Part 2.
- [ ] Tree committed clean — not committed this session (commits are made
      only on explicit request).
