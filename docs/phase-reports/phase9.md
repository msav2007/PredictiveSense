# Phase 9 — multi-object tracking & honest detection state semantics

Two problems, one phase: (A) identity was not continuous — every displayed
object was a fresh per-frame detection, so a missed detection made it vanish
and reappear as something new; (B) "grey" meant at least four unrelated
things on the overlay, so the visual state was not honestly readable.

Scope actually delivered: root-cause trace + measured frequency table
(published before any rendering change); a pure, dependency-free tracker
(`predictivesense/tracking/`) with every threshold measured, not invented;
wiring into both the real-time loop and the recorded driver;
`tracker_update_ms` as its own Phase 8 attribution stage with a measured
before/after showing no regression of the Phase 8 floor; a five-channel
state/visual mapping redesign of the overlay with presentation-only flicker
suppression; a Diagnostics per-track detail view; unit, integration and
browser tests for all of it. **No custom model training, no risk prediction,
no relationship engine, no voice, no Scene Snapshot Studio, no UI redesign
beyond the overlay/Diagnostics changes this phase required.**

---

## PART 1 — MEASURED

### 1. Root-cause trace (section 4 of the Phase 9 prompt) — before any rendering change

Traced by reading the actual code end to end — `perception/policy.py` →
`core/types.py` → `pipeline/loop.py` → `api/broadcast.py` →
`static/features/metrics.js` → `static/ui/store.js` →
`static/features/overlay.js` — with every claim below cited to a file:line at
the time of writing (see `docs/decisions.md`'s Phase 9 entry for the citation
list; kept out of this report to avoid rot as line numbers shift).

| # | Hypothesis | Verdict | Mechanism |
|---|---|---|---|
| 1 | `accepted_secondary` renders "de-emphasised" | **CONFIRMED** | `SECONDARY_COLOUR` (grey-blue), fixed alpha 0.62, fixed dash `[5,4]` — applied regardless of the detection's actual score. `clock`/`tv`/`bed`/`couch` are secondary tier; the prompt's example `remote` is actually primary tier. |
| 2 | `low_confidence_band` renders dashed/dimmed | **CONFIRMED**, narrower than expected | Only ever applied to `policy_state=="accepted"` detections — but it is a **separately configured** cutoff (`perception.detector.low_confidence_band`) from the policy's own `per_class_threshold_rule` (`policy.default_threshold`/`per_class_thresholds`). The two can and do drift apart: a detection can be `accepted` (above its per-class threshold) yet still render dashed/dimmed by the band — a second, uncoordinated "looks uncertain" signal. |
| 3 | `unknown_low_confidence`/`unknown_margin` omit the confidence percentage | **CONFIRMED**, and broader than the prompt's framing | The old overlay's only branch that appended a `%` to the label was `policy_state=="accepted"`. `accepted_secondary` never showed a percentage either — the omission wasn't specific to Unknown, it was "everything except plain accepted." |
| 4 | Snapshot staleness dims the overlay | **CONFIRMED**, and more severe than "dims" implies | `stale=True` at the pipeline level is only possible when no frame arrived that cycle, which by construction meant `detections == []` on that same snapshot — so a stale snapshot showed an **empty, dimmed** canvas, not a dimmed previous frame. Combined with the browser client unconditionally overwriting `runtime.lastSnapshot` on every message, any ingest gap over `stale_after_ms` (500 ms in the `dev` profile) made every box vanish, then reappear. |
| — | (not hypothesised) Zero hysteresis anywhere | **CONFIRMED** | `RecognitionPolicy` is stateless per frame; the overlay redrew directly off the raw per-frame `policy_state`/`score` on every WS message with no smoothing, no minimum dwell time. A detector score oscillating near a threshold flips accepted ↔ unknown, i.e. green ↔ grey, every frame it does. |

**Visual collision**: `SECONDARY_COLOUR` rgb(120,133,148) and
`UNKNOWN_COLOUR` rgb(148,163,184) are both desaturated blue-greys ~15–25 RGB
units apart at ~0.5–0.62 alpha — read as "the same grey" at a glance, despite
meaning "we know exactly what this is, we've chosen to mute it" vs "we have
no idea what this is." The `low_confidence_band`'s dash pattern `[6,4]` was
bit-for-bit identical to the `unknown` dash pattern — colour was the only
distinguishing signal between "accepted but a bit uncertain" and "no idea
what this is."

### 2. Measured frequency table (section 4, published before touching rendering)

Two independent measurements, both over the developer's own footage
(`data/raw`), full output in `results/grey_state_frequency.{json,md}`:

- **Batch** (`scripts/grey_state_audit.py`): the real detector + policy over
  every decoded frame of the one available clip (501 frames, content-driven,
  wants a large sample).
- **Live session** (same script): the real FastAPI app's real-time loop
  (real `/ws/ingest` → `/ws/state`, real Phase 8 timing) over the same clip
  for 20 s — the only way to observe staleness, a timing-driven property a
  batch pass over decoded frames cannot produce (there are no ingest gaps in
  a for-loop over `cv2.VideoCapture.read()`).

| `policy_state` | count | share |
|---|---|---|
| `accepted` | 561 | 0.616 |
| `accepted_secondary` | 350 | 0.384 |
| `unknown_low_confidence` | 0 | 0.000 |
| `unknown_margin` | 0 | 0.000 |
| `suppressed_implausible` | 0 | 0.000 |
| `rejected_size` | 0 | 0.000 |

`low_confidence_band` membership: **298 of 911** detections (0.327) — 40 of
those are `accepted` (where the band actually renders dashed/dimmed), 258 are
`accepted_secondary` (where the band check never fires because secondary's
own de-emphasis already dominates).

Live session (20 s, real-time loop): 250 snapshots, **8 stale (3.2%)**, in
**2 toggle runs** (lengths 1 and 7 consecutive stale snapshots — i.e. two
observed active→grey→active cycles), `pose_stale` 34.0%.

**Reading this table**: on the one clip available, `accepted_secondary` is
the dominant grey producer by a wide margin, not `unknown_*` (which never
fired at all on this footage) — the phase's own hypothesis ordering, keyed on
"Unknown," undersold how much of the observed greyness was actually
tier-de-emphasis. This directly shaped where the state/visual-mapping
redesign (section 4 below) put its weight: a distinct tier *badge*, separate
from the identity colour, rather than continuing to bundle tier into the box
colour. **This is a single-clip result** — `data/raw` has exactly one
recorded session; it should be re-derived once more footage exists.

### 3. Tracker thresholds derived from measurement (section 7.2)

`scripts/track_threshold_derivation.py`: a 40 s **live** real-time session
(same real `/ws/ingest`→`/ws/state` harness, not a batch/recorded pass —
recorded mode is lossless and analyses every frame at the video's native rate,
the wrong cadence to derive a real-time coasting budget from) over `data/raw`,
recording the presence/absence of each track-eligible raw class across the
ordered sequence of **non-stale** snapshots. Full output:
`results/track_thresholds.{json,md}`.

| measure | value |
|---|---|
| analysis-cycle interval p50 / p95 | 47.0 / 159.8 ms |
| non-stale snapshots used | 397 of 536 |
| presence runs / singleton (length-1) share | 291 runs, 0.323 |
| interior miss runs / p95 length | 286 runs, 11 cycles |

**Derived**: `n_init = 3` (rule: 3 if singleton-presence-run share > 0.20 else
2 — measured 0.323 > 0.20, so a single-cycle detection blip must not promote
a track to confirmed on one hit). `max_age_frames = 12` (rule:
`max(2, ceil(p95(miss_run_lengths)) + 1)` = `max(2, 11+1)` = 12).
`max_age_ms = 564.0` (`max_age_frames × interval_ms_p50`). Both rules and the
counts that produced them are in the result file, not just the final numbers.
**One clip, one 40 s session** — re-derive once more footage exists.

### 4. The tracker — `predictivesense/tracking/`

Pure, dependency-free (stdlib + numpy only — no tracking package added).
IoU + constant-velocity motion prediction, two-stage (ByteTrack-style) greedy
association (5 weighted signals: IoU, centre distance, size ratio, motion
consistency, class agreement — class is a signal, never a gate). Full shape,
lifecycle diagram, and the `Track` contract's 16 additive fields: this phase's
section of `docs/architecture.md`. Algorithm family cited (SORT, ByteTrack),
no implementation copied: `docs/attribution.md`. Every threshold's derivation
and every judgement default's rationale: `docs/decisions.md`.

**Unit tests** (`tests/unit/test_tracking.py`, 16 tests, pure — no camera, no
model, scripted sequences) cover section 14 items 1–11: `n_init`-consecutive-
hits promotion and the "a miss during tentative does not later promote"
boundary; id continuity across a moving box and monotonically-increasing,
never-reused ids; the exact miss-count boundary (`max_age` survives,
`max_age+1` expires); two crossing objects never swap ids; two same-class
boxes stay distinct; bounded `max_tracks` with a `dropped_max_tracks` counter
and 500-cycle sustained churn (created/expired/pruned, `active_track_count`
and the ghost pool both stay bounded throughout); class-flip (`cup`→`phone`→
`cup`) keeps one id and resolves `track_class` by majority vote while
`observed_class` tracks the current frame; re-association reclaims an id on a
strong match and spawns a new one on a weak match; border-exit expires fast
and is never re-associated, vs. mid-frame disappearance surviving the
ordinary `max_age` budget; a static (zero-velocity) object survives 50 frames
without penalty; a coasting track carries its previous confidence unchanged
with a correctly-computed age, and a track built with no observation reports
no confidence at all (never a fabricated value). Item 12 (state/visual
channel mapping) is a rendering-layer concern, covered under browser tests
below, not the pure module.

### 5. Wiring — both pipelines

`pipeline/loop.py`: `AnalysisLoop` holds one `Tracker` (built by `build_loop`
from `config.tracking` unless `tracking.enabled` is false), calls
`update()` only when a frame actually arrived this cycle — **never on a
stale one** — and populates `StateSnapshot.tracks`. On a stale cycle the
tracker performs no update (its internal state has not changed) and the
**last real track list is reported unchanged**, not wiped to `[]` — a
deliberate behaviour change from Phase 8 that directly fixes root-cause
finding 4 in practice (the overlay now dims the last known tracks through a
brief gap instead of every box vanishing).

`pipeline/recorded.py`: the same `Tracker` class, fed the file's own PTS
(never wall time); the previously-always-empty `tracks` field of the
recorded JSONL is now populated (`_track_to_dict`, fixed precision, fixed key
order, mirroring `_detection_to_dict`).

**Integration tests** (`tests/integration/test_tracker_wiring.py`, 3 tests,
`AnalysisLoop._iterate` exercised directly with a duck-typed fake perception
engine — no model weights needed, same style as the existing
`test_staleness_guard.py`): `StateSnapshot.tracks` is populated in a
real-time snapshot with `fresh=True` on the spawn frame (a real bug caught
and fixed here: the first implementation reported `fresh=False` on a track's
own creation frame, because the "fresh" set was computed before spawning ran
— fixed by threading the newly-spawned/reassociated ids into the same
fresh-id set returned by `update()`); the tracker never updates on a frame
that fails the Phase 8 staleness guard (seeded via direct
`tracker.update()` calls, verified by `consecutive_misses` staying flat
across the guard-dropped cycle and advancing by exactly one on the next
genuinely fresh cycle); `tracker_update_ms` is measured and stays under a
50 ms coarse regression bound over 30 cycles / 8 simultaneous objects.

**Determinism** (`tests/integration/test_recorded_determinism_with_models.py`,
`-m models`, real ONNX weights): two `RecordedDriver` runs of the same
fixture clip with the same tracker configuration produce **byte-identical**
JSONL, tracks included; a shape/bound test asserts every track dict's key set
and that `status` is one of the three live states and confidence is `None`
or in `[0,1]`.

### 6. Latency budget (section 11) — before/after, one consistent condition

`tracker` was threaded into the existing Phase 8 stage-attribution chain
(`FrameTrace.tracker_ms`, `telemetry/stages.py`'s `SERVER_STAGE_NAMES`,
between `policy` and `snapshot_build`) rather than measured separately, so it
shows up in the same `latency_stages_<label>.md` table Phase 8 already
produces. `scripts/benchmark_latency.py` gained a `--no-tracking` flag for
isolated before/after.

Same clip (`data/raw`), same machine, same session, 150 frames, 12 s
loopback each:

| | capture→snapshot p50 | analysis FPS | drop rate | tracker p50 / p95 |
|---|---|---|---|---|
| tracking **off** (`results/latency_stages_phase9_before_no_tracker.md`) | 108.73 ms | 14.22 | 0.0 | — |
| tracking **on** (`results/latency_stages_phase9_after_tracker_on.md`) | 112.63 ms | 13.77 | 0.0 | 0.19 ms / 0.3 ms |

Tracker's own cost is a fraction of a millisecond against detector
(~44–45 ms) and pose (~50–53 ms); the ~4 ms difference in end-to-end p50 is
within normal run-to-run detector/pose variance between the two runs (both
detector and pose shifted by a similar amount independently of tracking).
Both runs also beat the Phase 8 baseline cited in the Phase 9 prompt
(capture→snapshot p50 ~129.6 ms, ~12 FPS, drop rate ~0.04) — no regression of
the Phase 8 floor, in either direction of the comparison.

### 7. State/visual mapping (section 9) — the rendering change, done last

`overlay.js` now draws `snap.tracks` (every track-eligible detection this
frame is already represented by some track — nothing is lost switching away
from `snap.detections`). Five independent visual channels — full table in
`docs/architecture.md`'s Phase 9 section:

| concept | channel |
|---|---|
| Track identity | Hue, `identityColor(track_id)` — stable for the track's life, independent of class |
| Fresh vs coasting | Stroke: solid / dashed — and *only* that; no longer doubles as a confidence-band or uncertainty signal |
| Recognition uncertainty (`Unknown`) | Label text only, never a colour |
| Vocabulary tier (`secondary`) | A small `"tier: secondary"` badge, never a colour change |
| Snapshot staleness | Whole-overlay dim + `"STALE"` banner (unchanged from Phase 8) |

The `low_confidence_band`'s former dashed/dimmed treatment of `accepted`
detections is **retired** as a distinct signal (root-cause finding 2) — the
confidence value stays visible via the label percentage and the confidence
bar, so no information is lost, only the redundant/confusable channel.
Confidence percentage now renders for `accepted` **and** `accepted_secondary`
(previously secondary showed none at all — root-cause finding 3); a coasting
track's label appends its confidence's age (`"72% (0.3s old)"`) so the
viewer can always tell whether a number is this frame's or carried forward
(section 8.3). `"Unknown"` deliberately keeps no percentage — a pre-existing,
intentional design choice, unchanged by this phase.

**Flicker suppression** (section 9.4, presentation-only): a small per-track
hysteresis map in `overlay.js` (`stableKind`, 150 ms dwell — ~3 analysis
cycles at the measured interval) — a track's displayed *kind* only changes
once the new kind has persisted past the dwell window. Diagnostics reads the
raw, unsmoothed per-frame `policy_state` (`snap.detections`, untouched)
throughout. Most class-label flicker is already resolved upstream by the
tracker's own majority-vote `track_class`; the JS-side hysteresis covers the
`policy_state`-driven kind transitions specifically, which are
per-fresh-observation.

**Diagnostics** (section 9.5) gains a per-track detail panel
(`groups/diagnostics.js`): `track_id`, `status`, observed vs. track class,
vote counts, last detector confidence + its age, consecutive misses, track
age, `policy_state`, `tier` — built alongside, not replacing, the existing
per-detection panel. Click-to-select now selects a track (stable `track_id`,
no per-frame bbox re-matching needed); a revealed `suppressed_implausible`
detection (never tracked — excluded by `TRACK_ELIGIBLE_STATES`) is still
selectable by the old bbox-based path for that one diagnostic case.

**Browser tests** (`tests/browser/test_tracking_overlay.py`, 4 tests, real
Chromium via Playwright, synthetic `Track`-shaped snapshots injected directly
via `store.setSnapshot()` — same technique as the existing
`test_overlay_de_emphasises_a_stale_pose` — with the real canvas 2D context
spied on so assertions check exactly what was drawn):

- **17**: a fresh track renders with no dash; the same track_id going
  coasting renders dashed; a subsequent fresh hit reverts to solid.
- **18**: an `Unknown` track, a `secondary`-tier track, and a coasting
  `accepted` track in the same snapshot render three genuinely different
  things — `"Unknown"` exactly once and never with a `%`, a `"tier:
  secondary"` badge only on the secondary track, and `"...% (...s old)"`
  only on the coasting one.
- **19**: a track present in `tracks` across a coasting cycle is still drawn
  (never silently skipped just because it missed); a `policy_state` flicker
  injected and reverted within the dwell window never reaches the label as
  `"Unknown"`.
- **20**: clicking an injected track (via the same `detection-selected` event
  the real click handler emits) populates the Diagnostics per-track panel
  with the correct id, class, confidence and hit count.

Section 14 item 21 ("zero console errors on `/` and `/studio`") is already
covered by the existing
`test_phase8_responsiveness.py::test_slash_and_studio_load_with_zero_console_errors`
and needed no Phase-9-specific duplicate — it continues to pass.

Additionally verified with an ad-hoc script (not part of the test suite,
scratch tooling only) pushing real `data/raw` frames into the real live app
while driving the real page in headless Chromium: real tracks appeared
(`person` at 92–93% confidence, correctly coasting between detector hits;
a `secondary`-tier misclassification showing its tier badge), click-to-select
worked end to end into the Diagnostics panel, and zero console errors were
observed. Screenshots were reviewed visually during the session (not
committed — this was a manual verification aid, not a test artifact).

### 8. Full test suite

`pytest -q` (all markers): **437 passed, 4 skipped, 0 failed.** `-m models`:
18 passed, 2 skipped (weight-dependent). `-m browser`: 26 passed (22
pre-existing + 4 new). Two pre-existing flakes were observed once each during
a full-suite run under load and confirmed, by immediate re-run in isolation
(2/2 and 3/3 passes respectively) and by `git status` showing no file in
either area touched this session, to be **timing-sensitive and unrelated to
this phase**: `tests/integration/test_stage_instrumentation.py::test_stage_attribution_record_is_complete_for_an_analysed_frame`
(a real-WS-timing test) and
`tests/browser/test_studio_bulk_upload.py::test_discard_batch_removes_the_panel_and_staging`
(an Object Learning Studio async-proposal-vs-discard race, an area this phase
never touched). Neither reproduced in the final clean full-suite run
(437/4/0). One genuine regression was found and fixed during this phase: a
static-source-scan unit test (`test_ui_structure.py`) asserted a literal
`resting = "Unknown"` string that the overlay rewrite legitimately changed
shape (`return "Unknown"` inside the new `labelFor()` function) — updated to
match, same intent (`"Unknown"` is exact, never `(was X)`).

---

## PART 2 — PHYSICALLY OBSERVED BY THE DEVELOPER (pending, section 15)

Not yet performed. Automated tests are not evidence of real-world tracking
quality. Section 15's checklist, for the developer, on **both** the laptop
integrated camera and the OnePlus Nord 4 virtual camera, recorded separately:

1. Person sitting still — id stable, no recreation.
2. Person moving slowly, then quickly — id survives.
3–7. Phone / cup / mouse / pen moving in hand — object ids survive the
   motion and stay with the object, not the hand.
8. Object briefly occluded by a hand — the track coasts and recovers rather
   than being recreated.
9. Object leaving frame and returning — sensible behaviour, whichever way it
   resolves (new id or reclaim).
10. Two similar objects visible together — no id swapping.

For each: id continuity, unnecessary recreation, expiry behaviour, confidence
semantics, grey/active/stale legibility, and perceived responsiveness.
Confirm the grey↔active flicker is materially reduced and that what greying
remains is understandable.

---

## PART 3 — NOT VERIFIED / LIMITATIONS

- **No model trained, fine-tuned, or activated.** Recognition of watch,
  spectacles, charger, headphones, shaker remains unchanged and unrecognised
  — the tracker stabilises identity and presentation, it does not expand the
  detector's vocabulary.
- **Tracking quality on real footage is unmeasured against ground truth.**
  Only scripted-sequence ID-switch and fragmentation counts are computed
  (`tests/unit/test_tracking.py`). `MOTA`, `IDF1`, or any metric requiring
  labelled tracking ground truth is **not** claimed — that footage does not
  exist.
- **Every threshold derives from a single clip, a single session.**
  `n_init`, `max_age_frames`, `max_age_ms` (`results/track_thresholds.md`)
  and the grey-state frequency table (`results/grey_state_frequency.md`) are
  each one measurement on `data/raw`'s one recorded session — re-derive once
  more footage exists, as both scripts' own output already states.
  `class_vote_history`, `max_tracks`, `reassoc_window_frames` and the
  association weights are documented judgement/bounded-resource defaults,
  not measured thresholds — flagged as such in `TrackingConfig`'s comments
  and in `docs/decisions.md`.
- **`input_size: 480` remains provisional** — a Phase 8 carry-forward, not
  touched this phase; still needs a clip with small-in-frame objects.
- **No relationship engine, risk model, temporal risk classifier, preventive
  recommendations, or voice/TTS** — none started, none placeheld. No
  appearance-based re-identification (association is geometry + motion +
  class-vote only, by design — section 18's non-goals). No tracking
  dependency was added.
- **The "real end-to-end" browser verification described in Part 1 §7 used
  server-pushed frames, not the browser's own camera** (the fake-media-device
  test infrastructure generates a synthetic test pattern with no real COCO
  objects in it, so real frames were pushed directly into `/ws/ingest` from a
  separate process while the real page rendered the results) — this proves
  the rendering pipeline end to end with real detections and real tracks, but
  is **not** a substitute for the physical camera verification in Part 2.

---

## Acceptance criteria (Phase 9 prompt, section 17)

- [x] Root cause of `active→grey`, `grey→active` and missing-confidence
      identified in code and published, with a frequency table, before any
      rendering change.
- [x] Pure, dependency-free tracker; persons and objects tracked identically
      (no class special-casing in `Tracker`); many simultaneous stable ids
      (bounded by `max_tracks`, asserted by test).
- [x] Ids survive brief missed detections and a class flip; expiry bounded in
      both frames and ms; ids never reused (re-association is an explicit,
      evidence-gated reclaim of the *same* id, not reuse of a new one).
- [x] Every threshold derived from measurement, recorded with its result
      file (`n_init`/`max_age`); every other parameter individually
      commented as a judgement/bounded-resource default.
- [x] Detector confidence never fabricated; carried-forward values labelled
      with age; absent means absent.
- [x] Identity/freshness/uncertainty/tier/staleness each own a distinct
      visual channel, documented as a table, no two states render
      identically (asserted by browser test).
- [x] `Unknown` stays semantically distinct from stale, missed and
      predicted.
- [x] Flicker materially reduced (dwell-window hysteresis, browser-tested);
      confirmed track does not blink; suppression is presentational only;
      Diagnostics still exposes raw per-frame values.
- [x] `tracker_update_ms` measured as its own stage and bounded by test;
      Phase 8 latency not materially regressed, shown in one consistent
      measurement condition.
- [x] Recorded mode deterministic (byte-identical, `-m models`); one tracker
      serves both modes; CPU-only unaffected, provider-independent (the
      tracker never touches `onnxruntime` or a provider at all).
- [x] `StateSnapshot.tracks` populated with real tracks; no fabricated ids
      in the frontend.
- [x] Full suite green including browser; this report complete; tree to be
      committed clean.
- [ ] Physical verification (section 15) — pending, Part 2 above.
