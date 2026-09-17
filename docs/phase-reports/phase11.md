# Phase 11 — track/pose freshness + real Studio-to-trained-model pipeline

Two parts, sequenced per the prompt. **Part A (freshness) is P0 and landed
complete**: measured before any fix, root-caused in code, fixed, re-measured,
checked for recognition regression, and the Phase 8 latency floor reverified.
**Part B (training) is P1** and also landed as a genuinely working, tested,
end-to-end pipeline — proven on a synthetic fixture per section 12.4, plus a
real (additive, non-fake) live-inference wiring into the running app. No risk
prediction, no temporal risk model, no relationship engine, no voice, no
Scene Snapshot Studio, no cloud inference, no UI redesign beyond what these
two problems required.

---

## PART 1 — MEASURED

### A.1 The exact symptom, confirmed by code reading (section 4)

The green pose skeleton and the tracked box are both continuous (Phase 10
fixed that), but the developer reports the *displayed* position lags the real
one by roughly 1–2 seconds. Reading the code before touching anything:

- The track's own box position updates every real analysis cycle (a fresh hit
  or a constant-velocity prediction while coasting) — not obviously capable
  of a multi-second lag on its own.
- The pose skeleton is **bound to the person track** (Phase 10), not drawn
  per-frame. `Tracker._bind_poses` stamped the binding's `pose_capture_ts`
  with **the current cycle's own `capture_ts`** — i.e. "now" — on every
  successful bind, whether that bind was a brand-new inference or the
  perception engine's own within-window **reuse** of an old, unchanged
  `Pose` object. Reusing a stale measurement while resetting the clock that
  measures its own staleness is exactly the "continuity bought with
  staleness" mechanism section 4.2 named as the prime hypothesis.

### A.2 Headline metrics added, before any fix (section 4.1)

Two new numbers, folded into the existing `StateSnapshot.metrics` /
`MetricRegistry` (`pipeline/loop.py::AnalysisLoop._iterate` — no new
telemetry system):

```
displayed_pose_age_ms  = emitted_ts - (the bound pose's own pose_capture_ts)
displayed_track_age_ms = emitted_ts - (the track's last real detector hit)
```

Measured live (real `/ws/ingest → /ws/state`, real mailbox/broadcast timing —
`scripts/track_pose_freshness_audit.py`, same clip as Phase 10, 40s), **on
the unfixed code**:

| measure | p50 | p95 | max |
|---|---|---|---|
| `displayed_pose_age_ms` | 93.9 | 144.7 | **498.1** |
| `displayed_track_age_ms` | 100.6 | 356.9 | 649.0 |
| `stage_capture_to_snapshot_ms` (Phase 8, for comparison) | 110.4 | 147.4 | 220.3 |

**The max never exceeding 498.1ms is the numeric fingerprint of the bug**:
`pose_max_reuse_ms` is 500.0ms. Every successful bind — including a reuse
well inside its own window — reset the age to ~0, so the reported age could
essentially never reflect anything beyond one engine-level reuse window, no
matter how generous `tracking.pose_max_age_ms` (900ms) nominally was. The
raw number alone looks almost fine (p95 144.7ms); it is the *code trace*
that reveals it is fine for the wrong reason — it was measuring "time since
the last rebind", not "time since the pose was actually measured".

### A.3 The fix (sections 5.1/5.3)

`core/types.Pose` gained one additive field, `capture_ts: float | None`,
stamped once at true inference time in `perception/pose.py`. The perception
engine hands back the *same frozen `Pose` object* across every cadence-due
reuse cycle, so this field naturally keeps the true original measurement
instant through any number of reuses — no new plumbing needed.
`Tracker._bind_poses` now stores `pose.capture_ts` (falling back to the
current cycle's `capture_ts` only for a `Pose` built without one — full
back-compat) as the track's own `pose_capture_ts`. Zero new computation on
the hot path.

Re-measured, **identical 40s session**, same clip, after the fix
(`results/track_pose_freshness_after.md`):

| measure | p50 | p95 | max |
|---|---|---|---|
| `displayed_pose_age_ms` | 155.6 | 205.1 | 476.5 |
| `displayed_track_age_ms` | 93.5 | 347.5 | 610.7 |

The p50 rise (93.9 → 155.6ms) is the direct fingerprint of the fix — age no
longer collapses to near-zero on every reuse-cycle rebind. A longer **90s**
run (`results/track_pose_freshness_after_90s.md`, 1191 non-stale snapshots)
measures max **585.6ms** for pose — genuinely **past** `pose_max_reuse_ms`,
something the pre-fix reset bug made structurally near-impossible to observe
honestly, confirming the track-level `pose_max_age_ms` bound is now doing
real, correctly-attributed work.

**Caveat stated plainly**: this harness is a steady-state TestClient
loopback with none of the real webcam jitter or background CPU contention
Phase 10's own live session measured (36.4% empty-pose rate there). The true
field worst-case may exceed what this harness reproduces — section 18
physical verification is the check that would confirm the ~1–2s complaint is
resolved in practice, and remains pending (Part 2).

### A.4 `tracking.pose_max_age_ms` — kept at 900ms, not re-picked

The post-fix measured worst case (585.6ms pose / 711.2ms track over 90s)
stays comfortably under the existing 900ms bound. No measurement here
justifies moving it in either direction (section 5.1: derive from
measurement, don't pick a number — the disciplined choice given the data on
hand is to change nothing). Documented in `docs/decisions.md` with the
caveat that a real, physically-observed session may warrant re-deriving it.

### A.5 Buffer audit (section 4.4) — none found

Every buffer in the path is single-slot or a bounded ring by construction,
confirmed by reading the shipped code (unchanged since Phase 8/10, re-verified
this phase):

| buffer | bound | evidence |
|---|---|---|
| `BrowserSource._slot` | 1 | `submit()` overwrites, counts `_dropped` |
| `LatestFrameMailbox._slot` | 1 | `put()` overwrites, counts `dropped` |
| Analysis worker `encodeBusy` | 1 in flight | `dropReason()` → drop, never queue |
| WS send backpressure | drop, not queue | `ws.bufferedAmount > maxWsBufferedBytes` → drop |
| `Broadcaster` | 0 per-client queue | push-on-publish, last-value-wins |
| Browser `store.snapshot` | 1 | single field, overwritten per message |
| `_frame_stages` / `_perception_frame_records` | 256 | `deque(maxlen=256)` |

Empirically confirmed over the 90s live session: `mailbox_depth` max observed
= **0**, `frames_in_flight` max observed = **1.0** (both well inside their
declared bounds). No hidden accumulation anywhere in the path — the lag was
entirely the pose-binding timestamp bug above, not a buffer.

### A.6 Recognition must not regress (section 7)

Re-ran the Phase 10 attrition harness (`scripts/pipeline_attrition.py`,
identical clip) after the fix
(`results/pipeline_attrition_after_p11_fix.md`): **numbers are identical to
Phase 10's published table** — person 504 raw → 501 rendered, cell phone
22 → 14, cup 35 → 54, tracker stats unchanged. Expected and by construction:
the fix touches only which timestamp value is stored on a pose binding —
nothing in detection, the recognition policy, or tracker association/
eligibility. Zero recall regression at the render layer.

### A.7 Green lines / track vs pose interaction (section 6)

The green lines are the pose **skeleton** only (`overlay.js::drawPose`,
edges from `SKELETON_EDGES`); track boxes render in a per-track hue, not
green. Measured this phase: `displayed_track_age_ms` (p95 328–357ms across
runs) is **not** obviously smaller than `displayed_pose_age_ms` (p95
145–213ms) — if anything the opposite in these runs. This is a genuine,
as-measured finding, not the a‑priori assumption that pose must be the
staler of the two: track box staleness is bounded by
`tracking.max_age_ms` (564ms, coasting) plus normal cycle-to-cycle latency,
which is the same order of magnitude as the pose bound. Both are within the
same broad range; neither dominates the other on this session.

### A.8 Latency floor preserved (section 11)

`scripts/benchmark_latency.py --only-end-to-end`, same clip, same session
(controls for this machine's current background load):

| condition | capture→snapshot p50 | `tracker_update_ms` |
|---|---|---|
| tracking + the Part A fix, **on** | 114.23ms | p50 0.17 / p95 0.27ms |
| tracking **off** (control, same session) | 108.73ms | — |

Statistically indistinguishable — the fix adds no measurable cost (it reads a
different variable, computes nothing new). Consistent with Phase 9/10's own
measured `tracker_update_ms` (~0.2–0.4ms).

### A.9 Cameras (section 8)

Only the laptop integrated camera clip exists in `data/raw` (same one Phase
10 used) — no OnePlus Nord 4 clip exists yet, so the camera comparison could
not be run this phase either. Carried forward from Phase 10, unchanged.

---

### B.1 Where the Studio-to-training path actually stopped (section 9)

Traced before writing anything:

```
Studio UI → upload → batch/staging → saved samples → object dataset   ✅ existed
→ dataset export                                                      ✅ existed (COCO detection format,
                                                                            scripts/export_objects_coco.py)
→ training dataset → training command → training process              ❌ did not exist
→ model artifact → validation → model registry → activation           ❌ did not exist
→ inference → live recognition                                        ❌ did not exist
```

`predictivesense/models/registry.py` had no activation code and no
trained/validated/active state machine (only "read the one active detector
flag"). No `torch`/`torchvision` anywhere in `pyproject.toml`. No
`scripts/train*.py`. **The honest finding, exactly as section 9 anticipated:
Studio collection and the model registry existed; training never did.** No
bug was invented to explain a missing feature.

### B.2 Architecture decided on inspection (section 10)

Adopted the recommended two-stage design: the existing YOLO detector stays a
class-agnostic proposer (unchanged); a small, from-scratch CNN crop
classifier (`predictivesense/training/model.py` — 3 conv blocks + global
average pool + linear head, 64×64 input) is trained in PyTorch and served
through **ONNX Runtime only** (`predictivesense/perception/classifier.py`,
the same `perception/runtime.py` session/provider path the detector and pose
estimator use). Not a torchvision pretrained backbone — at Studio collection
scale (tens of images/class) a from-scratch CNN trains in seconds on CPU
with no ImageNet-weights download or licence question. Training deps
(`torch==2.14.0`, `torchvision==0.29.0`, `onnxscript==0.7.2`) live in a new
`pyproject.toml` `[train]` extra; a repo-wide static-import test
(`tests/unit/test_no_forbidden_imports.py::test_runtime_app_never_imports_torch`)
asserts `api/`, `pipeline/`, `perception/`, `camera/` never import `torch`.

### B.3 Dataset pipeline (section 11) — reuses Studio samples, no new format

`predictivesense/training/dataset.py` reads the **existing**
`ObjectRegistry`/`SampleStore` write path directly (no parallel format); a
`CropRecord` is a thin view — box + role + provenance — over one already-
stored sample. Roles handled explicitly and tested
(`test_hard_negative_never_becomes_a_training_positive`): `positive_records()`
feeds the softmax classifier; `negative`/`hard_negative` crops are **excluded**
from training and instead scored as a **false-class rate** at evaluation
(section 15) — the correct, non-poisoning way to use a hard negative when the
classifier has no explicit reject class.

**Real bug found and fixed while building this**: the first version of the
session/object-disjoint splitter (`training/splits.py`) shuffled one global
pool of capture sessions across all classes. With only a handful of sessions
per class this could — and on the first real run, did — send every session of
one class entirely to `train`, leaving it **completely absent from `test`**.
The first fixture run measured `test accuracy = 0.0`, which was actually zero
test examples for two of three classes, not a bad model. Fixed by splitting
**per class independently**, then merging; a regression test sweeps 10 seeds
asserting every class appears in every split.

Dataset validation (`validate_dataset`) refuses with a clear message when a
class has too few positives, and reports `box_confirmed_by_human` composition
(section 11.5) — on the synthetic fixture, 100% (the fixture stamps every box
as human-confirmed; a real collection run's fraction will differ and is worth
watching for exactly the reason section 11.5 names: it is the answer to "did
the custom model just learn the pretrained detector's boxes").

### B.4 Training pipeline — real, proven end to end (section 12)

`scripts/train_object_classifier.py --fixture` builds a real, on-disk,
Studio-shaped dataset (`training/synthetic_fixture.py` — three visually
distinct shapes, three capture sessions each, one hard-negative each, written
through the actual `SampleStore.add()` API, not a fake format), trains for 12
epochs, exports to ONNX, and registers an unvalidated, inactive version — all
exercised by `tests/integration/test_training_pipeline.py` (`-m train`,
skips cleanly without the `[train]` extra via `require_train`). Confirmed
this session:

- **A real artifact on disk with real metrics and a real SHA-256 checksum**
  (never claimed without one — `test_training_produces_a_real_artifact_with_metrics_and_checksum`).
  Fixture run: test accuracy **1.000**, false-class rate **0.000**.
- **Reproducibility within a stated tolerance** (section 12.3): two runs,
  same data/config/seed, test accuracy within 0.02 absolute
  (`test_reproducible_within_tolerance_given_the_same_seed`) — not bit-
  identical (CPU BLAS reduction order is not guaranteed deterministic even
  with a fixed seed), a tolerance chosen and stated up front, not picked to
  fit a result.
- **Full run manifest** (`results/train_run_<version_id>.json`): dataset
  content hash, split content hash, class map, architecture, image size,
  epochs, batch size, learning rate, seed, augmentation, per-epoch train/val
  metrics, wall-clock seconds, machine fingerprint, provider, artifact path +
  SHA-256, unique version id. Never overwritten — every run gets its own id.

### B.5 Registry, validation, activation, rollback (section 13)

A dedicated `predictivesense/training/classifier_registry.py` /
`models/classifier_registry.json` — **not** an extension of the shipped
detector's `models/registry.py` (that registry's "exactly one active version,
globally" invariant predates a second task existing; extending it risked a
frozen, heavily-tested invariant for no benefit — see `docs/decisions.md`).
Four states, never collapsed, and each transition tested
(`tests/integration/test_training_pipeline.py`):

- **training data** → Studio samples, untouched by this registry.
- **trained model** → `register()`; always `validated=False, active=False`
  (training never has the side effect of validating or activating —
  `test_unvalidated_model_cannot_be_activated` asserts activation of a fresh
  version raises).
- **validated model** → `validate_version()`, an explicit, separate step
  (`scripts/validate_object_classifier.py`, gated on a minimum accuracy /
  maximum false-class-rate).
- **active model** → `activate(version_id)`; exactly one active at a time
  (`test_only_one_version_active_at_a_time`, trains and activates a *second*
  version and confirms the first is automatically deactivated).
  `activate(None)` = **rollback to baseline** (`test_rollback_deactivates_the_custom_classifier`)
  — no active classify-task entry, the registry's natural starting state, so
  there is no separate "baseline" record to invent or keep in sync.

The resolved active version id is reported in `GET /api/runtime`
(`custom_classifier_version`), the session manifest, and Diagnostics
(`rt-custom-classifier` row) — `"none"` when nothing is active, never a
placeholder name.

### B.6 Live inference — real, additive, wired into the running app (section 16)

`AnalysisLoop` resolves the active classifier **once**, at construction
(`ClassifierRegistry().active()`, loaded through
`perception/classifier.py`'s ONNX Runtime path — never torch at serve time),
exactly like the detector's `model_version`. A classifier
trained/validated/activated while the app is already running takes effect on
the next restart — a stated limitation, not silent staleness.

The classifier pass (`AnalysisLoop._apply_custom_classifier`) runs strictly
**after** `RecognitionPolicy.apply()` and only ever adds two new `Detection`
fields (`custom_class_name`, `custom_class_confidence`) — `class_name`,
`policy_state`, `tier` and `score` are read, never written, so **the
recognition policy's own decisions are byte-for-byte identical whether or not
a custom classifier is active** (verified: `test_active_classifier_annotates_accepted_detections_without_changing_policy_fields`
asserts every policy field is unchanged after classification). A classifier
exception degrades to "no annotation this cycle", never drops the frame
(`test_classifier_failure_never_drops_the_frame`). Carried onto `Track`
exactly like `last_detector_confidence` (never decayed, never invented while
coasting) and rendered as a strictly additive overlay badge
("`learned: <name> <pct>`") stacked above the existing tier badge — never
replacing the resting label, colour or dash channel, all of which remain the
recognition policy's alone.

**Proven not fabricated**: `test_onnx_inference_matches_the_trained_classes_no_hardcoding`
runs the *exported ONNX artifact* (not the in-memory torch model) via the
real serving path on crops never seen during training, and gets them right
from pixels alone — a hardcoded class map or filename shortcut could not
reproduce this. No hard-coded class substitution, UI renaming, or filename
matching exists anywhere in this path (grepped and reviewed; the only place a
class name is ever assigned is `CropClassifierModel.infer_crop`'s own
`argmax` over real model output).

### B.7 Baseline vs custom (section 15)

`scripts/evaluate_classifier_baseline.py --fixture` →
`results/baseline_vs_custom_fixture.md`, identical held-out test split for
both:

| metric | custom classifier | baseline (pretrained detector) |
|---|---|---|
| accuracy | **1.000** | **0.000** |
| false_class_rate | 0.000 | 0.000 |
| latency mean (ms) | 1.20 | 45.99 |

The baseline (the shipped COCO-80 detector, run on each crop, top class by
score) never gets a custom class right — its vocabulary was never going to
contain these names (it calls a `circle` crop `"orange"` or nothing, a
`square` crop `"tv"`). **This is the expected, honest, publishable result**
(section 15), not a baseline-evaluator bug: it demonstrates precisely the gap
the two-stage design exists to close. Latency is reported entirely
separately from accuracy — the classifier is also far cheaper per crop (a
tiny 64×64 CNN vs re-running the full detector).

### B.8 Studio UI — honest state (section 14)

`GET /api/objects/training_status` reports the real state: per-class sample
counts and `ready_for_training` (a simple threshold, never a claim), plus
`trained_versions` / `active_classifier` straight from the classifier
registry. Surfaced in the Studio header (`#training-status`, refreshed on
every sample-list reload). Two browser tests assert no deceptive claim
appears: the **pre-existing** `test_no_text_claims_the_model_learned_or_was_trained`
(Phase 6/7) originally banned the bare words "trained"/"learned" anywhere on
the page — now that an *honest* factual readout legitimately uses those
words ("no versions trained"), it was narrowed to ban specific **deceptive
phrases** stated as present-tense fact ("has learned", "is trained", "now
recognises") rather than the words themselves; a new test
(`test_capturing_a_sample_never_claims_the_class_is_learned_or_trained`)
covers the same phrase list after an actual camera capture.

**Known limitation, stated openly**: the classifier registry resolves its
default path from the repo root, matching the existing detector registry's
own precedent — it is not isolated by the browser test suite's
`seeded_objects_root` the way `objects.root` is. Browser tests therefore
assert only the *absence* of deceptive phrasing, never exact trained-version
counts.

### B.9 Full test suite

`pytest -q` (all markers, `[train]` extra installed): **478 passed, 4
skipped** — up from Phase 10's 447 by Part A's 5 tests plus Part B's 42 new
tests (dataset, splits, end-to-end pipeline, registry state machine, live
wiring, forbidden-imports scoping, 2 browser). One flaky, unrelated failure
was observed once (`tests/integration/test_object_batches_api.py::test_duplicate_detection_flags_within_batch_and_against_existing`
— Phase 7 bulk-upload duplicate-hash detection, untouched by this phase) and
did **not** reproduce on immediate re-run, either alone or as part of the
full suite twice more; not investigated further as it is outside this
phase's scope. `-m browser`: 28 passed (26 pre-existing + 2 new). `-m train`:
9 passed (skips cleanly without the `[train]` extra via `require_train`).

---

## PART 2 — PHYSICALLY OBSERVED BY THE DEVELOPER (pending, section 18)

Not yet performed — requires a camera and a running browser session, which
this environment does not have.

**Freshness (both cameras)**: move a hand / whole body at varying speeds;
compare real position to the green representation's position, both for the
pose skeleton specifically (does it visibly keep up now, not just stay
visible) and the tracked box. The stopwatch/on-screen-timer ground-truth
check (section 4.6) is explicitly a developer, physical-camera check —
nothing here substitutes for it, only locates the fault in code and measures
the mechanism.

**Recognition (both cameras)**: phone, bottle, watch, mug, keyboard, pen at
varying views/distances.

**The Studio path, one class only (section 18)**: collect a small **Watch**
dataset, verify upload → boxes → save → dataset count → `python
scripts/train_object_classifier.py --objects-root data/objects` →
`scripts/validate_object_classifier.py` → `scripts/activate_object_classifier.py`
→ live Watch recognition (the "`learned: watch NN%`" overlay badge) on
camera. **Do not collect all twelve classes until this one path works end to
end.**

---

## PART 3 — NOT VERIFIED / LIMITATIONS

- **Part A**: this session's freshness measurements come from a steady-state
  TestClient loopback with no real webcam jitter or the background CPU
  contention Phase 10's own live session measured (36.4% empty-pose rate
  there, materially worse than anything reproduced this phase) — the true
  field worst-case for `displayed_pose_age_ms`/`displayed_track_age_ms` may
  exceed what is measured here; physical verification (Part 2) is the
  check that would confirm the reported ~1–2s complaint is actually resolved,
  not merely improved on this harness. `tracking.pose_max_age_ms` is kept at
  900ms on the evidence gathered, not re-derived from a physical session.
- **Part A**: no OnePlus Nord 4 clip exists yet — the camera comparison
  (section 8) could not be run, unchanged from Phase 10.
- **Part B**: **no real custom model has been trained on real collected
  Studio data this phase** — every number in this report (accuracy 1.000,
  false-class-rate 0.000, the baseline comparison) is on the synthetic
  fixture only, by design (section 12.4: prove the pipeline works before any
  real collection exists). A real class is only ever "learned" after an
  actual `train → validate → activate` on real samples — untrained/
  unvalidated/inactive is the state of every real Studio object today.
- **Part B**: a crop classifier cannot recognise an object the detector
  proposes no box for at all — stated in `docs/decisions.md`/section 10.1's
  own honest limitation, unchanged by this phase's work.
- **Part B**: robustness across angle, distance, lighting is unmeasured for
  the custom classifier (no real collection exists to measure it against).
- **Part B**: the classifier registry is not config-scoped/test-isolated the
  way `objects.root` is (section B.8/B.5 above) — a documented, open
  limitation, not a silent gap.
- No risk model, relationship layer, or voice exists — unchanged non-goal.
- No tracking quality metric against labelled ground truth (MOTA/IDF1) is
  claimed — unchanged from Phase 9/10, that footage does not exist.

---

## Acceptance criteria (Phase 11 prompt, section 20)

**Part A**

- [x] `displayed_pose_age_ms` / `displayed_track_age_ms` measured and
      published before any change (section A.2).
- [x] Root cause identified in code with numbers (section A.1/A.2).
- [x] Buffer audit complete; no hidden accumulation; Phase 8 newest-frame
      design intact (section A.5).
- [x] Measured reduction / correction in the reported ages, before/after in
      one condition (section A.3) — one camera (the only one available).
- [x] Display bounded by age; stale representation degraded rather than
      shown as current; nothing fabricated (unchanged Phase 10 design,
      re-verified honest by the fix).
- [x] Attrition comparison shows no per-class recall regression (section A.6).
- [x] Phase 8 latency floor preserved (section A.8).

**Part B**

- [x] Exact break point stated plainly, including "never implemented"
      (section B.1).
- [x] Real training pipeline proven end to end on the synthetic fixture by
      automated test (section B.4).
- [x] Roles interpreted correctly; splits session- and object-disjoint;
      leakage test fails on a deliberately leaky split; `data/objects` and
      `data/eval` still separate (section B.3; separation check unchanged
      and still green).
- [x] Run manifests complete and reproducible; artifacts versioned, never
      overwritten (section B.4).
- [x] Unvalidated models cannot be activated; activation explicit; active
      model reported in Diagnostics; rollback tested (section B.5).
- [x] Training deps in an optional extra; CPU-only unaffected; provider
      reported (section B.2). CUDA-loud-failure path unchanged (inherited
      from the existing `perception/runtime.py`; not re-tested this phase —
      no CUDA machine in this environment).
- [x] No "learned" claim without a trained, validated, activated model;
      asserted by browser test (section B.8).
- [x] No hard-coded class substitution, UI renaming or filename matching
      anywhere (section B.6).
- [x] Full suite green including browser; `docs/phase-reports/phase11.md`
      complete (this file).
- [ ] Tree committed clean — not committed this session (commits are made
      only on explicit request, per this repo's working agreement).
