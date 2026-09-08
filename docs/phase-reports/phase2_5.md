# Phase 2.5 - Recognition Reliability & Measurement

Scope: make perception **trustworthy and measurable**. A labelled evaluation set
from the project's own footage, an evaluation harness that reports real detection
metrics, and a switchable recognition policy layer that suppresses out-of-domain
classes and rejects weakly-supported predictions as `unknown`. One
permissively-licensed alternative detector wired for comparison. **No** tracking,
object-enrollment UI, training, fine-tuning, temporal reasoning, risk, or voice.

Three separated parts, as required.

---

## PART 1 - MEASURED (commands run on this machine)

Machine: Windows 11, Intel Core Ultra 5 125H (14C / 18T), 16 GB LPDDR5, Intel Arc
integrated graphics, no NVIDIA / no CUDA. Python 3.11.9. `onnxruntime==1.24.4`
(CPU) in `.venv`.

### Test suite

- `pytest -q` -> **250 passed, 2 skipped** (1 `hardware` - `--run-hardware`;
  1 `dataset` - skips cleanly, no labels yet). Baseline before this phase was
  205 passed / 1 skipped; +45 tests.
- `pytest -q -m models` -> **12 passed** (Phase 2's 11 + the new
  perception-with-policy determinism case). Skips cleanly when `models/` is empty.
- `pytest -q -m dataset` -> **1 skipped** with the message
  "no evaluation set ... run `scripts/build_eval_frames.py` and label at /label"
  (never fails, never silently passes).
- New unit tests: `test_coco_store` (round-trip, schema rejection, id allocation,
  seeded preserved, `.bak` on overwrite), `test_splits_no_leakage` (no shared
  session/image id, hash moves with content, a deliberately leaky split is
  rejected, single-session -> empty test, refuses when nothing labelled),
  `test_matching` (greedy IoU: correct assignment, duplicate -> FP, unmatched GT
  -> miss, score order decides the claim, threshold respected, empty-safe),
  `test_metrics` (P/R/F1 hand case, confusion totals reconcile = 5, false-class
  rate 1/3, mAP@0.5 = 1/6 hand-computed, empty-safe), `test_policy_rules` (each
  rule in isolation, counts reconcile exactly, every rule independently
  switchable, disabled = pass-through, per-class threshold override, a
  `domain_classes` typo fails loudly).
- New integration tests: `test_label_api` (post boxes -> read back, progress
  counts update, provenance preserved, bad category -> 400, 404, image bytes),
  `test_eval_harness` (fixture metric values + both output files written; a
  `dataset`-marked real-set consistency check).
- Phase 2 / Phase 1.6 regression tests updated where a precedent already exists:
  `test_ui_structure.py` now expects `research` registered and only `alerts`
  reserved (exactly as Phase 2 did for `analysis`); no other regression touched.

### Labelled evaluation set

`scripts/build_eval_frames.py --source data/raw --out data/eval/frames
--every-n 15 --max-per-clip 40` sampled **34 frames** from the **1** clip
currently in `data/raw/` (**1 recording session**, the Integrated Camera). Each
frame carries provenance (source clip, capture timestamp, session id, camera
device, condition tag `cabin-unlabelled`). The COCO store
(`data/eval/annotations.json`, 14 categories = `policy.domain_classes`) is
created with **0 annotations** - the labelling itself is the developer's step
(BLOCK 14). The `build` script warns loudly that only 1 session is available
against the BLOCK 3.1.7 target of >= 4.

**Everything downstream of labelling is built, tested, and blocked cleanly:**

| command | current result |
|---|---|
| `build_splits.py` | `ERROR ... 0 of 34 frames are labelled - label at /label` (exit 2) |
| `fit_thresholds.py --split val` | `ERROR ... no split file ... run scripts/build_splits.py` (exit 2) |
| `eval_detection.py --split val --model yolo11n --policy off` | same (exit 2) |

The harness refuses an empty / partial split and says what is missing (BLOCK
3.11), so the "baseline vs policy on `val`" table is **pending the developer's
labels**. What *can* be measured now is below.

### Recognition policy - mechanical effect on the 34 unlabelled frames

`scripts/policy_effect.py` (counts, **not accuracy** - there are no labels):

| model | raw dets | accepted | unknown (low-conf / margin) | rejected (domain / size) | reconciles | notable raw->decided |
|---|---|---|---|---|---|---|
| yolo11n | 43 | 39 | 0 | 4 (4 / 0) | ✔ | `umbrella -> unknown` ×4 |
| yolox_tiny | 71 | 37 | 0 | 34 (34 / 0) | ✔ | surfboard / bench / frisbee -> unknown |

`accepted + unknown + rejected == raw` holds for every frame (asserted). This is
the mechanism behind "watch -> donut": the out-of-domain raw label
(`umbrella`, `surfboard`, ...) becomes a dashed `Unknown` box instead of a
confident wrong label. The `unknown_low_confidence` / `unknown_margin` counts are
0 here because on this single close-up clip every surviving detection is a
high-confidence `person`; those rules will bite once `val` has more varied
content and `fit_thresholds.py` has set real per-class thresholds.

### Policy layer cost - BLOCK 11 ( < 1 ms p95 )

`scripts/benchmark_latency.py`, 150 frames: **policy p50 0.05 ms, p95 0.07 ms**
(threshold: < 1 ms p95). PASS.

### Latency - pose gating, thread sweep (isolated run, 150 frames of `data/raw/`)

`scripts/benchmark_latency.py --source data/raw --limit 150 --threads 4,6,8`,
isolated (nothing else running), 1080p frames, CPU, `intra_op_threads: 6`:

| run | frames | pose ran | detector p50 / p95 | pose p50 / p95 | policy p50 / p95 | combined p50 / p95 | implied FPS |
|---|---|---|---|---|---|---|---|
| before - pose every frame | 150 | 150 | 66.5 / 80.3 | 53.4 / 59.0 | 0.05 / 0.07 | 120.5 / 135.8 | 8.3 |
| after - pose gated on `person` | 150 | 150 | 70.3 / 101.9 | 56.6 / 80.5 | 0.05 / 0.08 | 127.4 / 180.8 | 7.9 |

intra-op thread sweep (pose every frame): **4 -> 134.1 ms**, **6 -> 125.1 ms**
(shipped), **8 -> 149.9 ms** combined p50 - the knee is 6, as in Phase 2.

**Pose gating** (`perception.pose_requires_person`): on this footage the
developer is in **every** frame, so `person` is always present and pose runs on
100 % of frames whether gating is on or off - **gating has no effect here** and
the default stays `false`. It is retained as a config knob for footage with
person-free stretches; the phase report's honest finding is that this clip cannot
exercise it.

**Detector input size** stays **640**: BLOCK 3.5.21 says "adopt the measured-best
input size from the Phase 2 sweep", and that sweep's conclusion
(`results/input_size_sweep.md`) was 640 for small-object sensitivity, with 480
the documented fallback. No labelled evidence yet supports changing it.

**ORT thread options** are now set explicitly (`inter_op_num_threads = 1`,
`execution_mode = ORT_SEQUENTIAL`, `intra_op` swept) - see the sweep row; it
re-confirms Phase 2's `intra_op_threads: 6` choice as the shipped default.

**End-to-end capture -> overlay latency** (p50 / p95) needs a live browser
painting the overlay canvas and is a developer-performed measurement (Part 2),
exactly as the equivalent Phase 2 check was. The backend-observable portion
(capture_ts -> snapshot `emitted_ts`, i.e. `frame_age_ms`) with perception +
policy active was re-checked by the existing browser-ingest loopback and is
unchanged from Phase 2 (~146 ms p50) - the policy adds < 0.1 ms.

### Model comparison

`results/model_comparison.md` - per-model size, licence, isolated latency, and
the unlabelled policy-effect counts. The **per-class F1 / false-class-rate**
columns are marked pending: they require `val` labels. See PART 3 and
`docs/attribution.md` - the AGPL-3.0 licence decision is **not** claimed settled
this phase; YOLOX-tiny (Apache-2.0) is wired and partially measured, the labelled
comparison decides it.

### Preview independence

`pytest -q -m models tests/integration/test_perception_pipeline.py::test_preview_
independence_holds_with_perception_running` passes with the recognition policy
also active (the policy runs inside `_run_perception`; `policy.apply` never raises
into the loop). Socket keeps accepting every frame, mailbox stays single-slot,
producer thread alive, `loop.error is None`.

### `test` split

**Not opened.** `results/test_set_openings.md` exists with its header and **no
data rows** - `eval_detection.py --split test` has never been run (and would fail
now for lack of labels). The first opening will be logged there with date +
reason.

---

## PART 2 - PHYSICALLY OBSERVED BY THE DEVELOPER

**Status: PENDING - none performed this pass.** The in-app Browser pane blocks
camera capture, so no browser ran the live overlay against a camera. Motivating
Phase 2 observations, attributed to the developer and **clearly observations, not
measurements**:

> watch -> *donut*; headphones -> *person*; mug -> *phone*; bowl -> *wine
> glass*; keyboard -> *TV remote*; can -> *phone*; bottle inconsistent;
> spectacles, charger, shaker not detected; small handheld objects unreliable;
> labels slow to update.

The developer must perform and record:

1. **Label the 34 sampled frames at `/label`** (and record more clips first if
   possible - both cameras, >= 4 sessions). **At least 30 % labelled with seeding
   OFF** (`min_unseeded_fraction: 0.30`); the harness reports the unseeded subset
   separately as the recall-bias check.
2. On the live overlay, confirm `unknown` is visually obvious - dashed outline,
   muted grey, label `Unknown` - and never looks like a confident detection.
3. Confirm the previously-observed confident errors (watch -> donut, bowl ->
   wine glass, keyboard -> remote) now render as `Unknown` or disappear rather
   than showing a confident wrong label.
4. Confirm labels update no slower than Phase 2 and the preview stays smooth with
   the policy active.

Then re-run `build_splits.py` -> `fit_thresholds.py --split val` -> the four
`eval_detection.py` runs (val policy off, val policy on, val yolox_tiny policy
on, and finally **once** `--split test`), and fill the tables in PART 1 and
`results/model_comparison.md`.

---

## PART 3 - NOT VERIFIED / LIMITATIONS

- **The evaluation set is not yet labelled.** Every "baseline vs policy on `val`"
  number, per-class P/R/F1, the confusion matrix over real ground truth, the
  false-class rate headline, mAP@0.5, and the fitted per-class thresholds /
  `margin_min` are **pending the developer's labelling**. The harness, splits,
  policy and reports are built and tested; the inputs are not there yet.
- **When labelled, the set will still be small and from one environment.** Only
  1 clip / 1 session / 1 camera exists in `data/raw/` today (34 sampled frames).
  BLOCK 3.1.7 targets ~200-400 frames across >= 4 sessions and both cameras. The
  harness prints the sample count beside every metric so a thin number is never
  read as solid; metrics from this set will **not generalise** beyond this room.
- **Detector-assisted seeding may bias recall upward.** A missed object the
  detector never proposed is easy to overlook. The tool makes adding a missed box
  exactly as cheap as accepting a proposed one, every frame records `seeded`, and
  the harness must report the >= 30 % unseeded subset separately as the check.
- **`unknown` handling reduces confident errors but adds no new class.** Objects
  absent from the model's COCO-80 vocabulary - watch, spectacles, charger,
  headphones, shaker - remain unrecognised (they become `Unknown` at best, or are
  not detected at all) and will require a custom-trained model in a later phase.
- **`policy.default_threshold` (0.35), `margin_min` (0.10) and
  `min_box_area_frac` (0.0005) ship at prompt-suggested defaults**, not
  val-fitted values, until the developer labels and `fit_thresholds.py` runs.
  They are flagged as judgement calls in `docs/decisions.md`; the profile comment
  names `results/fit_thresholds_val.md`.
- **The AGPL-3.0 licence decision is NOT settled.** YOLOX-tiny (Apache-2.0) is
  integrated through the identical wrapper / policy / harness and its *unlabelled*
  behaviour is measured (looser: 71 raw dets vs 43, 34 out-of-domain rejections
  vs 4), but the labelled per-class F1 / false-class comparison on `val` that
  would actually decide it has not been run. Default stays AGPL YOLO11n.
- **Pose gating could not be exercised** - the only footage has a person in every
  frame. The knob exists and is measured to be free (< 0.1 ms decision) but its
  benefit is unquantified on this machine.
- **End-to-end capture -> overlay latency** (the number the developer feels as
  "labels are slow") is a browser measurement and is developer-pending, like the
  equivalent Phase 2 check.
- **No tracking, temporal reasoning, risk inference, alert policy or voice
  exists.** `StateSnapshot.tracks` is still always `[]`. No object-enrollment UI.
  Phase 3 has **not** been started.
