# Phase 13 — Current-State Audit (independent verification)

**Auditor:** Claude (planning/audit layer), working over the connected-folder bridge into
`C:\Users\mummi\Documents\Projects\PredictiveSense`.
**Date:** 2026-09-17.
**Status:** AUDIT COMPLETE, WITH TWO STATED GAPS (see §0).
**Nothing in the repository was modified by this audit except the creation of this file.**

---

## 0. Scope and honest limits of this audit

Everything below was read from the working tree on this machine. Where a previous report or a
previous AI message is quoted, it is marked as a claim and separately marked verified or not.

**Two things this audit could NOT do, and did not fake:**

1. **The test suite was not executed by the auditor.** The audit shell runs in a Linux VM with the
   repository mounted; the project's interpreter is the Windows `.venv` (Python 3.11.9 with
   `onnxruntime`, `torch`, Playwright). Running `pytest` from the audit shell would have tested a
   different environment, so it was not attempted. Test evidence below is *circumstantial* (pytest
   cache state) and is labelled as such. **Exact pass/fail numbers must be produced by running the
   suite on Windows.** See §H.
2. **No live inference, no camera, no latency measurement.** Every performance number in this
   repository was produced by earlier phases; none was re-measured here. Anything not re-measured
   is reported as previously-recorded, not as current fact.

---

## A. Executive summary

**What the project currently is.** A working, well-instrumented *perception plumbing* system:
browser-owned camera capture → single-slot newest-frame mailbox → ONNX Runtime YOLO11n detector
(+ YOLO11n-pose on a cadence) → three-tier recognition policy → IoU/motion tracker with persistent
IDs → push-on-publish `/ws/state` snapshot → canvas overlay. 81 test files, ~515 collected tests,
extensive telemetry, a recorded/deterministic pipeline, and a documented decision log.

**What works (verified in code).** Studio isolation (Phase 13 Stage 2) is genuinely implemented and
genuinely tested, including a subprocess test that asserts the classifier modules are not even
*imported* by the live path. The tracker is detector-independent. The baseline detector is
registered, hashed and licence-documented. Dataset separation is enforced by dedicated tests.

**What does not work.** There is **no custom object detector**. There never has been. The only
trained artifact in the repository is a single-class crop classifier whose own registry entry
records `false_class_rate: 1.0` — it labels 100% of its rejection examples as "watch". It is now
switched off. Neither `watch` nor `comb` is detectable today.

**What has been isolated.** The Studio *classifier* (off, code-default and in every shipped
profile) and the misleading `learned: …` overlay badge (deleted outright, not gated). The Studio
*UI and data-collection routes* remain **enabled in the dev and eval profiles** — see §D, this is a
deliberate documented choice, not an oversight, but it means "Studio is off" is only half true.

**What remains.** Everything that makes this a detection project rather than a plumbing project:
a labelled evaluation set (currently **zero annotations**), an actual custom-class detector, a
detector training pipeline, and the licence decision.

---

## B. Repository map

| Path | Role |
|---|---|
| `predictivesense/camera/` | Browser ingest, `LatestFrameMailbox`, framing, device enumeration |
| `predictivesense/perception/` | `detector.py`, `pose.py`, `engine.py`, `runtime.py` (EP selection), `policy.py`, `vocabulary.py`, `preprocess.py`, `postprocess.py`, `classifier.py` (crop classifier — now dormant) |
| `predictivesense/pipeline/` | `loop.py` (AnalysisLoop), `perception_frame.py`, `recorded.py`, `scheduler.py` |
| `predictivesense/tracking/` | `tracker.py`, `track_state.py`, `association.py`, `geometry.py` — **untracked by git** |
| `predictivesense/training/` | Crop-classifier training: `train.py`, `dataset.py`, `splits.py`, `classifier_registry.py`, `evaluate.py` — **untracked by git** |
| `predictivesense/api/` | FastAPI app, `/ws/state`, `/ws/ingest`, Studio/objects routers, static dashboard |
| `config/profiles/` | `dev.yaml`, `eval.yaml`, others |
| `models/` | `yolo11n.onnx`, `yolo11n-pose.onnx`, `yolox_tiny.onnx`, `registry.json`, `manifest.json`, `classifier_registry.json`, `custom/` |
| `data/` | `eval/`, `external/`, `objects/`, `raw/`, `videos/` |
| `docs/` | `architecture.md`, `decisions.md`, `attribution.md`, `setup.md`, `phase-reports/` (phase0–phase13-stage1) |
| `results/` | ~170 measurement artifacts from Phases 2–12 |
| `tests/` | 81 test files: `unit/`, `integration/`, `browser/`, `hardware/` |

---

## C. Current pipeline (verified)

```
browser <video> capture (worker: MediaStreamTrackProcessor -> downscale -> JPEG)
  -> WS /ws/ingest  (api/ingest.py:43-105; refuses 4403 non-browser owner, 4409 while Studio active)
  -> BrowserSource.submit -> LatestFrameMailbox (single slot, overwrite, counted drops)
  -> AnalysisLoop consumer thread (pipeline/loop.py; threads spawned at loop.py:277-284)
     -> preprocess (letterbox) -> ONNX Runtime detector session
        models/yolo11n.onnx, input_size 480, nms_iou 0.5, default_conf 0.35,
        max_detections 100, provider: auto  (config/profiles/dev.yaml:118-137)
     -> postprocess/NMS -> Detection(bbox, class_id, class_name, score, frame_id,
        raw_class_name, policy_state, runner_up, tier, custom_class_* [now always None])
     -> recognition policy (perception/policy.py, vocabulary.py): three tiers
     -> pose on cadence every_n:2, reuse <= 500 ms (perception/pose.py)
     -> Tracker.update(detections, poses)  (tracking/tracker.py:89)
     -> PerceptionFrame -> StateSnapshot (core/types.py:307-345)
  -> Broadcaster.publish (push-on-publish, api/broadcast.py:54-65)
  -> WS /ws/state -> static/features/overlay.js drawTrack()
```

Staleness guard, drop accounting, preview independence and the recorded/deterministic path all
have dedicated tests (`test_staleness_guard.py`, `test_mailbox.py`,
`test_preview_independence.py`, `test_slow_detector_bounds_latency.py`,
`test_recorded_determinism_with_models.py`).

---

## D. Studio status — Stage 2 VERIFIED, with one important qualification

**Verified implemented:**

| Claim | Verdict | Evidence |
|---|---|---|
| `TrainingConfig.classifier_enabled` exists, default `False` | **TRUE** | `predictivesense/config/settings.py:636` |
| `StudioConfig.enabled` exists, default `False` | **TRUE** | `predictivesense/config/settings.py:639-653` |
| Classifier resolution gated in the loop | **TRUE** | `pipeline/loop.py:93-97` — `_resolve_active_classifier` called only if the flag is on |
| Studio routers gated in `create_app` | **TRUE** | `api/app.py:187-214` |
| `learned: …` overlay removed | **TRUE — deleted, not gated** | `static/features/overlay.js:397-410` is now an explanatory comment block; repo-wide grep for `learned` under `static/` returns only comments and Studio's own "never claim learned" notes |
| Studio source/data/registry preserved | **TRUE** | `predictivesense/training/`, `data/objects/`, `models/classifier_registry.json` all intact |
| Tests added for Studio-disabled behaviour | **TRUE** | `tests/integration/test_studio_isolation.py` — 7 tests incl. a **subprocess test** asserting `predictivesense.training.classifier_registry` and `predictivesense.perception.classifier` are absent from `sys.modules` after building a real `AnalysisLoop` |

**The qualification you need to know about:** `studio.enabled: true` is set explicitly in
**`config/profiles/dev.yaml:239-245` and `config/profiles/eval.yaml:166-169`**. Those are the
profiles the application actually runs with. So in practice:

- The Studio **classifier** is off everywhere (code default *and* both profiles) — this is the part
  that caused the false badge, and it is genuinely off.
- The Studio **UI, routes and data-collection workflow** are **on** whenever you run the app.

This is documented and deliberate (`test_shipped_dev_and_eval_profiles_opt_studio_back_in`
asserts it on purpose, so you keep your collection workflow). It is not a defect. But "Studio is
isolated" means *the classifier cannot influence inference*, not *Studio is absent*.

**One live landmine:** `models/classifier_registry.json` still has the degenerate model marked
`"validated": true, "active": true`. Flipping `training.classifier_enabled` back on would reload it
immediately. The registry entry should be deactivated or annotated before that flag is ever touched.

---

## E. Baseline detector

| Field | Value | Source |
|---|---|---|
| Model | Ultralytics **YOLO11n** ONNX, pre-exported v8.3.0 | `models/manifest.json` |
| Weights | `models/yolo11n.onnx`, 10,930,182 bytes | on disk, verified |
| SHA-256 | `634279b4…0c053b` | `models/registry.json` + `manifest.json` (both agree) |
| Pose | `models/yolo11n-pose.onnx`, 11,853,451 bytes, sha `93e2866b…ff6408` | same |
| Vocabulary | COCO-80 | `manifest.json` |
| Registry version | `v1`, `name: baseline-yolo11n`, `active: true` | `models/registry.json` |
| Input size (live) | **480** (model supports dynamic; 640 nominal) | `config/profiles/dev.yaml:128` |
| NMS IoU / conf | 0.5 / 0.35, low-confidence band [0.35, 0.50], max 100 | `dev.yaml:130-135` |
| Device | `provider: auto` → cuda → directml → cpu; explicit `cuda` when absent fails loudly | `dev.yaml:112-117`, `perception/runtime.py` |
| Licence | **AGPL-3.0-only** | `manifest.json` `license_note`: *"releasing this repository under a non-AGPL licence while shipping these weights is an OPEN, UNRESOLVED decision"* |
| Alternative | **YOLOX-tiny**, `models/yolox_tiny.onnx`, 20,219,662 bytes, sha `427cc366…70b0f7`, **Apache-2.0**, input 416, COCO-80, YOLOX decode | `manifest.json` `detector_alt` |

**Stage 3 (baseline freeze) status: PARTIALLY PRE-EXISTING, NOT DONE AS A STAGE.** A registry with a
hashed, named, `active` baseline already exists from Phase 4 and predates Phase 13. What does *not*
exist is a second, custom model to be independently selectable *against* it, or any
baseline-vs-custom selection switch at runtime. There is a `results/baseline_vs_custom_fixture.md`,
but it refers to the crop-classifier fixture experiment, not a detector comparison.

Note `models/registry.json`'s own `_comment`: *"No model has been trained: v1 is the pre-exported
Phase 2 detector + pose."* That is still accurate.

---

## F. Tracker

`predictivesense/tracking/tracker.py` — single entry point `Tracker.update(detections, poses, …)`
at `tracker.py:89`.

- **Input contract:** `Sequence[Detection]` from `predictivesense.core.types` — `bbox`, `class_id`,
  `class_name`, `score`, `frame_id`, `raw_class_name`, `policy_state`, `runner_up`, `tier`.
- **Output:** `Track` objects with persistent integer IDs, status (`TrackStatus`), class-vote
  history (`majority_class` in `track_state.py`), `last_detector_confidence`, `fresh` flag.
- **Association:** two-stage (high/low confidence) greedy IoU + constant-velocity motion
  (`association.py`, `geometry.py`), with re-association of recently lost tracks
  (`_try_reassociate`, `tracker.py:340`).
- **Eligible states:** `accepted`, `accepted_secondary`, `unknown_low_confidence`, `unknown_margin`.
  `rejected_size` and `suppressed_implausible` never reach the tracker (`TRACK_ELIGIBLE_STATES`).
- **No inference, no I/O, in-memory state only** (module docstring, and verified by reading the file).

**"Can the detector be replaced without rewriting the tracker?" — YES.** The tracker depends only on
the `Detection` dataclass, never on the model, the class vocabulary, or the ONNX session. A new
detector that emits `Detection` objects with the same fields plugs in unchanged. The only coupling
worth naming is the hard-coded `_POSE_TRACK_CLASS = "person"` (`tracker.py:23`) — any replacement
detector must keep emitting a class literally named `person` for pose binding to work.

The tracker also copies `custom_class_name`/`custom_class_confidence` from `Detection` to `Track`
(`tracker.py:239, 324, 366`). Those fields are now always `None`, and the overlay no longer renders
them, so this is inert — but it is the exact path the false badge travelled, so leave it visible
rather than quietly deleting it.

---

## G. Dataset inventory (counted from disk, not from a summary file)

| Store | Contents | Verdict |
|---|---|---|
| `data/eval/` | `annotations.json`: **34 images, 0 annotations, 14 categories**. `frames/`: 35 files | **EMPTY OF LABELS.** Unchanged since 2026-09-08 |
| `data/external/open-images-v7/` | **`watch/` only** — one class | Manifest: 260 samples = **220 positive boxes across 187 positive images + 40 hard negatives**; 2 rejected (`box_extreme_aspect_ratio` 1, `box_too_small` 1); all 260 CC-BY-2.0; all 260 `box_confirmed_by_human: true`; split `train`, MID `/m/0gjkl` |
| `data/objects/` | `comb` profile: **1 sample, 1 image**. `_deleted/`: `comb`, `watch`, `watch-2026-09-16T…` | Matches the previous report's "1 active comb sample"; the watch samples are confirmed soft-deleted |
| `data/raw/` | 1 session (`4918812e…`), one `.webm` + `.json` | 1 capture session total |
| `data/videos/` | empty | — |

**What the external audit actually found** (`results/external_dataset_audit.json`, 5,000-sample
Open Images V7 subset, 14,610,229 detection rows scanned, 37,471 annotations in subset):

| Open Images class | boxes | images |
|---|---:|---:|
| Glasses | 2975 | 2186 |
| Bottle | 3550 | 847 |
| Mobile phone | 607 | 410 |
| Computer keyboard | 578 | 409 |
| Coffee cup | 513 | 379 |
| Sunglasses | 556 | 372 |
| Goggles | 292 | 224 |
| **Watch** | **222** | **189** |
| Mug | 250 | 187 |
| Bowl | 476 | 165 |
| Tin can | 254 | 148 |
| Pen | 214 | 136 |
| Headphones | 141 | 120 |
| Mixing bowl | 100 | 59 |
| Musical keyboard | 6 | 6 |
| Cocktail shaker | 7 | 5 |
| Corded phone | 5 | 4 |
| Pencil case | 4 | 1 |
| Measuring cup | 1 | 1 |

Zero matches: `Can opener`, `Pencil sharpener`, `Salt and pepper shakers`, `Telephone`.

**COMB: confirmed absent.** A search of the 315 exact Open Images classes present in the subset for
`comb`, `pencil`, `charg`, `brush`, `cable` returns exactly one hit — `Pencil case` (4 boxes,
1 image). **There is no comb data, no charger data, and no pencil data in this download.**
Open Images V7 has no `Comb` class in the audited subset. This is the single hardest blocker for the
project's own stated acceptance test.

**Separation:** `data/external/` and `data/eval/` are physically separate, and
`tests/unit/test_dataset_separation.py` + `tests/unit/test_splits_no_leakage.py` exist to enforce it.
No contamination was observed. Studio data (`data/objects/`) is likewise separate.

---

## H. Test results — NOT RUN BY THIS AUDIT

**The auditor did not execute pytest** (see §0). Reporting a number here would be fabrication.

**Circumstantial evidence from the pytest cache**, timestamped 2026-09-17 06:26–06:27 on this machine:

- `.pytest_cache/v/cache/lastfailed` = `{}` — the most recent run recorded **no failures**.
- `.pytest_cache/v/cache/nodeids` contains **515 node ids**.

**How to read that carefully.** An empty `lastfailed` is consistent with a clean final run, but it
does not confirm the specific claim of *"501 passed, 1 failed, 4 skipped"*; 501+1+4 = 506 ≠ 515, and
`nodeids` reflects the last *collection*, which can include tests a run deselected by marker. The
two are not directly comparable. Treat both the claimed numbers and the cache as unconfirmed until
the suite is run.

**Commands to run on Windows to close this gap:**

```
.venv\Scripts\python -m pytest -q                       # full suite, all markers
.venv\Scripts\python -m pytest -q -m "unit or integration"
.venv\Scripts\python -m pytest -q tests/integration/test_studio_isolation.py -v
.venv\Scripts\python -m pytest -q tests/unit/test_no_forbidden_imports.py -v
.venv\Scripts\python -m pytest -q -m browser            # Playwright; the reported flake lives here
```

The previously-claimed pre-Stage-2 baseline was **494 passed / 4 skipped / 0 failed** in 284.4 s
(`docs/phase-reports/phase13-stage1-inspection.md` §0). The reported post-Stage-2 result was
**501 passed / 1 failed / 4 skipped**, the failure described as a pre-existing browser timing flake.
**Neither was verified here.** Classify the failure only after running the browser suite twice: a
failure that reproduces deterministically is not a flake, whatever a previous report called it.

---

## I. Git state — THE MOST URGENT FINDING

- **Branch:** `main`. **Remote:** `origin` → `https://github.com/msav2007/PredictiveSense.git` (exists).
- **`git rev-list --left-right --count origin/main...main` → `1  0`.** Local `main` is **1 behind**
  `origin/main` and **0 ahead** — meaning the *committed* local history is behind the remote, and
  none of the local work has been pushed.
- **Last commit: `b15c6d3` "Merge branch 'phase8-responsiveness' into main".**

**Everything from Phase 9 onward is uncommitted.** 64 modified files, plus these **untracked**
paths:

```
predictivesense/tracking/          <- the entire tracker (Phase 9/10)
predictivesense/training/          <- the entire training package (Phase 11/12)
predictivesense/perception/classifier.py
models/classifier_registry.json
docs/phase-reports/phase9.md  phase10.md  phase11.md  phase12.md
docs/phase-reports/phase13-stage1-inspection.md
scripts/activate_object_classifier.py  audit_external_dataset.py
scripts/download_openimages.py  evaluate_classifier_baseline.py
scripts/grey_state_audit.py  import_external_dataset.py
PredictiveSense-P13-Perception-Rebuild-Prompt.md
```

**Four phases of work exist in exactly one place: this working tree.** No commit, no push, no
backup. A bad `git checkout`, a disk failure, or an over-eager cleanup destroys Phases 9–13.

**On the diff size:** `git diff --stat` reports 64 files / 16,655 insertions / 14,214 deletions, but
`git diff --stat --ignore-cr-at-eol` reports **34 files / 2,638 insertions / 197 deletions**. The
difference is **CRLF/LF line-ending churn**, not real edits — `predictivesense/camera/mailbox.py`
shows 73 changed lines normally and **zero** ignoring whitespace. Set `core.autocrlf` / add a
`.gitattributes` before committing, or the real Phase 9–13 diff will be buried in 14,000 lines of
noise and the history will be unreviewable.

**No action was taken.** Nothing was committed, reset, staged or reverted by this audit.

---

## J. Known problems (verified only)

1. **Zero labelled evaluation data.** `data/eval/annotations.json` = 34 images, **0 annotations**.
   Every accuracy claim the project might make — mAP, precision, recall, baseline-vs-custom —
   is blocked on this one file, and only the developer can unblock it.
2. **No comb data anywhere.** 1 Studio image; nothing in Open Images V7's audited subset.
   The project's own headline acceptance test cannot currently be attempted.
3. **Four phases uncommitted and unpushed, with `main` 1 behind origin.** (§I)
4. **The degenerate classifier is still `active: true` in its registry** while the feature flag is
   off. Turning the flag on reinstates the defect instantly. (§D)
5. **The AGPL-3.0 licence decision is open**, and `models/manifest.json` says so in writing. Both
   candidate weights are already on disk; the decision has simply never been made.
6. **CRLF churn makes the working diff unreviewable.** (§I)
7. **Small duplication:** `api/ingest.py:55-59` re-probes `app.state.studio` inline instead of using
   the exported `studio_is_active()` helper (`api/studio.py:50-54`). Harmless; noted in Stage 1 and
   still present.
8. **`_POSE_TRACK_CLASS = "person"` is hard-coded** in the tracker — a constraint on any replacement
   detector, not a bug.

---

## K. What is ready to build on

- The **`Detection` / `Track` / `StateSnapshot` contracts** are stable, frozen and exercised by tests.
- The **tracker** is genuinely detector-independent. A new detector is a drop-in.
- The **ONNX Runtime provider layer** (`auto|cpu|cuda|directml`, loud failure on explicit-CUDA-absent)
  is implemented and tested (`test_provider_resolution.py`, `test_cuda_provider.py`).
- The **camera transport, mailbox, staleness guard and preview independence** are implemented and
  test-guarded.
- The **recorded/deterministic pipeline** exists, so a detector swap can be evaluated reproducibly.
- **Studio isolation** is real and test-enforced, including an import-leak subprocess test.
- **Dataset separation machinery** (importer, manifest schema with licence/provenance, split-leakage
  tests) exists and is reusable for detection data — it was built for crops, but the provenance
  fields are the right ones.
- **YOLOX-tiny weights (Apache-2.0) are already on disk and already integrated** enough to have
  produced `results/policy_effect_unlabelled_yolox_tiny.json`.

## L. What is NOT ready

- **Object identification is NOT good enough to proceed to the temporal/risk layers.** The pipeline
  *runs*; identification accuracy is **NOT YET MEASURED** on this project's own data, because there
  is no labelled data to measure it on. These are different statements and must not be conflated.
- **No custom detector exists.** Not trained, not partially trained, not scaffolded. The only
  training code in the repository (`predictivesense/training/`) trains a *crop classifier*, which
  Stage 1 proved cannot detect a class the base detector never proposes.
- **Watch: not detectable.** COCO-80 has no `watch` class. The only "watch" artifact is the
  degenerate single-class classifier, now off.
- **Comb: not detectable, and not trainable today** for lack of data.
- **No detector training pipeline, no detection-format dataset, no mAP/IoU harness run on real data.**
- **The licence decision is unmade**, and it gates what may be trained.
- **Work is unbacked.** (§I)

---

## M. Recommended next stages

Revised against what the repository actually contains. Stage numbering continues the
Perception-Rebuild prompt.

| Stage | Work | Blocking? |
|---|---|---|
| **2.5 — Preserve the work** | `.gitattributes` for line endings, then commit Phases 9–13 in coherent commits, reconcile with `origin/main` (currently 1 ahead of you), push. Also mark the degenerate classifier registry entry `active: false` / `superseded`. | **DO THIS FIRST. Cheap, and everything else is at risk until it is done.** |
| **3 — Baseline freeze proper** | Make baseline-vs-custom a runtime selection, not a single-registry assumption; record the baseline's behaviour before any custom model exists. | No |
| **3.5 — LABEL THE EVAL SET** | Label the 34–35 frames in `data/eval/frames/` with real boxes, and capture more from both cameras. The labelling tool already exists (`/label`, `api/labels.py`). | **YES — blocks every number in the paper.** Only the developer can do it. |
| **4 — Licence decision** | Evaluate YOLOX-tiny (Apache-2.0) against YOLO11n on the *labelled* eval set; decide AGPL or not. Both weights are already on disk. | **YES — blocks Stage 5.** Developer's call, not the implementer's. |
| **4.5 — Comb data** | Decide: capture and annotate comb images yourself (realistically a few hundred, across distance/angle/lighting), or locate another licence-clean source. There is no third option. | **YES for the comb acceptance test.** |
| **5 — Detection dataset build** | Convert Open Images watch (+ the classes in §G worth having) into a detection-format training set with source- and image-disjoint splits. Extend the existing importer. | No |
| **6 — Detector fine-tuning** | Real training run, full manifest, tiny-run proof first. Expect hours on CPU. | No |
| **7 — Validation gate + lifecycle** | trained → validated → active → rollback, gate defined numerically, rollback tested. | No |
| **8 — Live integration** | Custom detections through the existing contract; decide baseline/custom class ownership; measure the two-detector latency cost. | No |
| **9 — Tracker integration + real-time measurement** | Per-camera latency/FPS/frame-age/drop counts, before and after. | No |
| **10+** | Pose/motion → temporal/context → risk anticipation → recommendation → paper evaluation. | — |

**The two stages that only you can do — labelling the eval set and deciding the licence — are also
the two that block everything downstream.** Implementation work can continue around them for a
while, but it cannot produce a single defensible number until the eval set has labels.

---

## Acceptance-test status (§21 of the audit brief)

| Test | Current behaviour | Verdict |
|---|---|---|
| Physical comb → `comb` + box + confidence | No `comb` class in any active model; COCO-80 would propose `toothbrush` or nothing | **FAILS — expected at this stage, no custom training has occurred** |
| Physical watch → `watch` + box + confidence | No `watch` class in COCO-80; the single-class classifier that claimed it is off | **FAILS — expected at this stage** |
| `learned: watch 100%` never shown again | Overlay code deleted; classifier off in code default and in both shipped profiles | **PASSES (code-verified; not physically re-verified)** |

---

## Physical verification — PENDING, DEVELOPER-PERFORMED

Not performed by this audit and not reported as done.

1. Start the app on the laptop webcam; confirm the preview is smooth and **no `learned: …` badge appears anywhere**.
2. Repeat on the OnePlus Nord 4 virtual camera at 1280×720.
3. Hold a comb in view; record exactly what label and confidence appear. Expected: `toothbrush`, something else, or nothing. **Record the truth.**
4. Hold a watch in view; record the same. Expected: not detected as `watch`.
5. Confirm `/studio` still loads (it should — Studio is enabled in `dev.yaml`).
6. Confirm the previously observed failures (phone, bottle, bed→chair, laptop-webcam worse than OnePlus) — record current behaviour.
7. Run the full test suite on Windows and record the exact numbers (§H).
