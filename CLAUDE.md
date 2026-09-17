# PredictiveSense — Claude Code entry point

Lean, auto-loaded index. **Exhaustive module map, phase-by-phase history, and
the full command reference live in `projectContext.md`** — read that before
touching unfamiliar code. Keep this file lean; don't duplicate that one back
into it.

## What this project is

A one-month research prototype supporting a paper: observes an indoor cabin
scene, tracks objects/people, predicts developing safety risks early enough to
warn the user (spoken alert + fixed recommendation). Reproducibility,
measurement honesty, and attribution are hard requirements — never claim a
number not produced by a command run on this machine, and never claim a
physical check not actually performed.

Windows 11, Intel Core Ultra 5 125H (18 logical cores), integrated Arc GPU, no
NVIDIA GPU on the primary dev machine (a second NVIDIA machine exists for
`cuda` provider verification — never assume CUDA is present; always resolve
providers via `perception.provider: auto`). Python 3.11. Local-only: no cloud
service, no runtime download, no telemetry upload. Use `pathlib`; use
`time.monotonic()` for durations, never wall clock, except where a figure is
explicitly a cross-clock browser-vs-server comparison (see `docs/decisions.md`).

## Architecture invariants (never violate without a docs/decisions.md line)

- **Bounded everywhere.** `LatestFrameMailbox` and `BrowserSource._slot` are
  single-slot (depth ≤ 1). The WS broadcaster wakes clients, never buffers.
  Every `Rate`/`Samples` is a `deque(maxlen=...)`. No unbounded queue/list/cache.
- **Preview / client independence.** A slow or dead WebSocket client never
  slows the analysis loop; `Broadcaster.publish` is O(1) + notify, never awaits.
- **No global mutable state, no singletons, no import-time side effects.**
  Threads and wiring live on `app.state` (API) or in `pipeline/` (analysis
  loop), never as module-level globals — a module-level `ThreadPoolExecutor`
  shared across every app instance was a real bug (Phase 8 follow-up: it caused
  a cross-test flake by queuing unrelated tests' jobs onto one worker).
- **Detection / tracking separation.** `Detection`, `Pose`, `Track`, `Relation`,
  `RiskState`, `Alert` stay distinct types. `StateSnapshot.tracks` is populated
  by `predictivesense/tracking/` (Phase 9) when `config.tracking.enabled`;
  `Relation`/`RiskState`/`Alert` stay `[]`/unset until their own phase — never
  placehold those.
- **Local-only, no cloud.** Binding a local listening socket is fine;
  connecting outward at runtime is not (`test_no_outbound_network.py`).
- **Contracts are frozen.** `core/types.py` fields may be added, never
  renamed/reshaped, without a line in `docs/decisions.md`.
- **Mode branching lives only in `pipeline/`.** `realtime`/`recorded` share one
  perception → tracking → temporal → risk core.
- **UI**: application shell + group registry, not a panel stack. New control =
  `registerGroup(...)` in `app.js` + a module under `static/groups/` +
  `static/features/` — never edit the shell. `<video id="preview">` is
  `srcObject`-only, never read/drawn/replaced. No `console.log` outside
  `static/ui/log.js`. Full UI rules and the group-placement table: see
  `projectContext.md`.

## Module map (one line each; see `projectContext.md` for the exhaustive version)

- `core/` — `enums.py`, `types.py` (all frozen contracts).
- `config/settings.py` — typed settings, YAML profiles, `PS_`-prefixed env
  overrides. Phase 12: `ExternalDatasetConfig.root` (default `data/external`)
  and `TrainingConfig.classifier_registry_path` (default
  `models/classifier_registry.json` — closes the classifier-registry
  test-isolation gap the same way `objects.root` already works).
- `camera/` — `source.py`/`synthetic.py`/`mailbox.py`/`framing.py`/`browser.py`/
  `device.py`/`file_source.py`/`enumerate.py`/`_opencv.py`. `cv2` lives only here.
- `perception/` — detector + pose + `classifier.py` (Phase 11, the trained
  crop classifier's ONNX inference — same session/provider path, never
  torch), all ONNX/onnxruntime, imported only here. `policy.py` (recognition
  policy, vocabulary tiers, unchanged by Phase 11), `runtime.py` (provider
  resolution: `auto|cpu|cuda|directml`), `vocabulary.py`, `classes.py`.
- `pipeline/` — `loop.py` (`AnalysisLoop`, the only mode-branching code),
  `scheduler.py` (timer / completion-driven consumer), `recorded.py`
  (`RecordedDriver`, byte-deterministic), `perception_frame.py`
  (`PerceptionFrame`, tracking-ready, real-time and recorded parity).
- `tracking/` — Phase 9 multi-object tracker: `geometry.py`/`association.py`
  (pure bbox math + greedy 5-signal association)/`track_state.py`
  (mutable bookkeeping -> frozen `Track`)/`tracker.py` (`Tracker`, one
  instance per run, same class for both modes). Pure, stdlib + numpy only —
  no inference, no I/O, no tracking dependency.
- `telemetry/` — `metrics.py` (bounded registry), `writer.py`, `manifest.py`,
  `stages.py` (Phase 8 per-stage latency attribution, `stage_<name>_ms`).
- `dataset/` + `eval/` — COCO store, session-disjoint splits, greedy-IoU
  detection eval harness. `test`/`val` split discipline: fit thresholds on
  `val` only; `eval_detection.py --split test` logs every opening and refuses
  a second without `--allow-reopen`.
- `objects/` — Object Learning Studio store: `registry.py`/`samples.py`/
  `quality.py`/`vocab.py`/`batches.py` (bulk-upload staging)/`proposals.py`
  (class-agnostic box from raw detector output, policy bypassed on purpose).
  Trains nothing; a proposal is not ground truth.
- `models/registry.py` — the shipped DETECTOR's registry: exactly one
  `active` version, SHA-256-checked, no activation code. Unchanged and
  untouched by Phase 11 — the trained crop classifier has its own separate
  registry, see `training/` below.
- `training/` — Phase 11 Part B, gated behind the `[train]` extra
  (`torch`/`torchvision`/`onnxscript`; the only place they're imported —
  never `api/`/`pipeline/`/`perception/`/`camera/`). `dataset.py` (Studio
  samples -> crop records, role-aware), `splits.py` (session-**and**-
  object-disjoint, per-class stratified), `synthetic_fixture.py` (a real,
  on-disk fixture dataset for proving the pipeline without collection),
  `model.py` (a small from-scratch CNN), `train.py` (real training + ONNX
  export + run manifest), `classifier_registry.py` (trained -> validated ->
  active -> rollback state machine, `models/classifier_registry.json`),
  `evaluate.py` (accuracy/false-class-rate/latency, baseline vs custom).
  Phase 12 additions, all additive: `dataset.py::build_crop_records`/
  `validate_dataset` take an optional `external_manifests` sequence,
  `CropRecord.source`/`DatasetSummary.per_source_counts` report composition;
  `train.py::TrainConfig.external_manifests` passes them through. Still
  stdlib-only (reads a plain JSON manifest) — never `pandas`/`fiftyone`.
- `api/` — `app.py` (factory + lifespan; owns `app.state.batch_proposals_executor`,
  created + shut down per app, never a module-level singleton), `broadcast.py`
  (push-on-publish `Broadcaster`), `ingest.py`/`recorder.py`/`videos.py`/
  `labels.py`/`objects.py`/`object_batches.py`/`studio.py`. `static/` — no
  framework, no build step; `ui/` (store/shell/registry/controls), `groups/`
  (one module per panel section), `features/` (camera/worker/metrics/overlay),
  `studio/`, `label/` (standalone pages).
- `scripts/` — one-shot tools: `fetch_models.py`, `run_app.py`, `run_noop.py`,
  `run_recorded.py`, `benchmark_*.py`, `build_eval_frames.py`/`build_splits.py`/
  `fit_thresholds.py`/`eval_detection.py`, `export_objects_coco.py`,
  `class_coverage_audit.py`, `policy_audit.py`, `grey_state_audit.py` (Phase 9
  root-cause frequency table), `track_threshold_derivation.py` (Phase 9
  `n_init`/`max_age` from measurement), `pipeline_attrition.py`/
  `pose_continuity_audit.py`/`input_size_recall_compare.py` (Phase 10),
  `track_pose_freshness_audit.py` (Phase 11 Part A, `displayed_pose_age_ms`/
  `displayed_track_age_ms`), `train_object_classifier.py`/
  `validate_object_classifier.py`/`activate_object_classifier.py`/
  `evaluate_classifier_baseline.py` (Phase 11 Part B — need the `[train]`
  extra), `audit_external_dataset.py`/`import_external_dataset.py` (Phase
  12 — need the `[external]` extra for `pandas`; never import `fiftyone`,
  read the raw FiftyOne export files directly).

## Commands

PowerShell, from `C:\Users\mummi\Documents\Projects\PredictiveSense`:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,camera,cpu]"          # [cpu] xor [cuda] - never both (same ORT module name)
pip install -e ".[train]"                   # optional: torch/torchvision/onnxscript, training only

pytest -q                                   # full suite (hardware/cuda/train skip cleanly without their deps)
pytest -q -m "not slow"
pytest -q -m models                         # skips cleanly if models/ is empty
pytest -q -m browser                        # Playwright; skips cleanly if Chromium absent
pytest -q -m train                          # skips cleanly if the [train] extra is absent

python scripts\train_object_classifier.py --fixture            # prove the pipeline with no real collection
python scripts\train_object_classifier.py --objects-root data\objects
python scripts\validate_object_classifier.py <version_id>
python scripts\activate_object_classifier.py <version_id>      # or --rollback

python scripts\run_app.py --profile dev --source-kind browser   # http://127.0.0.1:8000/  (also /studio, /label)
python scripts\benchmark_latency.py --source data\raw --profile dev --limit 150 --label <name>
python scripts\benchmark_capture_paint.py --seconds 20 --label <name>   # real-browser capture->paint
python scripts\benchmark_providers.py --provider cpu|cuda|dml
python scripts\benchmark_recognition_paths.py --source data\raw --frames 300
```

Full command reference (per-phase eval/labelling/export workflows): `projectContext.md`.

## Current phase status

**Phase 12 (external dataset import + proven one-class training path) —
automated work complete on all three parts, physical verification pending.**
See `docs/phase-reports/phase12.md` (three-part: Measured / Physically
observed pending / Not verified).

**Part A (why the Comb is not recognised, P0)**: traced from the code — the
Comb reached the sample store fine (2 samples), but the only two prior
training runs were on Phase 11's synthetic fixture (never Comb, never any
real class), neither was active, and the active model was still the
pretrained baseline detector whose fixed COCO-80 vocabulary cannot emit
"comb" at all. **No bug in the recognition path** — exactly the "no custom
model has ever been trained" finding the prompt itself called most likely.
Adjacent real issue fixed: two stray fixture-trained artifacts had leaked
into the *production* `models/classifier_registry.json`/`models/custom/`
(test/dev-command residue) — removed, and the registry's long-open
test-isolation gap (`docs/decisions.md` Phase 11) is now closed via a new
`AppConfig.training.classifier_registry_path` field, used consistently by
the live loop, the Diagnostics API, and all three CLI scripts.

**Part B (external dataset import, P1)**: audited the developer's Open
Images V7 download (5000 images) directly from its raw export files before
importing anything (`results/external_dataset_audit.json`/`.md`) — the
master `detections.csv`/`image_ids.csv` turned out to be FiftyOne's full,
unfiltered multi-split files, filtered here to this download's own image
IDs. Produced an explicit, reviewable class mapping
(`results/external_class_mapping.json`): 10 classes mapped, `pencil` /
`charger` / `comb` confirmed unavailable, `shaker` explicitly **rejected**
(not merely unmapped — a genuinely different object). `data/external/` is a
new, third, git-ignored, strictly-separate store; nothing is copied, only
referenced by path + SHA-256. `scripts/import_external_dataset.py` filters
bad boxes with counted reasons, deduplicates within and across stores, and
**hard-fails the whole import on any external↔eval collision** (none
occurred for the real Watch import: 220 positive crops / 187 images + 40
hard-negative crops from Mobile-phone boxes). Feeds the **existing** Phase
11 pipeline unchanged — `CropRecord.source` (previously unused) now tags
external samples, session keys make them image-disjoint by construction (no
`splits.py` changes needed), and `DatasetSummary.per_source_counts` reports
composition per source in every run manifest.

**Part C (prove the complete path on one class, exit condition)**: chose
`watch` (189 external images; the developer's own Studio `watch` samples had
already been soft-deleted earlier this session with no restore path — given
the choice, the developer chose **external-only training, gap honestly
reported** over resurrecting deleted files or proving on `comb` (2 samples,
no external data)). Real artifact trained
(`models/custom/crop-clf-2026-09-16T1404-78a4bdb3/model.onnx`), validated
(with an explicitly documented, non-default `--max-false-class-rate 1.0`
override — the default 0.5 gate assumes an eventual multi-class deployment;
a genuine one-class softmax cannot structurally clear it, and reporting that
honestly was chosen over hiding it), and **activated**. `accuracy=1.000` /
`false_class_rate=1.000` are both mathematically forced by training exactly
one class, not a quality claim. **No held-out data of our own exists for
`watch`** — every number is against the external test split; this is stated,
not hidden, and is the developer's next physical-collection priority.

Full suite: **494 passed, 4 skipped, 0 failed** (`pytest -q`, all markers
including `browser`/`models`/`train`/`external`, CPU-only). Also found and
fixed a real, pre-existing environment contamination unrelated to this
phase's own code: `fiftyone` had been installed directly into this project's
shared `.venv`, silently upgrading `starlette` to a version incompatible
with the pinned `fastapi`, which was silently failing collection of most of
the API integration suite before this phase touched anything — fixed by
reinstalling the compatible `starlette`, with a strong recommendation
(`docs/decisions.md`) never to install `fiftyone` into this venv again.

Carried forward, not blockers:
- **Physical verification is the developer's, not yet performed** — section
  13 of the Phase 12 prompt: hold a real watch to both cameras and see how
  often it is actually recognised (this project's first physical test of
  any custom-trained recognition), confirm Diagnostics shows the active
  model, confirm baseline classes and tracking/pose are unchanged.
- **The Phase 8 latency floor was not cleanly reconfirmed** — this
  sandboxed VM's own session load drifted enough (detector-only inference,
  unchanged by this phase, varied 37–228ms p50 across back-to-back runs) to
  make the attempted A/B comparison inconclusive; a quiet-machine re-run is
  left to the developer.
- **Only `watch` is trained** — the other ten mapped classes (glasses,
  headphones, mug/cup, bottle, bowl, can, keyboard, phone, pen) are audited
  and mapped only, per section 10.6's explicit "one class only" scope.
- **`comb` remains Studio-data-only and untrainable** — 2 samples, 0
  external data, confirmed unavailable from Open Images V7 entirely.
- Every Phase 9/10 threshold is still derived from one clip/session each —
  unchanged this phase, re-derive once more footage exists.

Do not start risk prediction without the user's explicit go-ahead.

## Prohibitions

- No CUDA-dependent or CUDA-conditional code that assumes CUDA is present —
  always resolve through `perception.provider` / `resolve_provider`.
- No relationship/holding classifier, `Relation` field populated, temporal
  risk state, risk model, alert policy, or TTS — none started, none
  placeheld, until their phase. (The tracker, track IDs and association
  shipped in Phase 9 — this line no longer covers those.)
- No appearance-based re-identification and no tracking dependency added —
  `predictivesense/tracking/` stays pure geometry + motion + class-vote,
  stdlib + numpy only.
- Custom crop-classifier training is implemented (Phase 11 Part B,
  `predictivesense/training/`, gated behind the `[train]` extra) — this no
  longer covers that. Still prohibited: detector fine-tuning (the two-stage
  design deliberately avoids it — see `docs/decisions.md` Phase 11), and
  training/fine-tuning/activating anything outside `predictivesense/training/`'s
  own explicit train → validate → activate path (never as a side effect of
  another operation).
- Never report a confidence as accuracy, a detection frequency as P/R, a stored
  image as a trained model, a coverage heuristic as a validated metric, a box
  proposal as ground truth, or a tier assignment as a measured result.
- Never fit a threshold on `test`, or open `test` more than once.
- Never claim a performance number not produced by a command run on this
  machine, or a physical (camera/mic/speaker) check not actually performed.
- No module-level singleton, global mutable state, or import-time side effect —
  state lives on `app.state` or inside `pipeline/`.
