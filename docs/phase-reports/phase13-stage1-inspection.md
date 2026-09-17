# Phase 13 — Stage 1: Architecture Report (Perception Rebuild)

**Status: pre-implementation inspection only. No code has been changed. This document is the required Stage 1 deliverable of `PredictiveSense-P13-Perception-Rebuild-Prompt.md` §2 — it must be published before Stages 2+ begin.**

All file paths are relative to `C:\Users\mummi\Documents\Projects\PredictiveSense`. All claims below were verified by direct code reading (five parallel read-only inspection passes) on 2026-09-17, not inferred from `projectContext.md`/`CLAUDE.md`, both of which lag behind the current tree (they stop narrating at Phase 7 and Phase 12 respectively; several Phase 8–12 modules below are not in `projectContext.md` at all).

---

## 0. Baseline recorded before any change

- `pytest -q` (full suite, all markers, this machine): **494 passed, 4 skipped, 0 failed** — run to completion in 284.4s.
- Python **3.11.9**. Installed: `onnxruntime==1.24.4` (only ORT build present — no `onnxruntime-gpu`/`onnxruntime-directml` in `.venv`), `torch==2.14.0+cpu`, `torchvision==0.29.0+cpu`, `pandas==3.0.5`, `onnxscript==0.7.2`, `opencv-python==4.10.0.84`, `fastapi==0.115.6`, `starlette==0.41.3`.
- `pyproject.toml` extras: base install has no ORT; `[cpu]`→`onnxruntime==1.24.4`, `[cuda]`→`onnxruntime-gpu==1.24.4` (mutually exclusive, same module name, documented at `pyproject.toml:29-33`); `[train]`→`torch/torchvision/onnxscript` (training-only, never imported by `pipeline/`/`api/`); `[external]`→`pandas` (only `scripts/audit_external_dataset.py`/`scripts/import_external_dataset.py` use it — enforced by `tests/unit/test_no_forbidden_imports.py`); `[camera]`→`pygrabber` (Windows device-name enumeration, never fatal).
- pytest markers (`pyproject.toml:104-115`): `unit`, `integration`, `slow`, `hardware`, `models`, `dataset`, `browser`, `cuda`, `train`, `external`.

---

## 1. Entry points

- Process entry: `scripts/run_app.py:28-100` — parses `--profile/--host/--port/--source-kind`, builds `AppConfig`, calls `uvicorn.run(create_app(config, write_manifest=True), ...)`.
- App factory + lifespan: `predictivesense/api/app.py:114-312` (`create_app`). Builds/uses one `AnalysisLoop` (`app.py:129`, `pipeline.loop.build_loop`), one `Broadcaster` (`app.py:130`). `lifespan()` (`app.py:140-163`) starts the loop on startup, stops it and shuts down `app.state.batch_proposals_executor` on shutdown (`app.py:163`) — this executor is created per-app (`app.py:197-199`), not a module-level singleton (the Phase-8 bug class this project explicitly guards against per `CLAUDE.md`'s architecture invariants).
- Threads created: `AnalysisLoop.start()` spawns non-daemon `ps-producer`/`ps-consumer` threads (`predictivesense/pipeline/loop.py:277-284`); one dedicated 1-worker `ThreadPoolExecutor` (`batch-proposals`) lives on `app.state` for Object Learning Studio bulk-upload proposal inference (`api/app.py:197-199`).

## 2. Backend API surface

Routers mounted at `predictivesense/api/app.py:201-209`: `ingest`, `recorder`, `videos`, `labels`, `objects`, `object_batches`, `studio`, plus `/static`.

| Method + Path | Purpose | file:line |
|---|---|---|
| GET `/health` | liveness | `api/app.py:211-218` |
| GET `/api/config` | resolved config dump | `api/app.py:220-222` |
| GET `/api/runtime` | live EP, detector/classifier version, scheduler, machine fingerprint | `api/app.py:224-249` |
| GET `/api/cameras` | enumerate devices | `api/app.py:251-253` |
| POST `/api/debug/stall` | test hook | `api/app.py:255-258` |
| POST `/api/metrics/browser` | browser-measured metrics sample | `api/app.py:260-289` |
| GET `/` | dashboard shell | `api/app.py:291-293` |
| GET `/label` | labelling tool | `api/app.py:295-299` |
| WS `/ws/state` | last-value-wins `StateSnapshot` stream | `api/app.py:301-310` |
| WS `/ws/ingest` | binary analysis frames from browser worker | `api/ingest.py:43-105` |
| POST `/api/record/upload`, GET `/api/clips` | raw clip capture | `api/recorder.py:49,134` |
| GET `/api/videos`, POST `/api/analyze` | recorded-mode playback/analysis | `api/videos.py:41,59` |
| GET/POST `/api/labels/*` | detector-eval labelling tool backend | `api/labels.py:114-247` |
| GET `/api/objects/vocab`, `/training_status`, CRUD `/api/objects*`, coverage, sample image | Studio object/sample store | `api/objects.py:218-452` |
| CRUD `/api/objects/{id}/batches*` | Studio bulk-upload staging + proposals + save/discard | `api/object_batches.py:114-530` |
| GET `/studio`, `/api/studio/status`, POST `/enter`/`leave`, GET `/api/models/registry` | Studio page + pause/resume + detector registry read | `api/studio.py:57-155` |

## 3. WebSocket wire contract

- `/ws/ingest` (`api/ingest.py`): JSON handshake (`hello`/`hello_ack`), then binary frames (4-byte length + JSON header + JPEG) → `BrowserSource.submit`. Refused with close 4403 if `capture.owner != "browser"` (`ingest.py:48-51`), or **4409 while Studio is active** — checked via an inline re-probe of `app.state.studio` (`ingest.py:55-59`), not via the exported `studio_is_active()` helper defined for this exact purpose in `api/studio.py:50-54` (a small, harmless duplication to note for Stage 2).
- `/ws/state` (`api/app.py:301-310` → `api/broadcast.py:102-176`): each client blocks on an event woken by `Broadcaster.publish()` (O(1), never awaits — `broadcast.py:54-65`), then receives `snapshot.to_wire_json()` = `StateSnapshot.model_dump_json()` (`core/types.py:347-350`).
- `StateSnapshot` (`core/types.py:307-345`): `snapshot_id, mode, frame_id, capture_ts, emitted_ts, frame_age_ms, detections: list[Detection], poses: list[Pose], tracks: list[Track], risk, metrics, stale, pose_stale`.
  - `Detection` (`core/types.py:135-178`): `bbox, class_id, class_name, score, frame_id, raw_class_name, policy_state, runner_up, tier`, plus Phase-11-additive `custom_class_name: str|None`, `custom_class_confidence: float|None` (`:170-178`).
  - `Track` (`core/types.py:206-277`): carries the same `custom_class_name`/`custom_class_confidence` pair (`:271-277`) — **this is what the browser overlay renders.**

## 4. Frontend — root cause of the "learned: watch 100%" defect (§0 of the prompt)

This was traced end-to-end through real code, not assumed. **Repo-wide `grep "learned"` under `static/` hits exactly one live-overlay renderer** — `predictivesense/api/static/features/overlay.js:401-403`, inside `drawTrack(t, ...)`:

```js
if (t.custom_class_name) {
  const pct = `${Math.round((t.custom_class_confidence || 0) * 100)}%`;
  const badge = `learned: ${t.custom_class_name} ${pct}`;
```

`t` is one element of the live `/ws/state` snapshot's `tracks` array. **The gate is only `if (t.custom_class_name)` — no confidence floor, no check that it is compatible with the detector's own `class_name`.** (Studio's own pages, `studio.js:883-894` / `batch.js:450`, explicitly avoid ever claiming "learned" — the leak is confined to the main overlay.)

**Full data-flow trace, confirming this is real per-frame inference, not a stale Studio-state poll:**

1. `predictivesense/pipeline/loop.py:676-705` (`_resolve_active_classifier`, run once at `AnalysisLoop.__init__`) inline-imports `predictivesense.training.classifier_registry.ClassifierRegistry` (`loop.py:684`) and reads its `active()` entry from `models/classifier_registry.json`. The one active entry, `crop-clf-2026-09-16T1404-78a4bdb3`, has `class_map: {"watch": 0}` — **a single-class model.**
2. `loop.py:707-739` (`_apply_custom_classifier`), invoked from `_run_perception` at `loop.py:788-790`, runs on **every** detection whose `policy_state` is `accepted`/`accepted_secondary` (`loop.py:719-721`) — **it does not filter by `class_name`**, so a box the policy accepted as, say, "Toothbrush" is still cropped and fed to the watch-only classifier.
3. `predictivesense/perception/classifier.py:54-69` (`infer_crop`) computes softmax over the model's logits. Because the trained `class_map` has exactly one entry, the model has one output neuron — **softmax over a single logit is mathematically forced to `1.0`** (`exp(x−x)/exp(x−x)=1`). The classifier is structurally guaranteed to report `("watch", ~1.0)` for any crop whatsoever, independent of the pixels.
4. That `(name, confidence)` is copied onto `Detection.custom_class_name`/`custom_class_confidence` (`loop.py:732-735`) — additively; `class_name`/`policy_state`/`tier` are never touched.
5. `predictivesense/tracking/tracker.py:239,324,366` carries the pair from `Detection` onto the matching `Track`.
6. `StateSnapshot.tracks` is serialized and pushed over `/ws/state`; `overlay.js:401-403` renders it unconditionally.
7. `predictivesense/pipeline/recorded.py` has **no** classifier/`custom_class` wiring at all (confirmed by grep) — this defect **only manifests on the live path**, never in recorded/offline analysis.

**Conclusion:** the badge is not a UI bug reading stale enrollment state — it is the live crop-classifier genuinely running every frame, but (a) trained on exactly one class, making its softmax degenerate, (b) applied indiscriminately to every accepted detection regardless of the detector's own class, and (c) rendered with no confidence floor or class-compatibility check. This exactly matches, and refines, §0's framing: the classifier cannot invent a box (confirmed independently in §7 below), but it *can* mislabel any box the detector proposes, and today nothing stops it from doing so on an incompatible class.

## 5. Camera pipeline

- `LatestFrameMailbox` (`predictivesense/camera/mailbox.py`): single slot (`self._slot: Frame|None`, line 28); `put()` (32-39) increments `dropped` and overwrites rather than queues; `get()` (41-57) pops/clears; `stats()` (65-73) returns exact `consumed`/`dropped`/`depth∈{0,1}`.
- `BrowserSource` (`predictivesense/camera/browser.py`): identical single-slot newest-wins buffer one level upstream of the mailbox (`submit()` lines 91-151, drop-count at 145-148).
- **Staleness guard** — `predictivesense/pipeline/loop.py:432-436`: `age_ms = (now - frame.capture_ts) * 1000; if age_ms > self._max_frame_age_ms: dropped_stale += 1; frame = None` — threshold from `config.analysis.max_frame_age_ms` (0 disables it). A **separate** non-dropping `stale` annotation (for UI/telemetry only) is computed at `loop.py:441-443` from `config.consumer.stale_after_ms` — these are two distinct mechanisms, easy to conflate.

## 6. Current detector (baseline)

- `ObjectDetector` (`predictivesense/perception/detector.py:36-201`): one ONNX session (`runtime.create_session`, called at construction), per-class threshold LUT (`:58-64`). `infer()` (`:105-201`): letterbox → session run → per-class threshold mask (`:139-143`) → class-aware NMS (`:161-168`) → raw output becomes `Detection(...)` at `:178-200`.
- `predictivesense/perception/runtime.py`: `create_session()` (`:177-272`) builds the ORT session, **verifies** the requested EP is actually active post-construction (`:223-235`, raises rather than silently substituting), runs a timed warm-up (`:246-251`). Provider resolution `resolve_provider()` (`:118-163`): `"auto"` tries `("cuda","directml","cpu")` in order via `ort.get_available_providers()` (`:131-146`), falling back to `"cpu"` if neither GPU EP is present; an **explicit** request for an EP not actually built into this ORT raises `ProviderUnavailableError` (`:154-160`) — never a silent fallback. No hardcoded CUDA assumption anywhere in this module (confirmed by grep + the module docstring).
- `input_size` default: `predictivesense/config/settings.py:213` = **640**, validated as a multiple of 32; `config/profiles/dev.yaml:130` overrides the detector to **480** (pose stays 640 — the pose ONNX export locks that input size).
- `predictivesense/perception/classes.py:24-37` — the full COCO-80 list. **Confirmed: `"toothbrush"` is COCO class #80 (line 36). Neither `"comb"` nor `"watch"` appears anywhere in the 80-class list** (zero grep matches) — directly substantiating §0's and §5.6's premise.

## 7. The crop classifier (Phase 11) — architectural claim verified against code

`predictivesense/perception/classifier.py` (69 lines, read in full): `infer_crop(image_bgr)` (`:54-69`) takes an **already-cropped** image; module docstring (`:6-8`) states it "runs on a crop of an image (a box the class-agnostic detector already proposed), never a full frame, never re-detecting." Model loading (`:24-40`) goes through the identical `perception/runtime.create_session` path used by the detector and pose estimator — **no `torch` import anywhere in the file**; serving is ONNX Runtime only (the registry's own `provider` field states this explicitly: "training always runs in PyTorch on CPU; serving is ONNX Runtime").

**Call site is `predictivesense/pipeline/loop.py`, not `perception/engine.py`** — `_resolve_active_classifier` (`:676-705`) then `_apply_custom_classifier` (`:707-739`), invoked from `_run_perception` (`:788-790`). The classifier has **no independent proposal or localization mechanism** — there is no bounding-box logic anywhere in `classifier.py`; it is a pure classify-a-given-crop model, and its only call site iterates over detections the `ObjectDetector` already produced a box for **and** the `RecognitionPolicy` already accepted.

**Verdict: the §0 architectural claim is confirmed by the code, not merely asserted by prior-phase narrative.** The crop-classifier is strictly additive: it can annotate (`custom_class_name`) a box the base detector proposed, but it structurally cannot produce a detection for any region of the frame the base detector never proposed a box for. `comb` is not a COCO class (§6), so the base detector never proposes a box for a comb, so no crop is ever produced, so no classifier — however well trained — can ever recognize a comb under the current architecture. This is exactly why §0 states the detector itself, not the classifier, must be replaced or extended.

## 8. Recognition policy

`predictivesense/perception/vocabulary.py`: `PRIMARY_TIER` (`:44-47`, 14 classes), `SECONDARY_TIER` (`:49-54`, 26 classes), `IMPLAUSIBLE_TIER` (`:56-68`, 40 classes) — `validate_partition()` (`:77-134`) enforces they total/partition all 80 COCO classes disjointly; anything else resolves to `"unlisted"` (`TIERS`, `:70`).
`predictivesense/perception/policy.py`: six `policy_state` values (`POLICY_STATES`, `:44-51`): `accepted, accepted_secondary, unknown_low_confidence, unknown_margin, suppressed_implausible, rejected_size`. `apply(detections, *, frame_width, frame_height) -> PolicyOutcome` (`:196-202`); decision order: size rule → implausible-tier suppression → per-class threshold → margin rule → secondary-vs-accepted (`:239-305`).

## 9. Current tracker

`predictivesense/tracking/` — pure stdlib+numpy, no tracking library.
- `geometry.py`: `area, center, diagonal, iou, center_distance_score, size_ratio_score, shift, is_off_frame` (`:22-80`), all pure functions.
- `association.py`: 5-signal weighted cost `score_pair` (`:42-72`) — IoU (weight 0.5), center-distance (0.2), size-ratio (0.1), motion/constant-velocity consistency (0.2), class-agreement (0.1) — weights from `TrackingConfig` (`config/settings.py:489-493`); a geometric-plausibility gate (`iou<=0 and center<=0.5` → ineligible) runs before class can rescue a pair (`:52-55`). `greedy_match` (`:75-107`): highest-score-first, deterministic tie-break, no Hungarian solve.
- `track_state.py`: `TrackState` dataclass (`:40-76`) with a bounded `class_history` deque (`maxlen=config.class_vote_history`, default 20); `majority_class()` (`:19-32`) resolves ties by most-recent occurrence; `to_track()` (`:137-192`) converts to the frozen `Track` contract, dropping (not aging-in-place) any pose older than `pose_max_age_ms`.
- `tracker.py`: `Tracker.update(detections, *, frame_id, capture_ts, frame_width, frame_height, poses=(), pose_fresh=True) -> list[Track]` (`:89-99`). Two-stage ByteTrack-style association (high-score full-signal stage 1, low-score IoU-only recovery stage 2, never spawning new tracks — `:189-232`); lifecycle `TENTATIVE→CONFIRMED` (needs `n_init` hits) / `→COASTING` (on a survivable miss) / expire-and-drop (`:234-290`); border-exit gets only 1 miss-budget vs. 12 interior (`:267-278`); a bounded ghost re-association pool reclaims recently-expired (non-border) tracks above a stricter score floor (`:294-370`). `n_init=3`, `max_age_frames=12`, `max_age_ms=564.0` are **measured** (not invented) from a live session, documented in `results/track_thresholds.md`; `pose_bind_min_iou=0.1`/`pose_max_age_ms=900.0` likewise measured (`results/pose_continuity_raw.md`).
- **Wiring**: `pipeline/loop.py:459-469` calls `Tracker.update()` only `if frame is not None and self._tracker is not None`; on a stale/guard-dropped cycle, `tracks = self._last_tracks` (pre-set at `:458`) is reported unchanged rather than wiped — confirmed in code, matching CLAUDE.md's claim. `pipeline/recorded.py:250-263` calls it unconditionally per decoded frame (no staleness concept in recorded mode).
- The `Track` contract (`core/types.py:245-277`) has no `object_id` (identity is `track_id`) and no separate "direction" field beyond `velocity: (vx,vy)` — everything else required by later phases (class votes, age, pose binding, custom-classifier annotation) is already present and additive.

**This tracker interface already satisfies Stage 10's requirement** ("the detector must be swappable without touching tracker internals") — it consumes only `Detection` objects (bbox/class_name/score/frame_id/timestamp) and has no dependency on which model produced them.

## 10. Object Learning Studio — exhaustive coupling audit

`predictivesense/objects/` (vocab.py, registry.py, samples.py, quality.py, batches.py, proposals.py) is genuinely self-contained: a full import audit of every file in the package found **zero imports** of `perception/detector.py`, `perception/classifier.py`, `pipeline/`, or `tracking/`. `proposals.py` takes raw detections as a parameter and never imports the detector/onnxruntime/cv2 itself.

**Two load-bearing leaks into the live path were found, both outside `predictivesense/objects/` itself:**

1. **`predictivesense/api/object_batches.py:244,269`** — the bulk-upload background worker calls `app.state.perception.detect(frame)`, and `app.state.perception` (`api/app.py:129,178`) is **the same `PerceptionEngine` instance the live `AnalysisLoop` runs on its consumer thread** — not a second instance, and not gated by any `studio_is_active()` check at the route level.
2. **`predictivesense/pipeline/loop.py:93-94, 676-705, 707-739, 788-790`** — the live analysis loop unconditionally resolves and runs the Phase-11 crop classifier via inline imports of `predictivesense.training.classifier_registry.ClassifierRegistry` and `predictivesense.perception.classifier.CropClassifierModel`. **This is the single highest-priority coupling to sever**: today, `pipeline/loop.py` — squarely inside the live detection path — directly imports a Studio-trained artifact's registry, with no config flag gating it off.

**No existing config flag disables Studio or the live classifier application.** `StudioConfig.stop_monitoring_on_enter` (`config/settings.py:625-631`) only governs whether *entering* `/studio` pauses the loop — it is a lifecycle nuance, not an on/off switch. A new flag (e.g. `studio.enabled` / `training.classifier_enabled`) will need to gate: (a) the router mounts at `api/app.py:205-207`, and (b) the classifier-resolution call in `pipeline/loop.py:93-94`.

**Consolidated coupling table:**

| # | Point | file:line | Tag |
|---|---|---|---|
| 1 | `predictivesense/objects/*` package — no imports of detector/classifier/pipeline/tracking | `objects/__init__.py:24-29` etc. | INDEPENDENT |
| 2 | Batch proposal worker reuses the live `app.state.perception` instance, no lock, no activity gate | `api/object_batches.py:244,269`; shared at `api/app.py:129,178` | **SEVER** |
| 3 | No route-level gate on `/api/objects/*`, `/api/objects/{id}/batches*` requiring Studio to be active | `api/objects.py`, `api/object_batches.py` (all routes) | SEVER |
| 4 | `GET /api/objects/training_status` inline-imports the classifier registry (Studio-internal, read-only) | `api/objects.py:229,261` | KEEP BEHIND FLAG |
| 5 | `POST /api/studio/enter`/`leave` pause/resume the live loop; prior-state token | `api/studio.py:74-145` | KEEP BEHIND FLAG |
| 6 | `GET /api/models/registry` reads the **detector** registry, not the classifier registry | `api/studio.py:28,148-155` | INDEPENDENT |
| 7 | objects/object_batches/studio routers mounted unconditionally | `api/app.py:205-207` | SEVER (needs flag-gated mount) |
| 8 | `app.state.perception` exposed for Studio's batch worker to reuse | `api/app.py:129,178` | SEVER |
| 9 | `app.state.studio` dict + `batch_proposals_executor` lifecycle | `api/app.py:185,197-199,163` | KEEP BEHIND FLAG |
| 10 | `/ws/ingest` 4409 refusal inline-duplicates `studio_is_active()` instead of calling it | `api/ingest.py:55-59`; helper at `api/studio.py:50-54` | KEEP BEHIND FLAG (de-dup) |
| 11 | Main-shell link to `/studio` (no JS import of Studio's module graph) | `static/groups/dataset.js:44-49` | KEEP BEHIND FLAG |
| 12 | `/studio` standalone page + its own JS module graph | `api/studio.py:57-59`; `static/studio/index.html:209`, `studio.js:26-28` | INDEPENDENT |
| 13 | **Live pipeline directly imports and runs the Phase-11 crop classifier every frame** | `pipeline/loop.py:93-94,676-705,707-739,788-790` | **SEVER — highest priority** |
| 14 | `custom_class_name`/`custom_class_confidence` threaded through `Detection`/`Track`/wire snapshot | `core/types.py:177-178,276-277,347-350`; `tracking/tracker.py:239,324,366`; `api/broadcast.py:150` | SEVER (must become no-op/`None` when flag is off) |
| 15 | "learned: X %" badge rendered on the main overlay from live snapshot data | `static/features/overlay.js:401-403` | SEVER (naturally silenced once #13/#14 are gated, but the missing confidence floor / class-compatibility check should also be fixed regardless of the flag) |
| 16 | `custom_classifier_version` in `/api/runtime`, session manifest, Diagnostics panel | `api/app.py:243,339`; `static/groups/diagnostics.js:260,267` | KEEP BEHIND FLAG (degrade to "none", don't crash) |
| 17 | Detector registry legitimately used by the live pipeline for model-version tagging | `pipeline/perception_frame.py:31-42` | INDEPENDENT (separate registry, separate concern — not a Studio leak) |
| 18 | `policy.py`/`tracker.py` import neither registry directly | (full import list checked, no matches) | INDEPENDENT |
| 19 | No flag currently exists to disable Studio/classifier — must be added | `config/settings.py:625-631` (`StudioConfig`) | **Gap to close in Stage 2** |

**What happens to a Studio-derived class in the UI once severed**: with #13/#14 cut, `Detection.custom_class_name`/`custom_class_confidence` and their `Track` counterparts will always be `None` when the flag is off, so `overlay.js:401`'s `if (t.custom_class_name)` guard naturally renders nothing — no "learned: …" text will appear anywhere, matching the acceptance criterion. The Studio pages themselves (`/studio`, `/objects/*`) continue to function for data collection independent of this flag, per §12 above (INDEPENDENT).

## 11. Dataset code and paths

Three genuinely separate stores already exist and match the prompt's proposed shape (§5.1) — **no new parallel structure is needed, only extension**:

- **`predictivesense/dataset/` + `predictivesense/eval/`** — the *detector's own* held-out evaluation set at `data/eval/` (`DatasetConfig.root/coco_path/splits_path`, `config/settings.py:529-531`). `coco_store.py` (COCO JSON read/write, `seeded`/`labelled` provenance), `splits.py` (session-disjoint val/test, content-hash-guarded), `eval/matching.py`+`metrics.py` (greedy-IoU, per-class P/R/F1, confusion incl. background/unknown, false-class-rate headline, mAP@0.5), `eval/report.py` (json+md with full repro metadata).
- **`predictivesense/objects/`** — the developer's own captured Studio samples at `data/objects/` (`ObjectsConfig.root`, `settings.py:583-599`). Trains nothing itself.
- **`data/external/`** (Phase 12) — public dataset import (`ExternalDatasetConfig.root`, `settings.py:602-610`), whose own docstring states it is "strictly separate from `objects.root`... and `dataset.root`."
- **`predictivesense/training/`** (Phase 11 Part B, `[train]` extra) reads across all three: `dataset.py::build_crop_records`/`validate_dataset` take `external_manifests` and tag `CropRecord.source` (`dataset.py:147-149,229-233`); `splits.py::build_crop_splits` is session-**and**-object-disjoint, per-class stratified (session keys already embed `object_id`, so a class's whole capture burst can't straddle splits — `splits.py:83-151`); `model.py::CropClassifier` is a small from-scratch 3-conv CNN with GAP, **not** a pretrained backbone (`model.py:21-39`); `train.py::TrainConfig` (epochs/batch/lr/seed/fractions/`external_manifests`) writes a complete run manifest to `results/train_run_{version_id}.json` (note: augmentation is currently a hardcoded string in the manifest, not a `TrainConfig` field — see §20); `classifier_registry.py` implements the 4-state lifecycle `trained→validated→active→(rolled_back)` (`register()`/`validate_version()`/`activate()`, `:143-194`), with the numeric validation gate living in `scripts/validate_object_classifier.py` (`--min-accuracy` default 0.5, `--max-false-class-rate` default 0.5).
- **Hard-fail collision enforcement (Phase 12, already implemented)**: `scripts/import_external_dataset.py` raises `ImportAbortedError` and aborts the *entire* import, writing nothing, if any external image's SHA-256 or near-duplicate perceptual hash matches an eval-set image (`import_external_dataset.py:206-219`) — a genuine hard failure, not a warning, satisfying §5.1's inviolable rule already. (A collision against `data/objects/` instead just excludes that one image and continues — a deliberate, documented asymmetry.)

## 12. Model registries (two, deliberately separate)

- **`predictivesense/models/registry.py`** — the shipped **detector's** registry (`models/registry.json`). Enforces exactly one `active` version always (`registry.py:180-185`); no activation *code* — a human edits the JSON. Currently one entry, `v1 = baseline-yolo11n` (YOLO11n detect+pose, AGPL-3.0-only), SHA-256-pinned against `models/manifest.json`.
- **`predictivesense/training/classifier_registry.py`** — the **crop-classifier's** registry (`models/classifier_registry.json`), a materially different 4-state lifecycle. Confirmed as a genuinely separate file/class from the detector registry (both read in full). Currently one entry, `crop-clf-2026-09-16T1404-78a4bdb3`, `class_map: {"watch": 0}`, `validated: true`, `active: true`, `accuracy: 1.0` / `false_class_rate: 1.0` on 44 positive + 6 rejection examples.
- Neither registry is imported by `perception/policy.py` or `tracking/tracker.py` (confirmed by import audit) — only `pipeline/loop.py` (classifier registry, the §10 leak) and `pipeline/perception_frame.py` (detector registry, legitimate model-version tagging) touch either from the live path.

## 13. Python environment

Covered in §0. One venv, one ORT build, CPU-only torch. `models/yolox_tiny.onnx` (Apache-2.0, 20.2 MB) already exists on disk as the permissive-detector comparison candidate from Phase 2.5 (see §20 — the licensing decision itself is unresolved, not the technical evaluation).

## 14. Tests

`tests/{unit,integration,browser,hardware,fixtures}/`. Markers per §0. Baseline: **494 passed, 4 skipped**, 0 failures, on this machine, before any Phase 13 change.

## 15. Diagnostics/telemetry

`predictivesense/telemetry/{metrics.py,writer.py,manifest.py,stages.py}` — bounded `Counter`/`Rate`/`Samples`/`Timer` registry, append-only CSV writer (non-fatal on write failure), session manifest builder, per-stage latency attribution (`stages.py`, folds `tracker_update_ms` into `FrameTrace.tracker_ms` per Phase 9). `GET /api/runtime` (`api/app.py:224-249`) and `static/groups/diagnostics.js` surface EP, detector/classifier version, thread counts, machine fingerprint — this is the existing measurement surface Stage 11's per-camera latency/FPS/age/drop numbers will extend, not replace.

## 16. Dependency map — frame → detection → policy → tracker → wire → UI

```
FrameSource (camera/{device,browser,file_source,synthetic}.py)
  → LatestFrameMailbox (single-slot, drop-counted)  [camera/mailbox.py]
  → AnalysisLoop._iterate  [pipeline/loop.py]
       staleness guard (max_frame_age_ms) → may set frame=None, dropped_stale++
       ObjectDetector.infer(frame)  [perception/detector.py]  → list[Detection] (raw_class_name, no policy yet)
       RecognitionPolicy.apply(detections)  [perception/policy.py]  → policy_state/tier decided
       _apply_custom_classifier(detections)  [pipeline/loop.py]  ← LIVE-PATH LEAK (§10 #13)
         → CropClassifierModel.infer_crop(crop)  [perception/classifier.py]  → custom_class_name/_confidence (additive)
       Tracker.update(detections, poses, ...)  [tracking/tracker.py]  → list[Track] (carries custom_class_* too)
       StateSnapshot(detections, poses, tracks, ...)  [core/types.py]
  → Broadcaster.publish (O(1), never awaits)  [api/broadcast.py]
  → WS /ws/state  → snapshot.to_wire_json()
  → static/features/overlay.js: drawTrack()  → renders class/tier badge AND, unconditionally, the
    "learned: {custom_class_name} {pct}%" badge if custom_class_name is truthy  ← RENDER-SIDE LEAK (§10 #15)
```
Recorded mode (`pipeline/recorded.py`) runs detector → policy → tracker identically but has **no classifier wiring at all** and writes JSONL directly (no `StateSnapshot`, no WS, no overlay) — the defect in §4/§10 is exclusively a live-mode phenomenon.

## 17. Studio coupling list

See the consolidated table in §10 — repeated here as the required standalone deliverable per §2's Stage-1 checklist. 19 points identified; 2 tagged **SEVER** at the architectural level (object_batches.py's perception-instance reuse, and pipeline/loop.py's unconditional classifier resolution/application), 6 tagged KEEP BEHIND FLAG, 7 tagged INDEPENDENT, plus the confirmed absence of any existing on/off flag (#19).

## 18. Data audit (counted from disk, 2026-09-17)

- **`data/objects/`**: exactly **1 committed sample** exists across all *active* (non-deleted) object profiles — `data/objects/comb/images/` contains 2 files (1 original + 1 thumbnail) = 1 comb sample. **No active `watch` object profile exists** — all Studio-captured watch samples were soft-deleted earlier this session and now live only under `data/objects/_deleted/watch/` and `data/objects/_deleted/watch-2026-09-16T080614153787Z/` (6 image+thumb files total, unrecoverable via the normal API without a restore path).
- **`data/external/open-images-v7/watch/manifest.json`**: 220 positive + 40 hard-negative samples across 187 distinct positive images, license CC-BY-2.0, 2 boxes rejected (extreme aspect ratio / too small) — images are referenced by absolute path + SHA-256, never copied into the repo.
- **`results/external_class_mapping.json`**: 10 classes mapped from Open Images V7 (glasses, headphones, mug/cup, bottle, bowl, can, keyboard, phone, pen, watch); `pencil`/`charger`/`comb` confirmed **unavailable**; `shaker` explicitly **rejected** (a genuinely different object, not merely unmapped).
- **`data/eval/frames/`**: 35 image files on disk; `data/eval/annotations.json` has 34 images, 14 categories, **0 annotations** — the detector's own evaluation set exists but currently has zero labelled boxes. Any baseline-vs-custom accuracy comparison on this data cannot yet be run.
- **`models/custom/`**: exactly one trained artifact, `crop-clf-2026-09-16T1404-78a4bdb3/model.onnx` (the single-class "watch" classifier from §12).
- **Comb data gap (§5.6 of the prompt) — re-verified true, not assumed**: 1 Studio sample, 0 external samples, confirmed unavailable from Open Images V7 entirely. The comb test cannot be passed by training on public data alone under the current data. This is unchanged from the prior phase's finding and is re-confirmed here by direct filesystem inspection, not by trusting a summary file.

## 19. Where this report re-verifies or refines the prompt's own claims

- §0's core architectural claim ("the crop-classifier cannot name an object the base detector never proposes a box for") is **confirmed by code** (§7 above), not merely inherited from prior-phase narrative.
- §0's framing that the "learned: watch 100%" text "was not evidence of learning... it actively misled physical testing" is **more precise than the prompt states**: it is not disconnected from live inference — it *is* live inference, just from a degenerate (single-class) model applied without a class-compatibility gate. The fix is not "stop reading Studio state" (there was no such read) — it is (a) sever `pipeline/loop.py`'s unconditional dependency on the classifier registry behind a flag per §10, and (b) independently fix the missing confidence floor / class-compatibility check in `overlay.js:401` and the indiscriminate-class application in `loop.py:719-721`, regardless of the flag.
- §5.6's claim about `comb`/`pencil`/`charger` having no usable public source, and the prior "no substitution" rejection for `shaker`, are both **re-verified true from the current `data/external/` and `results/external_class_mapping.json` state** (§18), not assumed.
- Stage 10's tracker-interface requirement ("swappable detector, no tracker rewrite") is **already satisfied** by the existing `tracking/` package (§9) — it depends only on the generic `Detection` contract, with no assumption baked in about which model produced it.

## 20. Open questions requiring a decision before Stage 5 (not resolved by this report)

Per the prompt's own instruction (§19.6), this is flagged rather than decided unilaterally:

- **AGPL vs. permissive detector license (§6.2 of the prompt)** — `docs/attribution.md` and `docs/decisions.md` already record this as an **open, unresolved decision** carried over from Phase 2.5: `models/manifest.json` states outright *"shipping this repository under a non-AGPL licence while shipping these [YOLO11n] weights is an OPEN, UNRESOLVED decision."* The permissive alternative (`models/yolox_tiny.onnx`, Apache-2.0, YOLOX-tiny, Megvii) is already on disk and was already benchmarked for raw detection frequency in Phase 2.5, but the accuracy comparison needed to actually decide the question has never run, because `data/eval/annotations.json` has **zero labelled annotations** (§18) — there is nothing to evaluate against yet, on either license track. This phase's Stage 5/6 (train a real custom detector on the target classes) will make this decision load-bearing for the first time, since it is the first phase to *distribute a derived, fine-tuned* artifact rather than an off-the-shelf one. **This needs the developer's decision before Stage 5/6 begins.**

Everything else in Stages 2–15 is proceeding on a technically-justified, code-derived basis per this report, per the prompt's instruction to decide and continue rather than ask on non-essential points.
