# PredictiveSense - project context for Claude Code

## What this project is

PredictiveSense is a one-month research prototype that supports a paper. It
observes an indoor cabin scene, tracks objects and people over time, and predicts
developing safety risks early enough to warn the user with a spoken alert and a
fixed preventive recommendation. Reproducibility, measurement honesty, and
attribution are hard requirements.

## Hardware and platform

Windows 11. Intel Core Ultra 5 125H (14 cores / 18 logical), integrated Intel Arc
GPU, 16 GB LPDDR5 shared. **No NVIDIA GPU, no CUDA - never write CUDA-dependent
or CUDA-conditional code.** Python 3.11. All processing is local: no cloud
service, no runtime download, no telemetry upload. Use `pathlib`; use
`time.monotonic()` for every duration.

## Two modes

`realtime` (live camera) and `recorded` (recorded video, retrospective) share one
perception -> tracking -> temporal -> risk core. `mode` is validated in config
and carried on every `StateSnapshot`. Only `predictivesense/pipeline/` may branch
on it. Phase 1 implements the input layer for both: `capture.owner` (`browser` |
`backend`) picks who opens the live camera, and `RecordedDriver` drives Mode B
files losslessly.

## Architecture invariants

- **Bounded everywhere.** `LatestFrameMailbox` is single-slot (depth 0 or 1). The
  WS broadcaster drops rather than buffers. Every `Rate` / `Samples` is a
  `deque(maxlen=...)`. No unbounded queue, list, or cache anywhere.
- **Preview / client independence.** A slow or dead WebSocket client never slows
  the analysis loop; `Broadcaster.publish` is O(1).
- **Detection / tracking separation.** Contracts keep `Detection`, `Pose`,
  `Track`, `Relation`, `RiskState`, `Alert` as distinct types.
- **Local-only, no cloud.** No outbound network connection at runtime. Binding a
  local listening socket for the API is fine; connecting outward is not.
- **Contracts are frozen.** Later phases may add fields to `core/types.py`; they
  may not rename or reshape one without a line in `docs/decisions.md`.
- **No global mutable state, no singletons, no import-time side effects.** Threads
  and wiring live only in `pipeline/`.

## Module map

- `predictivesense/core/enums.py` - `Mode`, `SourceKind` (`SYNTHETIC`/`DEVICE`/`FILE`/`BROWSER` constructible; `WEBRTC` never), `RiskLevel`, `TrackStatus`, and the "constructible source kind" guard.
- `predictivesense/core/types.py` - all frozen contracts (Block 8). Imports only stdlib, numpy, Pydantic, and `core.enums`. Phase 1 added `SourceInfo`, `ClipManifest`, `IngestHeader`.
- `predictivesense/config/settings.py` - typed settings, YAML profile loading, strict validation, env overrides (`PS_` prefix, `__` nesting). Phase 1 sections: `capture` (incl. tuning knobs `fourcc`/`buffer_size`/`warmup_frames` and Phase 1.5's `max_ws_buffered_bytes` worker-backpressure ceiling, default 1 MB - all default to the measured-optimal value), `recorder`, `video`. Phase 2 section: `perception` (`detection_enabled`/`pose_enabled`/`pose_every_n`/`intra_op_threads` (6 - measured knee; auto over-subscribes)/`provider` (cpu), nested `detector` + `pose` sub-sections per Block 6). Phase 2.5: `perception.pose_requires_person` + `perception.detector.decode` (`yolo`|`yolox`); new sections `policy` (`enabled`/`domain_restriction`/`margin_rule`/`size_rule`/`per_class_threshold_rule` switches, `domain_classes` (validated vs COCO-80), `per_class_thresholds`, `default_threshold`, `margin_min`, `min_box_area_frac`, `emit_unknown`, `aspect_ratio_bounds`), `dataset` (`root`/`coco_path`/`splits_path`/`frames_dirname`/`min_unseeded_fraction`), `eval` (`iou_threshold`/`results_dir`/`val_fraction`/`test_fraction`/`split_seed`).
- `predictivesense/camera/_opencv.py` - `quiet_opencv_logging()` (idempotent `cv2.setLogLevel(ERROR)` - kills the VIDEOIO index-probe spam), `fourcc_to_str()`, and (Phase 2.5) `read_image_bgr()` (keeps `cv2` out of the API layer for the labelling seed path).
- `predictivesense/camera/source.py` - `FrameSource` interface + `create_frame_source` (builds only `SYNTHETIC`; other kinds raise `NotImplementedError` - built by `pipeline.build_camera_source` / `RecordedDriver`).
- `predictivesense/camera/synthetic.py` - `SyntheticSource`: deterministic, strictly increasing `frame_id` / `capture_ts`, paced to a target rate.
- `predictivesense/camera/mailbox.py` - `LatestFrameMailbox`: single-slot, overwrite-on-write, exact `consumed` / `dropped`.
- `predictivesense/camera/framing.py` - binary ingest message encode/decode (`FramingError`) + clock-offset arithmetic (`ClockOffset`, `clock_offset_seconds`, `capture_ts_seconds`).
- `predictivesense/camera/enumerate.py` - backend device enumeration (`enumerate_devices`, `as_api_rows`); optional `pygrabber` names, never fatal. Phase 1.5: `load_backend_hint`/`save_backend_hint` read/merge the per-index winning-backend cache `results/camera_backends.json` (written by `benchmark_camera_matrix.py`, machine-specific, not committed).
- `predictivesense/camera/browser.py` - `BrowserSource`: single-slot newest-wins buffer fed by `WS /ws/ingest`; decodes JPEG, stamps `capture_ts` from the clock offset.
- `predictivesense/camera/device.py` - `DeviceSource`: OpenCV camera, MSMF->DSHOW fallback (or cache-hinted backend first when `backend="auto"` + `backend_cache_dir`), reports *achieved* geometry. Reconnect is **non-blocking and `stop()`-interruptible** (one short reopen probe per `read()`, exponential backoff absorbed outside the lock via a `threading.Event`); no thread is added here.
- `predictivesense/camera/file_source.py` - `FileSource`: sequential decode, pts-based `capture_ts`, `asfast`/`realtime` replay, deterministic `restart()`; `video_duration_s()` helper.
- `predictivesense/pipeline/loop.py` - `AnalysisLoop` (+ `build_camera_source`, `request_consumer_stall`): producer thread + consumer + lifecycle + error surfacing + Phase 1 metric keys. Phase 2: takes an optional `PerceptionEngine`; the consumer runs `_run_perception(frame)` per iteration, populates `StateSnapshot.detections`/`.poses` and adds `detector_ms*`/`pose_ms*`/`perception_ms`/`detections_per_frame`/`poses_per_frame`/`*_warmup_ms`/`perception_frame_errors` metric keys. A perception exception is counted and dropped, never fatal. Phase 2.5: also takes an optional `RecognitionPolicy` (built by `build_loop` from `config.policy`), applied after `_run_perception`; adds `policy_accepted`/`policy_unknown_*`/`policy_rejected_*`/`policy_errors`/`policy_ms`/`policy_ms_p95` keys; `policy_raw_to_decided` property backs the Diagnostics readout.
- `predictivesense/pipeline/recorded.py` - `RecordedDriver`: Mode B, no mailbox, every frame, byte-deterministic `results/recorded_<run_id>.jsonl` + manifest. Phase 2: `run(..., perception=engine)` writes real `detections`/`poses` at fixed precision; determinism now includes the model (CPU ORT is deterministic). No `perception=` -> exactly the Phase 1 empty shape. Phase 2.5: `run(..., policy=RecognitionPolicy(...))` annotates each detection (`raw_class_name`/`policy_state`/`runner_up` in the JSONL); without `policy=` the JSONL shape is byte-identical to Phase 2.
- `predictivesense/perception/` - Phase 2 per-frame perception (detection + pose only; no tracking/identity/temporal/risk/voice). `runtime.py` (ORT session, explicit provider selection + verification, warm-up, `intra_op_threads`; Phase 2.5: `inter_op=1` + `ORT_SEQUENTIAL` set explicitly), `preprocess.py` (`letterbox` + coord mapping, pure NumPy; Phase 2.5: `to_rgb`/`scale`/`center` params for the YOLOX path, defaults = YOLO), `postprocess.py` (`xywh_to_xyxy`, `nms`, `class_aware_nms`; Phase 2.5: `yolox_decode` grid decode), `classes.py` (COCO-80, 12 required classes, alias map, 17 keypoints, skeleton, `class_color`), `detector.py` (`ObjectDetector.infer -> list[Detection]`, per-class thresholds, class list from ONNX `names` metadata; Phase 2.5: also sets `raw_class_name`/`runner_up` additively, `decode: yolo|yolox` variant - YOLO path unchanged), `pose.py` (`PoseEstimator.infer -> list[Pose]`), `engine.py` (`PerceptionEngine` + `build_perception(config, strict=)`; Phase 2.5: `pose_requires_person` gating), `policy.py` (**Phase 2.5** `RecognitionPolicy.apply(detections, frame_w, frame_h) -> PolicyOutcome`: domain / per-class-threshold / top-2-margin / size rules, each switchable and counted, `accepted+unknown+rejected==input`, never raises into the loop), `types.py` (`PerceptionResult` internal aggregate). `onnxruntime` + `cv2` are imported only here and under `camera/`.
- `predictivesense/dataset/` - **Phase 2.5** labelled eval set. `coco_store.py` (`CocoStore` read/write COCO detection JSON, id allocation, schema validation, `.bak` on overwrite, `seeded`/`labelled`/`ps_provenance` per image; `domain_categories`), `splits.py` (`build_splits` session-disjoint, `assert_no_leakage`, `load_splits` refuses a stale content hash), `quality.py` (`FrameProvenance`, `ProgressSummary`, unseeded-subset helpers). Stdlib + numpy only.
- `predictivesense/eval/` - **Phase 2.5** detection eval harness. `matching.py` (`iou_xyxy`, `greedy_match` - greedy score-ordered geometric assignment), `metrics.py` (`evaluate -> EvalMetrics`: per-class P/R/F1/support, confusion incl `background`+`unknown`, **false-class rate** headline, AP@0.5/mAP@0.5, top confusions; sample counts beside every metric; empty-safe), `report.py` (`run_metadata` + `write_reports` json+md). No onnxruntime/cv2 here.
- `predictivesense/telemetry/metrics.py` - `Counter`, `Rate`, `Samples`, `Timer`, `MetricRegistry` (all bounded).
- `predictivesense/telemetry/writer.py` - `MetricsWriter`: append-only CSV, write failure logged once and non-fatal.
- `predictivesense/telemetry/manifest.py` - `SessionManifest` + `build_manifest` + `git_state`.
- `predictivesense/api/app.py` - FastAPI factory and routes; mounts the ingest/recorder/videos/**labels** routers and `/static`; starts/stops the loop over the lifespan; `write_manifest=True` writes `results/session_<id>.json`. Phase 1.5: `POST /api/metrics/browser`. Phase 2.5: `GET /label` serves the labelling tool; `app.state.policy` shared with `POST /api/analyze`.
- `predictivesense/api/labels.py` - **Phase 2.5** `GET/POST /api/labels/frames|frame/{id}|progress|image/{id}|eval-summary`. Store from `config.dataset.coco_path` (503 with the fix command until `build_eval_frames.py` runs); write lock; optional detector seeding recorded per image.
- `predictivesense/api/ingest.py` - `WS /ws/ingest`: hello/ack/echo handshake + binary analysis frames -> `BrowserSource`.
- `predictivesense/api/recorder.py` - `POST /api/record/upload` + `GET /api/clips`: raw clip -> `data/raw/<session>/` + `ClipManifest`.
- `predictivesense/api/videos.py` - `GET /api/videos` + `POST /api/analyze` (path confined to `video.input_dir`). Phase 2: `/api/analyze` passes `app.state.perception` (the loop's engine, or None) to `RecordedDriver` so recorded runs use the identical perception code without re-creating sessions.
- `predictivesense/api/broadcast.py` - `Broadcaster` + `serve_state_client`: last-value-wins, slow client dropped on timeout.
- `predictivesense/api/static/` - dashboard. No framework, no build; ES modules served by the `/static` mount. **Phase 1.6** reorganised it into an application shell:
  - `index.html` - shell skeleton only (top bar, viewport with `<video id="preview" autoplay muted playsinline>` + `#overlay-layer`, empty `#panel-body`). `app.js` - composition root: fetch config, `registerGroup` x5, mount shell + registry.
  - `ui/` - `store.js` (observable UI state, `localStorage`), `shell.js`, `group.js`, `registry.js` (`registerGroup` / `registerAnalysisModule`), `controls.js` (`settingRow`/`actionButton{variant}`/`statusIndicator`/`metricRow`/`el`), `format.js` (label maps, `REPLAY_MODES`, `PREVIEW_UNAVAILABLE_TEXT`), `log.js` (the only `console.log`, gated by the Diagnostics toggle).
  - `groups/` - `constants.js` (`GROUP_ORDER`; `RESERVED_GROUP_IDS = ["alerts"]` after Phase 2 filled `analysis` and Phase 2.5 filled `research`), then one module per panel section: `input`, `camera`, `video`, `dataset`, `analysis` (Phase 2), `research` (**Phase 2.5** - links to `/label`, live labelling progress, latest eval summary), `diagnostics`. Each exports `id,title,order,modes,view,summary,render,update?`.
  - `features/` - logic moved out of `app.js` essentially verbatim: `runtime.js` (shared capture state + event bus), `camera-capture.js`, `analysis-client.js` (owns the Worker), `recording.js`, `videos.js`, `metrics.js`. Phase 2: `detection.js` + `pose.js` (analysis sub-modules via `registerAnalysisModule`), `overlay.js` (draws boxes/skeletons on `#overlay-layer`; never touches `<video>`; dashed+dimmed for the low-confidence band; `STALE` label when stale; no track IDs), `analysis-prefs.js` (per-viewer overlay-layer toggles, init from config, persisted). Phase 2.5: `policy.js` (per-viewer policy-view toggles - policy/domain/margin - re-derived from the additive `Detection` fields, **no API route**; `activeThresholds()` read-only); `overlay.js` renders `unknown` dashed/muted/`Unknown` keeping the box; `detection.js` gains the policy controls; `diagnostics.js` gains per-rule rejection counters + raw->decided list.
  - `label/` - **Phase 2.5** standalone labelling tool (`index.html` + `label.js`), served at `/label`, not part of the shell: canvas box editor (draw/move/resize/relabel/delete), class palette with keyboard shortcuts, next/prev, seed-from-detector, saves to `POST /api/labels/frame/{id}`.
  - `analysis-worker.js` - **unchanged** (newest-wins + `bufferedAmount` backpressure + `VideoFrame` close audit; `capture.max_ws_buffered_bytes`).
  - Input mode (`Real-time` / `Recorded video`) is **client-side view state** (`store`, persisted) - it does not touch the server `mode` config; recorded analysis still runs via `POST /api/analyze`, and the recorded-mode viewport plays a locally chosen file (no endpoint). Every engineering metric lives in the Diagnostics group (hidden until the top-bar toggle; raw keys shown next to renamed labels). `POST /api/metrics/browser` + the Browser-measurement block are in Diagnostics.
- `predictivesense/logging_setup.py` - `configure_logging` / `get_logger`.
- `scripts/run_app.py` - start the API for a profile (`--source-kind synthetic|browser|device`).
- `scripts/run_noop.py` - 60-second instrumented no-op run (synthetic); writes `results/noop_<id>.csv` and `results/manifest_<id>.json`. Phase 2: forces perception off (it measures the Phase 0 loop + its 25 MB RSS budget).
- `scripts/run_recorded.py` - run `RecordedDriver` over one file under `data/videos/`. Phase 2: builds perception from the profile (`--no-perception` to skip); fails loud (exit 2) if weights are missing.
- `scripts/fetch_models.py` - **manual** build-time model fetch: downloads + SHA-256-verifies `yolo11n.onnx` / `yolo11n-pose.onnx` / (Phase 2.5) `yolox_tiny.onnx` per `models/manifest.json`, writes the AGPL licence text. Not imported by the package; excluded from the outbound-network guard by path.
- `scripts/build_eval_frames.py` - **Phase 2.5** sample frames from `data/raw/` clips (`--every-n`, `--max-per-clip`) -> `data/eval/frames/*.jpg` + provenance + create the COCO store.
- `scripts/build_splits.py` - **Phase 2.5** session-disjoint `val`/`test` split -> `data/eval/splits.json` (content hash). Exit 2 when nothing is labelled.
- `scripts/fit_thresholds.py` - **Phase 2.5** per-class thresholds + `margin_min` on `val` only -> `results/fit_thresholds_val.{json,md}`. Refuses `test`.
- `scripts/eval_detection.py` - **Phase 2.5** `--split val|test --model yolo11n|yolox_tiny --policy on|off`: greedy-IoU harness -> `results/eval_<model>_<policy>_<split>.{json,md}`. Refuses empty/partial split; logs every `test` opening to `results/test_set_openings.md` and refuses a 2nd without `--allow-reopen`.
- `scripts/policy_effect.py` - **Phase 2.5** mechanical policy effect on unlabelled frames (counts, not accuracy) -> `results/policy_effect_unlabelled_<model>.{json,md}`.
- `scripts/benchmark_latency.py` - **Phase 2.5** pose-gating / intra-op thread sweep / policy cost (<1ms p95) over `data/raw/` -> `results/latency_2_5.{json,md}`.
- `scripts/_eval_common.py` - shared model registry + split-completeness checks for the eval scripts (needs `cv2.imread`, hence in `scripts/`, not the package).
- `scripts/benchmark_providers.py` - `--provider cpu|dml`: both models over a fixed clip -> warm-up / p50 / p95 / max / throughput / peak RSS + agreement check vs the CPU baseline -> `results/providers_<provider>.{json,md}`. DirectML runs from a separate `.venv-dml`.
- `scripts/class_coverage_audit.py` - `--source <clip-or-dir>`: detector over the developer's footage -> per required class: frames-with-detection, rate, conf p10/p50/p90, median box-area fraction, multi-count frames; top unexpected classes; a developer verdict column -> `results/class_coverage.{json,md}`. **Frequencies, not accuracy** - there are no labels.
- `scripts/benchmark_transport.py` - backend-owned camera transport benchmark -> `results/transport_<label>.json`.
- `scripts/benchmark_camera_matrix.py` - Phase 1.5: backend-owned matrix sweep (resolution x fps x backend x fourcc) -> `results/camera_matrix_<label>.{json,md}` + merges the winning backend into `results/camera_backends.json`.
- `scripts/benchmark_analysis_path.py` - Phase 1.5: in-process loopback sweep of `analysis_fps` x size x quality through the real `BrowserSource`->mailbox->loop -> `results/analysis_sweep_<label>.{json,md}`.

## Core contracts

Names only; definitions in `predictivesense/core/types.py`:
`BBox`, `Frame`, `Detection`, `Pose`, `Track`, `Relation`, `RiskState`, `Alert`,
`StateSnapshot`, `MailboxStats`, `SourceInfo`, `ClipManifest`, `IngestHeader`.
Interfaces: `FrameSource` (in `camera/source.py`), the mailbox `put` / `get` /
`stats` (in `camera/mailbox.py`).
Phase 2.5 added three **additive** fields to `Detection` (no rename/reshape):
`raw_class_name: str`, `policy_state: str`, `runner_up: tuple[str, float] | None`
(see `docs/decisions.md`).

## Commands

PowerShell, from `C:\Users\mummi\Documents\Projects\PredictiveSense`:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,camera]"

pytest -q                                    # full suite (hardware skipped)
pytest -q -m "not slow"                       # fast subset
pytest -q -m hardware --run-hardware          # only with a camera attached
python scripts\run_app.py --profile dev       # synthetic no-op API, http://127.0.0.1:8000/
python scripts\run_app.py --profile dev --source-kind browser   # Phase 1 browser input layer
python scripts\run_noop.py --profile dev --seconds 60           # writes results\
python scripts\run_recorded.py --profile eval --path <clip> --replay-mode asfast
python scripts\benchmark_transport.py --index 0 --seconds 30 --label laptop-cam

# --- Phase 2 perception ---
python scripts\fetch_models.py                                  # manual, once; verifies sha256
pytest -q -m models                                             # skips cleanly if models/ is empty
python scripts\benchmark_providers.py --provider cpu
py -3.11 -m venv .venv-dml; .\.venv-dml\Scripts\Activate.ps1
pip install -e "."; pip install onnxruntime-directml==1.24.4    # NEVER into the main .venv
python scripts\benchmark_providers.py --provider dml; deactivate
.\.venv\Scripts\Activate.ps1
python scripts\class_coverage_audit.py --source data\raw
python scripts\run_recorded.py --profile eval --path <clip>     # now populates detections/poses

# --- Phase 2.5 recognition reliability & measurement ---
python scripts\fetch_models.py                                  # also fetches yolox_tiny.onnx (Apache-2.0)
pytest -q -m dataset                                            # skips cleanly if data\eval\ is unlabelled
python scripts\build_eval_frames.py --source data\raw --out data\eval\frames --every-n 15 --max-per-clip 40
python scripts\run_app.py --profile dev --source-kind browser   # developer labels at http://127.0.0.1:8000/label
python scripts\policy_effect.py --model yolo11n                  # mechanical policy effect (counts, not accuracy)
python scripts\benchmark_latency.py --source data\raw           # pose gating / thread sweep / policy cost
# after labelling:
python scripts\build_splits.py
python scripts\fit_thresholds.py --split val
python scripts\eval_detection.py --split val  --model yolo11n    --policy off
python scripts\eval_detection.py --split val  --model yolo11n    --policy on
python scripts\eval_detection.py --split val  --model yolox_tiny --policy on
python scripts\eval_detection.py --split test --model yolo11n    --policy on --reason "final" # once, logged

# regenerate the lock file (UTF-8, no BOM; Windows PowerShell 5.1 has no utf8NoBOM):
$f = & .\.venv\Scripts\python.exe -m pip freeze --exclude-editable
[IO.File]::WriteAllText("$PWD\requirements.lock.txt", ($f -join "`n") + "`n", (New-Object Text.UTF8Encoding($false)))
```

## UI rules (permanent, Phase 1.6+)

- **The dashboard is an application shell, not a panel stack.** Three CSS-grid
  regions (top bar / viewport / collapsible right panel). No framework, no
  bundler, no build step, no CDN asset, no external font, no runtime dependency -
  everything ships from `static/`.
- **Add a control by registering a group, never by editing the shell.** One call:
  `registerGroup({ id, title, order, modes, view, summary, render, update? })` in
  `app.js`, with a module under `static/groups/` and its logic under
  `static/features/`. `order` comes from `static/groups/constants.js`. Analysis
  sub-modules use `registerAnalysisModule(...)`. Worked example in
  `docs/architecture.md`.
- `render` runs once at mount; `update(state)` must be cheap and must not rebuild
  the DOM; `summary(state)` must be a pure function of store state.
- **One logical home per control - no duplicates.** Camera selection only under
  Camera; video selection only under Recorded video; recording only under Dataset
  & recording; every engineering metric only under Diagnostics.
- **Never delete a metric - relocate it to Diagnostics.** Normal view keeps only
  mode, status, current-config summaries, primary actions, and warnings that need
  the user.
- **Preview `<video>` is `srcObject`-only** for the live camera (never read,
  drawn, or replaced for display); the sole read is `createImageBitmap` in the
  rVFC analysis fallback. `assertPreviewUncomposited` must keep passing. Keep
  `autoplay muted playsinline` on it in `index.html`; no `filter` / `transform` /
  `animation` / `opacity` / `transition` on `#preview`.
- **Action hierarchy is visible:** `btn-primary` (Start, Analyse, Record sample),
  `btn-secondary` (Advanced, Diagnostics, Capture sample), `btn-danger` (Stop,
  Delete, Clear).
- **No `console.log`** outside `static/ui/log.js` (the Diagnostics-gated logger).
  No unbounded listener list - `store.subscribe` returns an unsubscribe.
- **Input mode is client-side view state only** - it must not mutate the server
  `mode` config or add an API route. Reserved group ids `alerts` / `research` are
  constants; do not create a module for them, not even empty. (`analysis` was
  reserved in Phase 1.6 and is now filled by Phase 2 - `groups/analysis.js` hosts
  the Detection + Pose sub-modules via `registerAnalysisModule`; `overlay.js`
  draws on `#overlay-layer` only, never into `<video>`.)

### Where does a new control go?

| The control is about... | Group id | Modes | View |
|---|---|---|---|
| Choosing Real-time vs Recorded video | `input` | both | normal |
| Live camera: device, resolution, FPS, capture tuning | `camera` | realtime | normal |
| A recorded file: selection, replay speed, Analyse, output | `video` | recorded | normal |
| Recording a research clip, scenario tag, notes, consent, clip list | `dataset` | both | normal |
| Any FPS / drop / latency / byte / socket / worker counter, provider info | `diagnostics` | both | diagnostics |
| Detection / pose readouts | `analysis` *(order 50, Phase 2; Detection + Pose sub-modules via `registerAnalysisModule`)* | both | normal |
| Risk warnings, alert log, TTS state (later phase) | `alerts` *(reserved, order 60)* | both | normal |
| Labelling links, labelling progress, latest eval summary | `research` *(order 70, Phase 2.5)* | both | normal |

## Phase discipline

Current phase: **Phase 2.5 (Recognition Reliability & Measurement) COMPLETE** -
see `PredictiveSense-P2.5-Prompt.md`. Adds the labelled eval set (COCO), the eval
harness, and the switchable recognition policy. Still **no** tracker, track IDs,
association, `Relation`, temporal state, risk model, alert policy, TTS, or
object-enrollment UI - none started, none placeheld. `StateSnapshot.tracks` is
still always `[]`. **No training / fine-tuning.** Never report a confidence value
as accuracy, or a detection frequency as precision/recall. Never fit a threshold
on `test`, open `test` more than once, or omit the opening log
(`results/test_set_openings.md`). Never claim a performance number not produced by
a command run on this machine, or a physical check not performed. **Do not start
Phase 3.**

## Current phase status

Phase 2.5 complete. `pytest -q`: **250 passed, 2 skipped** (1 hardware
`--run-hardware`; 1 `dataset` - skips cleanly until the eval set is labelled).
`pytest -q -m models` = **12 passed**; `pytest -q -m dataset` = 1 skipped
cleanly. See `docs/phase-reports/phase2_5.md` (three-part format),
`docs/architecture.md` "Phase 2.5", `docs/decisions.md` "Phase 2.5",
`docs/attribution.md` (AGPL decision - evidence gathered, still deferred).

Phase 2.5 summary: `predictivesense/dataset/` (COCO store, session-disjoint
splits + hash, provenance) + `predictivesense/eval/` (greedy-IoU matching,
per-class P/R/F1, confusion incl `background`+`unknown`, **false-class rate**
headline, mAP@0.5) + `predictivesense/perception/policy.py` (`RecognitionPolicy` -
domain / per-class-threshold / top-2-margin / size rules, each switchable and
counted, `accepted+unknown+rejected==input`, never raises, applied in
`loop.py` + `recorded.py`). `Detection` gained `raw_class_name` / `policy_state` /
`runner_up` (additive). Labelling tool at `/label` + `api/labels.py`; `research`
group registered (order 70). YOLOX-tiny (Apache-2.0) wired via
`DetectorConfig.decode="yolox"` for the licence comparison. **Measured (this
machine):** policy cost 0.05 ms p50 / 0.07 ms p95 (< 1 ms target - PASS);
detector p50 ~66 ms / pose ~53 ms / combined ~120 ms (CPU, 1080p, threads 6 -
the knee, re-confirmed); 34 eval frames sampled from the 1 clip in `data/raw/`
(1 session), COCO store created with 0 annotations - **labelling is the
developer's step**. Everything downstream (`build_splits` / `fit_thresholds` /
`eval_detection`) is built + tested and **blocks cleanly** with the fix command
until labels exist; the "baseline vs policy on `val`" table, fitted thresholds,
and the labelled model comparison are **pending the developer's labels**. The
mechanical policy effect on the unlabelled frames is measured
(`results/policy_effect_unlabelled_*.md`: YOLO11n 43 raw -> 39 accepted / 4
out-of-domain -> `unknown`, reconciles). `test` split **never opened**.
The pre-Phase-2.5 status text follows.

Phase 2 summary: ONNX detector (`yolo11n`) + pose (`yolo11n-pose`), pre-exported
ONNX fetched + hash-verified by `scripts/fetch_models.py` (weights git-ignored;
**AGPL-3.0, open licence decision** in `docs/attribution.md`). `onnxruntime==
1.24.4` (CPU) added as a runtime dep, imported only under `perception/`. Both
models run on the same sampled frame in the analysis loop and in `RecordedDriver`
(byte-deterministic with models). `StateSnapshot.detections`/`.poses` populated;
contracts unchanged; metrics gained keys only. UI: `groups/analysis.js` fills the
reserved slot with Detection + Pose sub-modules; `features/overlay.js` draws
boxes/skeletons on `#overlay-layer` (dashed+dimmed low-confidence band, `STALE`
label, no track IDs); Diagnostics gained a Perception block. Toggling perception
off reproduces Phase 1.6 exactly. **Measured (this machine):** detector p50
~44 ms / pose p50 ~44 ms isolated (CPU, input 640); combined collapses to ~440 ms
under thread contention unless `perception.intra_op_threads` is capped -> shipped
at **6** (knee), giving combined ~90-110 ms; with perception on, browser-ingest
analysis holds **~10 fps** (matches the 10 fps feed) with **~2.7 % steady-state
drop** and frame age ~50 ms -> ~146 ms; DirectML (isolated `.venv-dml`) **agrees
with CPU to 0.0 px mean box diff but is ~2.7x slower** on these nano models ->
default `provider: cpu`; input-size sweep {320,480,640} tabled
(`results/input_size_sweep.md`). Preview independence re-verified with perception
running (`test_preview_independence_holds_with_perception_running`): socket keeps
accepting, mailbox stays single-slot, producer alive. Class-coverage audit run
over the one clip in `data/raw/` (`results/class_coverage.md`) - **detection
frequencies, not accuracy**; `person` 100 %, `cup`/`cell phone` low, others 0 on
that single clip; verdict column is the developer's. **Block 11 physical checks
pending** (no camera ran the overlay this pass - Browser pane blocks capture).

### Phase 1 (historical)

`pytest -q` was **145 passed, 1 skipped** at end of Phase 1.6. 10-minute
synthetic no-op RSS within budget; `DeviceSource` RSS flat over 30 s of real
capture.

Phase 1.6 (see `docs/phase-reports/phase1.md` "Phase 1.6" section,
`docs/architecture.md`, `docs/decisions.md`): the flat panel stack in
`static/` became a video-first application shell - top bar (mode, status pill,
Diagnostics + panel-collapse toggles), letterboxing viewport with an empty
`#overlay-layer`, collapsible right panel of collapsible groups. New `static/ui/`
component layer + one `static/groups/` module per section (`input`, `camera`,
`video`, `dataset`, `diagnostics`; `analysis`/`alerts`/`research` reserved, not
registered) + `static/features/` (camera / worker / recording / video / metrics
logic moved out of `app.js` verbatim; `app.js` is now a composition root). One
registration call - `registerGroup(...)` - is the whole extension contract for
later phases. Input mode is client-side view state (persisted, no backend
switch); every metric relocated to Diagnostics; `asfast`->Fastest,
replay `realtime`->Real-time speed, "backend owns the camera"->"Live preview
unavailable in backend-camera mode." No API route, payload, config or contract
changed; `analysis-worker.js` byte-unchanged; no new dependency. +18 static-asset
tests. Browser-pane render/interaction check clean (camera itself blocked
there - the eight Block 10 physical checks are the developer's).

Phase 1.5 (see `docs/phase-reports/phase1.md` "Phase 1.5" section and
`docs/decisions.md`): camera matrix on the Integrated Camera (index 0, 12 combos,
`benchmark_camera_matrix.py`) - **0 read failures / 0 reconnects / ~29 fps at
every resolution**; requested == achieved throughout; MJPG FOURCC is a no-op on
both MSMF (`raw:22`) and DSHOW (`YUY2`); DSHOW is the measured winner for index 0
(tighter p95, lower CPU) and is now cached to `results/camera_backends.json` and
opened first under `device_backend: auto`. Analysis-path sweep
(`benchmark_analysis_path.py`, 27 combos, loopback): drop rate ~0 everywhere,
frame age at the sampling floor - **defaults kept at 10 fps / 640x480 / q0.70**
(raising any axis only costs CPU/bandwidth). Worker got an explicit newest-wins +
`bufferedAmount` backpressure skip (`capture.max_ws_buffered_bytes`, 1 MB) + a
`VideoFrame` close audit; device switching stops every track and nulls
`srcObject` before re-acquiring and measures switch time; `applyConstraints` for
same-device size changes; `ondevicechange` re-enumerates. New:
`POST /api/metrics/browser`, dashboard "Browser measurement" panel,
`config/benchmarked_camera_combos.json` (defaults-can't-drift test).

The API adds `/api/cameras`, `/ws/ingest`, `/api/videos`, `/api/analyze`,
`/api/record/upload`, `/api/clips`, `/api/debug/stall`, `/api/metrics/browser`,
`/static/*`. `RecordedDriver` is byte-deterministic; preview independence holds
under the stall test. `cv2` is confined to `predictivesense/camera/`.

Developer physically verified both the Integrated Camera and the OnePlus virtual
camera (browser path) as smooth. Still pending developer verification (no browser
ran the camera path this pass; OnePlus virtual camera not registered): live
browser-side numbers (preview/worker-encode/switch-time), OnePlus **backend**
matrix, mid-run phone disconnect, stall-by-eye, clip-on-disk. Nothing from P2+
has been started.
