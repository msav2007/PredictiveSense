# PredictiveSense - project context

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
- `predictivesense/core/types.py` - all frozen contracts (Block 8). Imports only stdlib, numpy, Pydantic, and `core.enums`. Phase 1 added `SourceInfo`, `ClipManifest`, `IngestHeader`. Phase 5 added `Detection.tier` (additive).
- `predictivesense/config/settings.py` - typed settings, YAML profile loading, strict validation, env overrides (`PS_` prefix, `__` nesting). Phase 1 sections: `capture` (incl. tuning knobs `fourcc`/`buffer_size`/`warmup_frames` and Phase 1.5's `max_ws_buffered_bytes` worker-backpressure ceiling, default 1 MB - all default to the measured-optimal value), `recorder`, `video`. Phase 2 section: `perception` (`detection_enabled`/`pose_enabled`/`pose_every_n`/`intra_op_threads` (6 - measured knee; auto over-subscribes)/`provider` (cpu), nested `detector` + `pose` sub-sections per Block 6). Phase 2.5: `perception.pose_requires_person` + `perception.detector.decode` (`yolo`|`yolox`); new sections `policy` (`enabled`/`domain_restriction`/`margin_rule`/`size_rule`/`per_class_threshold_rule` switches; **Phase 5** replaced `domain_classes` with nested `vocabulary` (`primary`/`secondary`/`implausible` - a total, disjoint, load-validated partition of COCO-80; `domain_classes` kept as a `@computed_field` = `primary`) and added `thresholds_fitted` (false until `fit_thresholds.py` runs) + `show_suppressed_in_diagnostics`; `per_class_thresholds`, `default_threshold`, `margin_min`, `min_box_area_frac`, `emit_unknown`, `aspect_ratio_bounds`), `dataset` (`root`/`coco_path`/`splits_path`/`frames_dirname`/`min_unseeded_fraction`), `eval` (`iou_threshold`/`results_dir`/`val_fraction`/`test_fraction`/`split_seed`). Phase 4 sections: `objects` (`root`/`max_image_mb`/`thumbnail_px`/`blur_var_min`/`min_box_area_frac`/`duplicate_hamming_max` + nested `coverage_targets`), `studio` (`stop_monitoring_on_enter`), `ui.panel` (`default_width_px`/`min_width_px`/`max_width_px`/`max_width_frac`) - all with measured/prompt defaults, served read-only in `/api/config`.
- `predictivesense/camera/_opencv.py` - `quiet_opencv_logging()` (idempotent `cv2.setLogLevel(ERROR)` - kills the VIDEOIO index-probe spam), `fourcc_to_str()`, (Phase 2.5) `read_image_bgr()`, and (Phase 4) `decode_image_bgr()` / `encode_jpeg()` / `thumbnail_jpeg()` - the Object Learning Studio upload/capture codec path, keeping `cv2` out of `predictivesense/api/`.
- `predictivesense/camera/source.py` - `FrameSource` interface + `create_frame_source` (builds only `SYNTHETIC`; other kinds raise `NotImplementedError` - built by `pipeline.build_camera_source` / `RecordedDriver`).
- `predictivesense/camera/synthetic.py` - `SyntheticSource`: deterministic, strictly increasing `frame_id` / `capture_ts`, paced to a target rate.
- `predictivesense/camera/mailbox.py` - `LatestFrameMailbox`: single-slot, overwrite-on-write, exact `consumed` / `dropped`.
- `predictivesense/camera/framing.py` - binary ingest message encode/decode (`FramingError`) + clock-offset arithmetic (`ClockOffset`, `clock_offset_seconds`, `capture_ts_seconds`).
- `predictivesense/camera/enumerate.py` - backend device enumeration (`enumerate_devices`, `as_api_rows`); optional `pygrabber` names, never fatal. Phase 1.5: `load_backend_hint`/`save_backend_hint` read/merge the per-index winning-backend cache `results/camera_backends.json` (written by `benchmark_camera_matrix.py`, machine-specific, not committed).
- `predictivesense/camera/browser.py` - `BrowserSource`: single-slot newest-wins buffer fed by `WS /ws/ingest`; decodes JPEG, stamps `capture_ts` from the clock offset.
- `predictivesense/camera/device.py` - `DeviceSource`: OpenCV camera, MSMF->DSHOW fallback (or cache-hinted backend first when `backend="auto"` + `backend_cache_dir`), reports *achieved* geometry. Reconnect is **non-blocking and `stop()`-interruptible** (one short reopen probe per `read()`, exponential backoff absorbed outside the lock via a `threading.Event`); no thread is added here.
- `predictivesense/camera/file_source.py` - `FileSource`: sequential decode, pts-based `capture_ts`, `asfast`/`realtime` replay, deterministic `restart()`; `video_duration_s()` helper.
- `predictivesense/pipeline/loop.py` - `AnalysisLoop` (+ `build_camera_source`, `request_consumer_stall`): producer thread + consumer + lifecycle + error surfacing + Phase 1 metric keys. Phase 2: takes an optional `PerceptionEngine`; the consumer runs `_run_perception(frame)` per iteration, populates `StateSnapshot.detections`/`.poses` and adds `detector_ms*`/`pose_ms*`/`perception_ms`/`detections_per_frame`/`poses_per_frame`/`*_warmup_ms`/`perception_frame_errors` metric keys. A perception exception is counted and dropped, never fatal. Phase 2.5: also takes an optional `RecognitionPolicy` (built by `build_loop` from `config.policy`), applied after `_run_perception`; adds `policy_accepted`/`policy_unknown_*`/`policy_rejected_*`/`policy_errors`/`policy_ms`/`policy_ms_p95` keys; `policy_raw_to_decided` property backs the Diagnostics readout. Phase 4: `pause()` / `resume()` / `paused` (a dedicated `threading.Event`, distinct from the debug stall) - while paused the consumer runs **no** iteration (no perception, no snapshot) and the producer reads no frames; used by the Object Learning Studio lifecycle.
- `predictivesense/pipeline/recorded.py` - `RecordedDriver`: Mode B, no mailbox, every frame, byte-deterministic `results/recorded_<run_id>.jsonl` + manifest. Phase 2: `run(..., perception=engine)` writes real `detections`/`poses` at fixed precision; determinism now includes the model (CPU ORT is deterministic). No `perception=` -> exactly the Phase 1 empty shape. Phase 2.5: `run(..., policy=RecognitionPolicy(...))` annotates each detection (`raw_class_name`/`policy_state`/`runner_up` in the JSONL); without `policy=` the JSONL shape is byte-identical to Phase 2.
- `predictivesense/perception/` - Phase 2 per-frame perception (detection + pose only; no tracking/identity/temporal/risk/voice). `runtime.py` (ORT session, explicit provider selection + verification, warm-up, `intra_op_threads`; Phase 2.5: `inter_op=1` + `ORT_SEQUENTIAL` set explicitly), `preprocess.py` (`letterbox` + coord mapping, pure NumPy; Phase 2.5: `to_rgb`/`scale`/`center` params for the YOLOX path, defaults = YOLO), `postprocess.py` (`xywh_to_xyxy`, `nms`, `class_aware_nms`; Phase 2.5: `yolox_decode` grid decode), `classes.py` (COCO-80, 12 required classes, alias map, 17 keypoints, skeleton, `class_color`), `detector.py` (`ObjectDetector.infer -> list[Detection]`, per-class thresholds, class list from ONNX `names` metadata; Phase 2.5: also sets `raw_class_name`/`runner_up` additively, `decode: yolo|yolox` variant - YOLO path unchanged), `pose.py` (`PoseEstimator.infer -> list[Pose]`), `engine.py` (`PerceptionEngine` + `build_perception(config, strict=)`; Phase 2.5: `pose_requires_person` gating), `vocabulary.py` (**Phase 5** three-tier class partition `PRIMARY_TIER`/`SECONDARY_TIER`/`IMPLAUSIBLE_TIER`, `validate_partition` (total+disjoint over COCO-80, fails loudly), `VocabularyTiers.tier_for` -> `primary|secondary|implausible|unlisted`; stdlib-only), `policy.py` (`RecognitionPolicy.apply(detections, frame_w, frame_h) -> PolicyOutcome`: size / implausible-tier suppression / per-class-threshold / top-2-margin / secondary rules, each switchable and counted; **Phase 5** six reconciling `policy_state` values, `suppressed_implausible` distinct from `unknown`, sets `Detection.tier`; never raises into the loop), `types.py` (`PerceptionResult` internal aggregate). `onnxruntime` + `cv2` are imported only here and under `camera/`.
- `predictivesense/dataset/` - **Phase 2.5** labelled eval set. `coco_store.py` (`CocoStore` read/write COCO detection JSON, id allocation, schema validation, `.bak` on overwrite, `seeded`/`labelled`/`ps_provenance` per image; `domain_categories`), `splits.py` (`build_splits` session-disjoint, `assert_no_leakage`, `load_splits` refuses a stale content hash), `quality.py` (`FrameProvenance`, `ProgressSummary`, unseeded-subset helpers). Stdlib + numpy only.
- `predictivesense/eval/` - **Phase 2.5** detection eval harness. `matching.py` (`iou_xyxy`, `greedy_match` - greedy score-ordered geometric assignment), `metrics.py` (`evaluate -> EvalMetrics`: per-class P/R/F1/support, confusion incl `background`+`unknown`, **false-class rate** headline, AP@0.5/mAP@0.5, top confusions; sample counts beside every metric; empty-safe), `report.py` (`run_metadata` + `write_reports` json+md). No onnxruntime/cv2 here.
- `predictivesense/objects/` - **Phase 4** Object Learning Studio store (data collection only; trains nothing). `vocab.py` (fixed `view/distance/lighting/background/occlusion/held/frame_position` vocabularies, `ROLES`/`KINDS`/`STATUSES`, `CONFUSABLE_SEEDS` from the developer's observed failure modes, `validate_conditions`). `registry.py` (`ObjectProfile` + `ObjectRegistry` over `data/objects/objects.json`: slug ids with `-2`/`-3` collision suffixes, `create`/`get`/`list`/`update`/`set_counts`/`soft_delete`, atomic `os.replace` writes, `_deleted/` move, corrupt file -> `ObjectStoreError`). `samples.py` (`ObjectSample` + `SampleStore` over `data/objects/<id>/manifest.json`: one required `[x,y,w,h]` box validated vs image bounds, condition dict filled+validated, provenance `captured_utc`/`git_commit`/`source`/`device_label`/`original_filename`/`consent_ack`, atomic write, soft delete moves image files to `_deleted/`). `quality.py` (**numpy only** - `laplacian_variance` via a 3x3 valid correlation, `dhash` via area-average block-reduce, `hamming`, `nearest_duplicate`, `compute_sample_quality` -> blur/area-frac/phash/flags, `coverage_summary` -> counts + plain guidance strings, no composite score). Stdlib + numpy only.
- `predictivesense/models/` - **Phase 4** `registry.py`: `ModelRegistry.load/list/get/active/resolve_active_files/validate` over `models/registry.json`. Exactly one `active`; every file SHA-256 a 64-hex string and, when the filename is in `models/manifest.json`, matching it. **No activation code, no training.** v1 = the pre-exported YOLO11n detector + pose. `default_registry_path()`.
- `predictivesense/telemetry/metrics.py` - `Counter`, `Rate`, `Samples`, `Timer`, `MetricRegistry` (all bounded).
- `predictivesense/telemetry/writer.py` - `MetricsWriter`: append-only CSV, write failure logged once and non-fatal.
- `predictivesense/telemetry/manifest.py` - `SessionManifest` + `build_manifest` + `git_state`.
- `predictivesense/api/app.py` - FastAPI factory and routes; mounts the ingest/recorder/videos/**labels**/**objects**/**studio** routers and `/static`; starts/stops the loop over the lifespan; `write_manifest=True` writes `results/session_<id>.json`. Phase 1.5: `POST /api/metrics/browser`. Phase 2.5: `GET /label` serves the labelling tool; `app.state.policy` shared with `POST /api/analyze`. Phase 4: `app.state.studio = {"active", "token", "prior"}`.
- `predictivesense/api/objects.py` - **Phase 4** `GET/POST /api/objects`, `GET/PATCH/DELETE /api/objects/{id}`, `GET/POST /api/objects/{id}/samples`, `PATCH/DELETE /api/objects/{id}/samples/{sid}`, `GET /api/objects/{id}/coverage`, `GET /api/objects/{id}/samples/{sid}/image` (`?thumb=1`), `GET /api/objects/vocab`. Multipart image decode/encode via `camera/_opencv`; quality via `objects.quality`; profile counts synced after every sample write; soft delete for both profiles and samples.
- `predictivesense/api/studio.py` - **Phase 4** `GET /studio` (standalone page), `POST /api/studio/enter` (pauses the loop, returns a prior-state token; idempotent), `POST /api/studio/leave` (resumes if this session paused it, 409 on token mismatch), `GET /api/studio/status`, `GET /api/models/registry` (graceful `{"available": false}` when the registry is missing/invalid). `studio_is_active(app)` used by `api/ingest.py`.
- `predictivesense/api/labels.py` - **Phase 2.5** `GET/POST /api/labels/frames|frame/{id}|progress|image/{id}|eval-summary`. Store from `config.dataset.coco_path` (503 with the fix command until `build_eval_frames.py` runs); write lock; optional detector seeding recorded per image.
- `predictivesense/api/ingest.py` - `WS /ws/ingest`: hello/ack/echo handshake + binary analysis frames -> `BrowserSource`. Phase 4: refuses the connection (close 4409) while the Object Learning Studio is active.
- `predictivesense/api/recorder.py` - `POST /api/record/upload` + `GET /api/clips`: raw clip -> `data/raw/<session>/` + `ClipManifest`.
- `predictivesense/api/videos.py` - `GET /api/videos` + `POST /api/analyze` (path confined to `video.input_dir`). Phase 2: `/api/analyze` passes `app.state.perception` (the loop's engine, or None) to `RecordedDriver` so recorded runs use the identical perception code without re-creating sessions.
- `predictivesense/api/broadcast.py` - `Broadcaster` + `serve_state_client`: last-value-wins, slow client dropped on timeout.
- `predictivesense/api/static/` - dashboard. No framework, no build; ES modules served by the `/static` mount. **Phase 1.6** reorganised it into an application shell:
  - `index.html` - shell skeleton only (top bar, viewport with `<video id="preview" autoplay muted playsinline>` + `#overlay-layer`, empty `#panel-body`). `app.js` - composition root: fetch config, `registerGroup` x5, mount shell + registry.
  - `ui/` - `store.js` (observable UI state, `localStorage`), `shell.js`, `group.js`, `registry.js` (`registerGroup` / `registerAnalysisModule`), `controls.js` (`settingRow`/`actionButton{variant}`/`statusIndicator`/`metricRow`/`el`), `format.js` (label maps, `REPLAY_MODES`, `PREVIEW_UNAVAILABLE_TEXT`), `log.js` (the only `console.log`, gated by the Diagnostics toggle).
  - `groups/` - `constants.js` (`GROUP_ORDER`; `RESERVED_GROUP_IDS = ["alerts"]` after Phase 2 filled `analysis` and Phase 2.5 filled `research`), then one module per panel section: `input`, `camera`, `video`, `dataset`, `analysis` (Phase 2), `research` (**Phase 2.5** - links to `/label`, live labelling progress, latest eval summary), `diagnostics`. Each exports `id,title,order,modes,view,summary,render,update?`.
  - `features/` - logic moved out of `app.js` essentially verbatim: `runtime.js` (shared capture state + event bus), `camera-capture.js`, `analysis-client.js` (owns the Worker), `recording.js`, `videos.js`, `metrics.js`. Phase 2: `detection.js` + `pose.js` (analysis sub-modules via `registerAnalysisModule`), `overlay.js` (draws boxes/skeletons on `#overlay-layer`; never touches `<video>`; dashed+dimmed for the low-confidence band; `STALE` label when stale; no track IDs), `analysis-prefs.js` (per-viewer overlay-layer toggles, init from config, persisted). Phase 2.5: `policy.js` (per-viewer policy-view toggles - policy/domain/margin - re-derived from the additive `Detection` fields, **no API route**; `activeThresholds()` read-only); `overlay.js` renders `unknown` dashed/muted/`Unknown` keeping the box; `detection.js` gains the policy controls; `diagnostics.js` gains per-rule rejection counters + raw->decided list.
  - `label/` - **Phase 2.5** standalone labelling tool (`index.html` + `label.js`), served at `/label`, not part of the shell: canvas box editor (draw/move/resize/relabel/delete), class palette with keyboard shortcuts, next/prev, seed-from-detector, saves to `POST /api/labels/frame/{id}`.
  - `studio/` - **Phase 4** standalone Object Learning Studio (`index.html` + `studio.js` + `studio.css`), served at `/studio`, **not** part of the shell. On load `POST /api/studio/enter` (stops monitoring); on exit `POST /api/studio/leave` (+ a `pagehide` `sendBeacon` backstop). Object CRUD sidebar, camera preview (`<video>.srcObject` only, no worker) with a drag/resize box editor (4 corner handles + arrow-key nudge), file upload (staged one at a time, box adjusted before save), fixed condition-tag selects (defaults remembered in `localStorage`), positive/negative/hard-negative role, `confusable_with` hard-negative prompt, review strip with an inspector (retag / re-box / soft-discard), coverage guidance panel, read-only active-model badge. **Phase 5** `studio-state.js` state machine (`browsing|object_selected|capturing|reviewing`); `studio.js` a pure render of it. **Phase 6** the object list is selected by ONE delegated `click` listener on the stable `#object-list` container (`wireObjectList()`) - never a per-row handler; `save_and_return` is reachable from `browsing` (was a dead button); `showFatal()` + `#studio-error` banner surface a module-load fault; `refuseNote()` gives visible feedback on a refused transition; `#studio-diag` shows `{state, selected_object_id, has_pending, last_refused_transition}`; the box re-centres on returning to the camera stage. Browser-verified by `tests/browser/`.
  - `ui/resizer.js` - **Phase 4** operations-panel drag handle: pure `clampPanelWidth(width, winWidth, cfg)` (min `ui.panel.min_width_px`; max min(`max_width_px`, `max_width_frac` * window); viewport kept >= 45%), a `requestAnimationFrame`-throttled pointer drag batching one read + one write of `--panel-w` on `.app-shell`, keyboard (`role="separator"`, arrows step 16 px, `Home` / double-click reset), width persisted as `store.panelWidth`. Mounted by `shell.js`.
  - `analysis-worker.js` - **unchanged** (newest-wins + `bufferedAmount` backpressure + `VideoFrame` close audit; `capture.max_ws_buffered_bytes`).
  - Input mode (`Real-time` / `Recorded video`) is **client-side view state** (`store`, persisted) - it does not touch the server `mode` config; recorded analysis still runs via `POST /api/analyze`, and the recorded-mode viewport plays a locally chosen file (no endpoint). Every engineering metric lives in the Diagnostics group (hidden until the top-bar toggle; raw keys shown next to renamed labels). `POST /api/metrics/browser` + the Browser-measurement block are in Diagnostics.
- `predictivesense/logging_setup.py` - `configure_logging` / `get_logger`.
- `scripts/run_app.py` - start the API for a profile (`--source-kind synthetic|browser|device`).
- `scripts/run_noop.py` - 60-second instrumented no-op run (synthetic); writes `results/noop_<id>.csv` and `results/manifest_<id>.json`. Phase 2: forces perception off (it measures the Phase 0 loop + its 25 MB RSS budget).
- `scripts/run_recorded.py` - run `RecordedDriver` over one file under `data/videos/`. Phase 2: builds perception from the profile (`--no-perception` to skip); fails loud (exit 2) if weights are missing.
- `scripts/fetch_models.py` - **manual** build-time model fetch: downloads + SHA-256-verifies `yolo11n.onnx` / `yolo11n-pose.onnx` / (Phase 2.5) `yolox_tiny.onnx` per `models/manifest.json`, writes the AGPL licence text. Not imported by the package; excluded from the outbound-network guard by path.
- `scripts/build_eval_frames.py` - **Phase 2.5** sample frames from `data/raw/` clips (`--every-n`, `--max-per-clip`) -> `data/eval/frames/*.jpg` + provenance + create the COCO store.
- `scripts/build_splits.py` - **Phase 2.5** session-disjoint `val`/`test` split -> `data/eval/splits.json` (content hash). Exit 2 when nothing is labelled.
- `scripts/fit_thresholds.py` - **Phase 2.5** per-class thresholds + `margin_min` on `val` only -> `results/fit_thresholds_val.{json,md}`. Refuses `test`. **Phase 5**: refuses an unlabelled / empty split (checks the COCO store's labelled count first) before touching the detector or writing anything; prints `thresholds_fitted: true` in its paste-in block.
- `scripts/policy_audit.py` - **Phase 5** the policy-change evidence script. Detector + policy over real footage (`--source`, `--frames`) -> per-state + per-raw-class counts, per-state confidence distribution, top implausible-tier suppressions, `raw -> decided` relabels, + a synthetic-noise control (0 detections expected) -> `results/policy_audit_<label>.{json,md}`. **Frequencies, not accuracy.**
- `scripts/eval_detection.py` - **Phase 2.5** `--split val|test --model yolo11n|yolox_tiny --policy on|off`: greedy-IoU harness -> `results/eval_<model>_<policy>_<split>.{json,md}`. Refuses empty/partial split; logs every `test` opening to `results/test_set_openings.md` and refuses a 2nd without `--allow-reopen`.
- `scripts/policy_effect.py` - **Phase 2.5** mechanical policy effect on unlabelled frames (counts, not accuracy) -> `results/policy_effect_unlabelled_<model>.{json,md}`.
- `scripts/benchmark_latency.py` - **Phase 2.5** pose-gating / intra-op thread sweep / policy cost (<1ms p95) over `data/raw/` -> `results/latency_2_5.{json,md}`.
- `scripts/benchmark_recognition_paths.py` - **Phase 6** the three BLOCK 8 experiments under one canonical latency protocol (Protocol A isolated in-process / Protocol B end-to-end app loopback, both documented in `docs/decisions.md`): pose gating false vs true, detector input size 480 vs 640 (latency + primary-tier detection counts), and a reconciliation of the Phase 2 vs Phase 5 detector figures -> `results/recognition_paths.{json,md}`. **Latency and frequencies, not accuracy.** No config default changed.
- `scripts/_eval_common.py` - shared model registry + split-completeness checks for the eval scripts (needs `cv2.imread`, hence in `scripts/`, not the package).
- `scripts/benchmark_providers.py` - `--provider cpu|dml`: both models over a fixed clip -> warm-up / p50 / p95 / max / throughput / peak RSS + agreement check vs the CPU baseline -> `results/providers_<provider>.{json,md}`. DirectML runs from a separate `.venv-dml`.
- `scripts/class_coverage_audit.py` - `--source <clip-or-dir>`: detector over the developer's footage -> per required class: frames-with-detection, rate, conf p10/p50/p90, median box-area fraction, multi-count frames; top unexpected classes; a developer verdict column -> `results/class_coverage.{json,md}`. **Frequencies, not accuracy** - there are no labels.
- `scripts/benchmark_transport.py` - backend-owned camera transport benchmark -> `results/transport_<label>.json`.
- `scripts/benchmark_camera_matrix.py` - Phase 1.5: backend-owned matrix sweep (resolution x fps x backend x fourcc) -> `results/camera_matrix_<label>.{json,md}` + merges the winning backend into `results/camera_backends.json`.
- `scripts/export_objects_coco.py` - **Phase 4** `--out data/objects/coco_train.json`: converts object samples to COCO detection format (one category per object; positives contribute a box, negatives/hard-negatives contribute a box-less image) with a **sample-disjoint** train/val split (writes `coco_val.json` + `export_manifest.json`). Runs `check_dataset_separation()` first and **exits 2 writing nothing** if any object image content hash / path / source id overlaps `data/eval`. Never touches `data/eval`. No training.
- `scripts/benchmark_analysis_path.py` - Phase 1.5: in-process loopback sweep of `analysis_fps` x size x quality through the real `BrowserSource`->mailbox->loop -> `results/analysis_sweep_<label>.{json,md}`.

## Core contracts

Names only; definitions in `predictivesense/core/types.py`:
`BBox`, `Frame`, `Detection`, `Pose`, `Track`, `Relation`, `RiskState`, `Alert`,
`StateSnapshot`, `MailboxStats`, `SourceInfo`, `ClipManifest`, `IngestHeader`.
Interfaces: `FrameSource` (in `camera/source.py`), the mailbox `put` / `get` /
`stats` (in `camera/mailbox.py`).
Phase 2.5 added three **additive** fields to `Detection` (no rename/reshape):
`raw_class_name: str`, `policy_state: str`, `runner_up: tuple[str, float] | None`.
Phase 5 added a fourth: `tier: str = "primary"` (`primary|secondary|implausible|
unlisted`) and widened `policy_state` to six values
(`accepted`/`accepted_secondary`/`unknown_low_confidence`/`unknown_margin`/
`suppressed_implausible`/`rejected_size`; `rejected_out_of_domain` was renamed
`suppressed_implausible`). See `docs/decisions.md`. Phase 4 added **no**
`core/types.py` contract - the Object Learning Studio's `ObjectProfile` /
`ObjectSample` are their own schemas under `predictivesense/objects/` (a separate
concern, like `dataset/`); Phase 5 added additive capture-quality provenance
fields to `ObjectSample` (`capture_path`, `requested_resolution`,
`achieved_resolution`, `encoded_quality`, `original_bytes`).

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

# --- Phase 4 Object Learning Studio & operations panel ---
python scripts\run_app.py --profile dev --source-kind browser   # developer works at http://127.0.0.1:8000/studio
pytest -q tests\unit\test_object_registry.py tests\unit\test_object_quality.py tests\unit\test_model_registry.py tests\unit\test_dataset_separation.py tests\unit\test_panel_resize_contract.py
pytest -q tests\integration\test_objects_api.py tests\integration\test_studio_lifecycle.py
python scripts\export_objects_coco.py --out data\objects\coco_train.json   # after the developer collects samples

# --- Phase 6 Studio repair (browser-verified) & recognition responsiveness ---
pip install -e ".[dev]"                                         # now pulls Playwright
python -m playwright install chromium                           # once
pytest -q -m browser                                            # tests/browser/*; skips cleanly if Playwright/Chromium absent
python scripts\benchmark_recognition_paths.py --source data\raw --frames 300   # pose gating + input size, one protocol -> results\recognition_paths.{json,md}

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
| Object Learning Studio | **not a group** - a link in the `dataset` group + the top bar to the standalone `/studio` page (Phase 4). Entering it stops monitoring. |

## Phase discipline

Current phase: **Phase 6 (Studio Repair, Browser-Verified & Recognition
Responsiveness) COMPLETE** - see `PredictiveSense-P6-Prompt.md`. The Object
Learning Studio is now reachable and usable in a browser, proven by a Playwright
suite that loads the real page and clicks. Two root causes, both reproduced in
the browser first: (1) object row clicks were dead - rows built with
`el("li", { onClick })` and `el()`'s `on*` branch registers
`addEventListener("Click")` (case-sensitive; never fires); fixed by ONE delegated
`click` listener on the stable `#object-list` container; (2) "Save and return"
was dead - `studio-state.js` `TABLE.browsing` had no `save_and_return`, so
`dispatch` threw and the `async` click handler swallowed it; fixed by adding the
transition (now reachable from every state) + a `can()`/`try` guard so
navigation is never trapped. Additive: `showFatal()` + `#studio-error` banner
(module-load faults visible, not silent), `refuseNote()` (refused transitions
show feedback), `#studio-diag` readout `{state, selected_object_id, has_pending,
last_refused_transition}`, box re-centred on returning to camera. New `browser`
pytest marker + `tests/browser/*` (Playwright, the one new `[dev]` dependency;
Chromium fake media device; skips cleanly when absent). Recognition
responsiveness: `scripts/benchmark_recognition_paths.py` ran the three bounded
BLOCK 8 experiments under one canonical latency protocol (Protocol A isolated /
Protocol B end-to-end; Phase 2 vs Phase 5 detector figures reconciled) - **no
config default changed**, pose gating "no change justified", input size flip left
to the developer. **No model trained; no model swap / quantisation / threading
redesign.**

Phase 5 (still stands): fixed (1) the recognition policy
whitelist that rejected valid baseline classes -> replaced by a three-tier class
vocabulary (`primary`/`secondary`/`implausible`), a total disjoint partition of
COCO-80 validated at load; `primary` + `secondary` both shown, only
`implausible` suppressed - the general model works out of the box with no
enrolment; (2) "Unknown (was Bed)" -> `suppressed_implausible` is now distinct
from `unknown` in logic, contract and display; the main overlay label is exactly
`Unknown`, full detail relocated to Diagnostics; (3) unfitted thresholds are
labelled unfitted in the UI (`policy.thresholds_fitted`) and `fit_thresholds.py`
refuses an unlabelled split. Also: an explicit Studio state machine
(`browsing|object_selected|capturing|reviewing`) fixing Back-to-camera /
re-select / Save-&-Return, and Studio capture reusing the monitoring camera path
(full-resolution original + separate thumbnail, requested vs achieved recorded).
**No model trained or fine-tuned; no model swap / quantisation / threading
redesign.** Recognition of watch / spectacles / charger / headphones / shaker is
unchanged (outside the model's vocabulary). Still **no** tracker, track IDs,
association, `Relation`, temporal state, risk model, alert policy, TTS,
instance-recognition inference, environment scan, or Scene Snapshot Studio - none
started, none placeheld. `StateSnapshot.tracks` is still always `[]`. Object
images are training data, never merged with `data/eval`. Never report a
confidence value as accuracy, a detection frequency as P/R, a stored image as a
trained model, a coverage/quality heuristic as a validated metric, or a tier
assignment as a measured result. Never fit a threshold on `test` or open `test`
more than once. Never claim a performance number not produced by a command run on
this machine, or a physical check not performed. **Do not start Phase 7 (Scene
Snapshot Studio).**

## Current phase status

Phase 6 complete. `pytest -q`: **340 passed, 2 skipped** (baseline end of Phase
5: 321 passed / 2 skipped; +19 = 6 non-browser + 13 browser, 0 regressions).
`pytest -q -m browser` = **13 passed** (Playwright + Chromium fake media device;
skips cleanly when either is absent). `pytest -q -m "not browser"` = 327 passed /
2 skipped. `pytest -q -m models` = **12 passed** (unchanged - no inference code
touched); `pytest -q -m dataset` = 1 skipped cleanly. See
`docs/phase-reports/phase6.md` (three-part format), `docs/architecture.md`
"Phase 6", `docs/decisions.md` "Phase 6".

Phase 6 summary: both reported Studio failures reproduced in a real browser
first, then root-caused. (1) Object row clicks dead - `el("li", { onClick })`
registers `addEventListener("Click")` (case-sensitive; never fires) on nodes
`loadObjects()` replaces every refresh -> replaced by ONE delegated `click`
listener on the stable `#object-list` container (`wireObjectList()`). (2) "Save
and return" dead - `studio-state.js` `TABLE.browsing` had no `save_and_return`,
so `dispatch` threw and the `async` click handler swallowed it -> added
`save_and_return: "browsing"` (reachable from every state) + a `can()`/`try`
guard so navigation is never trapped. Additive: `showFatal()` + `#studio-error`
banner (BLOCK 14), `refuseNote()` (visible feedback on any refused transition,
BLOCK 5.13), `#studio-diag` readout `{state, selected_object_id, has_pending,
last_refused_transition}` (BLOCK 13), `studio-state.js` `lastRefused` +
`refuse()` (cleared by the next good `dispatch`, cannot latch), `render()`
re-centres the box on return to the camera stage (BLOCK 5.11). New: `browser`
pytest marker, `tests/browser/{conftest,test_studio_flow,test_main_page}.py`
(uvicorn in a thread, seeded temp objects store, Chromium fake media device,
console-error assertions), `scripts/benchmark_recognition_paths.py`.
`test_studio_state_machine.py` mirror + regression/static tests per root cause;
`test_studio_capture_quality.py` extended (decoded dims == achieved != thumb).
`playwright==1.62.0` added to `[dev]` (the one new dependency; base package
only). **Measured (this machine):** browser suite 13/13; fake-camera capture
stores a 1920x1920 original (`capture_path: imagebitmap`) with a separate
<=240 px thumbnail. Recognition experiments (`results/recognition_paths.md`, one
`data/raw` clip, 300 frames): canonical latency protocol defined (Protocol A
isolated-interleaved / Protocol B end-to-end loopback); Phase 2's 46.8 ms is
`benchmark_providers.py` timing models in **separate passes** (not Protocol A) -
Protocol A on this machine is ~93-113 ms detector p50, load-dependent, and that
distinction reconciles Phase 2 vs Phase 5. Pose gating: **no change justified**
(person in 295/295 frames). Input size 480 vs 640: 480 halves detector p50 with
0 primary-tier detection loss on this clip, but the default is **unchanged** -
left to the developer to confirm on representative footage. `config/profiles/*`
untouched.

### Pre-Phase-6 status (historical)

Phase 5 complete. `pytest -q`: **321 passed, 2 skipped** (1 hardware
`--run-hardware`; 1 `dataset` - skips cleanly until the eval set is labelled).
`pytest -q -m models` = **12 passed** (unchanged - Phase 5 touched no
detector/pose inference code); `pytest -q -m dataset` = 1 skipped cleanly.
Baseline at end of Phase 4 was 292 passed / 2 skipped; **+29 tests, 0
regressions**. See `docs/phase-reports/phase5.md` (three-part format),
`docs/architecture.md` "Phase 5", `docs/decisions.md` "Phase 5".

Phase 5 summary: `predictivesense/perception/vocabulary.py` (three-tier partition
+ `validate_partition` + `VocabularyTiers.tier_for`, stdlib-only);
`config.policy` gains `vocabulary` (nested `primary`/`secondary`/`implausible`,
partition-validated), `thresholds_fitted`, `show_suppressed_in_diagnostics`;
`domain_classes` is now a `@computed_field` = the `primary` tier (back-compat).
`perception/policy.py` reworked: six `policy_state` values reconciling exactly
(`rejected_out_of_domain` -> `suppressed_implausible`, new `accepted_secondary`);
`Detection.tier` additive. `core/types.py`, `pipeline/recorded.py` carry the new
fields; `pipeline/loop.py` metric keys renamed via `PolicyCounts.as_metrics`.
UI: `features/overlay.js` (label exactly `Unknown`, secondary de-emphasis,
suppressed hidden + Diagnostics reveal, click-to-select), `features/policy.js`
(six kinds, `thresholdsFitted`, `ruleLabel`, `thresholdFor`),
`features/detection.js` (unfitted-threshold `warn-note`), `groups/diagnostics.js`
(recognition-detail table + selected-detection inspector + renamed policy
counters). `scripts/policy_audit.py` (new), `scripts/fit_thresholds.py` (refuses
unlabelled), `scripts/benchmark_latency.py` (end-to-end loopback).
`static/studio/studio-state.js` (new state machine); `studio.js` rewritten as a
pure render of it; `objects/samples.py` + `api/objects.py` store a
full-resolution original (JPEG uploads verbatim) + separate thumbnail with
`capture_path` / `requested_resolution` / `achieved_resolution` /
`encoded_quality` / `original_bytes` provenance. **Measured (this machine):**
policy audit over 400 `data/raw` frames - 481 raw detections -> 452 accepted /
23 accepted_secondary / 1 unknown / 5 suppressed_implausible (`surfboard`) / 0
rejected_size, reconciles; with the old whitelist `couch` (3) + `umbrella` (20)
would have been "Unknown (was ...)". Synthetic-noise control: 0 detections.
Policy cost 0.06 ms p50 / 0.08 ms p95 (< 1 ms). Clean per-frame CPU inference
(first in-process run): detector 68.9/81.5, pose 55.1/63.2, combined
124.7/139.6 ms p50/p95 - unchanged from Phase 2.5 (no perception code touched);
no optimisation was justified by the measurements and none was applied.

Phase 4 summary: `predictivesense/objects/` (`vocab` / `registry` / `samples` /
`quality` - stdlib + numpy only; `ObjectProfile` + `ObjectSample` CRUD with slug
ids, atomic writes, soft delete to `_deleted/`, one required box validated vs
image bounds, Laplacian-variance blur + numpy dHash + near-duplicate + coverage
guidance strings). `predictivesense/models/registry.py` reads/validates/resolves
`models/registry.json` (exactly one `active`, SHA-256s cross-checked vs
`models/manifest.json`; **no activation code, no training**); v1 = the existing
pre-exported YOLO11n detector+pose. `api/objects.py` + `api/studio.py`;
`AnalysisLoop.pause()/resume()/paused` (Studio open -> consumer runs no
iteration, producer no read); `api/ingest.py` closes `/ws/ingest` (4409) while
the Studio is active; `camera/_opencv.py` gained `decode_image_bgr` /
`encode_jpeg` / `thumbnail_jpeg` (keeps `cv2` out of `api/`). New config
sections `objects.*` / `studio.*` / `ui.panel.*`. UI: `static/ui/resizer.js`
(drag + keyboard, rAF-throttled, `clampPanelWidth` min 300 / max min(560, 40%
window) / viewport >= 45%, width persisted as `store.panelWidth`), `shell.js`
mounts it, `groups/dataset.js` links the Studio and separates it from Record
Sample, `static/studio/` standalone page, topbar Studio link.
`scripts/export_objects_coco.py` -> `data/objects/coco_train.json` +
`coco_val.json` (sample-disjoint split; **refuses and exits 2** if any object
image hash / path / source id overlaps `data/eval`). **Measured (this machine):**
quality compute 3.87 ms p50 / 4.05 ms p95 per sample; server-side save
round-trip 34.2 ms p50 / 38.8 ms p95 (worst-case 640x480 random frame);
RSS +7.1 MB over a 50-capture session; **0 perception invocations over a bounded
interval with the Studio open** (asserted). Record Sample: root cause of the
reported weakness is environmental (browser WebM has no duration element OpenCV
can read -> `duration_s: null`, handled gracefully) - documented, **not
changed**. `data/objects/` is empty until the developer collects it.

### Pre-Phase-4 status (historical)

Phase 2.5 complete. `pytest -q`: **250 passed, 2 skipped**.
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
