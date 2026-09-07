# Decision log

One line per decision. Newest at the bottom. A contract rename or reshape in a
later phase must be recorded here.

## Phase 0

- **2026-09-07** Contracts are frozen **Pydantic v2** models (not dataclasses): gives the `StateSnapshot` JSON round-trip, field validation, and mutation-raises for free, and Pydantic is already a dependency via `pydantic-settings`.
- **2026-09-07** `Frame` sets `arbitrary_types_allowed=True` for `image: np.ndarray`; `Frame` is never compared by `==` or hashed in code or tests (numpy would make that ambiguous).
- **2026-09-07** Configuration uses `pydantic-settings` + one YAML file per profile under `config/profiles/`; `extra="forbid"` on every section so an unknown field is an error, not a silent ignore.
- **2026-09-07** `mode` accepts exactly `realtime` / `recorded` via the `Mode` enum; any other value is a `ValidationError`. No driver is implemented in Phase 0.
- **2026-09-07** Logging uses the **standard library `logging`** module (no new dependency): a single stdout `StreamHandler`, configurable level, format `"%(asctime)s %(levelname)-7s %(name)s | %(message)s"`. No `print()` in `predictivesense/`.
- **2026-09-07** `SyntheticSource.capture_ts` is a real `time.monotonic()` reading, nudged forward by 1e-9 s when the clock does not advance between reads, to guarantee a *strictly* increasing sequence.
- **2026-09-07** Per-frame synthetic image = the seeded base image rolled by `frame_id % width` columns: deterministic, cheap, visibly changing.
- **2026-09-07** `MailboxStats` lives in `core/types.py` (a frozen contract), so `camera/mailbox.py` imports it from `core` rather than defining its own.
- **2026-09-07** Non-synthetic `SourceKind` values raise `NotImplementedError` naming phase **P1**, from `camera/source.create_frame_source`.
- **2026-09-07** WebSocket broadcast is **poll-based, last-value-wins**: the broadcaster stores only the newest snapshot; each client's server task wakes at `broadcast.rate_hz` and sends the current snapshot, skipping any it missed. A send that exceeds `broadcast.client_send_timeout_s` drops that client. No per-client queue.
- **2026-09-07** `telemetry.Rate` and `telemetry.Samples` are `deque(maxlen=...)` (4096 / 100 000). At Phase 0 rates a 60 s run stays well under the latency cap; if it were ever exceeded, percentiles would be over the most recent samples.
- **2026-09-07** Metrics CSV schema: `elapsed_s, wall_utc, snapshot_id, frame_id, stale, loop_rate_hz, producer_rate_hz, consumed, dropped, mailbox_depth, iter_latency_ms, rss_mb`.
- **2026-09-07** `scripts/run_noop.py` always configures logging at `INFO` regardless of the profile's level, so the measurement summary is always emitted.
- **2026-09-07** Session manifest is a Pydantic model written as pretty JSON; `git_commit`/`git_dirty` come from `git rev-parse HEAD` and `git status --porcelain`, falling back to `("unknown", False)` with a logged warning if git is unavailable.
- **2026-09-07** pytest runs with `--import-mode=importlib`; `tests/conftest.py` puts the repo root on `sys.path` so the un-packaged `scripts/` directory is importable from `test_noop_run.py`.
- **2026-09-07** Python **3.11.9** is used, matching the Phase 0 prompt. It was installed on request because the machine previously had only 3.13 and 3.14.
- **2026-09-07** Durations never use `time.time()`. Coarse-grained timing (frame pacing, `capture_ts`, rate windows, uptime, the run deadline) uses `time.monotonic()`. **Sub-millisecond loop-iteration latency uses `time.perf_counter()`**, which `time.get_clock_info` reports as `monotonic=True` with ~100 ns resolution. On this Windows build `time.monotonic()` is `GetTickCount64()` at 15.625 ms resolution, so it reports every empty iteration as 0 ms and cannot satisfy Block 10's "measure loop iteration latency, p50 and p95". This is a deliberate departure from the prompt's literal "monotonic() only" wording, forced by the conflict with Block 10; `perf_counter` is still a monotonic clock, never wall-clock time. See `docs/phase-reports/phase0.md` section 4.
- **2026-09-07** `datetime.now(timezone.utc)` is used only for wall-clock timestamps in the manifest and the CSV `wall_utc` column, never to measure an elapsed duration.

## Phase 1

- **2026-09-07** `SourceKind` gains `BROWSER = "browser"`; `DEVICE`, `FILE` and `BROWSER` become constructible. `WEBRTC` is **kept** (removing it would reshape a frozen enum) but is now permanently non-constructible - `ensure_source_constructible(WEBRTC)` raises `NotImplementedError` explaining that the browser Web Worker ingest path replaces it. Additive change to `core/enums.py`.
- **2026-09-07** Contract additions to `core/types.py` (all new frozen models, no rename/reshape): `SourceInfo` (kind, source_id, label, width, height, achieved_fps, backend, extra), `ClipManifest` (clip_id, session_id, path, scenario_tag, device_label, width, height, nominal_fps, duration_s, size_bytes, recorded_utc, git_commit, config_profile, consent_ack, notes), `IngestHeader` (client_ts_ms, seq, w, h).
- **2026-09-07** `StateSnapshot.metrics` stays `dict[str, float]`; Phase 1 adds **keys only** - `capture_fps`, `analysis_fps`, `dropped_analysis_frames`, `drop_rate`, `frame_age_ms`, `decode_ms`, `ingest_bytes_per_s`, `reconnects`, `clock_offset_rtt_ms`. `frame_age_ms` inside `metrics` uses `-1.0` as the "no frame" sentinel because the map is `float`-valued; the nullable truth stays on the top-level `StateSnapshot.frame_age_ms` field.
- **2026-09-07** **Browser Web Worker ingest instead of WebRTC / aiortc.** The browser analysis worker downscales + JPEG-encodes sampled frames and pushes them over a binary `WS /ws/ingest` message (`4-byte BE length | UTF-8 JSON IngestHeader | JPEG`). This is a push path (not polling, not preview). Rationale: no extra dependency, no signalling/ICE/SDP, trivially inspectable on the wire, and it keeps the preview `<video>` element completely untouched (Block 4's defining constraint). WebRTC would have added `aiortc` and a signalling surface for no Phase 1 benefit.
- **2026-09-07** **Clock offset** is `offset = server_mono - client_hello_ms/1000`, measured at the hello/hello_ack exchange; the echo round-trip time is recorded **separately** as `clock_offset_rtt_ms` and is *not* folded into the offset (no half-RTT correction). `Frame.capture_ts` for browser frames is `client_ts_ms/1000 + offset`, i.e. the client's epoch clock mapped onto the server's `time.monotonic()` timeline so `frame_age_ms = emitted_ts - capture_ts` is meaningful. The RTT is the reported error bar on every frame age (Requirement 11).
- **2026-09-07** `create_frame_source(SourceConfig)` still builds **only** `SYNTHETIC` and raises `NotImplementedError` for every other kind - a bare `SourceConfig` carries no device index, file path or ingest dimensions. `DEVICE` / `BROWSER` are built by `pipeline.loop.build_camera_source(AppConfig)`; `FILE` is driven by `pipeline.recorded.RecordedDriver`. The Phase 0 test `test_factory_builds_synthetic_and_blocks_other_kinds` is unchanged and still passes.
- **2026-09-07** `dev.yaml` keeps `source.kind: synthetic` so `run_noop` and every Phase 0 test keep Phase 0 behaviour. The live Phase 1 input layer is selected with `run_app.py --source-kind browser|device` (env `PS_*` overrides cannot reach nested keys once the YAML supplies the `source` block, as noted for Phase 0). `capture.owner` must agree with the kind (`browser`<->`browser`, `device`<->`backend`); `build_camera_source` raises `ConfigError` otherwise.
- **2026-09-07** `FileSource.capture_ts` prefers `CAP_PROP_POS_MSEC` but falls back to `frame_index / fps` whenever the container reports `0` or a non-increasing timestamp. For the MJPG/AVI test fixture OpenCV returns `POS_MSEC` offset by one frame, so every frame takes the `idx/fps` fallback - still file-derived, still deterministic, and strictly increasing. `capture_ts` is rounded to 6 dp so the recorded JSONL is byte-identical across runs.
- **2026-09-07** `RecordedDriver` writes `results/recorded_<run_id>.jsonl` (one compact JSON object per frame, fixed key order) and a sibling `.manifest.json`. Only the **JSONL** is guaranteed byte-identical across two runs of the same file+config; the manifest records wall time and git state and is not part of that guarantee.
- **2026-09-07** `test_no_forbidden_imports.py` amended: `cv2` removed from the global ban and allowed **only** under `predictivesense/camera/`; added a self-check that the scanner catches a fabricated `cv2` import elsewhere, and a check that `cv2` really is used under `camera/`. The clip-duration probe was moved from `api/recorder.py` into `camera/file_source.video_duration_s` to keep `cv2` out of the API layer.
- **2026-09-07** The Phase 0 test `test_index_page_served` string assertions (`"state snapshot"`, `"/ws/state"`) were updated to `"PredictiveSense"` / `"/static/app.js"` because Block 6 mandates rewriting `index.html`. New `test_static_assets_served` and `test_cameras_endpoint_shape` were added to `tests/integration/test_api.py`.
- **2026-09-07** Debug consumer stall: `AnalysisLoop.request_consumer_stall(seconds)` (clamped to 10 s) and `POST /api/debug/stall` are the documented way to run Block 13 check 3 - the producer, ingest socket and browser preview keep running while the consumer pauses, mailbox drops climb and depth stays <= 1. Always available (local-only research tool; a brief consumer pause has no security surface).
- **2026-09-07** `create_app(..., write_manifest=True)` (set by `run_app.py`, not by tests) writes `results/session_<id>.json` on shutdown carrying `extra.ingest.clock_offset_rtt_ms` and the browser-source counters, satisfying "store the RTT in the session manifest" for the live app path.
- **2026-09-07** New dependencies: `opencv-python==4.10.0.84` (device/file capture, JPEG decode, MJPG fixture), `python-multipart==0.0.20` (multipart form parsing for `POST /api/record/upload`), optional `pygrabber==0.2` (+`comtypes`) for DirectShow camera names, installed via the `camera` extra. `opencv-python` resolves cleanly against the pinned `numpy==2.2.1`.

### Phase 1 camera optimization pass (2026-09-07)

- **Measured, not assumed: the existing capture defaults are near-optimal for the integrated camera, so they are unchanged.** `benchmark_transport.py` on the Integrated Camera (index 0), 12 s per config: 640×480, 1280×720 and **1920×1080 all deliver a stable ~29.4 fps with 0 read failures and 0 reconnects**; frame interval p50 ~32 ms, p95 ~49 ms. Resolution costs only CPU/RAM (480p 10.5 % CPU, 720p 34 %, 1080p 48 %), not FPS or latency. Default stays **1280×720 @ 30**.
- **`capture.fourcc` added, default `auto` - MJPG gives no benefit here.** MSMF ignores `CAP_PROP_FOURCC` entirely (reads back an internal id, `raw:22`); forcing `MJPG` produced byte-identical timing to `auto` and would only add JPEG-decode CPU on a camera whose backend already delivers an efficient raw stream. A 4-char code is accepted for a specific USB webcam that needs it.
- **`capture.buffer_size` added, default `1` (newest-frame-only).** No measurable effect on MSMF (which does not honour `CAP_PROP_BUFFERSIZE` on this build) but harmless, and it is honoured by some DSHOW / virtual-camera paths where it lowers latency. Set defensively.
- **`capture.warmup_frames` added, default `0`.** The integrated camera's first frame is clean at 15-31 ms; the knob exists for flaky virtual cameras that emit dark frames on open.
- **MSMF stays the auto-first backend.** Measured: DSHOW is actually *more consistent* on the integrated camera (max interval 50 ms vs MSMF's occasional 80 ms hitch) and uses far less process RSS (+25 MB vs +94 MB one-time), but `read()` always blocks the full ~29 ms frame time and DSHOW is less universally compatible with UVC / virtual cameras. MSMF-first is the safer default; `device_backend: dshow` is a documented, measured alternative for latency-sensitive backend-owned use.
- **`DeviceSource` reconnect reworked to be non-blocking and interruptible.** The previous version looped `time.sleep(backoff)` + a 5 s open-probe **inside `read()` while holding the lock**, so `stop()` (device switching, shutdown) blocked until a reconnect attempt finished and the producer thread could not make progress. Now: `stop()` sets a `threading.Event` that interrupts every wait and the open-probe loop; `read()` does at most one short (`0.4 s`) reopen probe per call and absorbs the exponential backoff with an interruptible `Event.wait(<=0.25 s)` **outside** the lock. `_reconnects` (successful recoveries), `_reconnect_attempts`, `reconnect_seconds_total` and `currently_down_s` are exposed in `info()`. No thread was added to `camera/` (the invariant that threads live only in `pipeline/` holds - the reopen runs on the caller's producer thread).
- **OpenCV VIDEOIO log spam suppressed** via `camera/_opencv.quiet_opencv_logging()` (`cv2.setLogLevel(ERROR)`, idempotent), called from `enumerate_devices`, `DeviceSource.start` and `FileSource.start`. The `[ WARN ] ... can't be used to capture by index` line no longer floods `GET /api/cameras` (which probes indices 0-9).
- **`benchmark_transport.py`** now also reports per-`read()` blocking time (p50/p95/min), a drop estimate vs the achieved FPS, `reconnect_attempts`, and process CPU% / RSS delta; it accepts `--fourcc` and `--buffer-size`.
- **Browser `openStream`** now sends `frameRate: { ideal: 30 }` and, when `getUserMedia({deviceId:{exact}})` fails (device vanished/busy), retries once with the default camera and refreshes the device list instead of dead-ending the preview.
- **No new dependency** was added in this pass (`psutil` was already a runtime dep).

## Phase 1.5 - camera/input optimization & verification (2026-09-07)

Every keep/reject below carries its measurement. Full tables: `results/camera_matrix_integrated.md`, `results/analysis_sweep_loopback.md` (git-ignored per `results/*`; the tables are also reproduced in `docs/phase-reports/phase1.md` §1.5).

- **`capture.max_ws_buffered_bytes` added (default `1_000_000`).** Was a hard-coded `1_000_000` literal in `analysis-worker.js`; now a typed config field wired through `init` to the worker. Same value, now measurable/tunable. The worker checks `ws.bufferedAmount` before *and* after the encode and skips (never queues) the frame when it is over the ceiling, incrementing `skipBackpressure`.
- **Worker newest-wins made explicit.** The worker now keeps at most one encode in flight (`encodeBusy`); a frame that arrives mid-encode is dropped (`skipBusy`), never queued. `drawImage` is synchronous so the `VideoFrame`/`ImageBitmap` is closed immediately after; `framesIn`/`framesClosed` are reported every 1 s and a non-zero `leaked` is surfaced. Contract pinned by `tests/unit/test_worker_backpressure_contract.py` (pure-Python port of `dropReason` + a no-backlog simulation + a shipped-JS guard scan).
- **Camera matrix (backend-owned), integrated camera, index 0, 10 s/combo, 12 combos (640x480 / 1280x720 / 1920x1080  x  30 fps  x  msmf/dshow  x  auto/MJPG):** every combo opened, **0 read failures, 0 reconnects, ~29 fps measured** at all three resolutions. Requested == achieved resolution and reported FPS in every row. FPS/latency do not move with resolution; only CPU (10.5 % -> 25 % -> ~45 %) and one-time RSS scale.
- **FOURCC = MJPG rejected again, now on both backends.** MSMF reads `CAP_PROP_FOURCC` back as `raw:22` (ignored); **DSHOW reads it back as `YUY2`** even when MJPG is requested - the integrated camera's DSHOW path ignores the request too. Timing byte-identical to `auto`. `capture.fourcc` stays `auto`; a 4-char code remains available for a UVC webcam that needs it.
- **`CAP_PROP_BUFFERSIZE = 1` kept (default).** Already set in `_open_locked`. Matrix max frame interval: DSHOW 49-56 ms vs MSMF's 78-93 ms occasional hitch - consistent with buffer_size=1 helping the DSHOW path and being a documented no-op on MSMF on this build. Harmless, set defensively.
- **Per-device winning-backend cache added and wired.** `benchmark_camera_matrix.py` writes `results/camera_backends.json` (`{index: "msmf"|"dshow"}`, the clean-read backend with the lowest median frame-interval p95, tie -> msmf). `camera/enumerate.py` gains `load_backend_hint` / `save_backend_hint`; `DeviceSource(backend="auto", backend_cache_dir=...)` opens the hinted backend first and still falls back to the other. `pipeline.build_camera_source` passes `config.results_dir`. **For index 0 the winner is `dshow`** (p95 ~48 ms vs MSMF ~49-62 ms, lower CPU, same 29 fps, 0 drops). The cache is machine-specific (device indices, per-camera backend preference) and is **not committed** - a fresh clone falls back to the historical msmf-first probe order, which is also the safest default for an unknown virtual camera.
- **Analysis-path sweep (in-process loopback, 27 combos: fps {5,10,15} x {480x360,640x480,800x600} x quality {0.5,0.7,0.85}, 6 s each).** `drop_rate` is ~0 for every combo (one 0.003 blip at 15 fps/640x480). Frame age in loopback is dominated by the 20 Hz consumer sampling phase (~20-55 ms regardless of parameters) and is not a discriminating signal without a real link. `decode_ms`, process CPU and ingest bytes/s scale cleanly and predictably with resolution / fps / quality. **Defaults kept at 10 fps / 640x480 / q0.70:** raising any axis did **not** reduce frame age or drops (both already at the floor) and each raises CPU and/or bandwidth - exactly the "not the highest numbers" rule. 15 fps roughly doubles CPU for no benefit; q0.85 roughly doubles bytes/s; 480x360 saves little and loses detail later phases will want.
- **Worker encode ms is browser-only.** `benchmark_analysis_path.py` reports a `cv2.imencode` **proxy** and labels it as such; the browser's `OffscreenCanvas.convertToBlob` time is not measurable from Python.
- **`POST /api/metrics/browser` added.** Pydantic `BrowserMetricsIn {label, sample}`; label slugified (`[^A-Za-z0-9_-]` -> `-`, dots included, so no path traversal), appended as a JSON array to `results/browser_metrics_<label>.json` with a `received_utc`. Feeds the new dashboard "Browser measurement" panel (requested vs achieved `track.getSettings()`, preview/analysis fps, worker encode ms + frame counters, `bufferedAmount`, ws rtt, backend decode ms, frame-age p50/p95, drop rate, switch times).
- **Device switching lifecycle hardened.** `openStream` now calls `stopCurrentStream()` first: terminate the worker, `track.stop()` every track, **`video.srcObject = null`**, *then* `getUserMedia` for the next device. Switch time (change event -> first `requestVideoFrameCallback` on the new stream) is measured per direction and shown as `switch ms`. `track.applyConstraints()` is used for a **preview-size change on the same device** (new `#preview-size` selector); a device change still does a full re-acquire. `navigator.mediaDevices.ondevicechange` re-enumerates and, if the in-use device vanished, falls back to the default without a reload.
- **Requirement 8/9 guard.** `assertPreviewUncomposited()` warns (console + note) if the `<video>` element has a non-`none` `filter`/`transform`/`box-shadow`/`animation`. The only read of `<video>` remains `createImageBitmap(video)` in the `MediaStreamTrackProcessor`-absent analysis fallback, now commented as such.
- **No new dependency.** `psutil` (already present) is used by both benchmark scripts. No WebRTC, no WebCodecs `VideoEncoder`, no new transport.
- **`config/benchmarked_camera_combos.json` added and committed.** Records the swept axes; `tests/unit/test_capture_config.py::test_shipped_defaults_are_benchmarked` asserts every shipped dev/eval capture default is a point on that grid so a future edit cannot silently drift from the evidence.

## Phase 1.6 - interface architecture (2026-09-07)

Presentation-layer only. No camera, transport, mailbox, telemetry, API route,
payload shape, config schema or contract changed. `pytest -q`: **127 -> 145
passed, 1 skipped** (+18 UI static-analysis tests).

- **The flat `<section class="panel">` stack is replaced with a CSS-grid
  application shell** (top bar / viewport / collapsible right panel) built from a
  small component layer under `static/ui/` and one group module per panel section
  under `static/groups/`. No framework, no bundler, no build step - ES modules
  served by the existing `/static` mount. Rationale: establish the interface
  architecture once so the six later phases register controls into an existing
  structure instead of redesigning the page each time.
- **The extension contract is one call: `registerGroup({id,title,order,modes,view,summary,render,update?})`**
  (+ `registerAnalysisModule(...)` for future Analysis sub-modules). `order`
  values live in `static/groups/constants.js`; `analysis` (50), `alerts` (60),
  `research` (70) are reserved there and **not** registered - no module file for
  them, not even empty. Documented with a worked example in `docs/architecture.md`.
- **Input mode is client-side view state, not a backend switch.** `store.mode`
  (`realtime` | `recorded`) is persisted in `localStorage` and only selects which
  groups show and what fills the viewport. The server `mode` config field is
  untouched; recorded analysis still runs through `POST /api/analyze` on a
  server-side path. The recorded-mode viewport plays a **locally chosen** file
  (`<input type="file">` + `URL.createObjectURL`) - no new endpoint, no
  `data/videos/` streaming route. Chosen over adding a video static-mount so the
  API surface stays frozen.
- **Logic moved, not rewritten.** `app.js` camera / worker / recording / video /
  metrics functions are relocated essentially verbatim into `static/features/*`;
  `app.js` is now a composition root. Behavioural deltas, both to decouple the
  modules and each listed in the phase report: (1) the analysis Worker starts /
  stops on `runtime` bus events (`stream` / `stream-stopped`) instead of a direct
  `startAnalysisWorker()` call from `openStream`; (2) the browser-metrics panel
  refreshes on a `sample-refresh` / `worker-metrics` event instead of a direct
  call; (3) `connectStateSocket` now also pushes each snapshot into the UI store
  and maps `stale` / socket state onto the top-bar status pill. The preview
  `<video>` is still only ever assigned `srcObject`; `assertPreviewUncomposited`
  is preserved and still called from the composition root.
- **User-facing renames (raw keys kept in Diagnostics):** `asfast` -> **Fastest**,
  replay `realtime` -> **Real-time speed** (the wire values sent to
  `/api/analyze` are unchanged - the map lives in `static/ui/format.js`);
  `worker_skips_t/b/p` -> **Worker skips** with the throttle/busy/backpressure
  breakdown shown in Diagnostics; "backend owns the camera" -> **"Live preview
  unavailable in backend-camera mode."**
- **`mimetypes.add_type("text/javascript", ".js"/".mjs")` at `api/app.py` import
  time.** Some Windows registries map `.js` to `text/plain`, which browsers
  refuse for `type="module"` and which would fail the new
  `test_static_assets.py` content-type check. Not an API change; it only fixes
  the MIME `StaticFiles` reports for the new `ui/` and `groups/` module tree.
- **`tests/integration/test_api.py` updated** (as Phase 1 updated it for the
  Block 6 rewrite): `test_static_assets_served` now asserts `app.js` contains
  `registerGroup` and that the Worker is spawned from
  `features/analysis-client.js`; `test_index_page_served` is unchanged and still
  passes. New tests: `test_ui_structure.py`, `test_ui_text.py`,
  `test_ui_mode_switch.py` (unit, static analysis of the shipped assets),
  `test_static_assets.py` (integration, every referenced module resolves + serves
  as JavaScript).
- **No new dependency, no CUDA, no cloud, no CDN/external font.** Everything
  ships from `static/`. `analysis-worker.js` is byte-unchanged.

## Phase 2 - perception (detection + pose) (2026-09-07)

Per-frame perception only: no tracking, identity, association, temporal state,
risk or voice. `core/types.py` shape is unchanged - `Detection` / `Pose` were
defined in Phase 0 for this. Full numbers in `docs/phase-reports/phase2.md`.

- **Models: Ultralytics YOLO11n + YOLO11n-pose, pre-exported ONNX, AGPL-3.0.**
  Taken from the `ultralytics/assets` `v8.3.0` GitHub release as ready ONNX, so
  **no `torch` / `ultralytics` install** is needed to run them. The AGPL-3.0
  licence is an **open, unresolved decision** recorded prominently in
  `docs/attribution.md` - not resolved in this phase. `models/*.onnx` are
  git-ignored; only `models/manifest.json` (names, URLs, sha256, sizes, licence,
  input sizes) is committed. `scripts/fetch_models.py` downloads + verifies the
  hashes and refuses to use a file on mismatch; it is a manual setup script, not
  part of the runtime package, so the outbound-network guard test is untouched
  (it scans `predictivesense/`, never `scripts/`).
- **onnxruntime pinned at 1.24.4 (CPU).** 1.20.x rejected the models (opset 22 >
  its max 21); 1.24.4 supports opset 22 and has a matching
  `onnxruntime-directml==1.24.4` for the isolated `.venv-dml`. Runtime dependency
  in `pyproject.toml`; imported **only** under `predictivesense/perception/`
  (amended `tests/unit/test_no_forbidden_imports.py`: `onnxruntime` scoped to
  `perception/`, `cv2` now allowed under `camera/` **and** `perception/`).
- **Model-agnostic wrappers.** `ObjectDetector` / `PoseEstimator` take model
  path, input size, thresholds and the NMS variant from `config.perception`; the
  class-name list is read from the ONNX `names` metadata when present, else the
  COCO-80 default in `perception/classes.py`. Swapping to a different ONNX model
  is a config change, not a code change. No per-model class-list *file* was wired
  (the metadata path covers the realistic cases); add one if a model without
  embedded names is ever used.
- **Per-class confidence thresholds.** `detector.default_conf` (0.35) with a
  `detector.class_thresholds` map for overrides - a single global threshold is
  not acceptable. Overrides are left empty until the class-coverage audit
  verdicts land (`results/class_coverage.md`).
- **Provider selection is explicit and verified.** `perception/runtime.create_
  session` requests `[chosen_ep, CPUExecutionProvider]` (CPU only for per-op
  fallback), then asserts the chosen EP is actually in `session.get_providers()`
  and raises otherwise - a requested provider is never reported as the one in
  use, and there is no silent fallback to a different provider. A missing model
  file or an input size a locked model rejects fails loudly.
- **Default provider = `cpu`, chosen from the measured table** (`results/
  providers_cpu.json` vs `results/providers_dml.json`). DirectML on the Arc iGPU
  **agrees with CPU numerically** (count-match 1.0, class-agreement 1.0, mean
  |box diff| 0.0 px, max 0.002 px) but is **~2.2x slower** for these nano models
  (combined detector+pose p50 216 ms vs 96 ms; ~9 fps vs ~21 fps per model, both at intra_op_threads=6) -
  the tiny kernels never saturate the iGPU while the 14-core CPU does. So the
  agreement check would *pass* but the speed check fails. `perception.provider`
  is `cpu` in both profiles with a comment naming the result files. OpenVINO /
  NPU were **not evaluated** (optional/deferred per Block 3.4.16).
- **Loop / API degrade; scripts fail loud.** `build_perception(config, strict=)`:
  `strict=False` (analysis loop, `POST /api/analyze`) logs one warning and runs
  **without** perception when weights are absent - never kills the loop or the
  preview (Block 7's harder constraint). `strict=True` (`fetch`/benchmark/audit
  scripts, `models`-marked tests) raises. With `perception.*_enabled` both off
  the pipeline is exactly Phase 1.6 (no session loaded, empty lists).
- **`StateSnapshot.metrics` gains keys only** (map stays `dict[str, float]`):
  `detector_ms`, `pose_ms`, `perception_ms`, `detector_ms_p50/p95`,
  `pose_ms_p50/p95` (small 600-sample window so the percentile sort is cheap on
  the hot path), `detections_per_frame`, `poses_per_frame`,
  `detector_warmup_ms`, `pose_warmup_ms`, `perception_frame_errors`. `-1.0` is
  the "not measured / not run" sentinel, matching the Phase 1 `frame_age_ms`
  convention. No key was renamed or removed.
- **`RecordedDriver.run(..., perception=)`** runs the identical engine; JSONL
  lines now carry real `detections` / `poses` at fixed precision (2 dp coords,
  4 dp scores) in fixed key order. Determinism now includes the model: CPU ORT
  is deterministic for identical input, so two runs over one clip + config +
  weights are byte-identical (`test_recorded_determinism_with_models.py`).
  Without a `perception=` argument the output is exactly the Phase 1 shape, so
  the existing `test_recorded_driver.py` is untouched.
- **`scripts/run_noop.py` forces perception off.** It measures the Phase 0
  synthetic loop and its 25 MB RSS budget; loading ~60 MB of ORT + weights for
  no signal on synthetic noise would break that budget. `benchmark_providers.py`
  is where perception cost is measured.
- **UI: the reserved `analysis` slot is filled, not renegotiated.**
  `groups/constants.js` keeps `GROUP_ORDER.analysis = 50`; only the
  `RESERVED_GROUP_IDS` array dropped `"analysis"` (still lists `alerts`,
  `research`). `groups/analysis.js` is a host that renders the Detection and Pose
  sub-modules registered via the existing `registerAnalysisModule(...)`. The
  Phase 1.6 shell / registry / `static/ui/*` are untouched. The Phase 1.6 guard
  `tests/unit/test_ui_structure.py` was updated to expect `analysis` registered
  and only `alerts`/`research` reserved - same precedent as Phase 1 updating the
  Phase 0 `index.html` string assertions.
- **Overlay draws on `#overlay-layer` only.** A `<canvas>` sized to the
  letterboxed `<video>` display rect; the `<video>` element is never read, drawn
  into, or replaced. Live overlay is browser-ingest only (backend-camera mode
  has no preview; recorded analysis is retrospective and does not stream
  `/ws/state`), so the coordinate reference is `capture.analysis_width/height`.
  Low-confidence-band detections render dashed + dimmed; low-visibility keypoints
  de-emphasised; a stale snapshot dims the overlay and labels it `STALE`. No
  track IDs (no tracker).
- **UI perception toggles are overlay-layer visibility**, initialised from
  `perception.detection_enabled` / `pose_enabled` and persisted per-viewer in
  `localStorage` (`features/analysis-prefs.js`). The server-side flags decide
  whether the backend runs each model at all. **No API route was added.**
- **Test fixtures `tests/fixtures/perception/*.jpg` are 3 frames of the
  developer's own `data/raw/` recording** (~55 KB each), committed so the
  `models`-marked pipeline test has real content (`person`, one `cup`). Pure
  synthetic images produce no detections and would make the test vacuous.
