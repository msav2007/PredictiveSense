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

## Phase 2.5 - recognition reliability & measurement (2026-09-08)

Adds the project's first labelled evaluation set (from its own footage), an
evaluation harness, and a recognition policy layer - all so the perception's
reliability *in this room* can be stated with numbers. No tracking, temporal
state, risk, or voice; no training or fine-tuning.

- **Annotation store is COCO detection JSON** (`images`/`annotations`/
  `categories`), in `predictivesense/dataset/coco_store.py`. Chosen so the same
  files feed standard fine-tuning tooling in a later phase with **no conversion**
  (COCO is what Ultralytics, Detectron2, MMDetection and the HF `object-detection`
  pipeline all read). Per-image extras beyond stock COCO: `seeded` (bool - was the
  frame detector-seeded during labelling), `labelled` (bool), and `ps_provenance`
  (source clip, capture timestamp, session id, camera device, condition tags).
  `CocoStore.save` writes a `.bak` before any overwrite (BLOCK 10). Categories are
  `policy.domain_classes` in order, ids from 1.
- **`Detection` gained four additive fields** (Phase 0 additive rule; no rename,
  no reshape): `raw_class_name: str` (what the model said), `policy_state: str`
  (`accepted` | `unknown_low_confidence` | `unknown_margin` |
  `rejected_out_of_domain` | `rejected_size`), `runner_up: tuple[str, float] |
  None` (second-best class for the anchor). `class_name` is unchanged in shape
  and after the policy holds the decided label (possibly `"unknown"`). Every
  pre-2.5 construction site is valid via defaults; the recorded-JSONL shape is
  unchanged for a `perception=`-only run and only grows these keys when a
  `policy=` is passed.
- **"Do not modify the detector's inference behaviour" is read as: the set of
  emitted detections and their `bbox` / `class_id` / `class_name` / `score`, and
  the NMS + thresholding that produce them, are unchanged.** `perception/
  detector.py` now *also* computes `raw_class_name` (mirrors `class_name`) and
  `runner_up` (the anchor's second-highest class score, already in the head
  output) as diagnostic metadata - additive, non-behavioural. Evidence it is
  non-behavioural: every Phase 2 `models`-marked test (detector validity,
  recorded determinism, person-detected) stays green byte-for-byte. Without this
  the Block 3.13 top-2 margin rule would be permanently inert in production.
- **YOLOX decode variant added, YOLO11 path untouched.** `DetectorConfig.decode`
  (`"yolo"` default | `"yolox"`). The `yolox` branch uses a raw-BGR, no-`/255`,
  top-left letterbox (`preprocess.letterbox` gained `to_rgb` / `scale` / `center`
  params, all defaulting to the YOLO convention) and `postprocess.yolox_decode`
  (grid decode for strides 8/16/32). This is the "adding a decode variant if
  needed" that BLOCK 3.18 anticipates.
- **Splits are session-disjoint, never frame-disjoint** (`dataset/splits.py`).
  Whole recording sessions go to `val` or `test`; a content hash over
  images+annotations+assignment lets `load_splits` refuse a stale split (BLOCK
  10). A leakage assertion (`assert_no_leakage`) is run on every build and load,
  and a test builds a deliberately leaky split and confirms it is rejected.
- **Thresholds and `margin_min` are fitted on `val` only** by
  `scripts/fit_thresholds.py` (best-F1 over a 0.05 grid per class, ties toward
  precision; `margin_min` = 10th-percentile of `score - runner_up` over correct
  detections). `test` is opened once, at the end, via
  `scripts/eval_detection.py --split test`, which appends the opening (date +
  `--reason`) to `results/test_set_openings.md` and refuses a second opening
  without `--allow-reopen`.
- **`policy.default_threshold` (0.35) is an explicit fallback, not a fitted
  value** - a judgement call, flagged here. It applies only to a class with no
  `val` support to fit from. `min_box_area_frac` (0.0005) and `margin_min` (0.10)
  ship at their prompt-suggested defaults until the developer's `val` labels
  allow `fit_thresholds.py` to replace them; the profile comment names
  `results/fit_thresholds_val.md`. `aspect_ratio_bounds` is empty by default
  (per-class, opt-in) - also a judgement call.
- **`domain_classes` is validated against COCO-80 at config load** (BLOCK 10 - a
  `domain_classes` entry the shipped model cannot emit fails loudly). This
  couples `config/settings.py` to `perception/classes.COCO_CLASSES`, which is
  stdlib-only, so no import cycle.
- **The policy runs in `pipeline/loop.py` and `pipeline/recorded.py`**, after
  `PerceptionEngine.infer`, before the snapshot / JSONL line. Built from
  `config.policy` by `build_loop`; `app.state.policy` is shared with
  `POST /api/analyze` so recorded runs apply the identical code. `policy.apply`
  never raises into the loop (it catches its own errors, counts the frame as a
  failure, passes detections through). New `StateSnapshot.metrics` keys
  (`policy_accepted`, `policy_unknown_*`, `policy_rejected_*`, `policy_errors`,
  `policy_ms`, `policy_ms_p95`) - keys only, map stays `dict[str, float]`.
- **Pose gating: `perception.pose_requires_person` (default false).** When true,
  pose runs on a sampled frame only if the detector found a `person` (still also
  subject to `pose_every_n`). Handled in `engine.infer` (where both models run);
  measured before/after in the phase report. Default off - it is only turned on
  if the measurement shows it helps on this machine without losing skeletons.
- **ORT thread options set explicitly** (`perception/runtime.py`):
  `inter_op_num_threads = 1`, `execution_mode = ORT_SEQUENTIAL`, `intra_op`
  swept. Confirms Phase 2's `intra_op_threads: 6` knee.
- **Detector input size stays 640.** BLOCK 3.5.21 says "adopt the measured-best
  input size from the Phase 2 sweep"; that sweep's *conclusion*
  (`results/input_size_sweep.md`) was 640 (small-object sensitivity), with 480
  the documented fallback. No labelled evidence yet supports changing it, so 640
  is retained and 480 remains the fallback.
- **Alternative model: Megvii YOLOX-tiny (Apache-2.0), evaluated on the identical
  set.** Fetched + SHA-256-verified by `scripts/fetch_models.py` (added to
  `models/manifest.json`; `models/*.onnx` still git-ignored). The AGPL-3.0
  licence decision and its evidence are in `docs/attribution.md` /
  `results/model_comparison.md`.
- **Labelling tool is a standalone page at `/label`** (`static/label/`), not part
  of the Phase 1.6 shell - it is a research tool. It talks to
  `predictivesense/api/labels.py` (`GET/POST /api/labels/...`). Optional
  detector-assisted seeding runs the loop's detector via a new
  `camera/_opencv.read_image_bgr` helper (keeps `cv2` out of the API layer) and
  is always recorded per image as `seeded`.
- **`research` group registered** (`static/groups/research.js`, order 70) - the
  slot reserved in Phase 1.6, filled exactly as Phase 2 filled `analysis`:
  `constants.js` moved `research` out of `RESERVED_GROUP_IDS` (only `alerts`
  remains). `test_ui_structure.py` updated to expect it (same precedent).
- **Policy UI controls are per-viewer overlay presentation** (`features/
  policy.js`), like `analysis-prefs.js` - they re-derive from the additive
  `Detection` fields already on the snapshot, so **no API route** was added. The
  authoritative policy switches are `config.policy`, shown read-only under
  Analysis -> Detection. `unknown` renders dashed, muted, labelled `Unknown` and
  keeps its box (BLOCK 3.14).
- **`data/eval/` is git-ignored** (under `data/**`), like `data/raw/` - frames
  and `annotations.json` never enter git. `results/test_set_openings.md` and
  `results/model_comparison.md` are the two `results/` files that *are* tracked
  (gitignore exceptions) because they are audit records, not derived metrics.
- **No new runtime dependency.** `predictivesense/dataset/` and
  `predictivesense/eval/` are stdlib + numpy only; the import guard
  (`test_no_forbidden_imports.py`) is unchanged and still passes (`cv2` /
  `onnxruntime` stay scoped to `camera/` + `perception/`; the eval *scripts*
  that need `cv2.imread` live in `scripts/`, outside the scanned package).

## Phase 4 - Object Learning Studio & operations panel (2026-09-08)

- **New packages `predictivesense/objects/` and `predictivesense/models/` are
  stdlib + numpy only.** No new runtime dependency. Laplacian variance and dHash
  are implemented directly in `objects/quality.py` (a 3x3 `valid` correlation and
  an area-average block-reduce). Image *codec* work (decode uploads, normalise to
  JPEG q92, thumbnail) is done via new helpers in `camera/_opencv.py`
  (`decode_image_bgr` / `encode_jpeg` / `thumbnail_jpeg`) so `cv2` stays out of
  `predictivesense/api/` - the same precedent Phase 2.5 set with
  `camera/_opencv.read_image_bgr`. `test_no_forbidden_imports.py` is unchanged
  and still passes.
- **The Studio stops monitoring by pausing the analysis loop.**
  `AnalysisLoop.pause()` / `resume()` set a dedicated `threading.Event`
  (distinct from the time-bounded debug stall): while paused the consumer runs
  **no** iteration (no perception, no snapshot) and the producer reads no
  frames. `POST /api/studio/enter` pauses it and records a prior-state token;
  `POST /api/studio/leave` resumes it (only if *this* Studio session paused it)
  and validates the token (409 on mismatch). Re-entering is idempotent (a page
  reload gets the same token). "No frames may reach `/ws/ingest`" is enforced
  additionally in `api/ingest.py`: while `app.state.studio["active"]` the socket
  is closed with code 4409 before `accept()`. The Studio page attaches no
  analysis worker. `test_studio_lifecycle.py` asserts zero perception calls and
  a refused ingest socket over a bounded interval, and that leaving restores the
  loop and the preview-independence stall invariant.
- **`ObjectProfile` / `ObjectSample` are their own contracts under
  `objects/`, not additions to `core/types.py`.** `core/types.py` is the frozen
  wire format for the perception/tracking pipeline; the object *collection*
  store is a separate concern (like `dataset/` for the eval set). Both schemas
  carry `kind: "class" | "instance"` now; **no instance-recognition inference
  exists** - the field is stored, nothing reads it.
- **Every object sample carries exactly one box, validated against the image
  bounds** (`objects/samples.py:_validate_box`, 1 px rounding tolerance). A box
  outside bounds, a non-positive size, or a missing box is a loud failure
  (400 / 422). Manifests (`objects.json`, per-object `manifest.json`) are written
  atomically (`os.replace` of a `.tmp` sibling). Deletion of a profile **or** a
  sample is always soft: the folder / image files move to `_deleted/` and the
  API says where.
- **Quality thresholds are data-collection heuristics, not scientific quality
  metrics, and there is deliberately no composite 0-100 score.**
  `objects.blur_var_min` (60.0, variance-of-Laplacian over the box crop),
  `objects.min_box_area_frac` (0.01), `objects.duplicate_hamming_max` (6, dHash
  Hamming), and every `objects.coverage_targets.*` value are guidance for *what
  to collect next* ("3 views, no far-distance samples, no occluded samples"). A
  flagged sample is marked (`quality.flags`), never auto-deleted - the developer
  decides. Documented as heuristics here per Block 4.4.16.
- **`confusable_with` is seeded from the developer's observed failure modes**
  (`objects/vocab.py:CONFUSABLE_SEEDS`): watch <-> clock/bracelet/hand/donut,
  headphones <-> person, mug/cup <-> phone/bowl/glass, bottle/shaker <->
  can/cylinder, keyboard <-> remote, can <-> phone. A new profile inherits the
  seed for its slug; the developer edits it per object. The Studio prompts for
  hard negatives when an object has a `confusable_with` list and zero
  `hard_negative` samples.
- **The object dataset (training data) and the Phase 2.5 evaluation set (test
  data) must never merge.** `data/objects/` is git-ignored (under `data/**`),
  strictly separate from `data/eval/`. `scripts/export_objects_coco.py` runs
  `check_dataset_separation()` before every export and **exits non-zero writing
  nothing** if any object image **content hash**, image **path**, or **source
  identifier** (an uploaded `original_filename` vs an eval image's
  `ps_provenance.source_clip`) appears in both. `tests/unit/test_dataset_separation.py`
  covers all three and the refusal. The exporter produces a **sample-disjoint**
  train/val split of its own (deterministic per `sample_id` hash, each object in
  both splits, >= 1 train sample per object); it never touches `data/eval/`.
- **`models/registry.json` is tracked** (a `.gitignore` exception alongside
  `models/manifest.json`); the `.onnx` weights stay ignored. `predictivesense/models/registry.py`
  reads / validates / resolves it: exactly one `active` version, every file
  SHA-256 a real 64-hex string and - when the filename is known to
  `models/manifest.json` - matching it. **There is no activation code**; a human
  edits `active`. v1 is the existing pre-exported YOLO11n detector + pose; its
  `metrics_ref` points at the pending Phase 2.5 `val` eval file.
  `GET /api/models/registry` degrades gracefully (`{"available": false, ...}`)
  rather than crashing when the file is missing/malformed.
- **Panel width is client-side UI state, persisted like the collapsed flag.**
  `store.js` gains `panelWidth` (px, `null` = config default) in `PERSISTED_KEYS`.
  `static/ui/resizer.js` owns the drag handle: a pure `clampPanelWidth(width,
  winWidth, cfg)` (min `ui.panel.min_width_px` 300; max the smaller of
  `max_width_px` 560 and `max_width_frac` 0.40 of the window; the viewport keeps
  >= 45%), a `requestAnimationFrame`-throttled pointer drag that batches its one
  read (`innerWidth`) and one write (`--panel-w`), keyboard support
  (`role="separator"`, arrows step 16 px, `Home` / double-click reset to the
  default), and width applied as an inline `--panel-w` on `.app-shell` (removed
  while collapsed so the CSS `46px` rule wins). No API route, no server state -
  `ui.panel.*` is served read-only in `/api/config`. `#preview` is untouched
  (still asserted by `test_ui_structure.py`).
- **Record Sample was investigated, not changed.** Root cause of the reported
  weakness: browser `MediaRecorder` WebM often has no duration element OpenCV's
  FFmpeg build can read, so `ClipManifest.duration_s` is stored `null` (handled
  gracefully everywhere). Environmental, larger than this phase - documented in
  `docs/phase-reports/phase4.md`, save location and manifest format unchanged.

## Phase 5 - recognition trust & Studio repair (2026-09-08)

### Policy rule audit (BLOCK 3.1) - intent vs actual, on unlabelled footage

`scripts/policy_audit.py --source data/raw --frames 400` (`results/policy_audit_raw.md`).
481 raw detections; **frequencies, not accuracy**.

- **domain restriction** (was: reject any class outside a 14-entry whitelist).
  Intended to reject classes that cannot be in an indoor cabin scene; was
  actually rejecting every emitted class outside the 14 - `couch` (3), `umbrella`
  (20) on this clip - and showing them as "Unknown (was Sofa)". Parameter
  support: none (a judgement call). **Change:** replaced by a three-tier
  partition; only the `implausible` tier is suppressed. Now only `surfboard` (5)
  is suppressed on this clip; `couch`/`umbrella` show de-emphasised.
- **per-class threshold** (`default_threshold` 0.35, `per_class_thresholds` {}).
  Rejected 0/481 at 0.35 on this clip. Parameter support: none - unfitted
  placeholder. **Change:** value unchanged; added `policy.thresholds_fitted`
  (false) so the UI stops presenting it as fitted, and `fit_thresholds.py`
  refuses an unlabelled split.
- **top-2 margin** (`margin_min` 0.10). Rejected 1/481 on this clip. Parameter
  support: none - prompt placeholder. **Change:** value unchanged, labelled
  unfitted.
- **size / aspect** (`min_box_area_frac` 0.0005, `aspect_ratio_bounds` {}).
  Rejected 0/481. A geometric sanity floor, not fitted. **Change:** none.

### Decisions

- **The `domain_classes` whitelist is replaced by a three-tier partition**
  (`policy.vocabulary.primary`/`secondary`/`implausible`), a total disjoint
  partition of COCO-80 validated at config load (typo / duplicate / overlap / gap
  -> `ValidationError`). `primary` and `secondary` are both shown (secondary
  de-emphasised and flagged); only `implausible` is suppressed. Rationale: the
  product must work out of the box on the general model - no enrolment of
  `person`/`laptop`/`keyboard` first. Canonical tier lists live in
  `perception/vocabulary.py` (stdlib-only); config defaults pull them in via
  `default_factory` (deferred import - avoids a settings<->perception cycle).
- **Tier assignment is an environment-specific judgement, not a measured result**,
  cheap to revise once labelled data exists. primary = 14 MVP-scenario classes;
  secondary = 26 indoor furniture/appliance/tableware/wearable/pet classes;
  implausible = 40 outdoor-vehicle/street/animal/sports/food classes.
- **`config.policy.domain_classes` kept as a `@computed_field` = the `primary`
  tier** (Phase 0 additive rule). Keeps `dataset/coco_store`, `_eval_common`,
  `fit_thresholds.py`, `eval_detection.py`, `api/labels.py` and `/api/config`
  working unchanged; the existing `data/eval` store still reconciles.
- **`Detection` gained one additive field `tier: str = "primary"`**
  (`primary|secondary|implausible|unlisted`). The policy always sets it, even on
  `accepted` and the disabled pass-through.
- **`policy_state` is a six-value set**; `rejected_out_of_domain` -> renamed
  `suppressed_implausible` (the model recognised a known class; calling that
  "Unknown" is a category error). Set: `accepted`, `accepted_secondary` (new),
  `unknown_low_confidence`, `unknown_margin`, `suppressed_implausible`,
  `rejected_size`. `PolicyCounts` reconciles
  `accepted + accepted_secondary + unknown + suppressed_implausible +
  rejected_size == input`. `StateSnapshot.metrics` policy keys renamed to match
  (`policy_suppressed_implausible`, new `policy_accepted_secondary`) - keys only.
- **Main overlay label for an unknown detection is exactly `Unknown`** - no
  `(was Clock)`, no raw class, no rule name. Decision / raw class / confidence /
  runner-up / rule that fired / effective threshold / tier are **relocated, not
  deleted**, to Diagnostics: a per-detection "Recognition detail" table for the
  latest frame plus a click-to-select inspector (the overlay wrapper hit-tests
  the last-drawn boxes; the canvas keeps `pointer-events:none`).
  `suppressed_implausible` boxes are hidden on the main overlay; a
  Diagnostics-only view toggle (`policy.js` localStorage state, no API route)
  reveals them muted with the raw class.
- **`policy.thresholds_fitted` (bool, default false)** set to `true` only by
  `scripts/fit_thresholds.py`. While false the Analysis -> Detection panel shows a
  `warn-note` and Diagnostics labels the thresholds `UNFITTED`. `fit_thresholds.py`
  checks the COCO store's labelled count first and refuses an unlabelled / empty
  split before touching the detector or writing any file.
- **`scripts/policy_audit.py`** is the Phase 5 evidence script (per-state and
  per-raw-class counts, per-state confidence distribution, top implausible
  suppressions, synthetic-noise control). `scripts/policy_effect.py` kept and
  updated to the new state names.
- **Studio: an explicit `browsing | object_selected | capturing | reviewing`
  state machine** (`static/studio/studio-state.js`; transitions `select`,
  `capture`, `review`, `back_to_camera`, `save_and_return`, `discard`; one owner,
  `hasPending()` guard). `studio.js` renders the DOM as a pure function of it -
  no reload, no forced re-fetch. Root causes of the three faults: (a) *Back to
  camera* left `editingSampleId` / the inspected box coords stale and never
  re-centred; (b) *re-selecting an object* never hid the dynamically created
  `#inspect-actions` bar or reset the stage; (c) *Save & Return* called `leave()`
  directly with no pending-capture check. Fixes: `back_to_camera`/`discard` reset
  the transient context but keep `objectId`; `select` fully resets;
  `save_and_return` is `hasPending()`-guarded and `saveAndReturn()` `window.confirm`s
  before discarding, with a `beforeunload` backstop.
- **Studio capture reuses the monitoring preview's acquisition path** -
  `getUserMedia({width:{ideal:1280},height:{ideal:720},frameRate:{ideal:30}})`,
  then `getCapabilities()` -> `applyConstraints()` up to 1920 -> `getSettings()`,
  requested vs achieved recorded. The frame is `createImageBitmap()` of the live
  `<video>` track at full `videoWidth/Height` (`capture_path` `imagebitmap` /
  `element` fallback), stored as the full-resolution original. A JPEG upload is
  stored **verbatim** (no re-compression - BLOCK 3.23); other formats -> JPEG q92
  once. The thumbnail is a separate smaller file; `SampleStore.add` refuses if
  its bytes equal the original's. New additive `ObjectSample` provenance:
  `capture_path`, `requested_resolution`, `achieved_resolution` (authoritative -
  from the decoded image), `encoded_quality`, `original_bytes`.
- **`benchmark_latency.py` gained an end-to-end capture -> snapshot loopback**:
  the real FastAPI app + browser-ingest + perception + policy, frames over
  `/ws/ingest` with a real capture clock, `frame_age_ms` from `/ws/state`, plus
  CPU% / RSS. Run in a clean subprocess so cumulative in-process ORT-session
  contention (which skews the legacy thread sweep) does not skew it.

## Phase 6

- **2026-09-09** Object row selection in the Studio is a **single delegated
  `click` listener on the stable `#object-list` container**, not a per-row
  handler. Root cause of the reported "clicking Watch does nothing": rows were
  built with `el("li", { onClick })` and `el()`'s generic `on*` branch registers
  `addEventListener(k.slice(2), v)` -> `addEventListener("Click", …)` - a
  case-sensitivity bug, the real `click` never fires - and `loadObjects()` also
  replaces the rows on every refresh. Delegation fixes both. `el()` itself is
  left unchanged: `studio.js` was the only caller passing `onClick`, and every
  other call site (`actionButton`, `label.js`) binds `click` directly.
- **2026-09-09** `studio-state.js` `TABLE.browsing` gains
  `save_and_return: "browsing"` - the transition is now reachable from **every**
  state. Root cause of the dead "Save and return" button: `browsing` had no
  entry, `dispatch("save_and_return")` threw, and the `async` click handler
  swallowed it as an unhandled rejection so `leave()` + navigation never ran.
  `saveAndReturn()` also guards with `sm.can()` + `try/catch` (belt-and-braces);
  the single `window.location.href` navigation is unconditional once any pending
  capture is confirmed/discarded. Python mirror in
  `test_studio_state_machine.py` updated to match.
- **2026-09-09** Additive Studio robustness (no API, no contract change):
  `showFatal()` + a `#studio-error` banner surface a module-load fault /
  unhandled rejection instead of a silent dead UI (BLOCK 14); `refuseNote()`
  writes a visible note on any refused/blocked transition (BLOCK 5.13); a
  `#studio-diag` readout shows `{state, selected_object_id, has_pending,
  last_refused_transition}` (BLOCK 13). `studio-state.js` gains `lastRefused`
  (exposed in `snapshot()`, set by `refuse()`, cleared by the next successful
  `dispatch` - it cannot latch). `render()` re-centres the box whenever the stage
  returns to `camera`, so a capture never inherits an inspected sample's box
  (BLOCK 5.11).
- **2026-09-09** **Playwright** added to `[dev]` (`playwright==1.62.0`) - the one
  new dependency this phase. Base package only; fixtures are hand-rolled (no
  `pytest-playwright`). New `browser` pytest marker; `tests/browser/*` skip
  cleanly when Playwright or its Chromium binary is absent. Camera-dependent
  steps use Chromium's fake media device.
- **2026-09-09** **One canonical latency protocol** (BLOCK 8.22).
  **Protocol A** - isolated, in-process, warm: one `PerceptionEngine`
  (`warmup=True`), a single ORT detector+pose session pair, frames decoded at
  native resolution, `intra_op_threads=6`, `provider=cpu`, pose every frame,
  first 5 frames discarded, timing read from `PerceptionResult.detector_ms` /
  `.pose_ms`. **Protocol B** - end-to-end app loopback: the real app +
  browser-ingest + perception + policy in a clean subprocess, `frame_age_ms`
  from `/ws/state`. Phase 2's `detector p50 46.8 ms` came from
  `benchmark_providers.py`, which runs the **detector and pose in separate
  passes** (no per-frame interleaving) - it is not Protocol A and under-reports
  the contended cost. Phase 5's `65-72 ms` is Protocol A. Both earlier reports
  now carry a one-line pointer to this definition;
  `scripts/benchmark_recognition_paths.py` re-reports the number.
- **2026-09-09** Pose gating (`perception.pose_requires_person`) and detector
  `input_size` **defaults unchanged**. Pose gating: no change justified - a
  person is present in essentially every frame of the one available clip, so
  gating never skips pose there. Detector input size: 480 vs 640 measured (see
  `results/recognition_paths.md` and the phase 6 report); a flip to 480 is
  plausible on latency with no primary-tier detection loss on this clip, but is
  left for the developer to confirm on representative footage before changing the
  default.

## Phase 7

- **2026-09-09** **Bulk-upload box proposals use the detector's RAW output,
  before the recognition policy** - deliberate. The policy exists to decide what
  to *show the user during monitoring*; for a proposal we only want a rectangle.
  A `watch` is routinely proposed as `donut` / `clock`, both in the policy's
  `implausible` tier, which the policy would suppress - suppressing exactly the
  box we want. `PerceptionEngine` does not apply the policy anyway (that happens
  in `pipeline/loop.py` / `recorded.py`), so `engine.detect(frame)` (new,
  detector-only, no pose) is the raw output. The predicted label is stored as
  `proposal_raw_class` / `proposal_score` only - a **hint**, shown as "detector
  hint: donut (0.42), not the label" - and is **never** the sample's class,
  which is always the selected object.
- **2026-09-09** **Proposal selection rule**: highest score above
  `objects.batch.proposal_min_score` (default **0.10** - deliberately low, we
  want a box not a confident class); ties broken by **larger box area, then box
  centre nearest the image centre**. Detector-empty or all-below-threshold ->
  the image is `manual_required` and seeded with `centred_default_box` (the same
  30%/40% centred box the camera path seeds), never silently skipped. One image
  failing (decode / inference) is caught: that item becomes `manual_required`
  with the error string recorded; the batch continues.
- **2026-09-09** **Staging lives at `data/objects/<id>/_staging/<batch_id>/`**
  with a `batch.json` manifest, entirely outside `manifest.json`. Consequences,
  all intentional: staged images are excluded from `SampleStore.counts()`,
  `coverage_summary(...)` and `scripts/export_objects_coco.py` by construction
  (the exporter also skips `_staging` paths explicitly now). Abandoning a batch
  leaves no committed sample; an explicit **Discard batch** removes it, and
  `POST /api/studio/enter` sweeps batches older than
  `objects.batch.staging_ttl_hours` (default 24 h).
- **2026-09-09** **One shared box editor.** The drag/resize/nudge editor was
  extracted verbatim from `studio.js` into `static/studio/box-editor.js`
  (`createBoxEditor({ stage, box, media, onChange })`). `studio.js` builds one
  instance (`state.boxImg` aliases `editor.box`, so the rest of that file is
  unchanged); the bulk-upload reviewer builds a second on its own DOM. No second
  editor was written. The Studio state machine (`studio-state.js`) is **not**
  touched - the bulk-upload panel is its own view owned by `batch.js`, shown /
  hidden independently and closed on object switch, so the transition table and
  its Python mirror (`test_studio_state_machine.py`) are unchanged.
- **2026-09-09** **`box_confirmed_by_human`** (`ObjectSample`, additive; `None`
  on the camera path) is a research-integrity field, not decoration: `false` for
  an untouched proposal even after a bulk Save all; `true` once the developer
  moved / resized / drew the box or pressed "Accept box". It lets a later
  training phase report what fraction of the training set was detector-proposed
  vs hand-drawn and test for proposal bias. Four more additive provenance fields
  on a bulk-committed sample: `source: "upload_batch"`, `batch_id`,
  `proposal_source` (`detector` | `manual` | `default_centred`),
  `proposal_raw_class`, `proposal_score`.
- **2026-09-09** **Progressive processing.** Upload returns immediately with a
  `batch_id`; proposals run on a **1-worker** `ThreadPoolExecutor` (one shared
  ORT session - never one per image; the Studio has paused the monitoring loop
  so there is no contention, and a one-shot annotation run is **not** resuming
  monitoring). The client polls `GET .../batches/{bid}` every 400 ms and rebuilds
  the grid only when an item's status actually changed. Config caps
  (`max_images` 60, `max_total_mb` 400, per-file `objects.max_image_mb`) fail
  with a 4xx and a clear message, never a silent truncation.
- **2026-09-09** **`persist_sample(...)` extracted** from `api/objects.py::
  add_sample` as the single decode -> quality -> normalise -> thumbnail ->
  `store.add` path, shared by the camera route and the bulk-upload save. The
  camera route's behaviour is unchanged (`test_objects_api.py` unchanged and
  green).
- **2026-09-09** No new runtime dependency. `objects/batches.py` and
  `objects/proposals.py` are stdlib + numpy only (like the rest of
  `objects/`); `cv2` stays confined to `camera/` + `perception/` (the batch API
  decodes/encodes via `camera/_opencv.py`, as the camera route already does).

## Phase 8 - perception responsiveness, hardware-portable runtime, tracking-ready output (2026-09-10)

Every latency decision below carries its measurement (`docs/phase-reports/phase8.md`
"Measured", `results/latency_stages_*.json`). Machine: Intel Core Ultra 5 125H,
18 logical cores, CPU provider, `onnxruntime` 1.24.4 (no CUDA / DirectML EP in
this build).

### Contracts (additive; Phase 0 rule - no rename, no reshape)

- **`core/types.py` `Frame.trace: FrameTrace | None = None`** - a mutable,
  never-serialised per-frame stamp carrier for end-to-end latency attribution.
  Default `None`; only the browser-ingest path populates it, so synthetic /
  device / file sources and every existing construction site are unchanged.
- **`IngestHeader.cap_ts_ms` / `enc_ms`** (optional, default `None`) - the
  worker's true capture instant (drawImage) and its real JPEG encode duration.
  Framing round-trips them only when present.
- **`StateSnapshot.pose_stale: bool = False`** - poses were reused from an
  earlier frame (`perception.pose_cadence`). `metrics` also gains `stage_*_ms`
  keys, `capture_client_ts_ms`, `dropped_stale`, `frames_decoded` /
  `frames_analysed` / `dropped_browser_buffer` / `dropped_mailbox` /
  `frames_in_flight`, `pose_reused` / `pose_age_ms` / `pose_src_frame_id`
  (keys only; the map stays `dict[str, float]`).
- **`PerceptionFrame`** (new frozen contract) - one frame's complete perception
  output as a single immutable record: `frame_id`, monotonic `seq`, `capture_ts`,
  frame `width`/`height`, `model_version`, active `provider`, deterministically
  ordered `detections` / `poses` (already carrying box-in-original-pixels,
  `raw_class_name`, `policy_state`, `tier`), and reused-pose provenance. A
  **pipeline-internal** contract: the loop exposes `perception_frames()`; the
  recorded driver writes `results/recorded_<run_id>.frames.jsonl` (a **sidecar** -
  the main recorded JSONL line shape and its byte-determinism guarantee are
  unchanged). Not added to the `/ws/state` wire. `StateSnapshot.tracks` stays
  `[]`; no `predictivesense/tracking/` package; no track ids.
- **`PerceptionResult`** (internal) gains `pose_reused` / `pose_frame_id` /
  `pose_capture_ts` / `pose_age_ms`.

### Instrumentation (before any optimization - section 5/6)

- **One frame's journey is stamped stage by stage on `FrameTrace` and folded
  into the existing `MetricRegistry`** (`telemetry/stages.py::record_stages` ->
  `stage_<name>_ms` `Samples`; an immutable `FrameStages` per frame kept in a
  bounded ring). **No parallel metrics system.** `scripts/benchmark_latency.py
  --label` writes `results/latency_stages_<label>.{json,md}` (per-stage p50/p95,
  % of end-to-end, machine fingerprint). `scripts/benchmark_capture_paint.py`
  (new; Playwright + `--use-fake-device-for-media-stream`) measures the two
  stages the server trace cannot see - real Web Worker encode and real
  capture->overlay-paint (a pure client-clock delta against
  `capture_client_ts_ms`). Real-camera absolute capture->paint and the OnePlus
  virtual-camera transport/jitter remain the developer's physical verification.
- **Attribution result:** the bottleneck is inference (detector ~42 % + pose
  ~28 % of end-to-end) then `mailbox_dwell` ~21 %; transport (worker encode + WS
  + decode) ~3 %. The second single-slot buffer (`BrowserSource._slot`) measures
  **0.0 ms** dwell - **kept**: it decouples the async WS-decode thread from the
  analysis producer thread for zero latency cost.

### Optimizations (kept / rejected on evidence - one change at a time)

- **Broadcast push-on-publish** (`api/broadcast.py`). Replaced the fixed
  `asyncio.sleep(1/rate_hz)` `/ws/state` poll with `Broadcaster.publish()` waking
  each client via `loop.call_soon_threadsafe`; client blocks on an
  `asyncio.Event` with a `2x rate_hz` send-rate ceiling (no busy wait) and a
  `period` fallback. `publish()` still O(1) and never touches a socket.
  **Measured `stage_ws_out_ms` p50 46.0 -> 0.0 ms, p95 93.5 -> 0.0 ms**
  (`results/latency_stages_{before,after}_pushpub.json`). **Kept.**
- **Completion-driven scheduling** (`consumer.scheduler`, `pipeline/scheduler.py`).
  Shipped and unit-tested (blocks on `LatestFrameMailbox.get(block=True)` - a
  `Condition.wait`, not a spin - with a `max_analysis_rate_hz` ceiling).
  **Default stays `timer`:** loopback measured `timer == completion`
  (cap->snapshot 251.8 vs 252.3 ms; `mailbox_dwell` 47/94 ms both). On this
  machine inference (~130-225 ms) exceeds the 50 ms tick, so the timer's
  catch-up clamp already picks up a frame the instant the previous iteration
  ends. Rejected as default *on evidence* (section 7).
- **Staleness guard** `analysis.max_frame_age_ms` (0 = off). **Set to 180 ms in
  `dev.yaml`**, derived from `frame_age_at_dequeue` p50 ~60 / p95 ~110 ms - never
  fires on steady jitter (`dropped_stale` = 0 in the loopback) but drops a frame
  once the consumer is ~2 inference cycles behind. `tests/integration/
  test_slow_detector_bounds_latency.py` proves it bounds the end-to-end age
  under a slow detector without unbounded growth. One reconciling frame-counter
  set is exposed and asserted exact at a drained point
  (`captured == analysed + dropped_overwrite + dropped_stale + in_flight`).
- **Pose cadence** `perception.pose_cadence` (`every_frame` | `every_n:<int>` |
  `interval_ms:<float>`) + `pose_max_reuse_ms`. **Set to `every_n:2` in
  `dev.yaml`.** A frame without a fresh pose reuses the last one as the **same
  immutable `Pose` objects** (single-owner consumer thread; no lock) with its own
  `frame_id`/`capture_ts`/`age_ms` and `pose_stale=true`; the overlay draws a
  stale skeleton dimmer/dashed; beyond `pose_max_reuse_ms` the pose is dropped,
  not shown wrong. Cadence uses frame capture time, not wall time, so recorded
  mode stays byte-deterministic. Measured `every_n:2`: cap->snapshot p50
  **225 -> 138 ms**, analysis FPS **6 -> 11.1**, drop rate **0.4 -> 0.016**, pose
  present on every snapshot. A pose worker thread was **not** added (Phase 2
  measured two ORT sessions contend badly). `eval.yaml` keeps `every_frame` -
  the evaluation harness must run pose on every frame.
- **Detector input size** `perception.detector.input_size` - **`dev.yaml` -> 480**
  (`eval.yaml` stays 640). `scripts/benchmark_recognition_paths.py` swept
  320 / 480 / 640 on the integrated-camera clip with primary-tier detection
  counts + score percentiles (`results/recognition_paths.md`): the primary-tier
  (safety-relevant) detection count is **identical (197) at every size** and the
  score distribution is flat (p50 0.90-0.91); only *non-primary* detections rise
  as size falls, and the policy already handles those. Detector p50:
  640 -> 73 ms, 480 -> 46 ms, 320 -> 25 ms. 480 halves detector cost with zero
  primary loss; 320's extra gain is small and ~10x's the non-primary noise. The
  pose model input is ONNX-locked to 640, so only the detector changes. The
  evaluation harness (`eval.yaml`) keeps 640 - most sensitive, not latency-bound.
  Developer confirms primary recall on the OnePlus 720p clip (phase 8 report
  Part 2).

### Hardware-portable runtime

- See the Phase 8 report's provider section: `perception.provider` accepts
  `auto | cpu | cuda | directml`, default `auto`; `auto` picks the first
  available EP in a documented order and always falls back to CPU, logging which
  and why; an explicitly requested unavailable provider is a **loud** failure.
  The **actually active** EP is reported (from the live session's
  `get_providers()`) in Diagnostics, the session manifest and every result file.
  `pyproject.toml` optional extras `[cpu]` / `[cuda]`; `docs/setup.md` covers a
  fresh clone on either machine. CUDA / DirectML could not be *executed* in this
  environment (`onnxruntime` build exposes CPU + Azure only); the `cuda`-marked
  tests skip cleanly and are ready for the developer's NVIDIA machine.

## Phase 9 - multi-object tracking & honest detection state semantics (2026-09-14)

Full trace, measurements and physical-verification status:
`docs/phase-reports/phase9.md`.

### Root-cause investigation before any rendering change

- **2026-09-14** Root cause of every "grey" symptom is traced by reading the
  actual code end to end (`perception/policy.py` -> `core/types.py` ->
  `pipeline/loop.py` -> `api/broadcast.py` -> `static/features/metrics.js` ->
  `static/ui/store.js` -> `static/features/overlay.js`) and published,
  file:line-cited, **before** touching any rendering code, per the Phase 9
  prompt's explicit gate. All four hypothesised causes confirmed, plus a fifth
  (the `low_confidence_band` and the policy's `per_class_threshold_rule` are
  two independently configured cutoffs that can drift apart) and a sixth
  (zero hysteresis anywhere - raw per-frame values reach the canvas on every
  WS message). See `docs/architecture.md`'s Phase 9 section for the summary
  and `docs/phase-reports/phase9.md` for the full citations.
- **2026-09-14** The frequency table (how often each grey-producing condition
  actually fires) is measured two ways, not invented: a **batch** pass
  (`scripts/grey_state_audit.py`) runs the real detector + policy over every
  decoded frame of the developer's own clip (`data/raw`) for `policy_state`/
  `tier`/band-membership frequency (content-driven, wants a large sample); a
  **live** pass runs the real FastAPI app's real-time loop (real `/ws/ingest`
  -> `/ws/state`, real Phase 8 timing) over the same clip for staleness
  frequency (a timing-driven property a batch pass cannot produce - there are
  no ingest gaps in a decoded-frame loop). Results:
  `results/grey_state_frequency.{json,md}`. On the one available clip,
  `accepted_secondary` dominates (38.4% of detections) and `unknown_*`/
  `suppressed`/`rejected` never fired at all - a single-clip result, re-derive
  once more footage exists.

### Tracker algorithm choice

- **2026-09-14** IoU + constant-velocity motion prediction with a two-stage
  (ByteTrack-style) greedy association pass, implemented in-repository rather
  than adding a tracking dependency (`predictivesense/tracking/`, stdlib +
  numpy only - no `scipy`/`lap`/`motpy`/etc). Cited (algorithm family, not
  copied implementation) in `docs/attribution.md`. Chosen over a Kalman-filter
  or appearance-based (Re-ID) tracker: the detector runs at ~10-14 Hz on this
  hardware (Phase 8), objects move at ordinary indoor speeds relative to that
  cadence, and an appearance embedding would need its own model and inference
  cost for a benefit not measured to be needed here - out of scope per the
  Phase 9 prompt's non-goals (no appearance-based re-identification).
- **2026-09-14** Greedy assignment, not an exact Hungarian solve
  (`scipy.optimize.linear_sum_assignment` or the `lap` package): avoids a new
  dependency, stays `O(n*m log(n*m))`, and is fully deterministic given a
  stable sort tie-break (`association.greedy_match`). For the track counts
  this project runs (a handful of objects per frame, `max_tracks` bounded at
  64), the two rarely disagree in practice and greedy is simpler to audit and
  unit-test.
- **2026-09-14** Class agreement is a signal, never a gate, in the association
  cost (`assoc_class_weight`, the smallest of the five weights) - the detector
  demonstrably flips class on the same physical object (`cup`/`phone`,
  `bottle`/`cylinder`); a tracker that required class equality to associate
  would shatter exactly the objects this project cares about. A pair is
  eligible only on geometric plausibility (`iou > 0` or centres within half a
  box diagonal); class agreement can never manufacture a match on its own.
- **2026-09-14** Stage 2 (low-score detections) is IoU-only, no class or
  motion signal, and **never spawns a new track** - only recovers an
  already-associated identity. A low-confidence single detection creating a
  brand-new id would bypass `n_init` entirely, exactly the noise-promotion
  `n_init` exists to prevent.
- **2026-09-14** A miss during `tentative` deletes the track immediately
  rather than letting it coast - a hit streak interrupted by a miss must not
  later resume and reach `n_init` as if uninterrupted (section 6.5 of the
  Phase 9 prompt: "an incorrectly reused id is worse than a new one").
  Confirmed tracks get the normal coasting budget; only tentative ones are
  held to this stricter standard, since they are not yet an established
  identity.
- **2026-09-14** A track whose predicted centre leaves the frame is flagged
  and given a much smaller miss budget (`border_exit_max_age_frames`, default
  1) than an ordinary mid-frame disappearance (`max_age_frames`), and is
  never added to the re-association "ghost" pool - it is expected to be gone,
  not occluded. This is a judgement default (not derived from measurement,
  unlike `n_init`/`max_age`), documented here for that reason.

### Thresholds derived from measurement, not invented (section 7.2)

- **2026-09-14** `n_init` and `max_age` are derived, not chosen by feel:
  `scripts/track_threshold_derivation.py` runs a **live real-time** session
  (real `/ws/ingest` -> `/ws/state`, real analysis-loop timing - not a batch/
  recorded pass, which is lossless and analyses every frame at the wrong
  cadence for this purpose) over `data/raw`, and measures, at the actual
  non-stale analysis-cycle interval: the presence-run-length distribution
  (singleton-run share decides `n_init`) and the interior miss-run-length
  distribution (its p95 decides `max_age_frames`). Result file:
  `results/track_thresholds.md`. Measured this session: interval p50/p95 =
  47.0/159.8 ms; singleton-presence share = 0.323 (> 0.20 threshold) ->
  **`n_init = 3`**; miss-run p95 = 11 cycles, `max(2, p95+1)` ->
  **`max_age_frames = 12`**, **`max_age_ms = 564.0`** (`max_age_frames *
  interval_ms_p50`). One clip, one session - re-derive once more footage
  exists (`docs/phase-reports/phase9.md`, Not verified / limitations).
- **2026-09-14** `class_vote_history` (default 20) is not independently
  measured but is grounded in the same session's measured interval
  (~1000 ms / 47 ms ≈ 21, rounded to 20) - enough history to resolve a
  majority vote without unbounded growth, documented as a judgement default
  rather than claimed as a second measured threshold.
- **2026-09-14** `max_tracks` (64), `reassoc_window_frames` (12, same order as
  `max_age_frames`) and the five association weights are bounded-resource /
  judgement defaults, not measurements - each individually commented in
  `TrackingConfig` (`config/settings.py`) with its rationale, cheap to revise.

### Contract and config changes (additive only)

- **2026-09-14** `core/types.Track` gains 16 fields additively (`observed_class`,
  `track_class`, `class_votes`, `velocity`, `fresh`, `age_frames`, `age_ms`,
  `hits`, `consecutive_misses`, `first_seen_frame_id`, `last_detection_frame_id`,
  `last_detection_capture_ts`, `last_detector_confidence`,
  `last_detector_confidence_age_ms`, `policy_state`, `tier`) on top of the
  Phase 0 minimal shape (`track_id`, `status`, `class_name`, `bbox`,
  `last_seen_frame_id`) - no rename, no reshape.
- **2026-09-14** `core/types.FrameTrace` gains `tracker_ms` (additive,
  `__slots__` updated); `telemetry/stages.py` threads `tracker` through
  `SERVER_STAGE_NAMES`, `FrameStages` and `record_stages` between `policy` and
  `snapshot_build`, exactly like every other stage - not a parallel metrics
  path. `scripts/benchmark_latency.py`'s own stage-name tuples updated to
  match, plus a `--no-tracking` flag for isolated before/after measurement.
- **2026-09-14** New `config.settings.TrackingConfig` section
  (`config.tracking`), mirroring `PolicyConfig`'s placement and style;
  `enabled: bool = True` by default. `enabled: False` reproduces Phase 8
  exactly (`StateSnapshot.tracks` stays empty) - no code path removed, only
  gated.
- **2026-09-14** `TRACK_ELIGIBLE_STATES` (`predictivesense/tracking/tracker.py`)
  excludes `rejected_size` and `suppressed_implausible` from ever becoming a
  track - the policy's own judgement that these are not candidate objects, and
  tracking them would waste bounded track slots on noise the policy already
  flagged. Consequence: the Diagnostics-only "reveal suppressed" toggle draws
  those boxes separately, straight from `snap.detections`, with no identity or
  freshness concept (there is none for something never tracked) -
  `overlay.js`'s `drawSuppressedDetection`.

### Wiring

- **2026-09-14** One `Tracker` instance per analysis run, held by
  `AnalysisLoop` (real-time) or passed into `RecordedDriver.run()` (Mode B) -
  never forked, never rebuilt per frame. The tracker reads no clock itself;
  `capture_ts` is passed in from `Frame.capture_ts`, so recorded mode's
  file-PTS timestamps keep the whole run deterministic (asserted by
  `tests/integration/test_recorded_determinism_with_models.py`).
- **2026-09-14** The tracker only ever updates on a frame that actually
  arrived this cycle (never a stale/guard-dropped one - section 10.2). On a
  stale cycle the last real track list is reported **unchanged** rather than
  wiped to `[]` - a deliberate behaviour change from Phase 8 (where
  `stale=True` implied empty `detections`): staleness is a property of the
  frame, not of any one object, so the overlay now dims the last known tracks
  through a brief gap instead of making every box vanish. This directly
  addresses root-cause finding 4.
- **2026-09-14** `predictivesense/pipeline/recorded.py`'s reserved but
  always-`[]` `tracks` JSONL field is now populated
  (`_track_to_dict`, mirroring `_detection_to_dict`'s fixed-precision,
  fixed-key-order style) when a `tracker=` is passed to `RecordedDriver.run()`.
  `scripts/run_recorded.py` gains a `--no-tracking` flag for parity with
  `--no-policy`/`--no-perception`.

### State/visual mapping (rendering change - last, after everything above)

- **2026-09-14** `overlay.js` now draws `snap.tracks`, not `snap.detections`
  directly - every track-eligible detection this frame is already represented
  by some track (continuing or newly spawned this cycle), so nothing is lost,
  and identity becomes the organising concept instead of a raw per-frame box
  list.
- **2026-09-14** Five independent visual channels replace the previous
  overlapping ones (`docs/architecture.md` has the table): identity = hue
  keyed by `track_id` (not class - two simultaneously Unknown objects now read
  as visibly different identities); freshness = stroke solid/dashed (and
  *only* that - no longer doubles as a confidence-band or uncertainty signal);
  uncertainty = label text `"Unknown"` only, never a colour; tier = a small
  badge, never a colour change; staleness = whole-overlay dim, unchanged from
  Phase 8. The `low_confidence_band`'s former dashed/dimmed treatment of
  `accepted` detections is **retired** as a distinct signal (root-cause
  finding 2 - it was a second, uncoordinated "looks uncertain" channel); the
  actual confidence value stays visible via the label percentage and the
  confidence bar, so no information is lost, only the redundant/confusable
  channel.
- **2026-09-14** Confidence percentage now renders for `accepted` **and**
  `accepted_secondary` (previously secondary showed no percentage at all -
  root-cause finding 3, generalised beyond just the Unknown case it was
  designed for). A coasting track's label appends its confidence's age
  (`"72% (0.3s old)"`) so the viewer can always tell whether the number is
  this frame's or carried forward (section 8.3). `"Unknown"` deliberately
  keeps no percentage - a pre-existing, intentional design choice (BLOCK
  3.10), not something this phase changes.
- **2026-09-14** Flicker suppression is a small per-track hysteresis map in
  `overlay.js` (`stableKind`/`KIND_DWELL_MS = 150ms`, ~3 analysis cycles at
  the measured interval) - a track's displayed *kind* only changes once the
  new kind has persisted past the dwell window. Presentation-only: Diagnostics
  reads the raw, unsmoothed per-frame `policy_state` from `snap.detections`
  throughout, untouched. Most class-label flicker is already resolved
  upstream by the tracker's own majority-vote `track_class`; this JS-side
  hysteresis specifically covers `policy_state`-driven kind transitions
  (accepted/secondary/unknown), which are per-fresh-observation and not
  otherwise smoothed.
- **2026-09-14** Click-to-select now selects a **track** (`selectedTrackId`,
  matched by stable `track_id`, no bbox IoU re-matching needed across frames)
  instead of a detection bbox; a revealed suppressed detection (never tracked)
  is still selectable by the old bbox-based path for that one diagnostic case.
  `groups/diagnostics.js` gains a per-track detail panel (section 9.5) built
  alongside, not replacing, the existing per-detection "Recognition detail"
  panel.

## Phase 10 - perception regression repair + track & pose continuity

- **2026-09-15** `scripts/pipeline_attrition.py` built and its output
  (`results/pipeline_attrition_integrated_camera.{json,md}`) published
  **before any fix**, per required class, on the only clip in `data/raw`
  (laptop integrated camera - no OnePlus clip exists yet, so that comparison
  is deferred to the developer, section 10/15). Findings: `bottle`, `laptop`,
  `keyboard`, `mouse`, `chair`, `book`, `backpack` had **zero raw detections**
  across 501 frames on this clip (a footage-content / model-capability
  question, not a pipeline bug - nothing to attribute attrition to for these
  classes on this footage). `cell phone` showed real tracker-layer attrition
  (22 raw -> 14 rendered): a low-score detection (`score <
  tracking.high_score_split`) with no already-open, non-tentative track to
  extend is dropped before ever becoming a track - it can neither spawn one
  nor match a still-tentative one (`Tracker._match_and_apply_hits`). The
  entire `accepted_secondary` bucket (350/350 detections) on this clip is one
  systematic misclassification of something in the room as `umbrella` -
  flagged, not fixed (a model error, out of scope per section 3.1's `bed ->
  chair` precedent).
- **2026-09-15** Section 3.1's "the overlay may now render only confirmed
  tracks" hypothesis is **REFUTED** by reading the shipped `overlay.js`
  (`draw()`/`drawTrack()`, pre-Phase-10): every entry of `snap.tracks` is
  drawn regardless of `status` - there is no confirmed-only gate, and the
  attrition table's `tracker output` and `rendered` columns are identical by
  construction, including nonzero tentative-status entries. Render predicate
  (section 6.1) is therefore **kept as-is, deliberately**: a real detection
  that spawns a track (any status) is never invisible purely because the
  track is unconfirmed. The actual "invisible despite being detected"
  mechanism is the tracker-input attrition finding above, one layer earlier.
- **2026-09-15** Green-line (pose skeleton) restart root-caused by
  `scripts/pose_continuity_audit.py` (real live `/ws/ingest`->`/ws/state`
  session, `results/pose_continuity_raw.md`): **36.4%** of non-stale live
  snapshots carried **zero** pose (not dimmed - genuinely absent), across 159
  distinct gap runs (max 9 consecutive snapshots, ~580ms at the measured
  interval). Hypothesis 1 (`pose_cadence`/`pose_max_reuse_ms` mismatch) is
  **not** the primary cause (measured fresh-pose refresh interval p95 406ms,
  under the 500ms reuse budget). The actual mechanism: `PerceptionEngine.infer`
  (`predictivesense/perception/engine.py`) overwrites `_last_poses` to an
  **empty** tuple whenever a cadence-due pose inference returns `[]` (person
  score momentarily under `pose.conf`=0.4) - the reuse branch's own guard
  (`elif ... and self._last_poses ...`) then finds nothing to reuse on every
  subsequent frame until the next successful due-cycle. Pose has no identity
  of its own in the pre-Phase-10 pipeline (drawn straight off the per-frame
  `StateSnapshot.poses`), so this empty-result gap is directly visible as the
  reported ~1s restart.
- **2026-09-15** Fix (section 5): pose is now **bound to the person track**
  it best overlaps (`Tracker._bind_poses`, highest IoU between the pose's own
  box and a person-class track's box, at or above the new
  `tracking.pose_bind_min_iou` (0.1, judgement default - low enough to
  tolerate the pose and detector boxes disagreeing slightly, high enough to
  reject a pose that plainly belongs to someone else); a tie between the top
  two candidate tracks assigns nothing (section 5.3, never a guess). The
  binding is stored on the mutable `TrackState` (`pose_keypoints`,
  `pose_source_frame_id`, `pose_capture_ts`, `pose_is_fresh_binding`) so a
  cycle where the ENGINE produces no pose at all (the measured 36.4% case)
  leaves the existing binding untouched - it ages instead of vanishing,
  bounded by the new `tracking.pose_max_age_ms` (900ms - derived from
  `results/pose_continuity_raw.md`'s measured max empty-gap-run length,
  ~580ms, with margin; same "re-derive once more footage exists" caveat as
  `n_init`/`max_age`). `Track` gains five additive fields (Phase 0 rule):
  `pose_keypoints`, `pose_frame_id`, `pose_capture_ts`, `pose_age_ms` (a
  **continuous** ms value, mirroring `last_detector_confidence_age_ms` -
  deliberately not a second binary stale flag, per section 5.2's "visibly
  reduced emphasis as it ages" and section 5.4's "do not interpolate,
  persisting the last pose with an age is honest"), `pose_fresh` (true only
  on the exact cycle a brand-new, non-engine-reused pose was bound).
  `Tracker.update()` gained optional `poses=()`/`pose_fresh=True` kwargs
  (backward compatible - every pre-Phase-10 call site is unaffected).
  `overlay.js` now draws each track's own bound pose (`drawPose` takes a
  continuous `ageFrac` in [0,1] instead of a boolean `stale`, fading emphasis
  smoothly toward `pose_max_age_ms`) instead of looping `snap.poses`
  per-frame; the pose layer toggle stays independent of the detection-box
  layer toggle, matching pre-Phase-10 behaviour. `pipeline/recorded.py`'s
  `_track_to_dict` gained the four wire-relevant pose fields (additive, JSONL
  determinism preserved and re-asserted by
  `test_tracks_are_shaped_and_bounded`). A person with no open track this
  frame still has no pose shown (never a floating unbound skeleton) - the
  same "no assignment rather than a guess" principle as section 5.3, just at
  the edge case of zero person tracks.
- **2026-09-15** Latency floor re-measured after the pose-binding change, one
  session, both conditions on this machine to control for its current load
  (elevated vs the Phase 8/9 baseline articles - `process_cpu_percent` ~920-960%
  in both runs, i.e. this run was measured under heavier background load than
  Phase 8/9's baseline sessions, not caused by this change):
  tracking+pose-binding on, capture->snapshot p50 159.22ms; tracking off
  (control, same session), p50 165.43ms - tracking+pose-binding is not slower
  than the no-tracking control measured in the same noisy conditions.
  `tracker_update_ms` (which now includes `_bind_poses`) stayed at p50
  0.21ms / p95 0.4ms, statistically unchanged from Phase 9's 0.19ms/0.3ms -
  the pose-binding addition itself adds no measurable cost. The elevated
  absolute p50 vs Phase 9's report (112.63ms) reflects this machine's current
  background load, not a regression from this change; a floor-clean
  re-measurement is recommended before the phase report is treated as final.
- **2026-09-15** `scripts/input_size_recall_compare.py` re-compared detector
  `input_size` 640 vs 480 on the same clip, by per-class raw-detection recall
  (`results/input_size_recall_integrated_camera.md`) - the missing evidence
  the Phase 8 sweep's identical 197/197/197 totals could not provide (that
  clip had nothing small enough to discriminate). Result, 501 frames: `person`
  504@480 vs 505@640 (a wash); **`cell phone` 22@480 vs 32@640 - a real,
  material recall loss at 480 (-31% vs 640) on this footage**; `cup` 35@480 vs
  17@640 (480 detected MORE here - the opposite direction, unexplained,
  possibly NMS/anchor-density interaction at the smaller input, not
  investigated further this phase); `bottle`/`laptop`/`keyboard`/`mouse`/
  `chair`/`book`/`backpack` at zero for both sizes (absent from this footage
  or beyond model capability regardless of resolution - this clip cannot
  distinguish the two). **Decision: `input_size` stays 480, unchanged**, given
  (a) the Phase 8 latency saving is real and load-bearing (~46ms vs ~73ms
  detector p50) and section 19 requires preserving it, (b) the evidence is
  single-clip and directionally mixed (phone favours 640, cup favours 480),
  and (c) `docs/decisions.md`'s existing "still needs a clip with small-in-
  frame objects" caveat is not fully resolved by one session - this
  measurement is data toward that resolution, not a substitute for it.
  `input_size: 480` remains explicitly **provisional**, now with a concrete,
  non-trivial counter-example (phone) on record rather than an untested
  assumption. Re-run this script once more/different footage exists,
  especially footage with a phone or watch at typical desk distance, before
  treating 480 as settled.

## Phase 11

- **2026-09-16** Root cause of the "continuous but not current" symptom
  (Part A section 4): `Tracker._bind_poses` stamped a track's
  `pose_capture_ts` with the CURRENT cycle's `capture_ts` ("now") on every
  successful pose bind - fresh inference OR the perception engine's own
  within-window reuse - instead of the pose's own true measurement instant.
  Because the engine hands back the identical frozen `Pose` object on every
  cadence-due reuse cycle (`pose_max_reuse_ms`), a track's reported
  `pose_age_ms` was silently reset to ~0 on almost every cycle, hiding true
  staleness that the (correctly implemented) `tracking.pose_max_age_ms`
  bound was meant to cap. Measured before any fix
  (`results/track_pose_freshness_before.md`, live 40s session, same clip as
  Phase 10): `displayed_pose_age_ms` p50/p95/max = 93.9/144.7/**498.1**ms -
  the max never exceeding `pose_max_reuse_ms` (500ms) is the numeric
  fingerprint of the reset, since every successful bind (including a reuse
  well inside its own window) clears the clock before the track-level bound
  ever gets exercised.
- **2026-09-16** Fix: added `capture_ts: float | None = None` to `Pose`
  (`core/types.py`, additive per the Phase 0 frozen-contracts rule), stamped
  once at true inference time in `perception/pose.py` and never touched
  again. `Tracker._bind_poses` now records `pose.capture_ts` (falling back to
  the current cycle's `capture_ts` only when a `Pose` was built without one -
  back-compat for existing test fixtures) as the track's `pose_capture_ts`.
  Zero new computation on the hot path (same values, different source
  variable) - `tracker_update_ms` unaffected (see latency re-measurement
  below). Re-measured, same clip, same 40s duration
  (`results/track_pose_freshness_after.md`): p50/p95/max =
  155.6/205.1/476.5ms; a longer 90s session
  (`results/track_pose_freshness_after_90s.md`) reaches max=585.6ms -
  **beyond** `pose_max_reuse_ms`, which was structurally almost unreachable
  under the pre-fix reset bug and is now correctly, honestly measured. The
  p50 rise (93.9 -> 155.6ms) on the identical before/after session is the
  direct fingerprint of the fix: age no longer collapses to near-zero on
  every reuse-cycle rebind.
- **2026-09-16** `tracking.pose_max_age_ms` (900ms) is kept unchanged. The
  post-fix measured worst case over a 90s live session (585.6ms pose /
  711.2ms track) stays comfortably under it, and no measurement in this
  session justifies moving the bound in either direction (section 5.1: derive
  from measurement, do not pick a number - the disciplined choice here is to
  change nothing without new evidence). Caveat carried forward explicitly:
  this harness is a steady-state TestClient loopback with no real webcam
  jitter or the background CPU contention Phase 10's own live session
  measured (which found a materially worse 36.4% empty-pose rate); a real
  physical session (section 18, developer, pending) may measure a worse
  worst-case than this harness reproduces, and the bound should be
  re-derived then if so.
- **2026-09-16** Two new headline metrics, `displayed_pose_age_ms` /
  `displayed_track_age_ms` (Part A section 4.1), added to the existing
  `StateSnapshot.metrics` dict and `MetricRegistry` Samples in
  `pipeline/loop.py` (`AnalysisLoop._iterate`) - no new telemetry system, per
  section 22. Computed as `emitted_ts - track.<field>` (max across all
  currently-rendered tracks, matching what `overlay.js` actually draws, per-
  track tentative included), which is deliberately different from Phase 8's
  `frame_age_ms` / `stage_capture_to_snapshot_ms` (how old THIS frame's own
  detection is): a coasting or pose-reusing track can display a measurement
  from several cycles earlier than the current frame. `-1.0` = nothing
  currently displayed to measure (no tracks, or no track carries a bound
  pose). Wired into Diagnostics (`groups/diagnostics.js` `PHASE8_METRICS`).

