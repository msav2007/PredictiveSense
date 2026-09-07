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
- `predictivesense/config/settings.py` - typed settings, YAML profile loading, strict validation, env overrides (`PS_` prefix, `__` nesting). Phase 1 sections: `capture` (incl. tuning knobs `fourcc`/`buffer_size`/`warmup_frames` and Phase 1.5's `max_ws_buffered_bytes` worker-backpressure ceiling, default 1 MB - all default to the measured-optimal value), `recorder`, `video`.
- `predictivesense/camera/_opencv.py` - `quiet_opencv_logging()` (idempotent `cv2.setLogLevel(ERROR)` - kills the VIDEOIO index-probe spam) and `fourcc_to_str()`.
- `predictivesense/camera/source.py` - `FrameSource` interface + `create_frame_source` (builds only `SYNTHETIC`; other kinds raise `NotImplementedError` - built by `pipeline.build_camera_source` / `RecordedDriver`).
- `predictivesense/camera/synthetic.py` - `SyntheticSource`: deterministic, strictly increasing `frame_id` / `capture_ts`, paced to a target rate.
- `predictivesense/camera/mailbox.py` - `LatestFrameMailbox`: single-slot, overwrite-on-write, exact `consumed` / `dropped`.
- `predictivesense/camera/framing.py` - binary ingest message encode/decode (`FramingError`) + clock-offset arithmetic (`ClockOffset`, `clock_offset_seconds`, `capture_ts_seconds`).
- `predictivesense/camera/enumerate.py` - backend device enumeration (`enumerate_devices`, `as_api_rows`); optional `pygrabber` names, never fatal. Phase 1.5: `load_backend_hint`/`save_backend_hint` read/merge the per-index winning-backend cache `results/camera_backends.json` (written by `benchmark_camera_matrix.py`, machine-specific, not committed).
- `predictivesense/camera/browser.py` - `BrowserSource`: single-slot newest-wins buffer fed by `WS /ws/ingest`; decodes JPEG, stamps `capture_ts` from the clock offset.
- `predictivesense/camera/device.py` - `DeviceSource`: OpenCV camera, MSMF->DSHOW fallback (or cache-hinted backend first when `backend="auto"` + `backend_cache_dir`), reports *achieved* geometry. Reconnect is **non-blocking and `stop()`-interruptible** (one short reopen probe per `read()`, exponential backoff absorbed outside the lock via a `threading.Event`); no thread is added here.
- `predictivesense/camera/file_source.py` - `FileSource`: sequential decode, pts-based `capture_ts`, `asfast`/`realtime` replay, deterministic `restart()`; `video_duration_s()` helper.
- `predictivesense/pipeline/loop.py` - `AnalysisLoop` (+ `build_camera_source`, `request_consumer_stall`): producer thread + no-op consumer + lifecycle + error surfacing + Phase 1 metric keys.
- `predictivesense/pipeline/recorded.py` - `RecordedDriver`: Mode B, no mailbox, every frame, byte-deterministic `results/recorded_<run_id>.jsonl` + manifest.
- `predictivesense/telemetry/metrics.py` - `Counter`, `Rate`, `Samples`, `Timer`, `MetricRegistry` (all bounded).
- `predictivesense/telemetry/writer.py` - `MetricsWriter`: append-only CSV, write failure logged once and non-fatal.
- `predictivesense/telemetry/manifest.py` - `SessionManifest` + `build_manifest` + `git_state`.
- `predictivesense/api/app.py` - FastAPI factory and routes; mounts the ingest/recorder/videos routers and `/static`; starts/stops the loop over the lifespan; `write_manifest=True` writes `results/session_<id>.json`. Phase 1.5: `POST /api/metrics/browser` appends a labelled browser-measured sample block to `results/browser_metrics_<label>.json`.
- `predictivesense/api/ingest.py` - `WS /ws/ingest`: hello/ack/echo handshake + binary analysis frames -> `BrowserSource`.
- `predictivesense/api/recorder.py` - `POST /api/record/upload` + `GET /api/clips`: raw clip -> `data/raw/<session>/` + `ClipManifest`.
- `predictivesense/api/videos.py` - `GET /api/videos` + `POST /api/analyze` (path confined to `video.input_dir`).
- `predictivesense/api/broadcast.py` - `Broadcaster` + `serve_state_client`: last-value-wins, slow client dropped on timeout.
- `predictivesense/api/static/{index.html,app.js,app.css,analysis-worker.js}` - dashboard: native `srcObject` preview, Web Worker analysis path, metrics strip, clip recorder, Mode B controls. No framework, no build. Phase 1.5: worker newest-wins + `bufferedAmount` backpressure skip + `VideoFrame` close audit; `#preview-size` selector (`applyConstraints` on the same device); device-switch stops every track and nulls `srcObject` first, and measures switch time; `ondevicechange` re-enumerates; "Browser measurement" panel + "Capture sample" -> `POST /api/metrics/browser`.
- `predictivesense/logging_setup.py` - `configure_logging` / `get_logger`.
- `scripts/run_app.py` - start the API for a profile (`--source-kind synthetic|browser|device`).
- `scripts/run_noop.py` - 60-second instrumented no-op run (synthetic); writes `results/noop_<id>.csv` and `results/manifest_<id>.json`.
- `scripts/run_recorded.py` - run `RecordedDriver` over one file under `data/videos/`.
- `scripts/benchmark_transport.py` - backend-owned camera transport benchmark -> `results/transport_<label>.json`.
- `scripts/benchmark_camera_matrix.py` - Phase 1.5: backend-owned matrix sweep (resolution x fps x backend x fourcc) -> `results/camera_matrix_<label>.{json,md}` + merges the winning backend into `results/camera_backends.json`.
- `scripts/benchmark_analysis_path.py` - Phase 1.5: in-process loopback sweep of `analysis_fps` x size x quality through the real `BrowserSource`->mailbox->loop -> `results/analysis_sweep_<label>.{json,md}`.

## Core contracts

Names only; definitions in `predictivesense/core/types.py`:
`BBox`, `Frame`, `Detection`, `Pose`, `Track`, `Relation`, `RiskState`, `Alert`,
`StateSnapshot`, `MailboxStats`, `SourceInfo`, `ClipManifest`, `IngestHeader`.
Interfaces: `FrameSource` (in `camera/source.py`), the mailbox `put` / `get` /
`stats` (in `camera/mailbox.py`).

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
# regenerate the lock file (UTF-8, no BOM; Windows PowerShell 5.1 has no utf8NoBOM):
$f = & .\.venv\Scripts\python.exe -m pip freeze --exclude-editable
[IO.File]::WriteAllText("$PWD\requirements.lock.txt", ($f -join "`n") + "`n", (New-Object Text.UTF8Encoding($false)))
```

## Phase discipline

Current phase: **Phase 1 (Input layer)** - see `PredictiveSense-P1-Prompt.md`.
Never implement a future phase or leave an empty placeholder for one. No
detector, pose, tracker, temporal state, risk model, policy, TTS, or overlay
drawing belongs here. If a requirement is ambiguous, **stop and ask** rather than
inventing one. Never claim a performance number that was not produced by a
command actually run on this machine, and never claim physical (camera / mic /
speaker) verification - Phase 1 has six developer checks still pending.

## Current phase status

Phase 1 complete, including Phase 1.5 (camera/input optimization & verification
closing pass). `pytest -q`: **127 passed, 1 skipped** (hardware; `--run-hardware`
to run). 10-minute synthetic no-op RSS within budget; `DeviceSource` RSS flat
over 30 s of real capture.

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
