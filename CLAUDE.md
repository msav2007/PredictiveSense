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
- `predictivesense/config/settings.py` - typed settings, YAML profile loading, strict validation, env overrides (`PS_` prefix, `__` nesting). Phase 1 sections: `capture`, `recorder`, `video`.
- `predictivesense/camera/source.py` - `FrameSource` interface + `create_frame_source` (builds only `SYNTHETIC`; other kinds raise `NotImplementedError` - built by `pipeline.build_camera_source` / `RecordedDriver`).
- `predictivesense/camera/synthetic.py` - `SyntheticSource`: deterministic, strictly increasing `frame_id` / `capture_ts`, paced to a target rate.
- `predictivesense/camera/mailbox.py` - `LatestFrameMailbox`: single-slot, overwrite-on-write, exact `consumed` / `dropped`.
- `predictivesense/camera/framing.py` - binary ingest message encode/decode (`FramingError`) + clock-offset arithmetic (`ClockOffset`, `clock_offset_seconds`, `capture_ts_seconds`).
- `predictivesense/camera/enumerate.py` - backend device enumeration (`enumerate_devices`, `as_api_rows`); optional `pygrabber` names, never fatal.
- `predictivesense/camera/browser.py` - `BrowserSource`: single-slot newest-wins buffer fed by `WS /ws/ingest`; decodes JPEG, stamps `capture_ts` from the clock offset.
- `predictivesense/camera/device.py` - `DeviceSource`: OpenCV camera, MSMF->DSHOW fallback, bounded exponential-backoff reconnect, reports *achieved* geometry.
- `predictivesense/camera/file_source.py` - `FileSource`: sequential decode, pts-based `capture_ts`, `asfast`/`realtime` replay, deterministic `restart()`; `video_duration_s()` helper.
- `predictivesense/pipeline/loop.py` - `AnalysisLoop` (+ `build_camera_source`, `request_consumer_stall`): producer thread + no-op consumer + lifecycle + error surfacing + Phase 1 metric keys.
- `predictivesense/pipeline/recorded.py` - `RecordedDriver`: Mode B, no mailbox, every frame, byte-deterministic `results/recorded_<run_id>.jsonl` + manifest.
- `predictivesense/telemetry/metrics.py` - `Counter`, `Rate`, `Samples`, `Timer`, `MetricRegistry` (all bounded).
- `predictivesense/telemetry/writer.py` - `MetricsWriter`: append-only CSV, write failure logged once and non-fatal.
- `predictivesense/telemetry/manifest.py` - `SessionManifest` + `build_manifest` + `git_state`.
- `predictivesense/api/app.py` - FastAPI factory and routes; mounts the ingest/recorder/videos routers and `/static`; starts/stops the loop over the lifespan; `write_manifest=True` writes `results/session_<id>.json`.
- `predictivesense/api/ingest.py` - `WS /ws/ingest`: hello/ack/echo handshake + binary analysis frames -> `BrowserSource`.
- `predictivesense/api/recorder.py` - `POST /api/record/upload` + `GET /api/clips`: raw clip -> `data/raw/<session>/` + `ClipManifest`.
- `predictivesense/api/videos.py` - `GET /api/videos` + `POST /api/analyze` (path confined to `video.input_dir`).
- `predictivesense/api/broadcast.py` - `Broadcaster` + `serve_state_client`: last-value-wins, slow client dropped on timeout.
- `predictivesense/api/static/{index.html,app.js,app.css,analysis-worker.js}` - dashboard: native `srcObject` preview, Web Worker analysis path, metrics strip, clip recorder, Mode B controls. No framework, no build.
- `predictivesense/logging_setup.py` - `configure_logging` / `get_logger`.
- `scripts/run_app.py` - start the API for a profile (`--source-kind synthetic|browser|device`).
- `scripts/run_noop.py` - 60-second instrumented no-op run (synthetic); writes `results/noop_<id>.csv` and `results/manifest_<id>.json`.
- `scripts/run_recorded.py` - run `RecordedDriver` over one file under `data/videos/`.
- `scripts/benchmark_transport.py` - backend-owned camera transport benchmark -> `results/transport_<label>.json`.

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

Phase 1 complete (code + automated tests). Clean venv install, full `pytest -q`
green (hardware skipped, runnable with `--run-hardware`), 10-minute no-op RSS
within budget. The API adds `/api/cameras`, `/ws/ingest`, `/api/videos`,
`/api/analyze`, `/api/record/upload`, `/api/clips`, `/api/debug/stall`,
`/static/*`. `RecordedDriver` is byte-deterministic. `cv2` is confined to
`predictivesense/camera/`. See `docs/phase-reports/phase1.md` for measured
numbers and the six pending physical checks (camera dropdown by name, preview
smoothness by eye, stall-by-eye, clip on disk, transport benchmarks, phone
disconnect/recover). Nothing from P2+ has been started.
