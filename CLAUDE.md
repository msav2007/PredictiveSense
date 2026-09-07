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
on it. Neither driver exists yet (P1).

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

- `predictivesense/core/enums.py` - `Mode`, `SourceKind`, `RiskLevel`, `TrackStatus`, and the "constructible source kind" guard.
- `predictivesense/core/types.py` - all frozen contracts (Block 8). Imports only stdlib, numpy, Pydantic, and `core.enums`.
- `predictivesense/config/settings.py` - typed settings, YAML profile loading, strict validation, env overrides (`PS_` prefix, `__` nesting).
- `predictivesense/camera/source.py` - `FrameSource` interface + `create_frame_source` factory (non-synthetic kinds raise `NotImplementedError` -> P1).
- `predictivesense/camera/synthetic.py` - `SyntheticSource`: deterministic, strictly increasing `frame_id` / `capture_ts`, paced to a target rate.
- `predictivesense/camera/mailbox.py` - `LatestFrameMailbox`: single-slot, overwrite-on-write, exact `consumed` / `dropped`.
- `predictivesense/pipeline/loop.py` - `AnalysisLoop`: producer thread + no-op consumer + lifecycle + error surfacing. The only place mode-awareness belongs.
- `predictivesense/telemetry/metrics.py` - `Counter`, `Rate`, `Samples`, `Timer`, `MetricRegistry` (all bounded).
- `predictivesense/telemetry/writer.py` - `MetricsWriter`: append-only CSV, write failure logged once and non-fatal.
- `predictivesense/telemetry/manifest.py` - `SessionManifest` + `build_manifest` + `git_state`.
- `predictivesense/api/app.py` - FastAPI factory and routes; starts/stops the loop over the app lifespan.
- `predictivesense/api/broadcast.py` - `Broadcaster` + `serve_state_client`: last-value-wins, slow client dropped on timeout.
- `predictivesense/api/static/index.html` - minimal snapshot viewer (no video, no canvas, no framework).
- `predictivesense/logging_setup.py` - `configure_logging` / `get_logger`.
- `scripts/run_app.py` - start the API for a profile.
- `scripts/run_noop.py` - 60-second instrumented no-op run; writes `results/noop_<id>.csv` and `results/manifest_<id>.json`.

## Core contracts

Names only; definitions in `predictivesense/core/types.py`:
`BBox`, `Frame`, `Detection`, `Pose`, `Track`, `Relation`, `RiskState`, `Alert`,
`StateSnapshot`, `MailboxStats`. Interfaces: `FrameSource` (in `camera/source.py`),
the mailbox `put` / `get` / `stats` (in `camera/mailbox.py`).

## Commands

PowerShell, from `C:\Users\mummi\Documents\Projects\PredictiveSense`:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"

pytest -q                                   # full suite
pytest -q -m "not slow"                      # fast subset
python scripts\run_app.py --profile dev      # http://127.0.0.1:8000/
python scripts\run_noop.py --profile dev --seconds 60   # writes results\
pip freeze > requirements.lock.txt           # after a dependency change
```

## Phase discipline

Current phase: **Phase 0 (Foundation)** - see `PredictiveSense-Phase0-Prompt.md`.
Never implement a future phase or leave an empty placeholder for one. No camera,
WebRTC, detector, pose, tracker, risk model, policy, TTS, overlay, or
recorded-video reader belongs here. If a requirement is ambiguous, **stop and
ask** rather than inventing one. Never claim a performance number that was not
produced by a command actually run on this machine, and never claim any physical
(camera / mic / speaker) verification - Phase 0 has none.

## Current phase status

Phase 0 complete. Clean venv install, full `pytest -q` green, 60-second no-op run
produces `results/` CSV + manifest, API serves `/health`, `/api/config`,
`/ws/state`, `/`. See `docs/phase-reports/phase0.md` for measured numbers.
Nothing from P1+ has been started.
