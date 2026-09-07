# PredictiveSense — Phase 0 Implementation Prompt (Claude Code) — rev 2

> Paste this entire file into Claude Code as a single message, in a **fresh session**. Do not summarise it.
> **Scope note: Phase 0 is one session and roughly one day of work.** It is deliberately small. Do not polish beyond what is written here.

---

## BLOCK 1 — CONTEXT

You are implementing **Phase 0 (Foundation)** of PredictiveSense.

PredictiveSense is a research prototype, built to a one-month deadline, that supports a research paper. It observes an indoor cabin scene, tracks objects and people over time, and predicts developing safety risks early enough to warn the user with a spoken alert and a fixed preventive recommendation. It has **two input modes**: Mode A (real-time camera) and Mode B (recorded video, analysed retrospectively), sharing one perception → tracking → temporal → risk core. Because it supports a paper, reproducibility, measurement honesty and attribution are hard requirements.

**Current repository state: empty**, apart from this prompt file. No source code, no backend, no frontend, no CV pipeline, no tracker, no model, no tests, no git history. You are creating the project from zero.

**Target machine:** Windows, Intel Core Ultra 5 125H (14 cores / 18 logical processors), 16 GB LPDDR5 shared with an integrated Intel Arc GPU. **No NVIDIA GPU, no CUDA.** Never write CUDA-dependent or CUDA-conditional code.

Phase 0 contains **no intelligence and no hardware access**. It builds the skeleton and the measuring apparatus every later phase depends on: contracts, configuration, a synthetic frame source, the bounded frame mailbox, a no-op analysis loop, an API with a state WebSocket, telemetry, a session manifest, tests, a project context file, and the phase report.

Environment: Python 3.11 on Windows. Use PowerShell syntax in all documentation.

---

## BLOCK 2 — OBJECTIVE

At the end of Phase 0, a clean clone must install into a fresh virtual environment with one documented command, pass its entire test suite, start a FastAPI application that streams state snapshots over a WebSocket, and produce — from a 60-second no-op run — a metrics CSV and a session manifest in `results/`. The typed core contracts that every later phase consumes must exist and must not import any model, camera, or network library. A `CLAUDE.md` must exist so later phases need no re-briefing.

No detector, pose model, tracker, risk model, WebRTC, camera device, TTS, or overlay exists yet.

---

## BLOCK 3 — REQUIREMENTS

1. Python package `predictivesense/` with the layout in Block 5.
2. `pyproject.toml` with pinned runtime dependencies and a `dev` extra for test dependencies. Commit `requirements.lock.txt` produced by `pip freeze` from the clean install.
3. `predictivesense/core/types.py` defining every contract in Block 8 as immutable, fully annotated dataclasses (or Pydantic models — choose one and be consistent). This module may import only the standard library, `typing`, `enum`, `dataclasses`, a numpy type alias, and — if chosen — Pydantic. It must not import OpenCV, torch, onnxruntime, any HTTP client, or anything from `predictivesense.api`, `.pipeline` or `.camera`.
4. Configuration: a Pydantic-settings model plus **two** YAML profiles, `dev` and `eval`, under `config/profiles/`. An unknown profile, an unknown field, or a value that fails validation must raise a clear error and exit non-zero — never fall back silently to a default.
5. Config includes a validated `mode` field accepting exactly `realtime` or `recorded`. Phase 0 implements neither driver; the field exists so later phases do not reshape the config. Any other value is a validation error.
6. `SyntheticSource` implementing `FrameSource`: deterministic given a seed, strictly increasing `frame_id`, strictly increasing monotonic `capture_ts`, configurable target rate.
7. `LatestFrameMailbox`: single-slot, overwrite-on-write. `put()` never blocks and never grows. `get()` returns the newest frame or `None`. Maintains exact `dropped` and `consumed` counts. Depth is 0 or 1 at all times.
8. A no-op analysis loop: a producer thread driving `SyntheticSource` into the mailbox, and a consumer sampling the mailbox at a configured rate, timing each iteration and emitting a `StateSnapshot`. Both shut down cleanly with no orphaned threads.
9. Telemetry: counters, rates, a timer context manager, a `MetricsWriter` appending CSV rows, and a `SessionManifest` written as JSON containing at minimum — session id, UTC start and end, git commit hash and dirty flag, resolved configuration, Python version, platform string, CPU count, total RAM, and the random seed.
10. FastAPI app exposing `GET /health`, `GET /api/config` (resolved configuration), and `WS /ws/state` broadcasting `StateSnapshot` JSON at a configured rate. The WebSocket is **last-value-wins**: a client that cannot keep up has snapshots dropped. Never buffer an unbounded per-client backlog, and never let a slow client slow the analysis loop.
11. Runtime code makes **no outbound network connections**. Binding a local listening socket for the API is expected and permitted; connecting outward to any non-loopback address is forbidden.
12. Deterministic behaviour under a fixed seed: two runs with the same seed and config produce the same synthetic frames.
13. Structured logging with a configurable level. No `print()` in library code.
14. A local git repository with meaningful commits. **Do not create a remote, do not create a GitHub repository, do not push.**
15. **`CLAUDE.md` at the repository root**, structured exactly as in Block 8 item 10. This file is how later phases avoid re-exploring the repo — write it carefully and keep it under roughly 150 lines.
16. `docs/decisions.md` — a running decision log, one line per decision, seeded with the choices you make in this phase (dataclasses vs Pydantic, logging library, CSV schema, etc.).
17. `docs/phase-reports/phase0.md` in the exact format of Block 14.

---

## BLOCK 4 — ARCHITECTURE CONSTRAINTS

- **Contracts are frozen once written.** Later phases may add fields; they may not rename or reshape what Phase 0 defines without a line in `docs/decisions.md`.
- **No unbounded queues anywhere.** The mailbox is single-slot. The broadcaster drops rather than buffers. Any collection that grows with runtime needs an explicit cap.
- **Stages are independent and injectable.** No global mutable state, no singletons, no import-time side effects. Threads and wiring exist only in `pipeline/`.
- **Purity where possible.** Mailbox, config, telemetry and contracts must be testable with no I/O, no threads, no hardware.
- **Mode-awareness lives in the driver, not the core.** Nothing outside `pipeline/` may branch on `mode`.
- **No hardware, no models, no network clients in Phase 0.** If a module needs one, it belongs to a later phase.
- **No CUDA, no NVIDIA assumptions, anywhere.**
- **Local processing only.** No cloud service, no runtime download, no telemetry upload.
- Windows is the target: use `pathlib`, never hard-code separators, and use `time.monotonic()` for all durations — never `time.time()`.
- **Do not overengineer.** No Docker, no CI config, no database, no message broker, no auth, no UI framework, no abstraction layer that has exactly one implementation and no second one planned in Block 8.

---

## BLOCK 5 — FILES TO CREATE

```
pyproject.toml
requirements.lock.txt
README.md
CLAUDE.md
.gitignore
config/profiles/dev.yaml
config/profiles/eval.yaml
predictivesense/__init__.py
predictivesense/core/__init__.py
predictivesense/core/enums.py            # Mode, SourceKind, RiskLevel, TrackStatus
predictivesense/core/types.py            # all shared contracts (Block 8)
predictivesense/config/__init__.py
predictivesense/config/settings.py       # typed settings + profile loading + validation
predictivesense/camera/__init__.py
predictivesense/camera/source.py         # FrameSource abstract interface
predictivesense/camera/synthetic.py      # SyntheticSource
predictivesense/camera/mailbox.py        # LatestFrameMailbox
predictivesense/pipeline/__init__.py
predictivesense/pipeline/loop.py         # producer thread + no-op consumer loop + lifecycle
predictivesense/telemetry/__init__.py
predictivesense/telemetry/metrics.py     # Counter, Rate, Timer, registry
predictivesense/telemetry/writer.py      # MetricsWriter (CSV)
predictivesense/telemetry/manifest.py    # SessionManifest
predictivesense/api/__init__.py
predictivesense/api/app.py               # FastAPI app factory + routes
predictivesense/api/broadcast.py         # last-value-wins WebSocket broadcaster
predictivesense/api/static/index.html    # minimal snapshot viewer (Block 8 item 9)
predictivesense/logging_setup.py
scripts/run_app.py
scripts/run_noop.py                      # 60-second instrumented no-op run
tests/__init__.py
tests/conftest.py
tests/unit/test_types.py
tests/unit/test_config.py
tests/unit/test_synthetic_source.py
tests/unit/test_mailbox.py
tests/unit/test_telemetry.py
tests/unit/test_no_forbidden_imports.py
tests/integration/test_api.py
tests/integration/test_ws_snapshots.py
tests/integration/test_noop_run.py
tests/integration/test_no_outbound_network.py
docs/architecture.md
docs/attribution.md
docs/decisions.md
docs/phase-reports/phase0.md
results/.gitkeep
data/.gitkeep
```

`.gitignore` must ignore `.venv/`, `__pycache__/`, `data/*` and `results/*` (except their `.gitkeep`), and IDE folders.

---

## BLOCK 6 — FILES TO MODIFY

None. Every file is new.

`PredictiveSense-Phase0-Prompt.md` already exists in the repository root. It is this prompt, not project source. **Do not delete it, do not modify it, and do not treat its presence as existing implementation.**

---

## BLOCK 7 — FILES THAT MUST NOT BE CREATED

Each belongs to a later phase. **Creating an empty placeholder for one is also forbidden** — scaffolding for future work is how phases leak into each other.

- `predictivesense/perception/`, `tracking/`, `scene/`, `temporal/`, `risk/`, `policy/`, `audio/`, `research/` — any of them, even empty.
- Any camera device access (`cv2.VideoCapture`, Media Foundation, DirectShow, `getUserMedia`).
- Any WebRTC signalling, peer connection, or video element.
- Any detector, pose, tracker, feature builder, risk model, alert policy, or TTS code.
- Any model weights, model download logic, or ONNX / OpenVINO / DirectML code.
- Any Dockerfile, CI workflow, database, message broker, or cloud SDK.
- Any recorded-video reader or writer.

If you believe one is required to satisfy Phase 0, **stop and ask** instead of creating it.

---

## BLOCK 8 — CONTRACTS

**1. `core/enums.py`**

- `Mode`: `REALTIME = "realtime"`, `RECORDED = "recorded"`
- `SourceKind`: `SYNTHETIC = "synthetic"`, `DEVICE = "device"`, `FILE = "file"`, `WEBRTC = "webrtc"` — only `SYNTHETIC` is constructible in Phase 0; the others must raise `NotImplementedError` naming the phase that will implement them (P1).
- `RiskLevel`: `SAFE`, `WARNING`, `HIGH_RISK`, `UNKNOWN` — defined, unused.
- `TrackStatus`: `TENTATIVE`, `CONFIRMED`, `COASTING`, `EXPIRED` — defined, unused.

**2. `Frame`**

| Field | Type | Meaning |
|---|---|---|
| `frame_id` | `int` | Monotonic per-source counter from 0 |
| `capture_ts` | `float` | `time.monotonic()` at capture |
| `image` | `np.ndarray` | HxWx3, BGR, uint8 |
| `width` | `int` | |
| `height` | `int` | |
| `source_id` | `str` | Opaque id of the producing source |
| `seq` | `int` | Sequence within the current source session (resets on reconnect) |

**3. `Detection`** — `bbox: tuple[float,float,float,float]` (x1,y1,x2,y2 px), `class_id: int`, `class_name: str`, `score: float`, `frame_id: int`. Defined, unused.

**4. `Pose`** — `keypoints: tuple[tuple[float,float,float], ...]` (x, y, visibility), `bbox`, `score`, `frame_id`. Defined, unused.

**5. `Track`** — `track_id: int`, `status: TrackStatus`, `class_name: str`, `bbox`, `last_seen_frame_id: int`. Minimal on purpose; P3 extends it.

**6. `Relation`** — `subject_id: int`, `object_id: int`, `kind: str`, `value: float`. Minimal; P3 extends it.

**7. `RiskState`** — `scenario: str`, `level: RiskLevel`, `confidence: float`. Minimal; P6 extends it.

**8. `Alert`** — `alert_id: int`, `level: RiskLevel`, `text: str`, `recommendation: str`, `issued_ts: float`. Minimal; P7 extends it.

**9. `StateSnapshot`** — the single object broadcast to the UI. The frozen wire format; document its JSON encoding in `docs/architecture.md`.

| Field | Type | Phase 0 value |
|---|---|---|
| `snapshot_id` | `int` | Monotonic |
| `mode` | `Mode` | From config |
| `frame_id` | `int \| None` | Frame this snapshot describes |
| `capture_ts` | `float \| None` | Capture time of that frame |
| `emitted_ts` | `float` | `time.monotonic()` at emission |
| `frame_age_ms` | `float \| None` | `(emitted_ts - capture_ts) * 1000` |
| `detections` | `list[Detection]` | Always empty |
| `poses` | `list[Pose]` | Always empty |
| `tracks` | `list[Track]` | Always empty |
| `risk` | `RiskState \| None` | Always `None` |
| `metrics` | `dict[str, float]` | Loop rate, drop count, consumed count, iteration latency ms |
| `stale` | `bool` | True when no frame arrived within a configured threshold |

**10. Interfaces**

- `FrameSource`: `start()`, `stop()`, `read() -> Frame | None`, `is_running -> bool`, `info() -> dict`, context-manager support, `stop()` safe to call twice.
- Mailbox: `put(frame) -> None`, `get() -> Frame | None`, `stats() -> MailboxStats(consumed:int, dropped:int, depth:int)`.

**11. HTTP / WS**

- `GET /health` → `200 {"status":"ok","uptime_s":float,"mode":"realtime|recorded","version":str}`
- `GET /api/config` → `200`, resolved configuration as JSON.
- `WS /ws/state` → server pushes `StateSnapshot` JSON. No client→server messages are interpreted in Phase 0.
- `GET /` → a **single static HTML page, under 60 lines**, that connects to the WebSocket and prints the last snapshot as formatted JSON. No video element, no canvas, no CSS framework, no build step, no styling work.

**12. `CLAUDE.md` structure** — exactly these sections, factual, no marketing prose:

```markdown
# PredictiveSense — project context for Claude Code
## What this project is            (3 sentences)
## Hardware and platform            (Windows, Core Ultra 5 125H, Arc iGPU, 16 GB, NO CUDA)
## Two modes                        (realtime / recorded, one shared core)
## Architecture invariants          (bounded queues, preview independence, detection/tracking
                                     separation, local-only, no cloud, contracts frozen)
## Module map                       (one line per package under predictivesense/)
## Core contracts                   (names only, pointing at core/types.py)
## Commands                         (install / test / run / noop — PowerShell)
## Phase discipline                 (current phase; never implement future phases; stop and ask
                                     if ambiguous; never claim unmeasured or unverified results)
## Current phase status             (updated at the end of each phase)
```

---

## BLOCK 9 — ERROR HANDLING

**Fail loudly, exit non-zero:** unknown or malformed config profile; unknown config field; invalid `mode`; unwritable `results/`; constructing a non-synthetic `SourceKind`.

**Degrade gracefully:** a WebSocket client that disconnects or stalls is dropped without affecting the loop; a metrics write failure is logged once and does not stop the run.

**Always log:** run start with resolved config, run stop with final counters, every dropped client, the final manifest path.

**Never:** swallow an exception silently, use a bare `except:`, or log a stack trace as normal operation. A thread that exits abnormally must set an error flag the loop surfaces and the process exit code reflects.

---

## BLOCK 10 — PERFORMANCE REQUIREMENTS

Phase 0 has **no performance target to meet**. It must be able to *measure*:

- Loop iteration latency, p50 and p95 over the run.
- Consumer loop rate (Hz) and producer frame rate (Hz).
- Dropped and consumed counts, and mailbox depth.
- Process RSS at start and end of the run.

Two assertions: a 60-second no-op run must not grow RSS by more than 25 MB, and mailbox depth must never exceed 1. Both are measured, and the actual numbers go in the report.

**Do not state, imply, or record any performance figure that was not produced by a run you actually executed on this machine.** No aspirational numbers in code comments, documentation, or the report.

---

## BLOCK 11 — TESTS

Configure pytest markers: `unit`, `integration`, `slow`, `hardware`. The `hardware` set is empty in Phase 0.

**Unit**

1. `test_types` — every contract constructible with explicit types; `StateSnapshot` round-trips to JSON with identical values; contracts are immutable (mutation raises).
2. `test_config` — both profiles load and validate; unknown profile raises; unknown field raises; `mode="invalid"` raises; resolved config is serialisable.
3. `test_synthetic_source` — `frame_id` strictly increasing with no gaps; `capture_ts` strictly increasing; image shape and dtype correct; same seed yields identical frames; `stop()` idempotent.
4. `test_mailbox` — fast producer / slow consumer: consumer receives only the newest frame and `dropped == produced - consumed` exactly; depth never exceeds 1 over a randomised put/get sequence; `get()` on empty returns `None` without blocking; counters exact under a concurrent producer thread.
5. `test_telemetry` — counter / rate / timer arithmetic; `MetricsWriter` emits one header and well-formed rows; `SessionManifest` contains every required key and is valid JSON.
6. `test_no_forbidden_imports` — parse every module under `predictivesense/` with `ast` and assert none imports `cv2`, `torch`, `onnxruntime`, `openvino`, `mediapipe`, `ultralytics`, `requests`, `urllib.request`, `httpx`, `aiohttp`, `pyttsx3`, or `aiortc`; and that `core/` imports nothing from `api/`, `pipeline/` or `camera/`.

**Integration**

7. `test_api` — `/health` and `/api/config` return the documented shapes.
8. `test_ws_snapshots` — a client receives at least N snapshots with strictly increasing `snapshot_id`; a deliberately slow client is dropped and the loop rate before and after stays within a stated tolerance.
9. `test_noop_run` (marked `slow`) — a 10-second in-process run starts and stops cleanly, leaves no live threads, writes a metrics CSV with the expected columns and a manifest with every required key. The full 60-second run is done by `scripts/run_noop.py`, not by the suite.
10. `test_no_outbound_network` — patch `socket.socket.connect` to raise on any non-loopback address, run the pipeline and API for a few seconds, assert nothing raised.

**Not in Phase 0:** any test needing a camera, a model, audio hardware, or a browser.

---

## BLOCK 12 — EXACT COMMANDS

Document these in `README.md` and `CLAUDE.md`, and verify each actually runs before reporting. Working directory is the repository root: `C:\Users\mummi\Documents\Projects\PredictiveSense`.

```powershell
# 0. Go to the project
cd C:\Users\mummi\Documents\Projects\PredictiveSense

# 1. Clean environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"

# 2. Full test suite
pytest -q

# 3. Fast subset
pytest -q -m "not slow"

# 4. Start the application (dev profile)
python scripts\run_app.py --profile dev

# 5. 60-second instrumented no-op run (writes results\)
python scripts\run_noop.py --profile dev --seconds 60

# 6. Regenerate the lockfile after a dependency change
pip freeze > requirements.lock.txt
```

Report the verbatim output of commands 2 and 5 in the phase report.

---

## BLOCK 13 — PHYSICAL VERIFICATION

**Phase 0 requires none and must claim none.** No camera, microphone, speaker or phone is involved.

The report's "Physically verified" section must read exactly: *"None required in this phase."* Running the app on this machine is a code-level check and belongs under "Measured".

Optional and non-blocking: open `http://127.0.0.1:<port>/` and confirm snapshot JSON is updating. If you cannot open a browser, say so and leave it to the developer.

---

## BLOCK 14 — REPORT FORMAT

Write `docs/phase-reports/phase0.md` with exactly these sections:

```markdown
# Phase 0 — Foundation — Report

**Date:** <UTC date>
**Commit:** <git sha>
**Machine:** <cpu / ram / os from the manifest>
**Python:** <version>

## 1. Measured
<Every number produced by a command you ran, with the command beside it: pytest summary
line, test count and duration; 60-second run loop rate, producer rate, dropped/consumed,
p50/p95 iteration latency, RSS start/end and delta; max mailbox depth observed. Include the
metrics CSV header and the manifest key list. No number here that a command did not print.>

## 2. Physically verified
None required in this phase.

## 3. Not verified
<Must not be empty. State explicitly that no camera, model, tracker, risk model, audio path,
WebRTC path or overlay exists or has been tested, and that no claim about detection accuracy,
real-world latency or voice output is supported by this phase.>

## 4. Deviations and decisions
<Anything done differently from this prompt, with the reason. "None." if none.>

## 5. Open questions for the developer
<Anything ambiguous you hit. "None." if none.>
```

Then, in your reply, list the Block 15 items with the evidence for each.

---

## BLOCK 15 — ACCEPTANCE CHECKLIST

Run the check; do not mark an item complete from inspection.

- [ ] Fresh venv + `pip install -e ".[dev]"` succeeds from a clean state.
- [ ] `pytest -q` passes with zero failures and zero errors.
- [ ] 60-second no-op run completes; RSS growth under 25 MB (record the actual number).
- [ ] Metrics CSV and session manifest written to `results/`, manifest containing the resolved config and git commit.
- [ ] `GET /health`, `GET /api/config` and `WS /ws/state` behave as specified; a slow client does not slow the loop.
- [ ] No forbidden imports; `core/` depends on nothing else in the package.
- [ ] Outbound-network guardrail test passes.
- [ ] Local git repository with meaningful commits and no remote.
- [ ] `CLAUDE.md`, `docs/architecture.md`, `docs/attribution.md`, `docs/decisions.md` and `docs/phase-reports/phase0.md` all exist and are accurate.

---

## BLOCK 16 — PROHIBITIONS (MANDATORY)

You must **not**:

1. Invent requirements. If anything is ambiguous, **stop and ask**.
2. Silently change any requirement, contract, or file path in this prompt.
3. Implement any part of P1 or later — no camera access, no WebRTC, no detector, no pose, no tracker, no risk model, no policy, no TTS, no overlay, no recorded-video reader — and no empty placeholder modules for them.
4. Claim any performance number not measured by a command you ran on this machine.
5. Claim any physical verification. Phase 0 has none.
6. Create an unbounded queue, buffer, list or cache anywhere.
7. Add a dependency not required by this prompt; every dependency is justified in `docs/attribution.md`.
8. Write CUDA-specific, NVIDIA-specific, or GPU-assuming code.
9. Contact any network service at runtime, download any model or asset at runtime, or use a cloud API.
10. Create a git remote, a GitHub repository, or push anything.
11. Use `print()` in library code, a bare `except:`, or swallow exceptions.
12. Use `time.time()` for durations — `time.monotonic()` only.
13. Write aspirational or placeholder numbers into code, docs, or the report.
14. Reorganise the module layout in Block 5 without asking first.
15. Mark an acceptance item complete without running the check that proves it.
16. Delete or modify `PredictiveSense-Phase0-Prompt.md`.

When Phase 0 is finished, **stop**. Do not begin P1. Report against Block 15 and wait.
