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
  `RiskState`, `Alert` stay distinct types. `StateSnapshot.tracks` is `[]` until
  a tracking phase actually ships one — never placehold it.
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
- `config/settings.py` — typed settings, YAML profiles, `PS_`-prefixed env overrides.
- `camera/` — `source.py`/`synthetic.py`/`mailbox.py`/`framing.py`/`browser.py`/
  `device.py`/`file_source.py`/`enumerate.py`/`_opencv.py`. `cv2` lives only here.
- `perception/` — detector + pose (ONNX/onnxruntime, imported only here),
  `policy.py` (recognition policy, vocabulary tiers), `runtime.py` (provider
  resolution: `auto|cpu|cuda|directml`), `vocabulary.py`, `classes.py`.
- `pipeline/` — `loop.py` (`AnalysisLoop`, the only mode-branching code),
  `scheduler.py` (timer / completion-driven consumer), `recorded.py`
  (`RecordedDriver`, byte-deterministic), `perception_frame.py`
  (`PerceptionFrame`, tracking-ready, real-time and recorded parity).
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
- `models/registry.py` — exactly one `active` model, SHA-256-checked. No
  activation code, no training anywhere in this repo.
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
  `class_coverage_audit.py`, `policy_audit.py`.

## Commands

PowerShell, from `C:\Users\mummi\Documents\Projects\PredictiveSense`:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,camera,cpu]"          # [cpu] xor [cuda] - never both (same ORT module name)

pytest -q                                   # full suite (hardware/cuda skip cleanly without hardware)
pytest -q -m "not slow"
pytest -q -m models                         # skips cleanly if models/ is empty
pytest -q -m browser                        # Playwright; skips cleanly if Chromium absent

python scripts\run_app.py --profile dev --source-kind browser   # http://127.0.0.1:8000/  (also /studio, /label)
python scripts\benchmark_latency.py --source data\raw --profile dev --limit 150 --label <name>
python scripts\benchmark_capture_paint.py --seconds 20 --label <name>   # real-browser capture->paint
python scripts\benchmark_providers.py --provider cpu|cuda|dml
python scripts\benchmark_recognition_paths.py --source data\raw --frames 300
```

Full command reference (per-phase eval/labelling/export workflows): `projectContext.md`.

## Current phase status

**Phase 8 (perception responsiveness, hardware-portable runtime,
tracking-ready output) — complete**, plus a follow-up investigation pass. See
`docs/phase-reports/phase8.md` (three-part: Measured / Physically observed
pending / Not verified). Highlights: push-on-publish broadcast
(`stage_ws_out_ms` p50 46→0 ms), pose cadence `every_n:2`, detector
`input_size: 480` (dev only — **provisional**, see below), `provider: auto`
resolving cuda→directml→cpu with a loud failure on an unavailable explicit
choice, `PerceptionFrame` (tracking-ready, no tracker yet). Full suite:
**412 passed, 4 skipped, 0 failed** (`pytest -q`, all markers, CPU-only).

Two open items carried forward, not blockers:
- **`input_size: 480` is provisional, not validated** — the one available clip
  has no distant/small primary-tier object, so identical 320/480/640 detection
  counts prove the sweep ran, not that recall holds at 480. Needs a clip with
  small-in-frame objects before being called settled.
- **OnePlus transport numbers, real-camera capture→paint, and CUDA/DirectML
  execution** are the developer's physical verification (Part 2 of the phase-8
  report) — not yet performed.

Do not start Phase 9 without the user's explicit go-ahead.

## Prohibitions

- No CUDA-dependent or CUDA-conditional code that assumes CUDA is present —
  always resolve through `perception.provider` / `resolve_provider`.
- No tracker, track IDs, association, `Relation`, temporal state, risk model,
  alert policy, or TTS — none started, none placeheld, until their phase.
- No model trained, fine-tuned, or activated from this repo's own code.
- Never report a confidence as accuracy, a detection frequency as P/R, a stored
  image as a trained model, a coverage heuristic as a validated metric, a box
  proposal as ground truth, or a tier assignment as a measured result.
- Never fit a threshold on `test`, or open `test` more than once.
- Never claim a performance number not produced by a command run on this
  machine, or a physical (camera/mic/speaker) check not actually performed.
- No module-level singleton, global mutable state, or import-time side effect —
  state lives on `app.state` or inside `pipeline/`.
