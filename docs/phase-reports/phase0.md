# Phase 0 — Foundation — Report

**Date:** 2026-09-07 (UTC)
**Commit:** 046641571f0046a940466d5e9335b37b96634a2a (all measurements below were taken against this commit with a clean working tree; this report is committed on top of it)
**Machine:** 18 logical CPUs / 16,766,066,688 bytes RAM (~15.6 GiB) / `Windows-10-10.0.26200-SP0` — values verbatim from the session manifest. The host is an Intel Core Ultra 5 125H with an integrated Intel Arc GPU; no NVIDIA GPU, no CUDA.
**Python:** 3.11.9

## 1. Measured

### `pytest -q` — full suite (`python -m pytest -q`)

```
...............................................                          [100%]
============================== warnings summary ===============================
.venv\Lib\site-packages\starlette\testclient.py:40
  C:\Users\mummi\Documents\Projects\PredictiveSense\.venv\Lib\site-packages\starlette\testclient.py:40: DeprecationWarning: The anyio.abc.BlockingPortal alias is deprecated, use anyio.from_thread.BlockingPortal instead.
    _PortalFactoryType = typing.Callable[[], typing.ContextManager[anyio.abc.BlockingPortal]]

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
47 passed, 1 warning in 16.64s
```

- **47 passed, 0 failed, 0 errors, 1 warning, 16.64 s.**
- Test count by kind: 6 unit files / 4 integration files; markers `unit`, `integration`, `slow` (1 test), `hardware` (0 tests).
- The lone warning is raised inside `starlette/testclient.py` (an `anyio` alias deprecation in a third-party package), not in `predictivesense/`.
- `pytest -q -m "not slow"` (`python scripts` command 3): **46 passed, 1 deselected, 1 warning in 6.66 s.**

### 60-second instrumented no-op run — `python scripts\run_noop.py --profile dev --seconds 60`

```
2026-09-07T10:41:48+0530 INFO    predictivesense.scripts.run_noop | no-op run start session=7c8c1ef6e83c42fab39834884a348f20 profile=dev seconds=60 results_dir=results
2026-09-07T10:41:48+0530 INFO    predictivesense.pipeline.loop | analysis loop starting profile=dev mode=realtime sample_rate=20.00 Hz
2026-09-07T10:41:48+0530 INFO    predictivesense.camera.synthetic | synthetic source started source_id=synthetic-12345-640x480 session=0 640x480 @ 30.00 fps seed=12345
2026-09-07T10:42:48+0530 INFO    predictivesense.camera.synthetic | synthetic source stopped source_id=synthetic-12345-640x480 frames_produced=1801
2026-09-07T10:42:48+0530 INFO    predictivesense.pipeline.loop | analysis loop stopped snapshots=1200 consumed=1198 dropped=602 max_mailbox_depth=0 loop_hz=20.05 producer_hz=30.00 error=None
2026-09-07T10:42:48+0530 INFO    predictivesense.telemetry.manifest | session manifest written: results\manifest_7c8c1ef6e83c42fab39834884a348f20.json
2026-09-07T10:42:48+0530 INFO    predictivesense.telemetry.writer | metrics csv written: results\noop_7c8c1ef6e83c42fab39834884a348f20.csv (61 rows)
2026-09-07T10:42:48+0530 INFO    predictivesense.scripts.run_noop | === no-op run summary ===
2026-09-07T10:42:48+0530 INFO    predictivesense.scripts.run_noop | session_id           : 7c8c1ef6e83c42fab39834884a348f20
2026-09-07T10:42:48+0530 INFO    predictivesense.scripts.run_noop | profile / mode       : dev / realtime
2026-09-07T10:42:48+0530 INFO    predictivesense.scripts.run_noop | requested seconds    : 60
2026-09-07T10:42:48+0530 INFO    predictivesense.scripts.run_noop | snapshots emitted    : 1200
2026-09-07T10:42:48+0530 INFO    predictivesense.scripts.run_noop | frames consumed      : 1198
2026-09-07T10:42:48+0530 INFO    predictivesense.scripts.run_noop | frames dropped       : 602
2026-09-07T10:42:48+0530 INFO    predictivesense.scripts.run_noop | loop rate (Hz)       : 20.049
2026-09-07T10:42:48+0530 INFO    predictivesense.scripts.run_noop | producer rate (Hz)   : 30.000
2026-09-07T10:42:48+0530 INFO    predictivesense.scripts.run_noop | iter latency p50 (ms): 0.0400
2026-09-07T10:42:48+0530 INFO    predictivesense.scripts.run_noop | iter latency p95 (ms): 0.0719
2026-09-07T10:42:48+0530 INFO    predictivesense.scripts.run_noop | iter latency max (ms): 0.1457
2026-09-07T10:42:48+0530 INFO    predictivesense.scripts.run_noop | max mailbox depth    : 0  (<=1 required: True)
2026-09-07T10:42:48+0530 INFO    predictivesense.scripts.run_noop | RSS start (MB)        : 50.061
2026-09-07T10:42:48+0530 INFO    predictivesense.scripts.run_noop | RSS end (MB)          : 51.282
2026-09-07T10:42:48+0530 INFO    predictivesense.scripts.run_noop | RSS growth (MB)      : 1.221  (<=25.0 required: True)
2026-09-07T10:42:48+0530 INFO    predictivesense.scripts.run_noop | csv                  : results\noop_7c8c1ef6e83c42fab39834884a348f20.csv (61 rows)
2026-09-07T10:42:48+0530 INFO    predictivesense.scripts.run_noop | manifest             : results\manifest_7c8c1ef6e83c42fab39834884a348f20.json
2026-09-07T10:42:48+0530 INFO    predictivesense.scripts.run_noop | PASS: no-op run within budget
```

| Measure | Value | Command that produced it |
|---|---|---|
| Consumer loop rate | **20.049 Hz** (configured 20 Hz) | `run_noop.py --profile dev --seconds 60` |
| Producer frame rate | **30.000 Hz** (configured 30 fps) | same |
| Frames consumed / dropped | **1198 / 602** — producer 30 Hz vs consumer 20 Hz, every unconsumed frame dropped by the single-slot mailbox | same |
| Snapshots emitted | **1200** | same |
| Iteration latency p50 | **0.0400 ms** (1200 samples) | same |
| Iteration latency p95 | **0.0719 ms** | same |
| Iteration latency max | **0.1457 ms** | same |
| Process RSS at start | **50.061 MB** | same |
| Process RSS at end | **51.282 MB** | same |
| RSS growth over 60 s | **1.221 MB** — assertion `< 25 MB`: **pass** | same |
| Max mailbox depth observed | **0** — assertion `<= 1`: **pass** | same |
| Script exit code | **0** | same |

### Metrics CSV (`results/noop_<session>.csv`)

1 header row + **61** data rows (sampled every `noop.csv_sample_period_s` = 1.0 s). Header:

```
elapsed_s,wall_utc,snapshot_id,frame_id,stale,loop_rate_hz,producer_rate_hz,consumed,dropped,mailbox_depth,iter_latency_ms,rss_mb
```

### Session manifest (`results/manifest_<session>.json`)

Top-level keys:

```
session_id, started_utc, ended_utc, git_commit, git_dirty, config,
python_version, platform, cpu_count, total_ram_bytes, seed, extra
```

Values from this run: `git_commit = 046641571f0046a940466d5e9335b37b96634a2a`,
`git_dirty = false`, `python_version = 3.11.9`,
`platform = Windows-10-10.0.26200-SP0`, `cpu_count = 18`,
`total_ram_bytes = 16766066688`, `seed = 12345`. `config` is the fully resolved
`dev` profile. `extra.run` holds the no-op run stats (`kind`, `requested_seconds`,
`actual_seconds`, `rss_start_bytes`, `rss_end_bytes`, `rss_growth_mb`,
`csv_path`, `csv_rows`, `snapshots`, `consumed`, `dropped`, `max_mailbox_depth`,
`loop_rate_hz`, `producer_rate_hz`, `iter_latency_ms_p50/p95/max`,
`iter_latency_samples`, `last_iter_latency_ms`, `error`).

### Clean install (`py -3.11 -m venv .venv` + `pip install -e ".[dev]"`)

Run twice from an empty `.venv`; both succeeded. `pip check` -> "No broken
requirements found." `requirements.lock.txt` holds **27** pinned packages,
produced by `pip freeze --exclude-editable` from that install.

### API — code-level check on this machine (`python scripts\run_app.py --profile dev`)

- `GET /health` -> `200 {"status":"ok","uptime_s":2.641,"mode":"realtime","version":"0.1.0"}`
- `GET /api/config` -> `200` with the resolved `dev` config JSON (`profile`, `mode`, `results_dir`, `logging`, `source`, `consumer`, `broadcast`, `api`, `noop`)
- `GET /` -> `200`, `content-type: text/html`
- `WS /ws/state` -> 5 messages received, `snapshot_id` = 198, 201, 203, 205, 207 (strictly increasing; gaps are the last-value-wins broadcaster at 10 Hz skipping the 20 Hz consumer's intermediate snapshots), `mode = "realtime"`, `detections = []`, `metrics.loop_rate_hz` ≈ 20.
- `tests/integration/test_ws_snapshots.py::test_slow_client_is_dropped_without_slowing_the_loop` (part of the passing suite) attaches a client whose `send_text` never returns, confirms it is dropped after a 0.3 s timeout (`broadcaster.dropped_clients == 1`) and that the loop's snapshot rate stays within ±50 % of the configured 20 Hz before and after (`LOOP_RATE_TOLERANCE_FRAC = 0.5`, stated in the test).

## 2. Physically verified

None required in this phase.

## 3. Not verified

- **No camera** exists or has been exercised: no device access, no Media Foundation / DirectShow / `getUserMedia`, no `cv2.VideoCapture`. The only frame source is `SyntheticSource`.
- **No detector, pose model, tracker, feature builder, risk model, or alert policy** exists. `Detection`, `Pose`, `Track`, `Relation`, `RiskState`, `Alert`, `RiskLevel`, `TrackStatus` are contract types only and are never populated; `StateSnapshot.detections/poses/tracks` are always empty and `risk` is always `null`.
- **No audio path**: no TTS, no speaker output, no `pyttsx3` or any audio library.
- **No WebRTC**: no signalling, no peer connection, no `<video>` or `<canvas>` in the UI.
- **No overlay / rendering** of any scene content.
- **No recorded-video reader or writer**, and neither the `realtime` nor the `recorded` driver is implemented — `mode` is validated and carried only.
- **No model weights, no model download, no ONNX / OpenVINO / DirectML / CUDA code.**
- Nothing in this phase supports any claim about detection accuracy, tracking quality, real-world end-to-end latency, risk-prediction lead time, or spoken-alert behaviour. The latency figures above are the cost of an **empty** consumer iteration (`mailbox.get()` plus `StateSnapshot` construction), not of any perception work.
- The RSS-growth, loop-rate and mailbox-depth results are from single 60-second `dev`-profile runs on this one machine. They are measurements of this skeleton, not a benchmark, and no performance target was set for them.

## 4. Deviations and decisions

- **Python 3.11 was installed on request.** The machine had only Python 3.13 and 3.14; per the Block 1 requirement Python 3.11.9 was installed and the entire phase was built and measured against it. No code change resulted.
- **Sub-millisecond iteration latency is timed with `time.perf_counter()`, not `time.monotonic()`.** This is a deliberate departure from the prompt's literal "monotonic() only" wording (Block 16.12), forced by a direct conflict with Block 10 ("measure loop iteration latency, p50 and p95"). On this build `time.get_clock_info("monotonic")` is `GetTickCount64()` with **15.625 ms** resolution, which reports every empty iteration as `0.0 ms` — the measuring apparatus cannot do its job. `time.perf_counter()` is `QueryPerformanceCounter()` with ~100 ns resolution and is reported by `get_clock_info` as `monotonic=True`; it is not wall-clock `time.time()`. All coarse-grained timing (frame pacing, `capture_ts`, rate windows, uptime, run deadline) stays on `time.monotonic()`. `time.time()` is used nowhere for a duration. Recorded in `docs/decisions.md`.
- **pytest uses a project-local `--basetemp=.pytest_tmp`** (set in `pyproject.toml`, git-ignored). The shared system temp root (`%LOCALAPPDATA%\Temp`) is not reliably writable in this environment: a stale, permission-locked `pytest-of-mummi` directory made the default `tmp_path` fixture fail with `WinError 5`. This only redirects pytest's scratch root; no contract, module, or command changes.
- **`scripts/run_noop.py` configures logging at `INFO`** regardless of the profile's `logging.level`, so the measurement summary is always emitted (the `eval` profile is otherwise `WARNING`).
- **Contracts are frozen Pydantic v2 models** rather than dataclasses (a choice the prompt left open); rationale in `docs/decisions.md`.
- No change to the Block 5 module layout, the Block 8 contracts, or any Block 12 command. No Block 7 file was created; no Block 16 prohibition was breached beyond the documented `perf_counter` clock choice above.

## 5. Open questions for the developer

- The stale `%LOCALAPPDATA%\Temp\pytest-of-mummi` directory cannot be removed by this user (owned/locked by another sandbox context). It is worked around with `--basetemp`, not fixed. Outside this sandbox you may want to delete it and drop the override.
- Confirm the `eval` profile values in `config/profiles/eval.yaml` before P1 consumes them: `mode: recorded`, 1280×720, `source.target_fps: 15`, `consumer.sample_rate_hz: 15`, `broadcast.rate_hz: 5`, `api.port: 8001`, `logging.level: WARNING`.
- The `perf_counter`-vs-`monotonic` clock decision (section 4) is the one place this phase does not follow the prompt to the letter. If you would rather the loop report `0.0 ms` latencies and stay literally on `monotonic()`, say so and it will be changed.
- No other ambiguity was hit.
