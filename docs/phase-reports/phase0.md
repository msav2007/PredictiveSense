# Phase 0 — Foundation — Report

**Date:** 2026-09-07 (UTC)
**Commit:** 99e0ae311c2a320ef4c4429d9fd389d8097b32df (measurements taken against this commit; this report is committed on top of it)
**Machine:** 18 logical CPUs / 16,766,066,688 bytes RAM (~15.6 GiB) / `Windows-10-10.0.26200-SP0` (values from the session manifest; the host is an Intel Core Ultra 5 125H with an integrated Intel Arc GPU, no NVIDIA GPU, no CUDA)
**Python:** 3.11.9

## 1. Measured

### `pytest -q` (full suite, `python -m pytest -q`)

```
...............................................                          [100%]
============================== warnings summary ===============================
.venv\Lib\site-packages\starlette\testclient.py:40
  C:\Users\mummi\Documents\Projects\PredictiveSense\.venv\Lib\site-packages\starlette\testclient.py:40: DeprecationWarning: The anyio.abc.BlockingPortal alias is deprecated, use anyio.from_thread.BlockingPortal instead.
    _PortalFactoryType = typing.Callable[[], typing.ContextManager[anyio.abc.BlockingPortal]]

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
47 passed, 1 warning in 16.70s
```

- **47 passed, 0 failed, 0 errors**, 1 warning, in **16.70 s**.
- The single warning originates in `starlette/testclient.py` (an `anyio` alias deprecation in a third-party package), not in `predictivesense/`.
- `pytest -q -m "not slow"`: **46 passed, 1 deselected** in 6.48 s.

### 60-second instrumented no-op run (`python scripts\run_noop.py --profile dev --seconds 60`)

```
2026-09-07T10:33:13+0530 INFO    predictivesense.scripts.run_noop | no-op run start session=bf8a84a83041442ba1a8c58d5bf0cc2c profile=dev seconds=60 results_dir=results
2026-09-07T10:33:14+0530 INFO    predictivesense.pipeline.loop | analysis loop starting profile=dev mode=realtime sample_rate=20.00 Hz
2026-09-07T10:33:14+0530 INFO    predictivesense.camera.synthetic | synthetic source started source_id=synthetic-12345-640x480 session=0 640x480 @ 30.00 fps seed=12345
2026-09-07T10:34:14+0530 INFO    predictivesense.camera.synthetic | synthetic source stopped source_id=synthetic-12345-640x480 frames_produced=1802
2026-09-07T10:34:14+0530 INFO    predictivesense.pipeline.loop | analysis loop stopped snapshots=1201 consumed=1200 dropped=601 max_mailbox_depth=0 loop_hz=19.99 producer_hz=30.00 error=None
2026-09-07T10:34:14+0530 INFO    predictivesense.telemetry.manifest | session manifest written: results\manifest_bf8a84a83041442ba1a8c58d5bf0cc2c.json
2026-09-07T10:34:14+0530 INFO    predictivesense.telemetry.writer | metrics csv written: results\noop_bf8a84a83041442ba1a8c58d5bf0cc2c.csv (61 rows)
2026-09-07T10:34:14+0530 INFO    predictivesense.scripts.run_noop | === no-op run summary ===
2026-09-07T10:34:14+0530 INFO    predictivesense.scripts.run_noop | session_id           : bf8a84a83041442ba1a8c58d5bf0cc2c
2026-09-07T10:34:14+0530 INFO    predictivesense.scripts.run_noop | profile / mode       : dev / realtime
2026-09-07T10:34:14+0530 INFO    predictivesense.scripts.run_noop | requested seconds    : 60
2026-09-07T10:34:14+0530 INFO    predictivesense.scripts.run_noop | snapshots emitted    : 1201
2026-09-07T10:34:14+0530 INFO    predictivesense.scripts.run_noop | frames consumed      : 1200
2026-09-07T10:34:14+0530 INFO    predictivesense.scripts.run_noop | frames dropped       : 601
2026-09-07T10:34:14+0530 INFO    predictivesense.scripts.run_noop | loop rate (Hz)       : 19.988
2026-09-07T10:34:14+0530 INFO    predictivesense.scripts.run_noop | producer rate (Hz)   : 30.000
2026-09-07T10:34:14+0530 INFO    predictivesense.scripts.run_noop | iter latency p50 (ms): 0.0399
2026-09-07T10:34:14+0530 INFO    predictivesense.scripts.run_noop | iter latency p95 (ms): 0.0722
2026-09-07T10:34:14+0530 INFO    predictivesense.scripts.run_noop | iter latency max (ms): 0.3009
2026-09-07T10:34:14+0530 INFO    predictivesense.scripts.run_noop | max mailbox depth    : 0  (<=1 required: True)
2026-09-07T10:34:14+0530 INFO    predictivesense.scripts.run_noop | RSS start (MB)        : 49.893
2026-09-07T10:34:14+0530 INFO    predictivesense.scripts.run_noop | RSS end (MB)          : 50.999
2026-09-07T10:34:14+0530 INFO    predictivesense.scripts.run_noop | RSS growth (MB)      : 1.106  (<=25.0 required: True)
2026-09-07T10:34:14+0530 INFO    predictivesense.scripts.run_noop | csv                  : results\noop_bf8a84a83041442ba1a8c58d5bf0cc2c.csv (61 rows)
2026-09-07T10:34:14+0530 INFO    predictivesense.scripts.run_noop | manifest             : results\manifest_bf8a84a83041442ba1a8c58d5bf0cc2c.json
2026-09-07T10:34:14+0530 INFO    predictivesense.scripts.run_noop | PASS: no-op run within budget
```

Numbers from that run:

| Measure | Value |
|---|---|
| Consumer loop rate | **19.988 Hz** (configured sample rate 20 Hz) |
| Producer frame rate | **30.000 Hz** (configured target 30 fps) |
| Frames consumed / dropped | **1200 / 601** (producer 30 Hz vs consumer 20 Hz; every unconsumed frame is dropped by the single-slot mailbox) |
| Snapshots emitted | **1201** |
| Iteration latency p50 | **0.0399 ms** |
| Iteration latency p95 | **0.0722 ms** |
| Iteration latency max | **0.3009 ms** (1201 samples) |
| Process RSS at start | **49.893 MB** |
| Process RSS at end | **50.999 MB** |
| RSS growth | **1.106 MB** (assertion: < 25 MB — pass) |
| Max mailbox depth observed | **0** (assertion: <= 1 — pass) |
| Script exit code | **0** |

### Metrics CSV

`results/noop_<session>.csv` — 1 header row + 61 data rows. Header:

```
elapsed_s,wall_utc,snapshot_id,frame_id,stale,loop_rate_hz,producer_rate_hz,consumed,dropped,mailbox_depth,iter_latency_ms,rss_mb
```

### Session manifest

`results/manifest_<session>.json` top-level keys:

```
session_id, started_utc, ended_utc, git_commit, git_dirty, config,
python_version, platform, cpu_count, total_ram_bytes, seed, extra
```

Selected values from this run: `git_commit = 99e0ae311c2a320ef4c4429d9fd389d8097b32df`,
`git_dirty = false`, `python_version = 3.11.9`,
`platform = Windows-10-10.0.26200-SP0`, `cpu_count = 18`,
`total_ram_bytes = 16766066688`, `seed = 12345`, and `config` is the fully
resolved `dev` profile. `extra.run` carries the no-op run stats listed above.

### Clean install

`py -3.11 -m venv .venv` + `pip install -e ".[dev]"` from an empty environment
succeeded; `requirements.lock.txt` (27 pinned packages) was produced by
`pip freeze --exclude-editable` from that install.

### API (code-level check on this machine)

`python scripts\run_app.py --profile dev` served:
`GET /health` -> `{"status":"ok","uptime_s":<float>,"mode":"realtime","version":"0.1.0"}`;
`GET /api/config` -> 200 with the resolved `dev` config JSON;
`GET /` -> 200 `text/html`;
`WS /ws/state` -> five snapshots received with strictly increasing `snapshot_id`
(198, 201, 203, 205, 207), `mode = "realtime"`, empty `detections`, `loop_rate_hz` ≈ 20.
`test_ws_snapshots.py` additionally verifies that a stalled client is dropped
after the send timeout while the loop's snapshot rate stays within
±50% of the configured 20 Hz before and after.

## 2. Physically verified

None required in this phase.

## 3. Not verified

- **No camera** exists or has been exercised: no device access, no Media Foundation / DirectShow / `getUserMedia`, no `cv2.VideoCapture`. The only frame source is `SyntheticSource`.
- **No detector, pose model, tracker, feature builder, risk model, or alert policy** exists. `Detection`, `Pose`, `Track`, `Relation`, `RiskState`, `Alert`, `RiskLevel` and `TrackStatus` are defined as contracts only and are never populated; `StateSnapshot.detections/poses/tracks` are always empty and `risk` is always `null`.
- **No audio path**: no TTS, no speaker output, no `pyttsx3` or any audio library.
- **No WebRTC**: no signalling, no peer connection, no video element or canvas in the UI.
- **No overlay / rendering** of any scene content.
- **No recorded-video reader or writer**, and neither the `realtime` nor the `recorded` driver is implemented — `mode` is validated and carried only.
- **No model weights, no model download, no ONNX / OpenVINO / DirectML / CUDA code.**
- Nothing in this phase supports any claim about detection accuracy, tracking quality, real-world end-to-end latency, risk-prediction lead time, or spoken-alert behaviour. The latency figures above are the cost of an *empty* consumer iteration (mailbox `get` + snapshot construction), not of any perception work.
- The RSS-growth and mailbox-depth results are from a single 60-second `dev`-profile run on this one machine; they are not a benchmark and no target was set for them.

## 4. Deviations and decisions

- **Python 3.11 was installed on request.** The machine had only Python 3.13 and 3.14; per the Block 1 requirement the developer installed 3.11.9 and the whole phase was built and measured against it. No code change resulted.
- **pytest uses a project-local `--basetemp=.pytest_tmp`** (in `pyproject.toml`, gitignored). The shared system temp directory (`%LOCALAPPDATA%\Temp`) is not reliably writable in this environment: a stale, permission-locked `pytest-of-mummi` directory made the default `tmp_path` fixture fail with `WinError 5`. This only redirects pytest's temp root and does not change any contract or module.
- **`scripts/run_noop.py` configures logging at `INFO`** regardless of the profile's `logging.level`, so the measurement summary is always emitted (the `eval` profile is otherwise `WARNING`).
- **Contracts are Pydantic v2 models rather than dataclasses** (a choice the prompt left open); rationale in `docs/decisions.md`.
- No change to the Block 5 module layout, the Block 8 contracts, or any Block 12 command. No file from Block 7 was created.

## 5. Open questions for the developer

- The stale `%LOCALAPPDATA%\Temp\pytest-of-mummi` directory cannot be removed by this user (owned/locked by another sandbox context). It is worked around, not fixed. If you later run pytest outside this sandbox you may want to delete it or keep the `--basetemp` override.
- `config/profiles/eval.yaml` currently mirrors `dev` with different rates, resolution (1280x720), `mode: recorded`, port 8001, and `logging.level: WARNING`. Confirm those `eval` values are what you want before P1 starts consuming them.
- No other ambiguity was hit.
