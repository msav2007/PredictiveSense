# PredictiveSense — Phase 1 Implementation Prompt (Claude Code)

> Paste this entire file into Claude Code in a **fresh session**. Do not summarise it.
> **Scope: Phase 1 is the input layer.** No detection, no pose, no tracking, no risk, no voice.

---

## BLOCK 1 — CONTEXT

PredictiveSense is a local-only research prototype with a one-month deadline, supporting a research paper. It observes an indoor cabin scene, will eventually track objects and people over time, and predicts developing safety risks early enough to warn the user. It has **two input modes** sharing one core: **Mode A** (real-time camera) and **Mode B** (recorded video, analysed retrospectively).

**Phase 0 is complete and must not be rebuilt.** The repository already contains: typed frozen contracts (`core/types.py`, `core/enums.py`), strict YAML config (`dev`, `eval` profiles), `SyntheticSource`, `LatestFrameMailbox` (single-slot, exact drop accounting), a no-op analysis loop, telemetry (Counter / Rate / Samples / Timer / MetricsWriter / SessionManifest), a FastAPI app with `/health`, `/api/config`, `WS /ws/state` and a minimal static page, 47 passing tests, and `CLAUDE.md`. Read `CLAUDE.md` and `docs/architecture.md` first. **Reuse these components; do not reimplement them.**

**Target machine:** Windows, Intel Core Ultra 5 125H, 16 GB RAM, Intel Arc integrated graphics + NPU, **no NVIDIA GPU, no CUDA**. Python 3.11.9 in `.venv`. PowerShell.

**Cameras available to the developer:** laptop integrated camera, and a OnePlus Nord 4 exposed to Windows as a virtual camera (over Wi-Fi via Phone Link, and possibly over USB). A USB webcam may also be attached.

---

## BLOCK 2 — OBJECTIVE

At the end of Phase 1: the user can open the dashboard, **see and choose among the camera devices Windows exposes**, and watch a live preview that is provably unaffected by a slow analysis loop. Analysis frames reach the Python backend through a path that is physically separate from the preview. The same backend can also drive a **recorded video file** deterministically, processing every frame with no drops. The user can **record raw clips to disk with a manifest**, so dataset collection can begin immediately. Capture, preview, analysis, drop and frame-age metrics are measured and displayed.

No object detection, pose, tracking, temporal state, risk model, alert policy, voice, or annotation tooling exists yet.

---

## BLOCK 3 — REQUIREMENTS

### 3.1 Camera enumeration and selection

1. The dashboard page enumerates cameras with `navigator.mediaDevices.enumerateDevices()` after permission is granted, and presents a `<select>` of device labels. Selection persists in `localStorage`.
2. The backend independently enumerates devices for headless use: probe OpenCV indices `0..9`, and resolve human-readable names via `pygrabber` when it imports successfully. If `pygrabber` is unavailable or throws, fall back to `Camera <index>` and log once at INFO — this must never be fatal.
3. `GET /api/cameras` returns the backend's enumeration: `[{index, name, available, backend}]`.

### 3.2 Camera ownership — exactly one owner at a time

4. Config field `capture.owner` accepts `browser` (default) or `backend`. Windows will not reliably share a camera device between two processes, so the two are mutually exclusive and this must be enforced: if `owner=browser`, the backend must never open a device; if `owner=backend`, the page must not call `getUserMedia` and must display "preview unavailable — backend owns the camera" rather than an empty panel.

### 3.3 Preview path — untouchable

5. In `browser` ownership the preview is a `<video>` element fed by `srcObject` from `getUserMedia`. **The video element is never read from, never drawn into a canvas for preview purposes, and never replaced.** No JPEG polling, no HTTP frame polling, no backend-rendered video streaming, no canvas preview substitution.
6. Preview FPS is measured in the page with `requestVideoFrameCallback` and displayed in the metrics strip.

### 3.4 Analysis path — separate from preview

7. Analysis frames are produced in a **Web Worker**, never on the page's main thread. Use `MediaStreamTrackProcessor` where available; fall back to `requestVideoFrameCallback` + `OffscreenCanvas` transferred to the worker if it is not.
8. The worker downscales each sampled frame to `capture.analysis_width` × `capture.analysis_height` (default 640×480), encodes JPEG at `capture.analysis_jpeg_quality` (default 0.7), and sends it as a **binary WebSocket message** to `WS /ws/ingest` at `capture.analysis_fps` (default 10). This is a *push* path for analysis only; it is not the preview and it is not polling.
9. Each binary message is framed as: a 4-byte big-endian header length, then a UTF-8 JSON header `{"client_ts_ms": float, "seq": int, "w": int, "h": int}`, then the JPEG bytes.
10. The backend decodes with `cv2.imdecode` and constructs a `Frame`, then `put()`s it into the existing `LatestFrameMailbox`. Malformed messages are counted and dropped, never fatal.
11. **Clock offset.** On ingest-socket open, the client sends `{"type":"hello","client_ts_ms":<epoch ms>}` and the server replies `{"type":"hello_ack","server_mono":<time.monotonic()>,"rtt_probe":<id>}`; the client immediately echoes, and the server records `offset = server_mono - client_epoch_ms/1000` together with the measured round-trip time. `Frame.capture_ts` is then `client_ts_ms/1000 + offset`. Store the RTT as `clock_offset_rtt_ms` in the session manifest — **frame-age numbers are only as good as this offset, and its error must be reported, not hidden.**

### 3.5 Backend-owned source (headless)

12. `DeviceSource(index, backend=...)` using `cv2.VideoCapture`. Default `cv2.CAP_MSMF`, automatic fallback to `cv2.CAP_DSHOW` if the first frame does not arrive within `capture.open_timeout_s`; log which backend won. Requested resolution and FPS come from config; **record what the device actually returned, and never report the requested value as if it were achieved.**
13. Capture runs on its own thread and `put()`s into the mailbox. It must never block on the consumer.
14. Reconnect with bounded exponential backoff (config: `capture.reconnect_initial_s`, `capture.reconnect_max_s`) when reads fail. Reconnect attempts and durations are counted.

### 3.6 Mode B — recorded video driver

15. `FileSource(path)` decodes a video file sequentially with OpenCV. Two replay modes: `realtime` (paced to the file's own timestamps) and `asfast` (no pacing). `capture_ts` derives from the file's presentation timestamps, not wall clock.
16. `RecordedDriver` processes **every frame with no drops** — it must not use the mailbox. It is deterministic: the same file plus the same config produces byte-identical output.
17. `GET /api/videos` lists video files under `data/videos/`. `POST /api/analyze` accepts `{path, replay_mode}` and runs the recorded driver, writing one JSON object per frame to `results/recorded_<run_id>.jsonl` plus a run manifest. In Phase 1 each line contains only what exists: `frame_id`, `capture_ts`, `pts_s`, and empty `detections`/`poses`/`tracks`. Event analysis and findings reports are later phases.

### 3.7 Raw clip recorder — the reason this phase exists now

18. The page records the **full-quality preview stream** with `MediaRecorder` (not the downscaled analysis frames) while a Record button is active, showing a visible recording indicator and elapsed time.
19. On stop, the blob is POSTed to `POST /api/record/upload` (multipart) with metadata: `scenario_tag`, `device_label`, `width`, `height`, `nominal_fps`, `notes`, `consent_ack` (boolean). The backend writes the video to `data/raw/<session_id>/<clip_id>.webm` and a sibling `<clip_id>.json` manifest containing that metadata plus UTC timestamp, git commit, config profile, and file size and duration.
20. `GET /api/clips` lists recorded clips with their manifests. Reject uploads over `recorder.max_clip_mb` (default 500) with a clear error.
21. `data/` stays git-ignored. Clips are never committed.

### 3.8 Metrics

22. Measure and expose in `StateSnapshot.metrics` and the metrics strip: `capture_fps`, `analysis_fps`, `preview_fps` (page-side), `dropped_analysis_frames`, `drop_rate`, `mailbox_depth`, `frame_age_ms`, `decode_ms`, `ingest_bytes_per_s`, `reconnects`.
23. **Do not call `Samples.percentile()` on the analysis loop's hot path** — it sorts up to 100 000 values under a lock. Percentiles are computed at report time only.

### 3.9 Transport benchmark

24. `scripts/benchmark_transport.py --index N --seconds S` opens a device backend-owned and reports: achieved resolution, achieved FPS, frame-interval p50/p95/max, time to first frame, and read-failure count, written to `results/transport_<label>.json`. The developer runs it once per transport; the comparison table is assembled in the phase report.

---

## BLOCK 4 — ARCHITECTURE CONSTRAINTS

- **Preview independence is the defining constraint of this phase.** A stalled analysis loop must not change preview FPS. This is proved by an automated test and by physical observation.
- **Mailbox stays single-slot** for Mode A. Mode B must not use it.
- **Contracts may be extended, never reshaped.** Every addition to `core/types.py` or `core/enums.py` gets a line in `docs/decisions.md`.
- **`cv2` is permitted only under `predictivesense/camera/`.** Update the forbidden-import test to enforce exactly that; every other ban stays.
- No unbounded queues, lists or caches anywhere. Ingest, decode and recording paths all have explicit caps.
- No CUDA, no NVIDIA assumptions, no cloud calls, no runtime downloads.
- Do not overengineer: no aiortc, no WebRTC, no Docker, no database, no UI framework, no build step for the frontend — plain HTML/JS modules served statically.
- `time.monotonic()` for coarse timing; `time.perf_counter()` only for sub-millisecond intervals, as established in Phase 0.

---

## BLOCK 5 — FILES TO CREATE

```
predictivesense/camera/device.py            # DeviceSource (OpenCV, MSMF/DSHOW fallback, reconnect)
predictivesense/camera/file_source.py       # FileSource (sequential decode, pts-based capture_ts)
predictivesense/camera/browser.py           # BrowserSource fed by the ingest socket
predictivesense/camera/enumerate.py         # backend device enumeration (+ optional pygrabber)
predictivesense/camera/framing.py           # binary ingest message encode/decode + clock offset
predictivesense/pipeline/recorded.py        # RecordedDriver: lossless, deterministic, no mailbox
predictivesense/api/ingest.py               # WS /ws/ingest endpoint
predictivesense/api/recorder.py             # clip upload, listing, manifests
predictivesense/api/videos.py               # video listing + /api/analyze
predictivesense/api/static/app.js           # page logic: device select, preview, metrics, record
predictivesense/api/static/analysis-worker.js  # worker: sample, downscale, encode, send
scripts/benchmark_transport.py
scripts/run_recorded.py
tests/unit/test_framing.py
tests/unit/test_clock_offset.py
tests/unit/test_file_source.py
tests/unit/test_enumerate.py
tests/integration/test_ingest_socket.py
tests/integration/test_recorded_driver.py
tests/integration/test_recorder_upload.py
tests/integration/test_preview_independence.py
tests/hardware/test_device_source.py        # marked `hardware`, skipped by default
tests/fixtures/make_fixture_video.py        # generates a tiny deterministic test clip
docs/phase-reports/phase1.md
data/videos/.gitkeep
data/raw/.gitkeep
```

## BLOCK 6 — FILES TO MODIFY

```
pyproject.toml                      # add opencv-python, python-multipart; pygrabber as optional
requirements.lock.txt               # regenerate; write UTF-8 without BOM
predictivesense/core/enums.py       # SourceKind: make DEVICE, FILE, BROWSER constructible
predictivesense/core/types.py       # add SourceInfo, ClipManifest, IngestHeader; extend StateSnapshot.metrics keys only
predictivesense/config/settings.py  # add capture.*, recorder.*, video.* sections
config/profiles/dev.yaml            # new sections with defaults
config/profiles/eval.yaml           # same, tuned for deterministic offline runs
predictivesense/camera/source.py    # only if the interface needs a documented addition
predictivesense/pipeline/loop.py    # accept any FrameSource; wire new metrics
predictivesense/api/app.py          # mount ingest, recorder, videos routers; serve static
predictivesense/api/static/index.html  # device select, video element, metrics strip, record button
tests/unit/test_no_forbidden_imports.py  # allow cv2 under camera/ only
CLAUDE.md                           # module map, commands, current phase status
docs/architecture.md                # ingest protocol, ownership model, clock offset
docs/decisions.md                   # contract additions; browser-worker ingest instead of WebRTC
docs/attribution.md                 # opencv-python, python-multipart, pygrabber + licences
```

## BLOCK 7 — MUST NOT BE CREATED OR MODIFIED

Do not create, even as an empty placeholder: `predictivesense/perception/`, `tracking/`, `scene/`, `temporal/`, `risk/`, `policy/`, `audio/`, `research/`. No detector, pose estimator, tracker, feature builder, risk model, alert policy, or TTS code. No bounding-box or skeleton drawing — there are no detections to draw. No annotation tooling (Phase 5). No aiortc or WebRTC. No Dockerfile, CI workflow, database, or cloud SDK.

Do not modify: `predictivesense/camera/mailbox.py`, `predictivesense/camera/synthetic.py`, `predictivesense/telemetry/*`, or `PredictiveSense-Phase0-Prompt.md`. If one of these genuinely blocks you, **stop and ask**.

---

## BLOCK 8 — CONTRACTS

**Config additions**

```yaml
capture:
  owner: browser            # browser | backend
  device_index: 0
  device_backend: auto      # auto | msmf | dshow
  request_width: 1280
  request_height: 720
  request_fps: 30
  open_timeout_s: 5.0
  reconnect_initial_s: 0.5
  reconnect_max_s: 8.0
  analysis_fps: 10
  analysis_width: 640
  analysis_height: 480
  analysis_jpeg_quality: 0.7
  max_ingest_message_bytes: 2000000
recorder:
  enabled: true
  max_clip_mb: 500
  output_dir: data/raw
video:
  input_dir: data/videos
  replay_mode: asfast       # realtime | asfast
```

**`SourceInfo`** — `kind: SourceKind`, `source_id: str`, `label: str`, `width: int`, `height: int`, `achieved_fps: float | None`, `backend: str | None`, `extra: dict[str, str]`.

**`ClipManifest`** — `clip_id: str`, `session_id: str`, `path: str`, `scenario_tag: str`, `device_label: str`, `width: int`, `height: int`, `nominal_fps: float`, `duration_s: float | None`, `size_bytes: int`, `recorded_utc: str`, `git_commit: str`, `config_profile: str`, `consent_ack: bool`, `notes: str`.

**`IngestHeader`** — `client_ts_ms: float`, `seq: int`, `w: int`, `h: int`.

**Endpoints**

| Method | Path | Behaviour |
|---|---|---|
| GET | `/api/cameras` | Backend enumeration |
| WS | `/ws/ingest` | Binary analysis frames + hello/hello_ack handshake |
| GET | `/api/videos` | Video files under `data/videos/` |
| POST | `/api/analyze` | `{path, replay_mode}` → runs RecordedDriver, returns `{run_id, jsonl_path, frames}` |
| POST | `/api/record/upload` | multipart clip + metadata → `ClipManifest` |
| GET | `/api/clips` | List clip manifests |

Existing `/health`, `/api/config` and `/ws/state` keep their Phase 0 shapes.

---

## BLOCK 9 — ERROR HANDLING

**Fail loudly, exit non-zero:** `owner=backend` with an unopenable device index; a video path outside `data/videos/`; an unwritable `data/raw/`; both owners active simultaneously.

**Degrade gracefully, log once:** `pygrabber` missing or throwing; a malformed ingest message (counted, dropped); a camera read failure (reconnect with backoff); an oversize upload (rejected with a clear HTTP error); a disconnected ingest client (the loop keeps running and snapshots go `stale=true`).

**Never:** let an ingest error kill the analysis loop; block the capture thread on a consumer; swallow an exception silently; use a bare `except:`.

---

## BLOCK 10 — PERFORMANCE REQUIREMENTS

Measure, do not target: capture FPS, preview FPS, analysis FPS, decode ms, frame age, drop rate, ingest bytes/s, reconnect time. Report p50 and p95 with the machine and the transport named.

Two assertions to verify by measurement:

1. A 10-minute run does not grow RSS by more than 50 MB (record the actual number).
2. With the analysis consumer artificially stalled for 3 seconds, page-side preview FPS stays within 10% of its pre-stall value.

**No performance figure may appear anywhere that was not produced by a command you ran on this machine.**

---

## BLOCK 11 — TESTS

Markers: `unit`, `integration`, `slow`, `hardware`. The `hardware` set is skipped unless `--run-hardware` is passed.

**Unit**

1. `test_framing` — header/payload round-trip; truncated, oversize and non-JPEG payloads are rejected without raising out of the handler.
2. `test_clock_offset` — offset arithmetic on synthetic hello/ack exchanges; `capture_ts` monotonic across a simulated sequence; RTT recorded.
3. `test_file_source` — replays the fixture clip with the expected frame count and ordering; `capture_ts` derives from pts; `asfast` and `realtime` yield identical frame sequences; seek/restart is deterministic.
4. `test_enumerate` — with `pygrabber` importable and monkeypatched, names resolve; with it raising, the fallback naming is used and nothing propagates.
5. Amended `test_no_forbidden_imports` — `cv2` allowed only under `camera/`; a synthetic violation elsewhere is detected.

**Integration**

6. `test_ingest_socket` — a fake client performs the handshake and sends N encoded frames; the backend decodes them, the mailbox receives them, `frame_id` increases, and `frame_age_ms` is finite and positive.
7. `test_recorded_driver` — every frame of the fixture clip is processed, zero drops, and two consecutive runs produce byte-identical JSONL.
8. `test_recorder_upload` — a small synthetic blob uploads, lands under `data/raw/`, and produces a manifest with every required key; an oversize upload is rejected.
9. `test_preview_independence` — ingest runs while the analysis consumer is stalled for 3 s; assert the ingest socket keeps accepting frames, mailbox drops rise, mailbox depth never exceeds 1, and neither the producer nor the socket blocks.

**Hardware (skipped by default)**

10. `test_device_source` — opens `capture.device_index`, reads 30 frames, reports the achieved backend, resolution and FPS.

**Fixture:** `tests/fixtures/make_fixture_video.py` generates a small deterministic clip (e.g. 3 s, 320×240, a moving rectangle) written to `tests/fixtures/`. Generate it in `conftest` if absent; do not commit a binary if generation is reliable.

---

## BLOCK 12 — EXACT COMMANDS

Working directory: `C:\Users\mummi\Documents\Projects\PredictiveSense`

```powershell
cd C:\Users\mummi\Documents\Projects\PredictiveSense
.\.venv\Scripts\Activate.ps1

# Install new dependencies
pip install -e ".[dev]"
pip freeze --exclude-editable | Out-File -Encoding utf8NoBOM requirements.lock.txt

# Tests
pytest -q
pytest -q -m "not slow"
pytest -q -m hardware --run-hardware        # only with a camera attached

# Run the app (browser owns the camera)
python scripts\run_app.py --profile dev

# Backend-owned transport benchmark, one per transport
python scripts\benchmark_transport.py --index 0 --seconds 30 --label laptop-cam
python scripts\benchmark_transport.py --index 1 --seconds 30 --label oneplus-wifi

# Recorded-video analysis
python scripts\run_recorded.py --path data\videos\<clip>.mp4 --replay-mode asfast

# 10-minute memory check
python scripts\run_noop.py --profile dev --seconds 600
```

Report the verbatim output of `pytest -q`, one benchmark run, and the 10-minute run.

---

## BLOCK 13 — PHYSICAL VERIFICATION

**Required this phase. You (Claude Code) cannot perform these — mark them "not verified" and list them for the developer.**

1. Open the dashboard; confirm the camera dropdown lists the laptop camera and the OnePlus virtual camera by name.
2. Select each in turn; confirm the preview appears and is smooth; wave a hand and judge the lag.
3. With the preview running, trigger the analysis stall (provide a documented way to do this — a config flag or a debug endpoint) and confirm **by eye** that the preview does not stutter while the drop rate climbs.
4. Record a 20-second clip with a scenario tag; confirm the file and manifest appear under `data/raw/`.
5. Run the transport benchmark for each available transport and record the results.
6. Disconnect the phone camera mid-run; confirm the UI shows a degraded state and recovers when reconnected.

---

## BLOCK 14 — REPORT FORMAT

Write `docs/phase-reports/phase1.md`:

```markdown
# Phase 1 — Input Layer — Report
**Date / Commit / Machine / Python:**

## 1. Measured
<Numbers with the command that produced them: pytest summary; capture/analysis FPS; decode ms
p50/p95; frame age p50/p95 with the clock-offset RTT; drop rate; ingest bytes/s; 10-minute RSS
delta; recorded-driver frame count and determinism check.>

## 2. Physically verified
<Only what the developer actually did. If they have not yet, write "Pending developer verification"
and list the six checks.>

## 3. Not verified
<Must not be empty. Include: no detection, pose, tracking, temporal, risk, or voice exists; preview
smoothness by eye; real camera behaviour; transport comparison until benchmarks are run.>

## 4. Deviations and decisions
## 5. Open questions for the developer
```

Then reply in this format only:

```
IMPLEMENTED: ...
TESTS: ...
MEASUREMENTS: ...
LIMITATIONS: ...
NEXT PHASE: ...
```

---

## BLOCK 15 — ACCEPTANCE CHECKLIST

- [ ] `pytest -q` passes; hardware tests skipped by default and runnable on demand.
- [ ] `/api/cameras` enumerates devices; the page lists them by label and selection persists.
- [ ] Preview is a native `srcObject` video element; no polling, no canvas substitution anywhere in the code.
- [ ] Analysis frames arrive over `/ws/ingest` from a Web Worker; mailbox stays single-slot; drops counted.
- [ ] `test_preview_independence` passes: a 3-second consumer stall does not block ingest or the producer.
- [ ] Clock offset established with RTT recorded; `frame_age_ms` finite and reported with its error.
- [ ] `RecordedDriver` processes every frame with zero drops and is byte-deterministic across two runs.
- [ ] Clip recorder writes video + manifest under `data/raw/`; `data/` remains git-ignored.
- [ ] Transport benchmark script runs and writes `results/transport_<label>.json`.
- [ ] 10-minute run: RSS growth under 50 MB (actual number recorded).
- [ ] `cv2` appears only under `camera/`; all other import bans still enforced.
- [ ] Contract additions recorded in `docs/decisions.md`; `CLAUDE.md` updated; attribution updated with licences.
- [ ] `docs/phase-reports/phase1.md` complete; working tree clean and committed; no remote.

---

## BLOCK 16 — PROHIBITIONS (MANDATORY)

1. Do not invent requirements. If genuinely blocked, **stop and ask**; otherwise decide from `CLAUDE.md` and this prompt and record the decision.
2. Do not rebuild Phase 0 components — reuse the mailbox, telemetry, config, contracts and loop as they are.
3. Do not implement Phase 2 or later: no detector, pose, tracker, temporal state, risk model, policy, voice, or overlay boxes.
4. Do not create placeholder modules for future phases.
5. Do not claim a performance number you did not measure on this machine.
6. Do not claim physical verification — that is the developer's, and this phase has six pending checks.
7. Do not implement the preview with JPEG polling, HTTP frame polling, backend video streaming, or canvas replacement.
8. Do not create an unbounded queue, buffer or cache.
9. Do not add aiortc, WebRTC, Docker, a database, a UI framework, or a frontend build step.
10. Do not add any dependency beyond opencv-python, python-multipart and optional pygrabber without asking.
11. Do not write CUDA-specific or GPU-assuming code.
12. Do not make outbound network calls or download anything at runtime.
13. Do not commit anything under `data/` or `results/`.
14. Do not use `print()`, a bare `except:`, or `time.time()` for durations.
15. Do not call `Samples.percentile()` on the analysis hot path.
16. Do not mark an acceptance item complete without running the check.

When Phase 1 is finished, **stop**. Do not begin Phase 2.
