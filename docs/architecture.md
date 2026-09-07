# Architecture (Phase 0)

## Shape

```
SyntheticSource --frames--> LatestFrameMailbox --newest--> AnalysisLoop consumer
                                                              |
                                                              v
                                                        StateSnapshot
                                                         /          \
                                              MetricsWriter (CSV)   Broadcaster --> WS clients
```

Everything is injectable. There is no global mutable state, no singleton, and no
import-time side effect. Threads exist only in `predictivesense/pipeline/`.

## Threading model

- **Producer thread** (`ps-producer`): calls `SyntheticSource.read()` in a loop
  and `put()`s each frame into the mailbox. `read()` paces itself to the
  configured `source.target_fps`.
- **Consumer thread** (`ps-consumer`): wakes at `consumer.sample_rate_hz`, does
  one `mailbox.get()`, times the iteration, builds a `StateSnapshot`, and hands
  it to every registered listener.
- **Shutdown**: `AnalysisLoop.stop()` sets an `Event`, joins both threads with a
  5 s timeout, then stops the source. A thread still alive after the join, or a
  thread that raised, sets `AnalysisLoop.error`; `run_for()` re-raises it and the
  scripts exit non-zero.

## Bounded everywhere

- `LatestFrameMailbox` holds exactly one frame. `put()` on a full slot drops the
  waiting frame and increments `dropped`. Depth is always 0 or 1.
- The broadcaster keeps only the single most recent snapshot. Each WS client is
  served by a poll loop at `broadcast.rate_hz` that sends the current snapshot;
  intermediate snapshots for a client that cannot keep up are simply skipped
  (last-value-wins). A client whose send does not finish within
  `broadcast.client_send_timeout_s` is dropped. `publish()` is an O(1) lock plus
  assignment and never touches a client, so a slow client cannot slow the loop.
- `telemetry.Rate` and `telemetry.Samples` are `deque`s with a fixed `maxlen`
  (rate window buffer 4096; latency reservoir 100 000).

## Mode awareness

`mode` (`realtime` | `recorded`) is validated in config and carried on every
`StateSnapshot`. Only `predictivesense/pipeline/` may branch on it. Phase 0
implements neither driver; the field exists so later phases do not reshape the
config or the wire format.

## `StateSnapshot` wire format

Broadcast over `WS /ws/state` and returned (shape-compatible) by the loop. JSON
is produced by `StateSnapshot.to_wire_json()` (Pydantic v2 `model_dump_json`)
and parsed by `StateSnapshot.from_wire_json()`. Enums serialise to their string
values; `None` is JSON `null`.

| Field | JSON type | Phase 0 value |
|---|---|---|
| `snapshot_id` | number | monotonic from 1 |
| `mode` | string | `"realtime"` or `"recorded"` |
| `frame_id` | number \| null | id of the frame this snapshot describes, else null |
| `capture_ts` | number \| null | `time.monotonic()` at capture, else null |
| `emitted_ts` | number | `time.monotonic()` at emission |
| `frame_age_ms` | number \| null | `(emitted_ts - capture_ts) * 1000`, else null |
| `detections` | array | always `[]` |
| `poses` | array | always `[]` |
| `tracks` | array | always `[]` |
| `risk` | object \| null | always `null` |
| `metrics` | object (string -> number) | `loop_rate_hz`, `producer_rate_hz`, `consumed`, `dropped`, `mailbox_depth`, `iter_latency_ms` |
| `stale` | boolean | `true` when no frame arrived within `consumer.stale_after_ms` |

Later phases may add fields. Renaming or removing one requires a line in
`docs/decisions.md`.

## Metrics CSV schema

Written by `scripts/run_noop.py` via `telemetry.MetricsWriter`. One header row,
then one row per `noop.csv_sample_period_s` seconds:

```
elapsed_s, wall_utc, snapshot_id, frame_id, stale, loop_rate_hz,
producer_rate_hz, consumed, dropped, mailbox_depth, iter_latency_ms, rss_mb
```

## Session manifest

Written by `scripts/run_noop.py` via `telemetry.SessionManifest` as pretty JSON.
Keys: `session_id`, `started_utc`, `ended_utc`, `git_commit`, `git_dirty`,
`config` (resolved), `python_version`, `platform`, `cpu_count`,
`total_ram_bytes`, `seed`, plus an `extra.run` block with the no-op run stats.

## What does not exist yet

No perception, tracking, scene, temporal, risk, policy, audio, or research code.
No model weights. No Docker, CI, database, or message broker. See Block 7 of the
Phase 1 prompt. (Phase 1 adds the camera / recorded-video input layer described
below; it adds no detection, pose, tracking, temporal, risk or voice code.)

---

# Phase 1 - the input layer

## Ownership model

`capture.owner` is `browser` (default) or `backend`, and the two are mutually
exclusive - Windows will not reliably share a camera between processes.

- `owner=browser`: the **page** opens the camera with `getUserMedia`. The
  backend never touches a device. Analysis frames arrive over `WS /ws/ingest`.
- `owner=backend`: a **`DeviceSource`** opens the camera with OpenCV. The page
  does not call `getUserMedia` and shows "preview unavailable - backend owns the
  camera". `WS /ws/ingest` refuses connections (close 4403).

The live source is chosen by `source.kind` (`build_camera_source`):
`synthetic` -> `SyntheticSource` (Phase 0 path, still used by `run_noop`);
`browser` -> `BrowserSource`; `device` -> `DeviceSource`; `file` is rejected for
the live loop (recorded video goes through `RecordedDriver`).

## Preview path (browser) - untouchable

```
getUserMedia --MediaStream--> <video>.srcObject      (native rendering only)
                          \--> MediaStreamTrackProcessor.readable --transfer--> Web Worker
```

The `<video>` element is never read from, drawn into a canvas for display, or
replaced. Preview FPS is measured in the page with `requestVideoFrameCallback`.
No JPEG/HTTP polling, no backend video stream, no canvas substitution anywhere.
A stalled analysis consumer does not change preview FPS - proved by
`tests/integration/test_preview_independence.py` and, physically, by the
developer (`POST /api/debug/stall`).

## Analysis path (browser) - separate from preview

```
Web Worker: sample frame -> OffscreenCanvas downscale (analysis_w x analysis_h)
         -> JPEG (analysis_jpeg_quality) -> binary WS message @ analysis_fps
              |
              v
WS /ws/ingest handler: decode framing -> asyncio.to_thread(BrowserSource.submit)
         -> cv2.imdecode -> Frame -> BrowserSource single-slot buffer
              |
   AnalysisLoop producer thread: BrowserSource.read() -> LatestFrameMailbox.put()
```

Two single-slot newest-wins buffers in series (the browser-source buffer and the
Phase 0 mailbox), each with exact drop accounting. Nothing queues.

The **worker** is also bounded (Phase 1.5): at most one encode+send is in flight
(`encodeBusy`) - a frame arriving mid-encode is dropped (`skipBusy`), never
queued; before *and* after the encode it checks `ws.bufferedAmount` against
`capture.max_ws_buffered_bytes` (default 1 MB) and skips over the ceiling
(`skipBackpressure`); `drawImage` is synchronous so every `VideoFrame` /
`ImageBitmap` is closed immediately (`framesIn === framesClosed`, reported each
1 s with a `leaked` count). The rule is pinned by
`tests/unit/test_worker_backpressure_contract.py`.

### Ingest wire format

One binary message:

```
| 4 bytes: uint32 big-endian header length H | H bytes: UTF-8 JSON | JPEG bytes |
```

The JSON header is `IngestHeader`: `{"client_ts_ms": float, "seq": int, "w": int, "h": int}`.
Malformed messages (truncated, oversize, bad JSON, non-JPEG) are counted
(`malformed`) and dropped; they never raise into the analysis loop.

### Handshake and clock offset

```
client -> {"type":"hello","client_ts_ms": <epoch ms>}
server -> {"type":"hello_ack","server_mono": <time.monotonic()>,"rtt_probe": <id>}
client -> {"type":"echo","rtt_probe": <id>}                (echoed immediately)
```

`offset = server_mono - client_hello_ms/1000`; `rtt_ms = (echo_recv - ack_send) * 1000`.
`Frame.capture_ts = client_ts_ms/1000 + offset` maps the client's epoch clock
onto the server monotonic timeline, so `frame_age_ms = emitted_ts - capture_ts`
is meaningful. The RTT is **not** folded into the offset - it is recorded as the
error bar (`clock_offset_rtt_ms` in the snapshot metrics and, for `run_app`, in
`results/session_<id>.json`).

## Backend-owned source (`DeviceSource`)

`cv2.VideoCapture`, `CAP_MSMF` first, automatic fallback to `CAP_DSHOW` if no
frame arrives within `capture.open_timeout_s`; the winning backend is logged.
With `device_backend: auto` and a `backend_cache_dir` (wired to
`config.results_dir` by `build_camera_source`), `DeviceSource` consults
`results/camera_backends.json` - written by `scripts/benchmark_camera_matrix.py`
with the measured winner per index - and opens that backend first, still falling
back to the other. The cache is machine-specific and not committed.
Requested width/height/fps come from config; `info()` reports what the device
**actually** returned. Read failures trigger a bounded exponential-backoff
reconnect (`reconnect_initial_s` .. `reconnect_max_s`); attempts and total
seconds are counted. The capture loop runs on the analysis producer thread and
`put()`s into the mailbox - it never blocks on the consumer; a reconnect backoff
pauses only this source.

## Mode B - recorded video (`FileSource` + `RecordedDriver`)

`FileSource` decodes a file sequentially. `capture_ts` derives from
`CAP_PROP_POS_MSEC` (falling back to `frame_index / fps`), never wall clock.
Replay modes: `asfast` (no pacing) and `realtime` (paced to file timestamps);
both yield an identical frame sequence.

`RecordedDriver` processes **every** frame with no drops - it does not use the
mailbox. It writes `results/recorded_<run_id>.jsonl` (one compact JSON object
per frame: `frame_id`, `capture_ts`, `pts_s`, empty `detections`/`poses`/`tracks`)
plus a `.manifest.json`. Two runs over the same file + config produce
byte-identical JSONL.

## Raw clip recorder

The page records the **full-quality** preview stream with `MediaRecorder` (not
the downscaled analysis frames). On stop the blob is POSTed to
`POST /api/record/upload` (multipart) with `scenario_tag`, `device_label`,
`width`, `height`, `nominal_fps`, `notes`, `consent_ack`. The backend writes
`data/raw/<session_id>/<clip_id>.webm` and a sibling `<clip_id>.json`
`ClipManifest` (adds UTC time, git commit, config profile, size, duration).
Uploads over `recorder.max_clip_mb` are rejected with 413. `data/` is
git-ignored; clips are never committed.

## Endpoints added

| Method | Path | Behaviour |
|---|---|---|
| GET | `/api/cameras` | Backend enumeration: `[{index, name, available, backend}]` |
| WS | `/ws/ingest` | Binary analysis frames + hello/ack/echo handshake |
| GET | `/api/videos` | Video files under `data/videos/` |
| POST | `/api/analyze` | `{path, replay_mode}` -> runs `RecordedDriver` -> `{run_id, jsonl_path, frames}` |
| POST | `/api/record/upload` | multipart clip + metadata -> `ClipManifest` |
| GET | `/api/clips` | List clip manifests |
| POST | `/api/debug/stall` | Stall the analysis consumer N s (preview-independence check) |
| POST | `/api/metrics/browser` | Phase 1.5: append a labelled browser-measured sample block to `results/browser_metrics_<label>.json` (label slugified; no path traversal) |
| GET | `/static/*` | Dashboard JS/CSS (`StaticFiles`) |

`/health`, `/api/config`, `/ws/state`, `/` keep their Phase 0 shapes.

## New `StateSnapshot.metrics` keys (additive)

`capture_fps`, `analysis_fps`, `dropped_analysis_frames`, `drop_rate`,
`frame_age_ms` (`-1.0` sentinel for "no frame"; the nullable truth is the
top-level field), `decode_ms` (last decode, never a percentile on the hot path),
`ingest_bytes_per_s`, `reconnects`, `clock_offset_rtt_ms`. `preview_fps` is
page-side only and is not in the backend snapshot.

## Still absent after Phase 1

No detector, pose estimator, tracker, feature builder, risk model, alert policy,
TTS, or overlay drawing. No `aiortc`/WebRTC, Docker, CI, database. `cv2` is
imported only under `predictivesense/camera/`.
