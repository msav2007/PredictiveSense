# Phase 8 — perception responsiveness, hardware-portable runtime, tracking-ready output

Three-part format, as required: **Measured** (code-level, on this machine),
**Physically observed by the developer** (pending — section 20), **Not verified /
limitations**.

> **Scope note carried from the pre-work exchange.** This machine has one
> recorded clip (`data/raw/4918…/53f57e…webm` — 1920×1080, "Integrated Camera",
> cabin-unlabelled) and **no OnePlus footage**, no live camera, and an
> `onnxruntime` build exposing only `['AzureExecutionProvider',
> 'CPUExecutionProvider']` (no CUDA, no DirectML). So this report delivers the
> full **code-level** phase plus **loopback + fake-camera-headless + CPU**
> measurements from the one available clip. OnePlus transport numbers,
> real-camera absolute capture→paint, and CUDA/DirectML execution are the
> developer's physical verification and are marked **pending** throughout.

Machine fingerprint (every figure below unless stated): Intel Core Ultra 5 125H
(Intel64 Family 6 Model 170), **18 logical cores**, 16.8 GB, Windows 10.0.26200,
Python 3.11.9, `onnxruntime` 1.24.4, providers `['AzureExecutionProvider',
'CPUExecutionProvider']`. Config: provider `cpu`, `intra_op_threads` 6, detector
input 640, analysis path 640×480 @ 10 fps q0.70, `pose_every_n` 1,
`pose_requires_person` false, `broadcast.rate_hz` 10, `consumer.sample_rate_hz`
20.

---

## PART 1 — MEASURED

### 1. Section 5 — answered from the actual code (before any change)

**Q1. Sampler scheduling — fixed timer or completion-driven?**
**Fixed timer.** `pipeline/loop.py::_run_consumer` runs a `next_due` clock at
`consumer.sample_rate_hz` (20 Hz / 50 ms). `_iterate()` calls `_run_perception()`
**synchronously inline**. Combined detector+pose+policy p50 is ~130–225 ms — far
over the 50 ms period — so in the loaded state the wait is skipped, a catch-up
clamp (`next_due = time.monotonic() + period`) fires every pass, and the loop
degenerates to "one iteration per perception-duration" with **no real rate
ceiling**. The mailbox `get()` at the top of `_iterate` happens immediately after
the previous ~180 ms inference returned, so the frame it pulls can already be up
to one inference-duration old before perception even starts.

**Q2. Where does a frame wait longest?** Measured — see §2. Loopback answer:
**inference dominates** (detector 42 % + pose 28 % of end-to-end), then
**mailbox dwell 21 %** (`LatestFrameMailbox` — a frame waits ~half an
inference-duration for the timer-driven consumer). Transport (worker encode + WS
transit + decode) is ~3 % combined. The second single-slot buffer
(`BrowserSource._slot`) contributes **0.0 ms** (§4).

**Q3. Where is `frame_age_ms` measured?** **At snapshot emission, server-side,
after perception** (`loop.py`: `emitted_ts = time.monotonic()` taken *after*
`_run_perception` returns; `frame_age_ms = (emitted_ts - capture_ts) * 1000`).
It is **not** measured at dequeue and **not** at overlay paint. The overlay
(`static/features/overlay.js`) computed no capture→paint age at all before this
phase — it only read `snap.stale`. Measured gap: capture→paint p50 **302 ms** vs
server capture→snapshot p50 **~225 ms** (§2) — the reported `frame_age_ms`
**understated the perceived latency by ~75–95 ms** (the `/ws/state` 100 ms poll +
the browser's own `store.subscribe → draw` dispatch + paint).

**Q4. Does the worker produce at a fixed rate regardless of server progress?**
**Yes.** `analysis-worker.js::dropReason()` gates only on the `analysis_fps`
throttle (10 fps), `encodeBusy` (newest-wins), and `ws.bufferedAmount` vs
`max_ws_buffered_bytes`. It has **no signal of consumer/analysis progress**. In
the loopback the server analysed at ~6.2 fps while frames were pushed at 10 fps;
**drop rate mean 0.35** — roughly a third of encoded+transported frames were
`dropped_overwrite` without ever being analysed.

**Q5. Does pose block detection?** **Yes.** `perception/engine.py::infer` runs
the detector to completion, then `self._pose.infer(frame)` **synchronously in the
same call on the same thread**. Per-frame cost is `detector_ms + pose_ms +
policy_ms` serially; the next detection cannot begin until pose returns.

### 2. Section 6 — stage-attribution tables (published BEFORE any optimization)

Instrumentation: a mutable `FrameTrace` travels attached to each browser
`Frame`; every pipeline stage stamps it; `telemetry/stages.py::record_stages`
folds each delta into the existing `MetricRegistry` (`stage_<name>_ms` samples —
no parallel metrics system) and freezes a `FrameStages` record. `worker_encode`
and `overlay_paint` cannot be stamped on the server trace, so they come from the
fake-camera headless run (§2b). Commands:

```
.venv/Scripts/python.exe scripts/benchmark_latency.py --source data/raw --profile dev \
    --limit 150 --end-to-end-seconds 20 --label loopback
.venv/Scripts/python.exe scripts/benchmark_capture_paint.py --seconds 18 --label fake_camera_headless
```

#### 2a. `results/latency_stages_loopback.json/md` — server pipeline, integrated-camera clip, CPU

Frames pushed over a real `/ws/ingest` with a real capture clock through the
real server pipeline; **no live camera, no browser paint**. 118 analysed frames.
**End-to-end capture→snapshot p50 224.66 ms** · frame_age p50/p95/max
224.66 / 280.56 / 406.24 ms · analysis FPS 6.21 · drop rate 0.35.

| stage | what | p50 ms | p95 ms | % of end-to-end |
|---|---|---|---|---|
| `worker_encode` | browser JPEG encode (imencode proxy) | 3.0 | 3.23 | 1.3 % |
| `ws_transit` | send → server receive | 1.97 | 9.48 | 0.9 % |
| `decode` | `cv2.imdecode` | 2.42 | 3.99 | 1.1 % |
| `src_buffer_dwell` | **`BrowserSource` slot dwell** | **0.0** | **0.0** | **0.0 %** |
| `producer_handoff` | producer read → mailbox put | 0.0 | 0.0 | 0.0 % |
| `mailbox_dwell` | **`LatestFrameMailbox` dwell** | **47.0** | **94.0** | **20.9 %** |
| `detector` | detector inference | 94.83 | 133.39 | **42.2 %** |
| `pose` | pose inference | 62.73 | 99.76 | **27.9 %** |
| `policy` | recognition policy | 0.07 | 0.09 | 0.0 % |
| `snapshot_build` | snapshot assembly | 0.03 | 10.62 | 0.0 % |

Server stages attributed (Σ p50): 212.05 ms of 224.66 ms (12.6 ms
unattributed / measurement overlap).

#### 2b. `results/latency_stages_fake_camera_headless.json/md` — real browser, fake camera, full capture→paint

Headless Chromium with `--use-fake-device-for-media-stream`; **real** Web Worker
`OffscreenCanvas.convertToBlob` encode and **real** overlay canvas paint;
perception ON. 113 paint samples. **capture → overlay paint p50 302.1 ms / p95
378.1 ms.**

| stage | p50 ms | p95 ms | % of capture→paint |
|---|---|---|---|
| `worker_encode` (real) | 2.3 | 4.08 | 0.8 % |
| `ws_transit` | 1.8 | 8.98 | 0.6 % |
| `decode` | 2.29 | 4.06 | 0.8 % |
| `src_buffer_dwell` | 0.0 | 0.0 | 0.0 % |
| `mailbox_dwell` | 78.0 | 134.0 | 25.8 % |
| `detector` | 85.26 | 116.4 | 28.2 % |
| `pose` | 78.01 | 120.49 | 25.8 % |
| `snapshot_build` | 0.0 | 10.34 | 0.0 % |
| **`overlay_paint`** (capture→paint) | **302.1** | **378.1** | **100 %** |

The ~55–95 ms between `overlay_paint` (302 ms) and the server capture→snapshot
sum (~225–247 ms) is the `/ws/state` broadcast poll + the page's
`store.subscribe → draw` dispatch. **This is the honest "age of the world the
overlay is describing".** Every before/after figure in this report uses
**capture→paint** (or, for the pure server view, capture→snapshot, labelled).

#### 2c. OnePlus Nord 4 virtual camera — **PENDING (developer)**

No OnePlus footage exists in the repo and a virtual camera's transport latency
and capture jitter cannot be reconstructed from a recorded clip. The developer
will record a 720p clip for **image characteristics** (noise/blur/exposure) so
the input-size sweep (§?) can include it; absolute transport/latency for that
camera is physical verification only.

### 3. Baseline (before any Phase 8 optimization)

| measure | value | command |
|---|---|---|
| capture→snapshot p50 / p95 (loopback) | **224.66 / 280.56 ms** | `benchmark_latency.py … --label loopback` |
| capture→paint p50 / p95 (fake-camera headless) | **302.1 / 378.1 ms** | `benchmark_capture_paint.py … --label fake_camera_headless` |
| analysis FPS p50 (loopback) | 6.21 | " |
| drop rate mean (loopback) | 0.35 | " |
| detector p50 (Protocol A, isolated) | 73.2 ms | `benchmark_latency.py` `before__pose_every_frame` |
| pose p50 (Protocol A, isolated) | 56.5 ms | " |
| combined p50 (Protocol A) | 130.9 ms | " |
| full test suite | 374 passed / 2 skipped | `pytest -q` |

### 4. Two single-slot buffers in series — reconciliation (user correction #3)

The analysis path has **two** single-slot newest-wins buffers in series:
`BrowserSource._slot` (fed by the async WS-decode thread) → producer thread →
`LatestFrameMailbox._slot` (drained by the consumer).

Measured dwell, reported separately: **`src_buffer_dwell` p50/p95 = 0.0 / 0.0 ms**;
**`producer_handoff` p50/p95 = 0.0 / 0.0 ms**; **`mailbox_dwell` p50/p95 =
47 / 94 ms** (loopback) / 78 / 134 ms (headless). The producer thread blocks on
`BrowserSource._frame_ready`, which `submit()` sets, so it wakes and hands the
frame to the mailbox in <0.01 ms — the second buffer costs nothing.

**Verdict: the two-buffers-in-series design is justified and is kept.**
`BrowserSource._slot` decouples the asyncio WS-decode thread from the analysis
producer thread (the decode must not block the socket; the producer must not
block on the socket). Collapsing to one buffer would couple them for **zero**
measured latency gain. Drop accounting is unified into one reconciling set (§ —
lands with the staleness guard): `frames_decoded == analysed + dropped_stale +
dropped_browser_buffer + dropped_mailbox + in_flight`, exact.

### 5. Change log — before / after per change

_(one row per change; measured, kept or reverted on evidence)_

| # | change | before (capture→paint p50 / capture→snapshot p50) | after | verdict |
|---|---|---|---|---|
| — | baseline | 302.1 / 224.66 ms | — | — |
| 1 | broadcast push-on-publish | _pending_ | _pending_ | _pending_ |

---

## PART 2 — PHYSICALLY OBSERVED BY THE DEVELOPER (pending, section 20)

Not performed by the assistant. The developer must confirm, on real hardware:

1. Laptop integrated camera — labels follow the scene noticeably faster than
   before; record the achieved resolution and the active provider shown in
   Diagnostics.
2. OnePlus Nord 4 virtual camera — the same; confirm the **achieved** resolution
   the device really delivers (it has previously given 1280×720 when 1920×1080
   was requested).
3. Move an object quickly across the frame — the box follows the *current*
   position, not a past one.
4. Preview stays smooth during a deliberate 3 s analysis stall
   (`POST /api/debug/stall`).
5. A stale skeleton is visibly distinguishable from a fresh one.
6. Baseline classes still recognised; no label reads `Unknown (was …)`.
7. When the NVIDIA machine is available — repeat 1–6, record the active provider,
   compare CPU vs GPU side by side with both machine fingerprints shown.

---

## PART 3 — NOT VERIFIED / LIMITATIONS

- **OnePlus virtual-camera transport latency and capture jitter** — not
  measurable here (no footage, no device). Recorded-clip image characteristics
  only, once the developer provides a 720p clip.
- **Real-camera absolute capture→paint** — the fake device has no sensor
  exposure / auto-focus / auto-exposure cost, so §2b is a **floor**, not a
  real-camera figure.
- **CUDA / DirectML execution** — this `onnxruntime` build has neither EP.
  Provider *resolution* logic and the `cuda`-marked tests are implemented and
  unit-verified; the GPU benchmark and cross-provider agreement check **run on
  the developer's NVIDIA machine only**.
- Cross-clock subtraction is avoided end to end: `capture→paint` is a pure
  browser-`performance`-clock delta against `capture_client_ts_ms`; the ingest
  clock offset RTT remains the error bar on any server↔client figure.
