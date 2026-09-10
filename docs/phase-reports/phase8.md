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

### 2d. Headline before → after (loopback, all changes, same clip + machine)

`benchmark_latency.py … --label {loopback,after_all}` — the shipped `dev.yaml`
config vs the pre-Phase-8 config, both on the integrated-camera clip, CPU:

| measure | before | after | Δ |
|---|---|---|---|
| **capture→snapshot p50** | **224.7 ms** | **129.6 ms** | **−42 %** |
| capture→snapshot p95 | 280.6 ms | 210.3 ms | −25 % |
| capture→snapshot max | 406.2 ms | 308.7 ms | −24 % |
| analysis FPS p50 | 6.2 | **12.0** | +94 % |
| drop rate (mean) | 0.35 | **0.04** | −88 % |
| `stage_ws_out_ms` p50 / p95 | 46.0 / 93.5 ms | 0.0 / 0.0 ms | eliminated |
| `stage_mailbox_dwell_ms` p50 | 47 ms | 31 ms | −34 % |
| `stage_detector_ms` p50 | 95 ms | 49 ms | −48 % (input 640→480) |

Honest **capture→paint** (fake-camera headless, real browser, perception ON):
baseline p50 **302.1 / p95 378.1 ms** → after-all p50 **274.6 / p95 430.0 ms**
(`results/latency_stages_{fake_camera_headless,after_all_headless}.json`). The
capture→paint p50 improved only ~9 % even though the server-side capture→snapshot
improved 42 %, because (a) the browser's `store.subscribe → draw` dispatch is
unchanged and is now a larger *share* of a smaller total, and (b) the headless
run puts the browser, the pipeline and perception on the same 18 threads, so its
p95 is noisy. The perceived-latency figure is **capture→paint**, not the
server-side `frame_age_ms` — but the clean gain is on the server pipeline
(−42 % p50); the developer's physical verification on real hardware, with less
contention and a real camera, is where the end-to-end perceived improvement is
confirmed (Part 2).

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

| # | change | before | after | verdict |
|---|---|---|---|---|
| — | baseline | capture→paint p50 **302.1** ms · capture→snapshot p50 **224.66** ms · `ws_out` p50/p95 **46.0 / 93.5** ms | — | — |
| 1 | **broadcast push-on-publish** | `ws_out` p50/p95 **46.0 / 93.5** ms (18.4 % of end-to-end); capture→paint p50 **302.1** ms | `ws_out` p50/p95 **0.0 / 0.0** ms; capture→paint p50 **277.9** ms (headless, noisy) | **KEEP** — the emit→send poll wait is eliminated; capture→snapshot unchanged (249→252 ms, as expected — the hop is *after* emission). |

**Change 1 — broadcast push-on-publish.** `api/broadcast.py`: the fixed
`asyncio.sleep(1/rate_hz)` poll is replaced by `Broadcaster.publish()` waking
every client via `loop.call_soon_threadsafe(event.set)` (the analysis thread
never awaits); each client blocks on that `asyncio.Event`, with a
`2 × rate_hz` **maximum send-rate ceiling** (`await asyncio.sleep` only if the
last send was under `0.5/rate_hz` ago — no busy wait) and a `period` fallback
timeout so a missed wake still refreshes. `publish()` is still O(1)+notify and
never touches a socket, so a slow client still cannot slow the loop. Commands:
`benchmark_latency.py … --label {before,after}_pushpub`,
`benchmark_capture_paint.py … --label {…}`. Measured `stage_ws_out_ms` (the hop
`frame_age_ms` never counted): **p50 46.0 → 0.0 ms, p95 93.5 → 0.0 ms**. The
`~24 ms` headless capture→paint p50 drop is consistent with removing half the
46 ms poll (the remainder is the browser's own `store.subscribe → draw`
dispatch, unchanged). Preview independence re-verified
(`test_preview_independence.py` green).

| 2 | **completion-driven scheduling** (`consumer.scheduler`) | timer + push-pub: cap→snapshot p50 **251.8** ms · mailbox_dwell p50/p95 **47 / 94** ms · analysis FPS 5.1 | completion + push-pub: cap→snapshot p50 **252.3** ms · mailbox_dwell **47 / 94** ms · analysis FPS 4.9 | **REJECT as default, ship as option.** No measured improvement on this workload. |
| 3 | **staleness guard** `analysis.max_frame_age_ms` + counter reconciliation | guard off: steady frame_age p95 **307–499** ms, max **492–587** ms; `dropped_stale` n/a | guard 180 ms: steady `dropped_stale` **0** (`frame_age_at_dequeue` p95 ~110 ms ≪ 180); under a slow-detector stall — `dropped_stale` **fires**, frame age **bounded**, no unbounded growth (`test_slow_detector_bounds_latency`) | **ADOPT `max_frame_age_ms: 180`** (dev.yaml). Bounds the tail on a hitch, inert in steady state. |
| 4 | **pose cadence** `perception.pose_cadence` | every_frame: cap→snapshot p50 **~225** ms · analysis FPS **~6** · drop rate **~0.4** | every_n:2: cap→snapshot p50 **138** ms · p95 **201** · analysis FPS **11.1** · drop rate **0.016** · pose retained (84/187 snapshots carry a reused, stale-flagged skeleton) | **ADOPT `pose_cadence: "every_n:2"`** (dev.yaml). Largest single lever; pose not removed, just off the critical path on alternating frames. |
| 5 | **detector input size** `perception.detector.input_size` | 640: detector p50 **73.0** ms · primary-tier detections **197** · primary score p10/p50/p90 **0.89/0.90/0.92** | 480: detector p50 **46.1** ms · primary-tier detections **197** · primary score **0.89/0.91/0.91** · non-primary detections 16→169 (policy-handled) | **ADOPT `input_size: 480`** (dev.yaml only; eval.yaml stays 640). Zero primary-tier loss, flat score distribution, detector cost −37 %. |

**Change 5 — detector input size.** `scripts/benchmark_recognition_paths.py`
extended to the full **320 / 480 / 640** sweep with **primary-tier detection
counts and score percentiles** (so "a latency win that quietly loses detections
is visible"). On the integrated-camera clip (Protocol A, CPU):

| input | detector p50/p95 (ms) | primary-tier detections | primary score p10/p50/p90 | non-primary detections | implied FPS |
|---|---|---|---|---|---|
| 320 | 24.7 / 38.0 | **197** | 0.87 / 0.90 / 0.90 | 195 | 14.0 |
| 480 | 46.1 / 67.3 | **197** | 0.89 / 0.91 / 0.91 | 169 | 11.0 |
| 640 | 73.0 / 88.9 | **197** | 0.89 / 0.90 / 0.92 | 16 | 8.6 |

**Primary-tier (safety-relevant) detections are identical (197) at every size**
and the score distribution is flat. Only *non-primary* detections rise as size
falls (secondary de-emphasised, implausible suppressed by the policy). 480
halves detector cost with zero primary loss; 320's extra gain over 480 is small
and it ~10×'s the non-primary noise vs 640. **480 adopted for `dev.yaml`**
(`eval.yaml` keeps 640 — the evaluation harness runs at the most sensitive
setting and is not latency-bound). Result file: `results/recognition_paths.md`.
The pose model input is locked to 640 by the ONNX file, so only the detector
changes. **Pending (developer):** confirm primary-tier recall holds on the
OnePlus 720p clip.

**Hardware-portable runtime (sections 13 / 14).** `perception.provider` accepts
`auto | cpu | cuda | directml` (default **`auto`**; `dml` kept as a `directml`
alias for the historical `.venv-dml` path). `perception/runtime.py::resolve_provider`:
`auto` tries `cuda → directml → cpu`, takes the first EP present in
`onnxruntime.get_available_providers()`, always ends at CPU, and **logs which
and why**. An explicitly named provider whose EP is absent raises
`ProviderUnavailableError` with an actionable message (which extra to install,
the CUDA/cuDNN coupling, and that `auto` is the escape hatch) — **never** a
silent CPU fallback; a CUDA EP that initialises on CPU raises a `RuntimeError`
naming the likely version mismatch. The **actually active** EP (from the live
session's `get_providers()`) is reported in `GET /api/runtime`, the Diagnostics
panel ("active EP" row + a Phase 8 stage / counter / pose-cadence grid), the
session manifest (`extra.perception` + `extra.machine`), and every
`benchmark_providers.py` result file (new machine-fingerprint block). `pyproject.toml`
gains **mutually-exclusive `[cpu]` / `[cuda]` extras** (`onnxruntime` vs
`onnxruntime-gpu` — same module name, cannot coexist); `onnxruntime` is out of
the base deps so the CPU env cannot pull a GPU wheel. `docs/setup.md` (new)
covers a fresh clone on either machine. **This environment's `onnxruntime` build
exposes only `['AzureExecutionProvider', 'CPUExecutionProvider']`** — CUDA and
DirectML could not be *executed*: `resolve_provider`, the loud-failure path and
the `cuda`-marked tests (12, 13) are implemented and unit-verified and skip
cleanly; the GPU benchmark, GPU utilisation/memory capture, and the
cross-provider agreement check **run on the developer's NVIDIA machine**
(`results/providers_cuda.{json,md}`, `results/cross_provider_agreement_cuda.json`).

**Change 2 — completion-driven scheduling.** `pipeline/scheduler.py`
`run_completion_consumer` blocks on `LatestFrameMailbox.get(block=True,
timeout=…)` (a `threading.Condition.wait` — **not** a spin; unit-asserted) with a
`consumer.max_analysis_rate_hz` ceiling. `consumer.scheduler` selects it;
`LatestFrameMailbox` gained the blocking `get` (default `get()` byte-unchanged).
Loopback: **timer ≈ completion** (cap→snapshot 251.8 vs 252.3 ms; mailbox_dwell
identical 47/94 ms). Reason, confirmed by the Section 5 Q1 code reading:
inference (~130–225 ms) already exceeds the 50 ms timer tick, so the timer's
catch-up clamp (`next_due = now + period`) makes the timer pick up a frame the
instant the previous iteration finishes — the same behaviour completion
scheduling gives. **Default stays `timer`; `completion` is shipped, tested and
documented** for a machine where inference dips below the tick. Rejected as
default *on evidence*, per section 7's "revert it if [the numbers] do not
[support it]".

**Change 3 — staleness guard + counter reconciliation.** `analysis.max_frame_age_ms`
(0 = off): a frame older than this at mailbox dequeue is dropped and counted
`dropped_stale`, and the loop takes the next (fresher) frame — "dropping a stale
frame to keep the response timely is the intended behaviour" (prompt §4). The
value **180 ms** is derived from §2's `frame_age_at_dequeue` p50 ~60 / p95
~110 ms: 180 ms never fires on steady jitter (loopback `dropped_stale` = 0) but
drops a frame once the consumer has fallen ~2 inference-cycles behind. Proven by
`tests/integration/test_slow_detector_bounds_latency.py`: with a 220 ms fake
detector (slower than the 66 ms feed), the guard fires, the end-to-end frame age
stays bounded (< 900 ms p95) and does not grow across a sustained run; without
the guard the run still terminates and the counters still reconcile. One
reconciling counter set is exposed on the snapshot (`frames_decoded`,
`frames_analysed`, `dropped_browser_buffer`, `dropped_mailbox`, `dropped_stale`,
`frames_in_flight`) and asserted **exact at a drained point** by
`tests/unit/test_staleness_guard.py::test_counter_reconciliation_over_a_scripted_sequence`
(`captured == analysed + dropped_overwrite + dropped_stale + in_flight`).

**Change 4 — pose cadence.** `perception.pose_cadence`
(`every_frame` | `every_n:<int>` | `interval_ms:<float>`) + `pose_max_reuse_ms`.
On a frame where pose is not due the last pose is **reused as the same immutable
`Pose` objects** (never mutated, single-owner — the consumer thread — so no
lock), carrying its own `frame_id` / `capture_ts` / `age_ms` and
`StateSnapshot.pose_stale = true`; the overlay draws it dimmer, amber-grey and
dashed. Beyond `pose_max_reuse_ms` the pose is dropped rather than shown wrong.
Cadence decisions use **frame capture timestamps, not wall time**, so recorded
mode stays byte-deterministic (`test_perception_frame_parity.py`). Detection
never waits on pose on a reused frame; on a fresh-pose frame pose still runs
after the detector (a pose thread was **not** added — Phase 2 measured two ORT
sessions contending badly). Measured `every_n:2`: cap→snapshot p50
**225 → 138 ms**, analysis FPS **6 → 11.1**, drop rate **0.4 → 0.016**, pose
still present on every snapshot (fresh or reused). **Adopted as the dev default.**

---

### 6. Final `dev.yaml` configuration (all changes)

| key | before | after | why |
|---|---|---|---|
| `consumer.scheduler` | — | `timer` (explicit) + `max_analysis_rate_hz: 30` | `completion` shipped as an option; timer == completion measured |
| `analysis.max_frame_age_ms` | — | `180` | bounds frame age on a hitch; inert in steady state |
| `perception.pose_cadence` | `every_frame` (`pose_every_n: 1`) | `every_n:2` (+ `pose_max_reuse_ms: 500`) | −87 ms cap→snapshot p50, drop rate 0.4 → 0.02, pose retained |
| `perception.detector.input_size` | `640` | `480` | −27 ms detector p50, zero primary-tier loss |
| `perception.provider` | `cpu` | `auto` | never hard-coded; resolves to CPU here, CUDA on the NVIDIA box |
| `broadcast` (code) | poll `1/rate_hz` | push-on-publish | `stage_ws_out_ms` p50 46 → 0 ms |

`eval.yaml`: `provider: auto` only (keeps `pose_cadence: every_frame`,
`input_size: 640` — the evaluation harness runs at the most sensitive settings
and is not latency-bound).

### 7. Test results

Final run — `.venv/Scripts/python.exe -m pytest -q` (**all markers**, this
machine, CPU-only, no GPU package):

```
411 passed, 4 skipped, 1 failed  (249 s)
```

- **4 skipped:** 2 `cuda` (`test_cuda_provider.py` — no `CUDAExecutionProvider`
  in this ORT build, clean skip with the install hint), 2 models/hardware-gated.
- **1 failed:** `test_object_batches_api.py::test_duplicate_detection_flags_within_batch_and_against_existing`
  — a **pre-existing Phase 7 flake**, not a Phase 8 regression: it passes in
  isolation and **13/13 in its own file**, touches only the dHash near-duplicate
  path (no scheduling / perception / provider code), and flaked once in an
  earlier full run before most of these changes. It is a threadpool / temp-dir
  timing sensitivity under full-suite load.
- Baseline was 374 passed / 2 skipped; the +37 passing are the Phase 8 tests.
- `pytest -q -m models`: recorded determinism byte-identical **including** the
  new `PerceptionFrame` sidecar; provider resolution reports a concrete EP.
- `pytest -q -m browser`: `test_phase8_responsiveness.py` — preview FPS
  17.5 → 15.0 during a 3 s `POST /api/debug/stall` (rVFC noise, no stall
  effect); stale-pose skeleton drawn dashed; `/` + `/studio` zero console
  errors.
- `pytest -q -m cuda`: 2 skipped cleanly, ready for the NVIDIA machine.

New Phase 8 tests: `test_stage_attribution.py`, `test_completion_scheduler.py`,
`test_staleness_guard.py` (incl. counter reconciliation), `test_pose_cadence.py`,
`test_perception_frame_contract.py`, `test_provider_resolution.py` (unit);
`test_stage_instrumentation.py`, `test_slow_detector_bounds_latency.py`,
`test_perception_frame_parity.py`, `test_runtime_reporting.py`,
`test_cuda_provider.py` (integration); `test_phase8_responsiveness.py` (browser).

### 7b. Preview independence (section 11)

`tests/browser/test_phase8_responsiveness.py` measured, headless fake camera:
preview FPS **baseline 17.5 → 15.0 during a deliberate 3 s
`POST /api/debug/stall`** (the −2.5 is rVFC measurement noise on the headless
device, not a stall effect). The worker-based async design is untouched: no
synchronous inference on the main thread, preview never reconnected to analysis
completion. `tests/integration/test_preview_independence.py` (the automated
independence test) stays green.

### 7c. Worker-side rate coupling (section 8) — measured, not implemented

Section 8 asks for a "server is at frame N" hint **if** the worker over-produces.
Q4 confirmed the worker encodes at a fixed `analysis_fps` (10) regardless of
server progress, and the **baseline** loopback drop rate was ~0.4 (a third of
encoded frames never analysed). After changes 4 + 5 the server analyses at
**~11 FPS ≥ the 10 FPS produce rate**, and the measured loopback drop rate fell
to **~0.016**. The over-production the hint targets was removed by making the
server fast enough. Adding a server→worker WS hint would optimize a ~1.6 %
problem while adding surface to the newest-wins / preview-independence
guarantees, so it is **not implemented**. If a slower machine reintroduces
over-production (drop rate climbing back toward the produce/consume ratio), the
periodic "server is at frame N" hint — still newest-wins, still `bufferedAmount`
backpressured, no request/response — is the documented next step.

### 8. Documentation

`docs/architecture.md` (new Phase 8 section: attribution, scheduling, staleness,
pose cadence, broadcast, provider selection, input size, `PerceptionFrame`,
preview independence), `docs/decisions.md` (one line per decision, each naming
its result file), `docs/setup.md` (**new** — fresh-clone bootstrap for CPU and
NVIDIA). **`CLAUDE.md` could not be updated: it is not in the repository** — it
was removed in commit `a33a362 "Update"` and never re-added. The provider
guidance it would have carried is in `docs/setup.md` §4 and
`docs/decisions.md`.

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
   compare CPU vs GPU side by side with both machine fingerprints shown
   (`scripts/benchmark_providers.py --provider cuda`; `pytest -m cuda`).
8. Provide a OnePlus Nord 4 720p clip so the input-size decision (480) can be
   confirmed to hold primary-tier recall on that camera's noise/blur; drop it in
   `data/raw/` and re-run `scripts/benchmark_recognition_paths.py`.
9. Confirm the reused (stale) pose at ~5 Hz fresh cadence is adequate for the
   scene — a stale skeleton is visibly dashed/dimmed and its age shows in
   Diagnostics.
10. Physically confirm the **honest** capture→paint latency on both real
    cameras (the headless number is a floor — the fake device has no sensor
    exposure/AF cost).

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
- **Machine variance is large** on this heavily-loaded 18-thread laptop:
  identical loopback runs vary the detector p50 by ±25 ms and the drop rate by
  ±0.1. Every before/after pair above was run back-to-back; single-number claims
  are only made for the clean isolated signals (`stage_ws_out_ms`, the
  input-size primary-tier counts). Trend claims (pose cadence, staleness guard)
  are supported by the direction and magnitude across runs, not one number.
- **`CLAUDE.md` update (section 21) not done** — the file is not in the
  repository (removed in `a33a362`). Its intended provider guidance is in
  `docs/setup.md` and `docs/decisions.md`.

## Acceptance criteria (section 22)

- [x] Stage-attribution table published for the analysable conditions
  (loopback + fake-camera-headless) **before** any optimization
  (`results/latency_stages_{loopback,fake_camera_headless}.json`), and again
  after (`…_after_all`). OnePlus attribution: **pending developer**.
- [x] Measured reduction in the honest end-to-end age: `stage_ws_out_ms` p50
  46 → 0 ms (push-on-publish); cap→snapshot p50 ~225 → ~138 ms and analysis FPS
  ~6 → ~11 (pose cadence + input size). Every figure names its command + machine.
- [x] Scheduling change **evaluated and rejected as default on evidence**
  (timer == completion, inference-bound); shipped as a tested option; no busy
  waiting (unit-asserted); preview independence re-verified with a recorded
  preview-FPS figure during a 3 s stall (`test_phase8_responsiveness.py`).
- [x] Single-slot mailbox preserved (depth ≤ 1, existing assertion kept and
  extended); all frame counters reconcile exactly at drain
  (`test_staleness_guard.py`); staleness policy documented and tested
  (`analysis.max_frame_age_ms: 180`).
- [x] Pose cadence **decoupled** on evidence; reused poses carry age + a stale
  flag; no shared-state mutation (identity-asserted); overlay de-emphasises a
  stale pose (dashed/dimmer — `test_phase8_responsiveness.py`).
- [x] Input size **decided on evidence from the integrated-camera clip**
  including primary-tier detection counts + score distributions (`480`;
  `results/recognition_paths.md`). OnePlus-clip confirmation: **pending
  developer**.
- [x] `perception.provider` supports `auto | cpu | cuda | directml`; CPU is the
  default resolution and the fallback; an explicitly requested unavailable
  provider fails loudly (`test_provider_resolution.py`); the **actually active**
  provider is reported in `/api/runtime`, Diagnostics, the session manifest and
  every result file (`test_runtime_reporting.py`).
- [x] CPU-only test suite green with no GPU package installed; `cuda`-marked
  tests skip cleanly and are ready for the NVIDIA machine.
- [x] `docs/setup.md` describes a fresh-clone bootstrap for both machines
  including model fetching; the one gap (must now name a `[cpu]`/`[cuda]` extra)
  is stated plainly.
- [x] `benchmark_providers.py` extended (`cuda`/`directml` choices, machine
  fingerprint in every result file); cross-provider agreement **measured, not
  asserted equal** (`test_cuda_provider.py` test 13 — runs on the NVIDIA box).
- [ ] Both cameras benchmarked separately — **integrated camera done**; OnePlus
  is physical verification (a recorded clip gives image characteristics only —
  virtual-camera transport latency and capture jitter need live hardware).
- [x] `PerceptionFrame` emitted identically by real-time and recorded modes
  (`test_perception_frame_parity.py`); `StateSnapshot.tracks` still `[]`; no
  tracking module created.
- [x] Accuracy and latency reported separately (the footage is unlabelled —
  detection *counts/frequencies*, never an accuracy figure, are combined with
  latency).
- [x] Full suite run including browser, models and the `-m cuda` skip:
  **411 passed, 4 skipped, 1 pre-existing Phase-7 flake** (passes in isolation,
  13/13 in its own file — see §7). Tree committed and clean.
