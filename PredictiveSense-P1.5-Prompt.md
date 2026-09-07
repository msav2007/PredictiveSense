# PredictiveSense — Phase 1.5: Camera/Input Optimization & Verification Pass

> Paste this entire file into Claude Code in a **fresh session**. Do not summarise it.
> **This is not a new phase.** It is the closing pass on Phase 1. Do NOT start Phase 2.

---

## BLOCK 1 — CONTEXT

Phase 0 and Phase 1 are implemented and committed (`1115801` on `main`). Read `CLAUDE.md`, `docs/architecture.md`, `docs/decisions.md` and `docs/phase-reports/phase1.md` before touching anything, then **inspect the existing implementation and work from what is actually there.** Do not rebuild working components.

**Physical observations already made by the developer — treat these as established, do not ask him to repeat them:**

- **Integrated Camera:** working, very smooth, very little noticeable latency. Currently the better-feeling source.
- **OnePlus Nord 4 via Windows virtual camera:** working, stable, usable. Only a very small, barely noticeable delay versus the integrated camera. No major lag or freezing.

**Target machine:** Windows, Intel Core Ultra 5 125H, 16 GB RAM, Intel Arc integrated graphics, no NVIDIA GPU, no CUDA. Python 3.11.9 in `.venv`.

---

## BLOCK 2 — OBJECTIVE

Make both camera sources perform as well as reasonably possible **on measured evidence**, choose defaults from that evidence, and close Phase 1 with a report that separates what was measured, what the developer observed physically, and what remains unverified.

The existing architecture stays: browser owns the camera, preview is a native `srcObject` video element, analysis frames go through the Web Worker to `WS /ws/ingest`, backend keeps the single-slot mailbox. Change it only if a measurement proves a change is worth it.

---

## BLOCK 3 — REQUIREMENTS

### 3.1 Measure before changing anything

1. Build `scripts/benchmark_camera_matrix.py` (backend-owned) that sweeps, per device index: resolution × requested FPS × capture backend (MSMF, DSHOW) × FOURCC (default, MJPG). For each combination measure over ≥10 s: time-to-first-frame, achieved resolution from the device, achieved FPS, frame-interval p50/p95/max, read-failure count, process CPU% and RSS (via `psutil`). Write `results/camera_matrix_<label>.json` and a readable markdown table.
2. Add a browser-side measurement page section that records, for the currently selected device: `track.getSettings()` (actual width, height, frameRate), preview FPS, analysis FPS, worker encode ms, WebSocket `bufferedAmount`, backend decode ms, frame age p50/p95, drop rate. Add `POST /api/metrics/browser` so the page can push a labelled sample block that the backend writes to `results/browser_metrics_<label>.json`.
3. **Requested is not achieved.** Everywhere a resolution or FPS is reported, report the requested value and the value the device actually returned, side by side. Never present a requested value as an outcome.

### 3.2 Specific optimizations to implement and measure

Implement each, measure before/after, and keep only what the numbers justify. Record every keep/reject decision with its measurement in `docs/decisions.md`.

**Worker / analysis path**

4. **`VideoFrame.close()` audit.** Every `VideoFrame` obtained from `MediaStreamTrackProcessor` must be closed on every path, including early returns and errors. A leaked frame stalls the whole reader after a few frames. Add a test or an assertion counter that proves frames in == frames closed.
5. **Newest-wins inside the worker.** If an encode or send is still in flight when the next frame arrives, drop the older one rather than queueing. The worker must never accumulate a backlog.
6. **Send skip on backpressure.** Before sending, check `socket.bufferedAmount`; if it exceeds `capture.max_ws_buffered_bytes` (new config, default 1 MB), skip the frame and increment a counter. This is the direct fix for frame age creeping upward.
7. Sweep `analysis_fps` ∈ {5, 10, 15}, `analysis_width×height` ∈ {480×360, 640×480, 800×600}, `analysis_jpeg_quality` ∈ {0.5, 0.7, 0.85}. Report frame age, drop rate, encode ms, ingest bytes/s and CPU for each. Choose defaults that minimise frame age and drops at acceptable quality — **not** the highest numbers.

**Preview path**

8. Confirm the video element carries `autoplay`, `muted`, `playsinline` and has no CSS filter, transform, box-shadow or opacity animation applied to it — compositor work on the video surface costs latency for no benefit.
9. Confirm nothing reads the video element for display purposes. The only permitted read of `<video>` is the `createImageBitmap` analysis fallback when `MediaStreamTrackProcessor` is unavailable.

**Device switching and lifecycle**

10. On switching device: stop every track of the previous stream (`track.stop()`) and null `srcObject` **before** calling `getUserMedia` for the new device. Holding the old device is the usual cause of slow or failed switches and of a virtual camera appearing busy.
11. Measure switch time: from the `change` event to the first frame presented on the new device. Report it per direction (integrated → OnePlus, OnePlus → integrated).
12. Prefer `track.applyConstraints()` when only resolution or frame rate changes on the *same* device; full re-acquisition only when the device changes.
13. Re-enumerate devices on `navigator.mediaDevices.ondevicechange` so the OnePlus appearing or disappearing updates the dropdown without a page reload.

**Backend-owned source**

14. Set `cv2.CAP_PROP_BUFFERSIZE = 1` where the backend supports it — this is one of the largest latency reductions available on the OpenCV path. Measure the effect.
15. Try `CAP_PROP_FOURCC = MJPG` for the integrated and any USB camera. Many UVC cameras deliver 30 fps at 720p in MJPG but only 5–10 fps in YUY2. Measure, and keep it only if the device actually improves.
16. Cache the winning backend (MSMF or DSHOW) per device in the benchmark output so `capture.device_backend: auto` can resolve quickly rather than re-probing with a timeout every start.

### 3.3 Defaults

17. After the sweeps, write the measured-best defaults into `config/profiles/dev.yaml` and `eval.yaml`, with a comment beside each naming the result file that justified it. Add a test asserting the shipped defaults are one of the benchmarked combinations, so a future edit cannot silently drift from the evidence.

### 3.4 Robustness

18. Verify and, if needed, fix: camera becomes unavailable mid-run → UI shows a degraded state, backend keeps running, snapshots go `stale=true`; device returns → automatic recovery without a page reload. Cover with an automated test using a fake source that fails and recovers.
19. Confirm no regression: the full test suite still passes, preview independence still holds under the stall hook, and the recorded-video driver is still byte-deterministic.

---

## BLOCK 4 — CONSTRAINTS

- **Preview independence is inviolable.** A stalled or slow analysis path must never affect the preview. Re-run `test_preview_independence` after every change.
- Do not add dependencies. `psutil` is already present; use it. No WebRTC, no WebCodecs `VideoEncoder`, no new transport.
- Do not raise FPS or resolution as an end in itself. A configuration that increases frame age, drops, stutter, CPU or instability is a worse configuration even if its FPS number is higher.
- Do not rebuild Phase 0 or Phase 1 components that already work.
- No CUDA, no cloud calls, no runtime downloads.
- Keep bounded queues everywhere, including the new worker backpressure logic.

---

## BLOCK 5 — FILES

**Create:** `scripts/benchmark_camera_matrix.py`; `tests/unit/test_worker_backpressure_contract.py` (pure-logic test of the skip/newest-wins rule); `tests/integration/test_camera_recovery.py`; `results/` outputs as generated.

**Modify:** `predictivesense/api/static/analysis-worker.js`, `app.js`, `index.html`; `predictivesense/camera/device.py`; `predictivesense/camera/enumerate.py`; `predictivesense/config/settings.py`; `config/profiles/dev.yaml`, `eval.yaml`; `predictivesense/api/app.py` (browser-metrics endpoint); `docs/phase-reports/phase1.md`; `docs/decisions.md`; `docs/architecture.md`; `CLAUDE.md`.

**Must not be created or modified:** anything under `perception/`, `tracking/`, `scene/`, `temporal/`, `risk/`, `policy/`, `audio/`, `research/` — none of these exist and none may be created, not even empty. Do not touch `camera/mailbox.py`, `camera/synthetic.py`, `telemetry/*`, or the Phase 0/Phase 1 prompt files.

---

## BLOCK 6 — WHAT TO REPORT

Update `docs/phase-reports/phase1.md` with a **Phase 1.5** section containing three clearly separated parts:

**Measured** — the camera matrix table (per device: requested vs achieved resolution and FPS, interval p50/p95, time-to-first-frame, CPU, RSS); the analysis-path sweep table; before/after numbers for each optimization applied; switch times both directions; recovery time; final test run.

**Physically observed by the developer** — quote the two observations from Block 1 verbatim and attribute them. Do not add observations he did not make, and do not upgrade "barely noticeable delay" into a number.

**Not verified / limitations** — everything else, including: latency contributed by the Phone Link virtual-camera pipeline itself (phone encode → wireless → Windows) is outside PredictiveSense and cannot be fixed in this codebase; virtual cameras commonly ignore requested resolution and frame rate; browser preview latency is the browser's compositor and is not measurable from Python; no detection, pose, tracking, temporal, risk or voice exists.

---

## BLOCK 7 — COMMANDS

```powershell
cd C:\Users\mummi\Documents\Projects\PredictiveSense
.\.venv\Scripts\Activate.ps1

pytest -q
python scripts\benchmark_camera_matrix.py --index 0 --label integrated --seconds 10
python scripts\benchmark_camera_matrix.py --index 1 --label oneplus-vcam --seconds 10
python scripts\run_app.py --profile dev --source-kind browser
```

Run everything yourself. Fix your own failures and re-run. Do not ask the developer to run routine commands.

---

## BLOCK 8 — EXIT CRITERIA

- [ ] Both cameras remain selectable **by name**; `ondevicechange` re-enumeration works.
- [ ] Both sources work; switching works in both directions with a measured switch time.
- [ ] Camera matrix benchmarked for both devices; results written to `results/`.
- [ ] Analysis-path sweep run; defaults chosen from measurements and justified in config comments.
- [ ] `VideoFrame` close audit passes; worker newest-wins and backpressure skip implemented and tested.
- [ ] Preview independence re-verified by the automated stall test.
- [ ] Recovery from device loss tested automatically.
- [ ] Full suite passes; recorded driver still byte-deterministic; no regression.
- [ ] Phase 1 report updated with the three-part Phase 1.5 section.
- [ ] Working tree committed, no remote. **Phase 2 not started, no Phase 2 file created.**

---

## BLOCK 9 — PROHIBITIONS

1. Do not start Phase 2 or create any Phase 2 module, even empty.
2. Do not claim a performance improvement without before/after numbers from this machine.
3. Do not present a requested resolution or FPS as an achieved one.
4. Do not turn the developer's "barely noticeable delay" into a measured latency figure.
5. Do not claim to have fixed a limitation that lives in the Phone Link virtual-camera pipeline — document it instead.
6. Do not add dependencies, WebRTC, WebCodecs encoders, or a new transport.
7. Do not break preview independence, or introduce any unbounded queue or buffer.
8. Do not ask the developer to repeat physical tests that already passed, unless a change you made genuinely invalidates one — and then say exactly which and why.
9. Do not stop for routine errors; diagnose, fix, re-run.
10. Do not rewrite working Phase 0/Phase 1 components for style.

---

## BLOCK 10 — FINAL RESPONSE FORMAT

Reply with only this:

```
CHANGED: ...
FINAL MEASURED CAMERA RESULTS: ...
TESTS / BENCHMARKS PASSED: ...
GENUINE LIMITATIONS: ...
PHASE 2 NOT STARTED: confirmed
```
