# PHASE 8: PERCEPTION RESPONSIVENESS + HARDWARE-PORTABLE RUNTIME + TRACKING-READY OUTPUT

> Paste this entire file into Claude Code in a **fresh session**. Do not summarise it.
> **This phase is about latency, portability and output shape.** No custom-model training, no tracker, no relationship engine, no risk, no voice, no Scene Snapshot Studio, no UI redesign.

---

## 1. OBJECTIVE

Three outcomes, in this order:

1. **Responsiveness.** Make the live system react to the *current* scene rather than a scene that has already passed. The target is **low perceived latency with fresh analysis**, not a higher frame count. A system that processes 30 frames per second while reacting to a 300 ms-old frame has failed this phase.
2. **Hardware portability.** The same codebase must run on the current CPU-only laptop and, later, on an NVIDIA laptop cloned from GitHub — with the execution provider detected and configured, never hard-coded, and CPU always a working fallback.
3. **Tracking-ready output.** Shape perception output so a future tracker and relationship layer can consume it, without building either.

Every change must be justified by a measurement taken on the machine that ran it.

---

## 2. CURRENT VERIFIED ARCHITECTURE (confirm by inspection; do not assume)

Real-time path as understood from prior phases — **confirm each hop in the code before changing it**:

```
browser getUserMedia → <video srcObject>            (preview — never read, drawn or replaced)
                     ↘ Web Worker: MediaStreamTrackProcessor → downscale → JPEG
                       → binary WS /ws/ingest       (bufferedAmount backpressure, newest-wins in worker)
                       → server: framing decode → cv2.imdecode → Frame(capture_ts via clock offset)
                       → LatestFrameMailbox         (single slot, overwrite, drops counted)
                       → analysis loop sampler → PerceptionEngine (ONNX detector + pose)
                       → RecognitionPolicy          (vocabulary tiers, unknown, suppressed)
                       → StateSnapshot → WS /ws/state (last-value-wins) → overlay canvas
```

Known configuration and prior measurements: provider `cpu`; `intra_op_threads: 6`; detector `input_size: 640`; `pose_every_n: 1`; `pose_requires_person: false`. Detector p50 ≈ 65–113 ms and pose p50 ≈ 52–57 ms depending on measurement protocol; combined ≈ 129–206 ms; end-to-end frame age p50 ≈ 177 ms / p95 ≈ 231 ms; analysis ≈ 8.5 FPS; DirectML measured ~2.2× slower than CPU for these nano models.

**Inspect before modifying:** `CLAUDE.md`, `docs/architecture.md`, `docs/decisions.md`, `docs/phase-reports/phase5.md`, `phase6.md`, `phase7.md`; `predictivesense/pipeline/loop.py`, `pipeline/recorded.py`; `camera/mailbox.py`, `camera/browser.py`, `camera/framing.py`, `camera/device.py`; `perception/runtime.py`, `detector.py`, `pose.py`, `policy.py`, `vocabulary.py`; `api/ingest.py`, `api/broadcast.py`, `api/app.py`; `telemetry/*`; `core/types.py`; `config/settings.py`, `config/profiles/*`; `static/analysis-worker.js`, `static/features/analysis-client.js`, `features/overlay.js`, `groups/diagnostics.js`; `pyproject.toml`; `models/manifest.json`, `models/registry.json`; `scripts/benchmark_latency.py`, `benchmark_analysis_path.py`, `benchmark_camera_matrix.py`, `benchmark_providers.py`, `benchmark_recognition_paths.py`, `fetch_models.py`.

---

## 3. PHYSICAL OBSERVATIONS DRIVING THIS PHASE

Reported by the developer from real testing with the **laptop integrated camera** and the **OnePlus Nord 4 Windows virtual camera**. These are observations, **not measured accuracy or latency figures**:

- Detection and identification work on both cameras.
- The live system feels noticeably laggy; identification feels slow.
- Frame drops happen frequently during analysis.
- The system sometimes reacts to a scene state that has already passed — old frames appear to be processed.
- Inference sometimes falls behind capture.
- The OnePlus camera feels slightly more delayed than the integrated camera.
- Preview itself remains smooth and must stay that way.

---

## 4. LATENCY PROBLEM DEFINITION

The quantity that matters is **the age of the world the overlay is describing**: the time from photons at the sensor to a box drawn on screen. That is not detector milliseconds, and it is not analysis FPS.

Three distinct failures are suspected and must be separated by measurement, not argument:

- **Accumulated pipeline cost** — every stage adds a little and the total is large.
- **Scheduling staleness** — the loop reads a frame that was already old when it started, so latency exceeds the sum of the stages.
- **Wasted work** — frames are captured, encoded and transported that will never be analysed, or are analysed after they stopped being useful.

A frame drop is **not automatically a failure**. Dropping a stale frame to keep the response timely is the intended behaviour, and the report must say so where it applies.

---

## 5. INSTRUMENTATION PLAN — DO THIS BEFORE CHANGING ANYTHING

**No scheduling, cadence, input-size or provider change may be made until the attribution table in section 6 exists.**

First, confirm in the actual code and record the answers in the report:

1. **How does the sampler decide when to run** — a fixed timer, or on completion of the previous inference? If it is a timer and inference exceeds the interval, the loop runs late and may read an already-old frame.
2. **Where does a frame wait longest** — worker encode, WebSocket buffer, server decode, mailbox dwell, inference, or snapshot broadcast?
3. **Where is `frame_age_ms` measured** — at dequeue, at snapshot emission, or at overlay paint? These are three different numbers; only the last is what the developer perceives.
4. **Does the worker produce frames at a fixed rate regardless of server progress?** If so, frames may be encoded and sent that will never be analysed.
5. **Does pose block detection?** If both run in the same call for the same frame, detection cadence is capped by their sum.

Then extend the **existing** telemetry — no parallel metrics system — so one frame's journey is attributable end to end. Stamp and record, per analysed frame: browser capture time, worker encode ms, WS send time, server receive time, decode ms, mailbox enqueue time, mailbox dwell ms, detector ms, pose ms (or `reused`), policy ms, snapshot build ms, WS-out time, and — from the page — overlay paint time.

Browser-side stamps return through the existing browser-metrics endpoint. Keep the existing clock-offset handling and keep reporting the offset RTT as the error bar. **Never present a cross-clock difference as exact.**

Extend `scripts/benchmark_latency.py` rather than writing a new script, if its structure allows.

## 6. ATTRIBUTION TABLE — THE FIRST DELIVERABLE

Produce `results/latency_stages_<label>.json` plus a readable markdown table that **attributes the total**: each stage's p50 and p95, and its percentage share of end-to-end. Produce it for **both cameras**.

This table answers "where does the frame age actually go". Publish it before any optimization, and re-publish it after. Do not assume the bottleneck is the mailbox. Do not assume it is the detector.

---

## 7. SCHEDULING INVESTIGATION

If section 5 confirms timer-driven sampling, evaluate **completion-driven scheduling**:

```
inference completes → read newest frame in the mailbox → if eligible, process it → repeat
```

Constraints:

- Keep a configurable **maximum rate ceiling** so the loop cannot peg the CPU.
- **No busy waiting** — block on a condition/event, never spin.
- The loop must never sit idle holding a stale frame while a newer one is available.
- Preview independence guarantees are unchanged.

**Measure before and after.** Do not adopt this because it sounds faster; adopt it if the numbers support it, and revert it if they do not.

## 8. NEWEST-FRAME AND STALENESS STRATEGY

- **Keep the single-slot mailbox.** No queue, bounded or otherwise. Depth stays 0 or 1 and the existing assertion stays.
- Prioritise the newest available frame at every opportunity.
- **Staleness guard**, introduced only if the measurements support it: `analysis.max_frame_age_ms` — a frame older than this at dequeue is dropped and counted as `dropped_stale` rather than analysed. Derive the initial value from section 6, not from intuition, and document the exact policy.
- **Worker-side rate coupling:** if the worker encodes at a fixed rate regardless of server progress, add a lightweight periodic "server is at frame N" hint so it produces at roughly the consumable rate — still newest-wins, still with `bufferedAmount` backpressure. Do not build a request/response protocol.
- Counters, all exposed and reconciling exactly: frames captured, encoded, sent, received, decoded, enqueued, `dropped_overwrite`, `dropped_stale`, analysed, in-flight.
- Expose and report: frame age, dropped-frame counts by reason, mailbox depth, analysis FPS, capture FPS, preview FPS.

## 9. POSE CADENCE STRATEGY

Inspect the actual pose cost first. If pose is a substantial share of per-frame time, decouple its cadence from detection:

- Config `perception.pose_cadence` (`every_frame` | `every_n` | `interval_ms`) alongside the existing `pose_requires_person`. **Detection must never wait on pose.**
- When a frame is analysed without a fresh pose, the snapshot carries the **last pose with its own `frame_id`, `capture_ts` and `age_ms`, plus `stale: true`**. Consumers must be able to distinguish a fresh pose from a reused one, and the overlay must visually de-emphasise a stale skeleton. **Never present old pose information as current.**
- Bound reuse with `perception.pose_max_reuse_ms`; beyond it the pose is dropped and the snapshot carries no pose rather than a wrong one.
- **No race conditions.** The reused pose is an immutable value object — never mutate a shared pose in place, never read a half-written one. Prefer single ownership and value semantics over locks.
- Measure detection-only cadence versus combined cadence and report both. **Do not remove pose functionality to improve a benchmark number.**

## 10. INPUT-SIZE BENCHMARKING

Review the existing input-size benchmarks first (`docs/phase-reports/phase2.md`, `results/input_size_sweep.md`, and the Phase 6 re-run). Prior evidence: 480 gave detector p50 ≈ 58 ms versus ≈ 113 ms at 640 with zero primary-tier detections lost — on **one unrepresentative clip**.

Re-run 320 / 480 / 640 on representative footage from **both cameras**, measuring detector latency, end-to-end frame age, analysis FPS, CPU, memory, and — critically — **detection counts and score distributions for primary-tier classes**, so a latency win that quietly loses detections is visible.

Choose the default on evidence, name the result file in the config comment, and record the decision in `docs/decisions.md`. **The objective is not the smallest possible detector.** "No change justified" is an acceptable outcome.

## 11. PREVIEW INDEPENDENCE — HARD REQUIREMENT

A slow or stalled detector must never affect the video preview, UI interaction, camera controls, the Studio, or browser event processing. Preserve the existing worker-based asynchronous design. **Do not introduce synchronous inference on the browser main thread. Do not reconnect preview rendering to analysis completion.**

Re-verify after every change using the existing stall hook and the automated independence test, and record preview FPS during a deliberate stall.

---

## 12. CPU RUNTIME SUPPORT

CPU-only must remain **fully supported and the default**. It must never require a GPU package, a CUDA toolkit, or a GPU-capable driver. The full test suite must pass on a CPU-only machine with no GPU-specific dependency installed.

## 13. NVIDIA / CUDA RUNTIME SUPPORT

The same codebase must run on an NVIDIA laptop cloned from GitHub, with **no architectural change** — same interfaces, same model registry, same telemetry, same perception contracts, same tests.

1. **Never hard-code a provider.** Config `perception.provider` accepts `auto | cpu | cuda | directml`. Default `auto`.
   - `auto` selects the first available provider in a documented preference order, always falling back to CPU, and **logs which one was chosen and why**.
   - An explicitly requested provider that is unavailable is a **loud failure with an actionable message**, not a silent fallback. Silent fallback is how a "GPU result" turns out to have been CPU all along.
2. **Detect, do not assume.** Use `onnxruntime.get_available_providers()` at startup. Never import a GPU-specific package at module import time.
3. **Report the provider that is actually in use** — from the live session's `get_providers()`, not from config — in Diagnostics, in the session manifest, and in every benchmark result file.
4. **Packaging.** `onnxruntime`, `onnxruntime-gpu` and `onnxruntime-directml` install the same module name and **cannot coexist in one environment**. Do not try. Express this as explicit optional dependency extras in `pyproject.toml` (for example `[cpu]`, `[cuda]`) with a documented, maintainable install path — never undocumented manual edits. The CPU environment must not pull GPU dependencies.
5. **Bootstrap documentation.** Write or extend `docs/setup.md` so a fresh clone runs on either machine. **Verify by inspection what a fresh clone actually contains**: model weights and datasets are git-ignored, and the repository may have no git remote configured — so state exactly what a new machine needs (clone, virtual environment, which extra to install, `scripts/fetch_models.py` to fetch and hash-verify weights, how to select the provider, how to confirm which provider is live). If a fresh clone would not currently produce a working install, say so plainly in the report and fix the documented path.
6. **CUDA version coupling.** ORT's CUDA execution provider requires matching CUDA and cuDNN runtime versions. Record the versions the chosen `onnxruntime-gpu` build requires, and make provider initialisation failure produce a clear diagnostic naming the mismatch rather than a stack trace.
7. **Determinism is per-provider.** Different providers use different kernels, so byte-identical output across providers must not be assumed. Scope the recorded-mode determinism assertion **within** a provider, and **measure** cross-provider agreement (class-agreement rate and mean absolute box difference, as done previously for DirectML) rather than asserting equality. A provider that is faster but disagrees numerically must not be adopted silently.

## 14. PROVIDER BENCHMARKING

Extend `scripts/benchmark_providers.py` rather than writing a new one. For each available provider measure: warm-up ms, detector and pose latency p50/p95/max, throughput, end-to-end frame age, analysis FPS, peak RSS, CPU utilisation where practical, GPU utilisation and GPU memory where practical, and the numerical agreement check against the CPU baseline.

**Do not assume GPU is better** — for these nano models on small inputs, kernel launch overhead can dominate, and DirectML was already measured ~2.2× slower than CPU on this machine. Record what is actually measured.

Every result file must carry a **machine fingerprint**: CPU model, logical cores, RAM, GPU if any, OS, Python version, ONNX Runtime version and build, provider, thread settings, model version and hash. Results from the two laptops must never be conflated in a table without that fingerprint visible.

## 15. CAMERA BENCHMARKING

Benchmark the **laptop integrated camera** and the **OnePlus Nord 4 Windows virtual camera** separately and end to end. Do not assume they behave alike, and do not tune for one at the other's expense.

Per camera record: requested resolution, **achieved resolution**, capture FPS and jitter, preview FPS, analysis FPS, frame age p50/p95, end-to-end latency, dropped frames by reason.

**Never report a requested value as achieved.** The OnePlus has previously delivered 1280×720 when 1920×1080 was requested; record what the device actually gave.

---

## 16. TRACKING-READY PERCEPTION CONTRACTS

Ensure every detection and pose leaving perception carries what a future tracker needs, **without building a tracker**: class name and raw class, confidence, box in original frame pixel coordinates, `frame_id`, `capture_ts`, monotonic frame sequence, frame width and height, `policy_state` and `tier`, model version and active provider, and a stable within-frame ordering.

Group them into one **immutable per-frame record** (a `PerceptionFrame` or equivalent) so a future tracker consumes frames rather than reassembling parallel lists. Additive to existing contracts, with a `docs/decisions.md` line per addition. Real-time and recorded modes must emit the identical structure.

`StateSnapshot.tracks` stays empty. **Do not create `predictivesense/tracking/`, not even empty. Do not invent track IDs. Do not claim tracking exists.**

## 17. FUTURE PERSON–OBJECT RELATIONSHIPS — PREPARATION ONLY

A later phase will associate a person with a held object — person holding cup, phone, mouse, pen, bottle, charger, headphones — using box geometry, hand/keypoint proximity, temporal persistence, motion and occlusion handling. **None of that is implemented here.**

The only requirement now is that the contracts in section 16 preserve enough information for that layer to be built: person boxes and object boxes in the same coordinate space, pose keypoints with their own timestamp and staleness, and a per-frame identity that a tracker can key on. Do not fake any association, and do not add relationship fields that nothing populates.

## 18. REGRESSION CONSTRAINTS

Do not "fix" latency by weakening recognition. Specifically forbidden: lowering thresholds blindly, renaming detector classes in the UI, hard-coded class substitutions, suppressing confusing detections to improve appearance, faking tracking, faking custom recognition.

Preserve unchanged: the recognition policy and vocabulary tiers (overlay label exactly `Unknown`; `suppressed_implausible` hidden, never shown as `Unknown`; baseline classes working with no enrolment); the Object Learning Studio including bulk upload; dataset separation between `data/objects` and `data/eval`; the model registry; the evaluation harness; deterministic recorded processing (per provider); detection/tracking separation; local-first operation with no outbound network; the resizable panel.

**Studio uploads still do not change the active model, and nothing in this phase may imply otherwise.**

---

## 19. TESTS

**All existing tests must stay green** (374 passed / 2 skipped at last report). Markers as existing, plus a new `cuda` marker.

**Unit**

1. Completion-driven scheduling as a pure scheduler test: with inference slower than the ceiling, the loop takes the newest available frame each time and never waits on a tick; no busy-wait (assert the wait is event-based).
2. Staleness guard: frames older than `max_frame_age_ms` dropped and counted as `dropped_stale`; fresh frames pass; boundary case exact.
3. Counter reconciliation over a scripted sequence: captured = analysed + dropped_overwrite + dropped_stale + in-flight, exactly.
4. Mailbox depth never exceeds 1 (existing test extended, not weakened).
5. Pose reuse: reused pose carries its own `frame_id` / `capture_ts` / `age_ms` and `stale: true`; a pose beyond `pose_max_reuse_ms` is dropped; the reused object is not mutated (identity/immutability assertion).
6. `PerceptionFrame` contract: all required fields present, ordering stable, identical shape from real-time and recorded paths.
7. Provider resolution: `auto` picks the documented preference order and falls back to CPU; an explicitly requested unavailable provider fails loudly with an actionable message; the reported active provider comes from the live session, not from config.

**Integration**

8. Slow-detector simulation: end-to-end frame age stays bounded, drops rise, latency does not grow without limit over a sustained run.
9. Recorded mode remains deterministic **within the active provider**, with pose cadence configured.
10. Stage instrumentation produces a complete attribution record for an analysed frame, with no missing stage.
11. Diagnostics and the session manifest report the active provider and the machine fingerprint.

**CUDA-marked** — must skip cleanly with a clear message when `CUDAExecutionProvider` is unavailable, and must actually run when it is present:

12. CUDA session initialises; the reported active provider is `CUDAExecutionProvider`; inference produces valid detections.
13. Cross-provider agreement against the CPU baseline is computed and recorded (class-agreement rate, mean absolute box difference) — **measured, not asserted equal**.

**Browser**

14. Preview FPS unaffected during a deliberate 3-second analysis stall; recorded as a number.
15. Overlay marks a stale pose distinctly; zero console errors on `/` and `/studio`.

**Regression** — camera capture, Studio lifecycle and bulk upload, dataset separation, recognition policy behaviour, no-outbound-network, panel resize, model registry.

**The whole suite must pass on CPU-only hardware with no GPU package installed.**

## 20. PHYSICAL VERIFICATION (developer, not you — mark pending)

Automated tests are not sufficient evidence for any latency claim. Distinguish code-level verification from physical verification everywhere.

1. Laptop integrated camera: labels follow the scene noticeably faster than before; note the achieved resolution and the active provider shown in Diagnostics.
2. OnePlus virtual camera: the same, and confirm the reported achieved resolution matches what the device really delivers.
3. Move an object quickly across the frame; confirm the box follows the *current* position, not a past one.
4. Confirm the preview stays smooth during a deliberate stall.
5. Confirm a stale skeleton is visibly distinguishable from a fresh one.
6. Confirm baseline classes are still recognised and no label reads `Unknown (was …)`.
7. When the NVIDIA machine is available: repeat 1–6 there, record the active provider, and compare CPU and GPU results side by side with both machine fingerprints shown.

## 21. DOCUMENTATION AND RESULTS

Update `docs/architecture.md` (scheduling, staleness policy, pose cadence, provider selection), `docs/decisions.md` (one line per decision, each naming the result file that justified it), `CLAUDE.md` (current phase status, provider guidance), and `docs/setup.md` (bootstrap for both machines).

Write `docs/phase-reports/phase8.md` in the established three-part format: **Measured** (attribution tables per camera, before/after per change, counter reconciliation, input-size and pose-cadence experiments, provider benchmarks with machine fingerprints, test results), **Physically observed by the developer** (pending, section 20), **Not verified / limitations**.

Record baseline, each change, the final configuration, CPU results, GPU results where available, camera results, and known limitations. **Do not invent numbers.** Every figure names the command and machine that produced it.

## 22. ACCEPTANCE CRITERIA — MEASURED, NOT MERELY IMPLEMENTED

- [ ] Stage-attribution table published for both cameras **before** any optimization, and again after.
- [ ] **A measured reduction in end-to-end frame age p50 and p95 on both cameras**, with before/after figures and the command that produced each. If no change proves justified, the report states that explicitly with the evidence — but the attribution table must still identify where the time goes.
- [ ] Scheduling change adopted or rejected on evidence; no busy waiting; preview independence re-verified with a recorded preview-FPS figure during a stall.
- [ ] Single-slot mailbox preserved; all frame counters reconcile exactly; staleness policy documented and tested.
- [ ] Pose cadence decoupled or explicitly left alone on evidence; reused poses carry age and a stale flag; no shared-state mutation; overlay de-emphasises stale poses.
- [ ] Input size decided on evidence from both cameras including detection counts, or explicitly unchanged with the measurement recorded.
- [ ] `perception.provider` supports `auto | cpu | cuda | directml`; CPU is the default and the fallback; an explicitly requested unavailable provider fails loudly; the **actually active** provider is reported in Diagnostics, the session manifest and every result file.
- [ ] CPU-only test suite fully green with no GPU package installed; `cuda`-marked tests skip cleanly and are ready to run on the NVIDIA machine.
- [ ] `docs/setup.md` describes a bootstrap that makes a fresh clone runnable on both machines, including model fetching; any gap in the current clone-to-running path is stated.
- [ ] Provider benchmark extended; every result carries a machine fingerprint; cross-provider agreement measured, not assumed.
- [ ] Both cameras benchmarked separately with requested vs achieved recorded honestly.
- [ ] `PerceptionFrame` (or equivalent) emitted identically by real-time and recorded modes; `StateSnapshot.tracks` still empty; no tracking module created.
- [ ] Accuracy and latency reported separately, never combined into one score.
- [ ] Full suite green including browser and models; `docs/phase-reports/phase8.md` complete; tree committed and clean.

## 23. NON-GOALS

Custom model training or fine-tuning; custom classifier training; dataset export for training; model activation changes; Scene Snapshot Studio; risk prediction; temporal risk model; preventive recommendations; voice or TTS; a full object tracker; a person–object relationship engine; large UI redesign; cloud inference; external camera-processing services; any new runtime dependency beyond the optional provider extras.

Only tracking-**ready** output contracts and interfaces are in scope.

## 24. EXECUTION INSTRUCTIONS FOR CLAUDE CODE

Inspect before modifying, and answer every question in section 5 from the actual code before touching it. Reuse the existing telemetry, mailbox, benchmark scripts, config system, contracts and browser-test fixtures — do not fork or reimplement them, and do not create a parallel metrics system.

Implement **incrementally, one change at a time**: instrument first and publish the attribution table; then scheduling; then pose cadence; then input size; then provider portability. **Measure baseline, make one meaningful change, measure again, keep or revert on evidence.** Do not bundle unrelated optimizations into a single before/after. A change without a number is not finished.

Run the targeted tests after each step, then the full suite plus `-m browser`, `-m models` and `-m cuda`, and the performance benchmarks. Fix your own routine failures automatically and re-run — do not stop for ordinary test failures. Do not ask for approval on routine steps; stop only for a genuinely blocking ambiguity, and say precisely what is blocking.

Report: changes made, tests passed, tests skipped and why, warnings, benchmark results, the active provider, and measured before/after figures. Clearly distinguish code-level verification from physical verification throughout. Never claim a measurement you did not run, a physical check you did not perform, or a capability that does not exist.

**Do not start Phase 9. Do not train a model. Do not claim tracking or custom object recognition exists. Do not remove the CPU path.** When Phase 8's acceptance criteria are met, stop.
