# PredictiveSense — Phase 2: Perception (Detection + Pose)

> Paste this entire file into Claude Code in a **fresh session**. Do not summarise it.
> **Scope: per-frame perception only.** No tracking, no IDs, no relations, no temporal state, no risk, no voice, no fine-tuning.

---

## BLOCK 1 — CONTEXT

PredictiveSense is a local-only research prototype on a one-month deadline, supporting a research paper. Phases 0, 1, 1.5 and 1.6 are complete and committed: typed contracts, strict config, telemetry, single-slot mailbox, browser-owned camera with native `srcObject` preview, Web Worker analysis path over `WS /ws/ingest`, recorded-video driver, clip recorder, camera benchmarking and recovery, and an application shell with a group registry (`static/ui/`, `static/groups/`, `static/features/`).

Read first and work from what is present: `CLAUDE.md`, `docs/architecture.md`, `docs/decisions.md`, `docs/phase-reports/phase1.md`, `predictivesense/core/types.py`, `predictivesense/pipeline/loop.py`, `predictivesense/pipeline/recorded.py`, `predictivesense/api/static/ui/registry.js`.

**Target machine:** Windows, Intel Core Ultra 5 125H (14C/18T), 16 GB LPDDR5, Intel Arc integrated graphics + Intel NPU, **no NVIDIA GPU, no CUDA**. Python 3.11.9 in `.venv`.

**What this phase is for.** Everything downstream — tracking, relations, temporal features, the risk model — consumes per-frame perception output. This phase produces that output, measures what it costs on this machine, and answers the question that decides the final scenario list: **which required object classes does a pretrained detector actually recognise in this room?**

---

## BLOCK 2 — OBJECTIVE

Run an ONNX object detector and an ONNX pose estimator on the sampled analysis frame, publish their results through the existing `StateSnapshot`, draw them on the viewport overlay with honest uncertainty styling, register Detection and Pose as Analysis sub-modules in the existing UI, benchmark execution providers on this machine, and produce a **class-coverage audit** over the developer's own footage.

Both Mode A (real-time) and Mode B (recorded) must use the identical perception code.

---

## BLOCK 3 — REQUIREMENTS

### 3.1 Model acquisition — build-time, never runtime

1. Create `models/` (git-ignored) and `scripts/fetch_models.py`. The script downloads the detector and pose weights **once**, verifies a SHA-256 recorded in `models/manifest.json` (committed), exports to ONNX if needed, and writes the model's licence text beside it. If a hash mismatches, it fails loudly and deletes nothing.
2. **Runtime must never download anything.** The existing outbound-network guardrail test stays green; `fetch_models.py` is a setup script, not part of the runtime package, and must be excluded from that guard by path, not by weakening the guard.
3. Default models: a small YOLO-family detector (`yolo11n` or `yolov8n`) and the matching pose variant (`yolo11n-pose` / `yolov8n-pose`), both exported to ONNX at a fixed input size. **Record the licence prominently in `docs/attribution.md`** — the Ultralytics models are AGPL-3.0 and that has consequences for how this repository may be released. Note it as an open decision; do not resolve it.
4. The detector and pose wrappers must be **model-agnostic**: model path, input size, class-name list, and pre/post-processing variant all come from config. Swapping to a different ONNX model must be a config change, not a code change. State this in `docs/decisions.md`.

### 3.2 Perception modules

5. Create `predictivesense/perception/` with: `detector.py` (ONNX object detector), `pose.py` (ONNX pose estimator), `runtime.py` (ONNX Runtime session creation, provider selection, warm-up), `classes.py` (class-id → name map plus the user-facing alias map), `types.py` if a `PerceptionResult` aggregate is needed beyond the existing `Detection` / `Pose` contracts.
6. `ObjectDetector.infer(frame) -> list[Detection]`: letterbox resize to the model input, normalise, run, decode, class-aware NMS, map back to original pixel coordinates. Per-class confidence thresholds from config with a documented default; a single global threshold is not acceptable.
7. `PoseEstimator.infer(frame) -> list[Pose]`: keypoints with visibility, person box, score. Keypoint names and skeleton edges defined once in `classes.py` and reused by the overlay.
8. Both run on the **same sampled frame** inside the existing analysis loop. Config `perception.pose_every_n` (default 1) allows pose on alternate frames if the latency budget demands it — measure before changing the default.
9. **Warm-up:** run each model once on a synthetic frame at startup so the first real frame is not an outlier in the latency measurement. Record warm-up time separately.
10. Session creation must be explicit about providers, must never assume CUDA, and must log which provider was actually used — not which was requested.

### 3.3 Wiring into both modes

11. Extend the analysis loop so `StateSnapshot.detections` and `.poses` carry real values. Contracts do not change shape — they were defined in Phase 0 for this.
12. `RecordedDriver` runs the identical perception path, still lossless and still byte-deterministic for a given model, config and file. Determinism now includes the model: assert two runs over the same clip produce identical detections.
13. Perception is switchable at runtime through config flags `perception.detection_enabled` and `perception.pose_enabled`, surfaced as toggles in the UI. With both off, the system must behave exactly as it did in Phase 1.6.

### 3.4 Execution-provider benchmark

14. **Packaging reality:** `onnxruntime`, `onnxruntime-directml` and `onnxruntime-openvino` provide the same module name and cannot coexist in one environment. Do not try. Keep the main `.venv` on plain `onnxruntime` (CPU). For the GPU path, create a **separate** `.venv-dml` with `onnxruntime-directml` and run the same benchmark script there via its own interpreter.
15. `scripts/benchmark_providers.py --provider cpu|dml` runs both models over a fixed fixture clip and reports, per model: warm-up ms, inference p50/p95/max ms, throughput, peak RSS, and a **numerical agreement check** against the CPU baseline (mean absolute box difference and class-agreement rate) so a faster provider that changes results is not silently adopted. Writes `results/providers_<provider>.json` plus a markdown table.
16. OpenVINO and the NPU are **optional and deferred**. If time permits, benchmark them the same way; if not, record in the report that they were not evaluated. Do not spend the phase on packaging archaeology.
17. Choose the default provider from the measured table, write it into the config with a comment naming the result file, and state the choice in `docs/decisions.md`.

### 3.5 Class-coverage audit — the phase's most important output

18. `scripts/class_coverage_audit.py --source <clip-or-directory>` runs the detector over the developer's recorded clips (`data/raw/`) and/or a directory of stills, and produces `results/class_coverage.json` and a readable `results/class_coverage.md`.
19. For each **required class** — `person`, `cup`, `bottle`, `laptop`, `chair`, `backpack`, `handbag`, `book`, `cell phone`, `keyboard`, `mouse`, `scissors` — report: number of frames with ≥1 detection, detection rate as a fraction of frames, confidence distribution (p10/p50/p90), median box area as a fraction of frame, and the count of frames where the class was detected more than once.
20. Also report the **top unexpected classes** by frequency — these are the likely false positives (a hand read as a person, a book read as a TV, and so on).
21. **This audit produces detection frequency and confidence, not accuracy.** There are no labels yet. The output file must say so in its own header, and the phase report must not present any number from it as precision, recall or mAP.
22. The audit ends with a **per-class verdict column** the developer fills in or confirms: `reliable` / `marginal` / `unusable`. The MVP scenario list from blueprint addendum v3 section C depends on this table, so it must be legible on its own.

### 3.6 Overlay and UI

23. Draw on the existing `#overlay-layer` canvas above the video — never into the `<video>` element, never replacing it. The overlay is sized to the displayed video box and must stay correct when the window resizes or the panel collapses.
24. Per detection: box, class name (user-facing alias, not the raw model label), confidence value and a short confidence bar. Colour is per class, deterministic. **No track IDs — there is no tracker yet.**
25. **Uncertainty must look uncertain.** Detections in the configured low-confidence band render dashed and at reduced opacity. Never style a marginal detection like a confident one.
26. Per pose: skeleton edges and keypoints, with low-visibility keypoints visually de-emphasised.
27. Register an **Analysis** group and two sub-modules through the existing registry — `registerAnalysisModule({id:"detection"...})` and `{id:"pose"...}`. Each carries: enable toggle, model name (read-only), input size, confidence threshold, and a live summary line. **Do not modify the shell, the group taxonomy, or any Phase 1.6 file beyond registering into them.**
28. Diagnostics gains: detector ms p50/p95, pose ms p50/p95, provider in use, model names and input sizes, detections per frame, poses per frame, warm-up times.
29. The overlay must degrade honestly: if the last snapshot is stale beyond the existing threshold, dim the overlay and label it stale rather than showing old boxes as current.

### 3.7 Measurement

30. Measure and report: detector latency p50/p95/max, pose latency p50/p95/max, combined per-frame perception cost, achieved analysis FPS with perception on versus off, drop rate change, frame age change, peak RSS, and CPU%. Sweep detector input size over {320, 480, 640} and report the latency/detection-count trade-off.
31. **Preview independence must be re-verified with perception running** — this is the first phase where the analysis path is genuinely slow, so it is the first real test of the architecture. Run the existing stall test and the automated independence test, and record the drop rate under load.

---

## BLOCK 4 — CONSTRAINTS

- Preview independence is inviolable. A slow detector must never affect preview FPS.
- Single-slot mailbox, newest-wins, worker backpressure: unchanged.
- No CUDA, no NVIDIA assumptions, no runtime network access, no cloud inference.
- Detection and tracking stay separate — this phase produces per-frame detections only, with no identity, no association, no history.
- No unbounded queues, caches, or per-frame accumulating lists.
- Do not modify Phase 1.6 shell files except to register into them.
- Do not rewrite working camera, worker, recording or recorded-driver logic.
- ONNX Runtime session objects are created once and reused; never per frame.
- Model files never enter git — only `models/manifest.json` with hashes and licences.

---

## BLOCK 5 — FILES

**Create**

```
predictivesense/perception/__init__.py
predictivesense/perception/runtime.py          # ORT session, provider selection, warm-up
predictivesense/perception/detector.py
predictivesense/perception/pose.py
predictivesense/perception/classes.py          # class map, alias map, keypoint names, skeleton edges
predictivesense/perception/preprocess.py       # letterbox, normalise, coordinate mapping back
scripts/fetch_models.py
scripts/benchmark_providers.py
scripts/class_coverage_audit.py
predictivesense/api/static/groups/analysis.js  # Analysis group + registerAnalysisModule host
predictivesense/api/static/features/detection.js
predictivesense/api/static/features/pose.js
predictivesense/api/static/features/overlay.js # canvas drawing on #overlay-layer
models/manifest.json                           # committed: names, sha256, licences, input sizes
tests/unit/test_preprocess.py
tests/unit/test_postprocess_nms.py
tests/unit/test_classes.py
tests/integration/test_perception_pipeline.py
tests/integration/test_recorded_determinism_with_models.py
tests/fixtures/perception/                     # 2–3 small committed JPEGs with known content
docs/phase-reports/phase2.md
```

**Modify**

```
pyproject.toml                          # add onnxruntime (CPU) and any minimal helper; pin versions
requirements.lock.txt                   # regenerate, UTF-8 no BOM
predictivesense/config/settings.py      # perception.* section
config/profiles/dev.yaml, eval.yaml     # perception defaults with justifying comments
predictivesense/pipeline/loop.py        # call perception, populate snapshot
predictivesense/pipeline/recorded.py    # same perception path
predictivesense/api/static/app.js       # register the analysis group
predictivesense/api/static/groups/diagnostics.js   # perception rows
tests/unit/test_no_forbidden_imports.py # allow onnxruntime under perception/ only; cv2 stays camera/ + perception/
docs/architecture.md, docs/decisions.md, docs/attribution.md, CLAUDE.md
```

**Must NOT be created or modified**

- `predictivesense/tracking/`, `scene/`, `temporal/`, `risk/`, `policy/`, `audio/`, `research/` — none exist; none may be created, not even empty.
- `predictivesense/camera/mailbox.py`, `camera/synthetic.py`, `telemetry/*`, `core/types.py` (shape unchanged), `analysis-worker.js`.
- `static/ui/*` — register into the registry; do not edit it.
- Any prior prompt file.

If one of these genuinely blocks you, **stop and ask**.

---

## BLOCK 6 — CONFIG

```yaml
perception:
  detection_enabled: true
  pose_enabled: true
  pose_every_n: 1
  provider: cpu                 # chosen from results/providers_*.json
  detector:
    model_path: models/yolo11n.onnx
    input_size: 640
    nms_iou: 0.5
    default_conf: 0.35
    class_thresholds: {}        # per-class overrides, set from the audit
    low_confidence_band: [0.35, 0.50]   # rendered dashed / reduced opacity
    max_detections: 100
  pose:
    model_path: models/yolo11n-pose.onnx
    input_size: 640
    conf: 0.4
    max_persons: 4
    keypoint_visibility_threshold: 0.3
```

---

## BLOCK 7 — ERROR HANDLING

**Fail loudly, exit non-zero:** a configured model file missing or failing its hash; an unsupported provider requested; an input size the model does not accept.

**Degrade gracefully, log once:** an inference exception on a single frame (count it, drop that frame's results, keep the loop alive); a malformed frame; models disabled by config.

**Never:** create the ORT session per frame; let a perception failure kill the analysis loop or the preview; silently fall back to a different provider without logging it; report a requested provider as the one in use.

---

## BLOCK 8 — TESTS

Markers: `unit`, `integration`, `slow`, `hardware`, plus a new **`models`** marker for tests requiring weights on disk. `models` tests skip cleanly with a clear message when `models/` is empty — never fail, never silently pass.

**Unit (no weights needed)**

1. `test_preprocess` — letterbox is aspect-preserving; coordinate mapping round-trips box corners back to original pixels within tolerance for several aspect ratios.
2. `test_postprocess_nms` — class-aware NMS on hand-built boxes: overlapping same-class suppressed, overlapping different-class kept, `max_detections` respected, empty input safe.
3. `test_classes` — every required class in Block 3.5 resolves; alias map has no duplicate targets; keypoint names and skeleton edge indices are consistent.

**Integration (`models` marker)**

4. `test_perception_pipeline` — fixture image through detector and pose: returns valid `Detection`/`Pose` objects, boxes inside frame bounds, scores in [0,1], keypoint count correct.
5. `test_recorded_determinism_with_models` — the same clip twice yields identical detections, byte for byte.
6. Perception disabled → snapshot detections and poses are empty and behaviour matches Phase 1.6.

**Regression (must stay green)**

7. `test_preview_independence`, `test_ingest_socket`, `test_recorded_driver`, `test_recorder_upload`, `test_camera_recovery`, `test_no_outbound_network`, all Phase 1.6 UI structure tests.

8. Amended `test_no_forbidden_imports` — `onnxruntime` permitted only under `perception/`; `cv2` permitted under `camera/` and `perception/`; every other ban unchanged; `scripts/` excluded from the runtime guard by path.

---

## BLOCK 9 — COMMANDS

Working directory `C:\Users\mummi\Documents\Projects\PredictiveSense`.

```powershell
cd C:\Users\mummi\Documents\Projects\PredictiveSense
.\.venv\Scripts\Activate.ps1

pip install -e ".[dev]"
pip freeze --exclude-editable | Out-File -Encoding utf8NoBOM requirements.lock.txt

python scripts\fetch_models.py

pytest -q
pytest -q -m "not slow"
pytest -q -m models

python scripts\benchmark_providers.py --provider cpu

# DirectML in an isolated environment — do not install it into .venv
py -3.11 -m venv .venv-dml
.\.venv-dml\Scripts\Activate.ps1
pip install -e "." ; pip install onnxruntime-directml
python scripts\benchmark_providers.py --provider dml
deactivate

.\.venv\Scripts\Activate.ps1
python scripts\class_coverage_audit.py --source data\raw
python scripts\run_app.py --profile dev --source-kind browser
```

Run everything yourself. Fix your own failures and re-run. Do not ask the developer to run routine commands.

---

## BLOCK 10 — ACCEPTANCE CRITERIA

- [ ] `fetch_models.py` retrieves and hash-verifies both models; `models/manifest.json` committed with licences; runtime makes no network call and the guardrail test still passes.
- [ ] Detector and pose run on the same sampled frame in both real-time and recorded modes.
- [ ] `StateSnapshot.detections` and `.poses` populated; contract shapes unchanged.
- [ ] Overlay draws boxes, labels, confidence and skeletons on `#overlay-layer`, correct after resize and panel collapse; low-confidence detections visibly dashed; stale overlay labelled stale; no track IDs shown.
- [ ] Analysis group registered with Detection and Pose sub-modules; no Phase 1.6 shell file modified beyond registration; Diagnostics carries the new rows.
- [ ] Toggling perception off reproduces Phase 1.6 behaviour exactly.
- [ ] Provider benchmark run for CPU and DirectML with an agreement check against CPU; default chosen from the table and justified in config comments and `docs/decisions.md`.
- [ ] Input-size sweep {320, 480, 640} measured and tabulated.
- [ ] **Class-coverage audit produced** over the developer's own footage, with the "these are frequencies, not accuracy" header and a verdict column.
- [ ] Preview independence re-verified with perception running; drop rate under load recorded.
- [ ] Recorded-mode determinism holds with models loaded.
- [ ] Full suite green; `models` tests skip cleanly when weights are absent.
- [ ] `docs/phase-reports/phase2.md` written in the three-part format; tree committed; no remote. **Phase 3 not started.**

---

## BLOCK 11 — PHYSICAL VERIFICATION (developer, not you)

Mark as pending and list:

1. Place each required object on the desk in normal cabin lighting and confirm on the overlay which classes appear and which flicker or never appear.
2. Sit in frame and confirm the skeleton tracks posture changes.
3. Confirm the preview stays smooth with perception running, and during the stall hook.
4. Confirm the overlay stays aligned when the window is resized and the panel is collapsed.
5. Fill in the verdict column of `results/class_coverage.md`.

---

## BLOCK 12 — REPORT

`docs/phase-reports/phase2.md`, three separated parts: **Measured** (latencies, FPS with/without perception, provider table with agreement check, input-size sweep, RSS, drop rate, test results), **Physically observed by the developer** (pending until performed), **Not verified / limitations** — which must state explicitly that no accuracy, precision, recall or mAP figure exists because no labelled data exists yet, that class-coverage numbers are detection frequencies on unlabelled footage, and that no tracking, temporal reasoning, risk inference or voice exists.

Reply with only:

```
IMPLEMENTED: ...
TESTS: ...
MEASUREMENTS: ...
CLASS COVERAGE SUMMARY: ...
LIMITATIONS: ...
PHASE 3 NOT STARTED: confirmed
```

---

## BLOCK 13 — PROHIBITIONS

1. Do not implement tracking, track IDs, association, relations, temporal state, risk inference, alert policy, or voice.
2. Do not fine-tune, train, or modify any model. Do not create a dataset pipeline.
3. Do not report detection frequency as accuracy, precision, recall or mAP.
4. Do not claim a class works because it appeared once; the audit reports rates, and the verdict is the developer's.
5. Do not download anything at runtime; only `fetch_models.py` may fetch, and only when run manually.
6. Do not install `onnxruntime-directml` or `onnxruntime-openvino` into the main `.venv`.
7. Do not write CUDA-specific or NVIDIA-assuming code.
8. Do not create the ORT session per frame, or any unbounded queue or cache.
9. Do not break preview independence, the mailbox, or worker backpressure.
10. Do not modify the Phase 1.6 shell, registry, or group taxonomy — register into them.
11. Do not commit model weights.
12. Do not claim a performance number you did not measure on this machine, or a physical check you did not perform.
13. Do not stop for routine errors — diagnose, fix, re-run.
14. Do not mark an acceptance item complete without running the check.

When Phase 2 is finished, **stop**. Do not begin Phase 3.
