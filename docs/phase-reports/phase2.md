# Phase 2 - Perception (detection + pose)

Per-frame perception only. An ONNX object detector and an ONNX pose estimator run
on the same sampled analysis frame in both Mode A (real-time) and Mode B
(recorded), through identical code. **No tracking, identity, association,
relations, temporal state, risk model, alert policy or voice** - none started,
none placeheld. This report has three separated parts, as required.

---

## PART 1 - MEASURED (commands run on this machine)

Machine: Windows 11, Intel Core Ultra 5 125H (14C / 18T), 16 GB LPDDR5, Intel Arc
integrated graphics, no NVIDIA / no CUDA. Python 3.11.9. `onnxruntime==1.24.4`
(CPU) in `.venv`; `onnxruntime-directml==1.24.4` in an isolated `.venv-dml`.

### Models

`scripts/fetch_models.py` downloaded and SHA-256-verified two pre-exported ONNX
files from the `ultralytics/assets` `v8.3.0` GitHub release (no `torch` /
`ultralytics` installed):

| file | sha256 (prefix) | size | task | input | licence |
|---|---|---|---|---|---|
| `yolo11n.onnx` | `634279b4…` | 10 930 182 B | detect (COCO-80) | 640, dynamic | AGPL-3.0-only |
| `yolo11n-pose.onnx` | `93e2866b…` | 11 853 451 B | pose (17 kpt) | 640, **locked** | AGPL-3.0-only |

Only `models/manifest.json` is committed. Runtime makes no network call - the
guard test `test_no_outbound_network.py` (scans `predictivesense/`, never
`scripts/`) still passes.

### Test suite

- `pytest -q` → **205 passed, 1 skipped** (hardware; `--run-hardware` to run).
- `pytest -q -m models` → **11 passed** (detector/pose validity on 3 fixture
  frames, recorded determinism with models, person-detected sanity,
  preview-independence-with-perception). Skips cleanly with a clear message when
  `models/` is empty (`require_models` fixture) - never fails, never silently
  passes.
- `pytest -q -m "not slow"` → green.
- Amended `test_no_forbidden_imports.py`: `onnxruntime` permitted only under
  `perception/`; `cv2` under `camera/` **and** `perception/`; every other ban
  unchanged; a self-check that a fabricated violation is still caught.
- `test_recorded_driver.py` (Phase 1) unchanged and green - `RecordedDriver`
  without a `perception=` argument still produces the exact Phase 1 empty shape.

### Perception latency - `scripts/benchmark_providers.py` (150 frames of the
developer's 1920×1080 clip, warm)

> **Latency protocol note (added Phase 6).** These figures come from
> `benchmark_providers.py`, which times the **detector and pose in separate
> passes** with no per-frame interleaving. That is *not* the canonical
> **Protocol A** defined in `docs/decisions.md` (Phase 6) - Protocol A runs both
> ORT sessions per frame, as the analysis loop does, and measures ~2× higher
> detector p50 on the developer's footage. The `detector p50 46.75 ms` below is
> a lower bound (isolated single-model), not the in-loop cost. See
> `docs/phase-reports/phase6.md` and `results/recognition_paths.md` for the
> reconciled numbers.

**CPU** (`results/providers_cpu.json`, shipped `intra_op_threads: 6`):

| model | warm-up ms | p50 ms | p95 ms | max ms | throughput fps | peak process RSS MB |
|---|---|---|---|---|---|---|
| detector | 44.0 | 46.75 | 53.84 | 63.73 | 21.4 | 1145 |
| pose | 52.0 | 49.70 | 54.37 | 57.83 | 20.1 | 1177 |

Combined detector+pose p50 **96.5 ms**; process CPU ≈ 634 % (≈ 6 cores).
Peak RSS is the whole process (Python + numpy + cv2 + ORT + weights), not the
model alone.

**DirectML** (`results/providers_dml.json`, isolated `.venv-dml`):

| model | warm-up ms | p50 ms | p95 ms | max ms | throughput fps |
|---|---|---|---|---|---|
| detector | 216.9 | 110.32 | 123.54 | 132.61 | 9.1 |
| pose | 173.7 | 106.08 | 112.92 | 125.03 | 9.4 |

Combined p50 **216.4 ms** (~2.2x slower than CPU).

**Agreement check, DirectML vs the CPU baseline (150 frames):**
detection-count match rate **1.0**, class-agreement rate **1.0**, mean |box
diff| **0.0 px**, max |box diff| **0.002 px**. DirectML would *pass* the
agreement bar - it is rejected purely on speed: the nano-model kernels never
saturate the Arc iGPU while the 14-core CPU does. **Default `perception.provider:
cpu`** (comment in `dev.yaml` / `eval.yaml` names the two result files).
OpenVINO / the NPU were **not evaluated** (optional/deferred, Block 3.4.16).

### ORT thread contention (why `intra_op_threads` exists)

With ORT `intra_op_num_threads` left at auto (all 18 logical cores), running two
sessions back-to-back **under a competing background thread** collapses combined
detector+pose p50 from ~90 ms to **~440 ms** (measured). Capping each session's
intra-op threads: 4 → ~148 ms, **6 → ~107 ms**, 8 → ~116 ms. Shipped default
`perception.intra_op_threads: 6` (the knee). Auto is fine only for an isolated
one-shot; the analysis loop is never that.

### Detector input-size sweep {320, 480, 640} - `results/input_size_sweep.md`

(CPU, `intra_op_threads: 6`, 150 frames, 1920×1080)

| input | p50 ms | p95 ms | max ms | mean det/frame | total det | person frame rate | warm-up ms |
|---|---|---|---|---|---|---|---|
| 320 | 16.23 | 20.62 | 25.61 | 2.013 | 302 | 1.0 | 13.0 |
| 480 | 30.27 | 34.70 | 36.05 | 1.880 | 282 | 1.0 | 26.7 |
| 640 | 48.12 | 52.35 | 54.41 | 1.113 | 167 | 1.0 | 48.7 |

Latency scales ~linearly with input area. On **this** clip (dominated by one
large, close person) the person is found in every frame at every size, so
person-recall does not discriminate. Detection *count* falls as input grows -
that is fewer fragmented / duplicate boxes, not fewer true objects; the small
objects (`cup`, `cell phone`) that need resolution are under-represented in this
single clip. **Default kept at 640** for small-object sensitivity; 480 is the
obvious fallback if a slower machine needs the ~18 ms.

### Achieved analysis FPS - perception ON vs OFF (browser-ingest loopback,
10 fps feed, 640×480, 20 s each)

| | analysis_fps p50 | drop rate | frame age p50 / p95 ms | mailbox depth |
|---|---|---|---|---|
| perception OFF | 20.0 | 0.000 | 49.5 / 69.2 | ≤ 1 |
| perception ON | **10.08** | **0.027** | **145.7 / 192.3** | ≤ 1 |

With perception on, the analysis loop keeps up with the 10 fps feed (10.08 fps,
2.7 % steady-state drop); frame age rises from ~50 ms to ~146 ms (the ~96 ms
perception cost plus one sample period of queueing). The consumer sample rate
(dev: 20 Hz) is now the ceiling only when perception is off.

### Preview independence re-verified WITH perception running

`tests/integration/test_perception_pipeline.py::test_preview_independence_holds_
with_perception_running` (models-marked): browser-ingest source + detection +
pose enabled, a `POST /api/debug/stall` of 3 s, ~170 frames pushed over
`/ws/ingest` during the stall.

- socket kept accepting every frame (`frames_submitted == sent`), each
  `send_bytes` returned in < 1 s;
- mailbox stayed single-slot (`depth ≤ 1`, `max_mailbox_depth ≤ 1`);
- producer thread stayed alive; `loop.error is None`;
- **drop rate under the artificial stall: 0.97** (consumed 5, dropped 164) - by
  design: a paused consumer drops, the preview and ingest do not. Steady-state
  (no stall) drop rate with perception on is 0.027 (table above).

The Phase 1 automated `test_preview_independence.py` and the stall test also stay
green (those fixtures run with perception off - the Phase 1.6 baseline).

### Recorded-mode determinism WITH the model

`test_recorded_determinism_with_models.py`: two `RecordedDriver` runs over the
same clip + config + weights produce **byte-identical** JSONL, detections and
poses included (CPU ORT is deterministic for identical input; coords written at
2 dp, scores at 4 dp, fixed key order). A full CLI run
(`scripts/run_recorded.py --profile eval`) over the 501-frame `data/raw/` clip:
501 frames → 588 detections / 502 poses, 0 frame errors, manifest carries the
perception block.

### Class-coverage audit - `results/class_coverage.md` /
`results/class_coverage.json`

`scripts/class_coverage_audit.py --source data/raw` over the **one** clip present
in `data/raw/` (501 frames, stride 1, detector `yolo11n.onnx` @ 640, default_conf
0.35). The file header states in full that these are **detection frequencies on
unlabelled footage, not accuracy**.

| class | frames w/ ≥1 | rate | conf p10 / p50 / p90 | median box area frac | frames w/ >1 | verdict |
|---|---|---|---|---|---|---|
| person | 501 | 1.000 | 0.88 / 0.902 / 0.92 | 0.357 | 4 | *(developer)* |
| cup | 17 | 0.034 | 0.397 / 0.444 / 0.607 | 0.014 | 0 | *(developer)* |
| cell phone | 32 | 0.064 | 0.382 / 0.544 / 0.748 | 0.009 | 0 | *(developer)* |
| bottle | 0 | 0 | – | – | 0 | *(developer)* |
| laptop | 0 | 0 | – | – | 0 | *(developer)* |
| chair | 0 | 0 | – | – | 0 | *(developer)* |
| backpack | 0 | 0 | – | – | 0 | *(developer)* |
| handbag | 0 | 0 | – | – | 0 | *(developer)* |
| book | 0 | 0 | – | – | 0 | *(developer)* |
| keyboard | 0 | 0 | – | – | 0 | *(developer)* |
| mouse | 0 | 0 | – | – | 0 | *(developer)* |
| scissors | 0 | 0 | – | – | 0 | *(developer)* |

Top unexpected classes (likely false positives): `umbrella` 21 frames (0.042),
`surfboard` 5, `couch` 3, `frisbee` 2, `bench` 2, `remote` 1.

The zeros mean *only* that those objects do not appear (or are not detected) in
this single 20-second clip - **not** that the class is unusable. The verdict
column (`reliable` / `marginal` / `unusable`) is the developer's to fill after
running the audit over more footage and watching the overlay (Part 2 below). The
MVP scenario list from blueprint addendum v3 §C depends on that filled table.

### UI (checked in the in-app Browser pane; camera capture is blocked there)

- The reserved `analysis` slot is filled by `groups/analysis.js` (order 50).
  Detection and Pose register as sub-modules via the existing
  `registerAnalysisModule(...)`. Each shows an overlay toggle, read-only model /
  input size / thresholds, and a live summary line.
- `#overlay-canvas` is created inside `#overlay-layer`; a simulated snapshot with
  detections + a pose drew boxes and a skeleton (coloured pixels 8533, skeleton
  green 2831) with no console error; a `stale` snapshot dimmed the overlay and
  drew the `STALE` label. The `<video>` element is never read, drawn into, or
  replaced (`test_overlay_never_touches_the_video_element`).
- Diagnostics gained a Perception block: provider, detector/pose model + input
  size, `pose_every_n`, detector/pose p50 & p95, per-frame counts, warm-up times,
  frame errors.
- With `perception.detection_enabled` / `pose_enabled` both false the loop loads
  no session, snapshots carry `detections: []` / `poses: []`, none of the new
  metric keys appear, and the UI Analysis group shows "Perception idle" -
  `test_perception_off_reproduces_phase_1_6` asserts this.
- All Phase 1.6 UI static-analysis tests pass; `test_ui_structure.py` was updated
  to expect `analysis` registered and only `alerts` / `research` reserved (same
  precedent as Phase 1 updating the Phase 0 `index.html` assertions).

---

## PART 2 - PHYSICALLY OBSERVED BY THE DEVELOPER

**Status: PENDING - none performed this pass.** The in-app Browser pane blocks
camera capture, so no browser ran the real detector/pose/overlay path against a
live camera or a played-back file this pass. The developer must perform and
record:

1. Place each required object (`person`, `cup`, `bottle`, `laptop`, `chair`,
   `backpack`, `handbag`, `book`, `cell phone`, `keyboard`, `mouse`, `scissors`)
   on the desk in normal cabin lighting and confirm on the overlay which classes
   appear solidly, which flicker (land in the dashed low-confidence band), and
   which never appear.
2. Sit in frame and confirm the skeleton tracks posture changes (stand / sit /
   lean / arms up), and that low-visibility keypoints render de-emphasised.
3. Confirm the live preview stays visibly smooth with perception running, and
   during a `POST /api/debug/stall` - the overlay should freeze/label `STALE`
   while the preview does not stutter.
4. Confirm the overlay boxes stay aligned to the video when the window is resized
   and when the control panel is collapsed.
5. Run `scripts/class_coverage_audit.py` over several real clips and fill the
   `verdict` column of `results/class_coverage.md` (`reliable` / `marginal` /
   `unusable` per class).

---

## PART 3 - NOT VERIFIED / LIMITATIONS

- **No accuracy, precision, recall, mAP, F1 or any error rate exists** for the
  detector or the pose model, because **there is no labelled data**. Every number
  in `results/class_coverage.*` is a detection *frequency* or a *confidence
  distribution* on unlabelled footage. Nothing in this phase may be cited as
  precision/recall/mAP.
- The class-coverage audit ran over **one** 20-second clip (the only file in
  `data/raw/`). The zeros for 9 of the 12 required classes reflect that clip's
  content, not model capability. The table is not usable for scenario selection
  until it is re-run over footage that actually contains those objects.
- **No tracking, no track IDs, no association, no `Relation`, no temporal
  features, no risk inference, no alert policy, no TTS, no overlay of any of
  those.** `StateSnapshot.tracks` is still always `[]`. The overlay shows no IDs.
- **No fine-tuning, training, or dataset pipeline.** The models are used
  as-published.
- The live overlay path is **browser-ingest only**. Backend-owned camera mode has
  no preview by design; recorded-mode analysis is retrospective (JSONL, not
  `/ws/state`) so it has no live overlay. The overlay's coordinate reference is
  `capture.analysis_width/height`; if a future non-browser live path streams
  snapshots it will need the frame dimensions carried too.
- DirectML numbers are from a **separate `.venv-dml`**; they are not part of the
  shipped environment and are not re-checked by the test suite.
- Peak RSS reported by `benchmark_providers.py` is the whole Python process
  (~1.15 GB with both sessions warm), not an isolated model footprint.
- **The AGPL-3.0 licence of the model weights is an open, unresolved decision**
  (`docs/attribution.md`). It is recorded, not resolved.
- The eight Block 10 physical camera/mic/speaker checks from Phase 1, and all of
  Part 2 above, remain the developer's and are **pending**.
- Phase 3 has **not** been started.
