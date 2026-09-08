# Phase 5 - Recognition Trust & Studio Repair

Scope: fix the three things that were wrong (an over-aggressive recognition
whitelist; "Unknown (was Bed)" conflating a suppressed known class with a genuine
unknown; unfitted thresholds presented as fitted), plus the Object Learning
Studio's lifecycle bugs and its inferior capture path. **No model trained or
fine-tuned. No model swap, quantisation, or threading redesign. No tracker,
temporal reasoning, risk, or voice. No Scene Snapshot Studio (that is Phase 6).**

Three separated parts, as required.

---

## PART 1 - MEASURED (commands run on this machine)

Machine: Windows 11, Intel Core Ultra 5 125H (14C / 18T), 16 GB LPDDR5, Intel Arc
integrated graphics, no NVIDIA / no CUDA. Python 3.11.9, `.venv`.

### Test suite

- `pytest -q` -> **321 passed, 2 skipped** (1 `hardware` - needs `--run-hardware`;
  1 `dataset` - skips cleanly, the evaluation set is still unlabelled). Baseline
  at the end of Phase 4 was 292 passed / 2 skipped; **+29 tests, 0 regressions**.
- `pytest -q -m models` -> **12 passed** (unchanged - Phase 5 touched no
  detector/pose inference code; `perception/detector.py` and `perception/pose.py`
  were not modified).
- `pytest -q -m dataset` -> **1 skipped** with the "label frames at /label"
  message.
- New tests:
  - `tests/unit/test_vocabulary_tiers.py` - the default tiers are a total,
    disjoint partition of COCO-80 (14 + 26 + 40 = 80); an unknown class name in a
    tier, a missing class, an overlap, and a within-tier duplicate each fail
    loudly; `tier_for` is correct for every tier and returns `unlisted` for a
    non-COCO class; both shipped profiles carry a valid partition and
    `domain_classes == primary`; a config with a broken partition raises.
  - `tests/unit/test_policy_states.py` - each of the six `policy_state` values is
    produced by a constructed detection; the counts reconcile exactly across all
    six; `suppressed_implausible` is not `unknown` (and keeps its `tier`); a
    disabled policy is a pass-through that still annotates `tier`; a rule that
    raises is caught, counted (`errors == 1`), and the frame passes through.
  - `tests/unit/test_policy_rules.py` - the *extended* Phase 2.5 reconciliation
    test: implausible -> suppressed (not unknown), secondary -> shown
    de-emphasised, below-threshold / small-margin -> unknown, tiny box / bad
    aspect -> rejected_size, confident primary -> accepted; all six states in one
    frame reconcile; each rule independently switchable; disabled = pass-through;
    per-class override beats the default.
  - `tests/integration/test_studio_state_machine.py` - an executable Python
    mirror of `studio-state.js` (kept in step with the shipped JS by a static
    guard) drives every transition and the three reported faults: back-to-camera
    preserves the selected object and clears stale inspect state; re-selecting an
    object clears a stale/dirty inspect; Save & Return is blocked while a capture
    is staged and unblocks once it is saved/discarded. Static assertions that
    `studio.js` uses the machine, dispatches all six transitions, and **never
    calls `location.reload`** (exactly one `window.location.href`, in
    `saveAndReturn`).
  - `tests/integration/test_studio_capture_quality.py` - a captured sample stores
    a full-resolution original at the achieved resolution with a **separate**
    thumbnail; `capture_path` / `requested_resolution` / `achieved_resolution`
    (authoritative, from the decoded image) / `encoded_quality` / `original_bytes`
    are all recorded; the original decodes back at full resolution and the
    thumbnail is genuinely downscaled; a JPEG upload is stored **verbatim** (no
    re-compression).
  - `tests/unit/test_ui_structure.py` gained
    `test_overlay_unknown_label_is_exactly_unknown` (BLOCK 11.7): no `(was ` in
    `overlay.js`, the resting label literal is `"Unknown"`; and the
    policy-controls test now asserts the unfitted-threshold notice and the
    relocated recognition-detail rows.

### Policy audit (BLOCK 3.1 / 3.2) - evidence for every change

`python scripts/policy_audit.py --source data/raw --frames 400 --synthetic-control 30`
-> `results/policy_audit_raw.{json,md}`. **These are detection frequencies on
UNLABELLED footage - not accuracy, precision, recall or mAP.** One clip, one
session, 400 frames, 481 raw detections.

Per-state, after the redesigned policy:

| state | count | share of raw |
|---|---|---|
| accepted | 452 | 0.940 |
| accepted_secondary | 23 | 0.048 |
| unknown_low_confidence | 0 | 0.000 |
| unknown_margin | 1 | 0.002 |
| suppressed_implausible | 5 | 0.010 |
| rejected_size | 0 | 0.000 |
| **reconciles (all six == 481 raw)** | **True** | — |

Per raw class (the whole clip's vocabulary): `person` 404 accepted, `cell phone`
32 accepted, `cup` 16 accepted, `couch` 3 -> **accepted_secondary**, `umbrella`
20 -> accepted_secondary (+1 unknown_margin tie), `surfboard` 5 ->
**suppressed_implausible**.

**What changed vs the Phase 2.5 whitelist:** under the old 14-entry
`domain_classes`, `couch` (3) and `umbrella` (20) were rejected as
out-of-domain and rendered on screen as "Unknown (was Sofa)" / "Unknown (was
Umbrella)". They are now shown de-emphasised with their real class. `surfboard`
(a clear misfire in an indoor room) is the only class still suppressed on this
clip, and it is hidden rather than labelled "Unknown". Every policy change traces
to a number in this table or to the documented rule audit in
`docs/decisions.md` (Phase 5). **No threshold value was changed** (they are
unfitted - see Part 3).

**Synthetic-frame control:** 30 deterministic-noise frames -> **0 raw
detections** (the pipeline does not hallucinate objects on noise); reconciles.

### Per-rule rejection counts and confidence (from the audit)

| rule | rejected on this clip | confidence of what it caught (p50) |
|---|---|---|
| implausible-tier suppression | 5 (`surfboard`) | 0.499 |
| per-class threshold (0.35, unfitted) | 0 | — |
| top-2 margin (0.10, unfitted) | 1 (`umbrella`/other) | 0.363 |
| size / aspect | 0 | — |

### Latency (BLOCK 3.8)

`python scripts/benchmark_latency.py --source data/raw --limit 150 --end-to-end-seconds 14`
-> `results/latency_2_5.{json,md}`.

**End-to-end capture -> snapshot loopback** (the real FastAPI app + browser-ingest
source + perception + policy, frames pushed over `/ws/ingest` with a real capture
clock, `StateSnapshot.frame_age_ms` collected from `/ws/state`; run in a clean
subprocess). This is the server-side portion of "recognition feels slow"; the
live browser adds only its own capture-encode and one paint (a Block 13 check).

| measure | value |
|---|---|
| samples | 117 |
| **frame age p50 / p95 / max** | **177.4 / 230.6 / 248.3 ms** |
| detector p50 | 65.2 ms |
| pose p50 | 51.7 ms |
| analysis FPS p50 | 8.5 |
| drop rate (mean) | 0.14 |
| process CPU % | ~1065 (of 1800 = 18 logical cores) |
| RSS MB (before -> after the loopback) | 1023 -> 1212 |

Per-frame CPU inference (first, uncontended in-process run):

| | detector p50/p95 | pose p50/p95 | policy p50/p95 | combined p50/p95 | implied FPS |
|---|---|---|---|---|---|
| pose every frame | 72.2 / 88.8 ms | 57.0 / 67.7 ms | **0.06 / 0.08 ms** | 129.2 / 153.4 ms | 7.7 |

intra-op thread sweep (clean, in-process): threads 4 -> combined 132.9 ms,
6 -> 133.1 ms, 8 -> 173.5 ms. The shipped `perception.intra_op_threads: 6` is
confirmed; 4 is equivalent, 8 is worse.

**Optimisations applied: none.** The measurements do not justify one. The policy
adds 0.06 ms p50 (< 1 ms target). Detector and pose per-frame cost is unchanged
from Phase 2.5 (Phase 5 modified no inference code). The ~130 ms combined CPU
inference and the 8-10 fps sampling floor are the inherent cost characterised in
Phase 2.5; Block 3.8.27 forbids a model swap / quantisation / threading redesign
this phase, and nothing cheaper is on the table.

### Studio capture quality - before / after (BLOCK 3.7 / 25)

| | Before (Phase 4) | After (Phase 5) |
|---|---|---|
| `getUserMedia` request | `{ video: true }` - **no resolution constraint** | `{ width:{ideal:1280}, height:{ideal:720}, frameRate:{ideal:30} }` - identical to the monitoring preview, then `getCapabilities()` -> `applyConstraints()` up to 1920 |
| frame source | `drawImage(<video>)` at `videoWidth` | `createImageBitmap(<video>)` of the live track at full `videoWidth` (`element` fallback), path recorded |
| original stored | `cv2.imencode('.jpg', img, q92)` **re-encoding** a canvas JPEG (double compression) | JPEG kept **verbatim** (no generation loss); non-JPEG normalised to q92 once |
| resolution recorded | none | `requested_resolution` + `achieved_resolution` (authoritative, from the decoded image) + `capture_path` + `encoded_quality` + `original_bytes` |
| thumbnail | separate file (unchanged) | separate file; `SampleStore.add` refuses if its bytes equal the original's |

**The dominant before/after is the request.** With no constraint the browser
hands the Studio its default capture resolution (commonly 640x480); the main
monitoring preview asks for 1280x720. Reusing the same request is the fix.
Whatever blur remains after that is the camera/lens itself, not the pipeline -
`achieved_resolution` in each sample's provenance makes that explicit and is
never inflated to the requested value. The actual on-camera comparison is a
developer check (Part 2) - no camera ran the Studio in this environment.

### Studio lifecycle - root causes and fixes (BLOCK 3.6 / 16)

`static/studio/studio.js`, `api/objects.py`, `api/studio.py` and the object store
were read in full. `api/studio.py` and `api/objects.py` had **no** lifecycle bug
- all three faults were in the client's implicit state. Root causes:

1. **Back to camera loses the box editor.** The only "Back to camera" control was
   created dynamically inside `ensureInspectActions()` during a sample inspect.
   Its handler set `stageKind = "camera"` but never cleared `state.editingSampleId`,
   never re-centred the box (`state.boxImg` still held the *inspected sample's*
   pixel coordinates), and never re-shown `#upload-preview` state consistently -
   so the next capture saved a stale box.
2. **Re-selecting an object leaves stale controls.** `selectObject()` reset
   `state.objectId` / `stageKind` but never hid the dynamically-appended
   `#inspect-actions` bar from the previous object's inspect session, and never
   re-established the camera stage - so "Discard sample / Back to camera" buttons
   from object A stayed visible while object B was selected.
3. **Save & Return silently discards a staged upload.** `btn-return` called
   `leave()` -> `stopCamera()` + `POST /api/studio/leave` directly, with no check
   for `state.uploadQueue.length` or an unsaved inspect edit.

Fixes: `static/studio/studio-state.js` is an explicit
`browsing | object_selected | capturing | reviewing` machine with one owner and a
`hasPending()` guard; `studio.js` renders the DOM as a pure function of
`(state, context)` - no reload, no forced re-fetch. `#inspect-actions` is now a
permanent element in `index.html` shown/hidden only by `render()`.
`back_to_camera` / `discard` reset the transient context but keep `objectId` and
re-centre the box; `select` fully resets and re-`ensureCamera()`s;
`save_and_return` is `hasPending()`-guarded and `saveAndReturn()` `window.confirm`s
before discarding, with a `beforeunload` backstop. Regression coverage in
`test_studio_state_machine.py`.

### Preview independence + panel (BLOCK 3.8.28 / 3.9)

- `test_preview_independence.py`, `test_studio_lifecycle.py` (incl. the
  stall-hook invariant after returning from the Studio),
  `test_perception_pipeline.py::test_preview_independence_holds_with_perception_running`
  all pass. Mailbox stays single-slot, the producer never blocks on a stalled
  consumer, `loop.error is None`.
- Panel: `test_panel_resize_contract.py` and `test_ui_structure.py` pass
  unchanged. `static/ui/resizer.js` / `group.js` / `registry.js` were not
  touched (Block 8). No regression found.

---

## PART 2 - PHYSICALLY OBSERVED BY THE DEVELOPER

**Pending - not performed by this pass.** The Browser pane blocks camera capture,
so no live camera ran the overlay or the Studio here.

Motivation for the phase, quoted from the developer's own notes and marked there
as **observations of failure modes, not measured accuracy**: "watch -> *donut* or
undetected; spectacles, charger not detected; headphones -> *person*; mug ->
*phone*; bowl -> *wine glass*; keyboard -> *TV remote*; can -> *phone*; bottles
inconsistent; recognition feels slow. Person and pose generally work." These
classes (watch, spectacles, charger, headphones, shaker) are absent from the
COCO-80 vocabulary; this phase does not add them. `watch -> donut` is now
*suppressed* (donut is implausible-tier) instead of shown; `bowl -> wine glass`
now shows de-emphasised (wine glass is secondary-tier) instead of as a confident
primary label.

Developer checklist (BLOCK 13), all pending:

1. On the live overlay, confirm `person`, `laptop`, `keyboard`, `chair`,
   `bottle`, `clock` are recognised **with no enrolment**, and that no label
   reads `Unknown (was ...)`.
2. Confirm a genuinely unsupported object (watch, charger, spectacles) shows as
   `Unknown` or nothing - not as a confident wrong class.
3. Open Diagnostics; confirm the raw class, confidence and rule are visible for a
   clicked detection, and that the thresholds are labelled UNFITTED.
4. In the Studio: select an object, capture, use **Back to camera**, re-select
   the object, confirm nothing is lost; then **Save & Return** and confirm
   monitoring resumes. Try leaving with a staged upload and confirm the warning.
5. Compare a Studio capture with the main preview and confirm the sharpness gap
   is gone (or identify what remains as the camera itself - `achieved_resolution`
   in the sample provenance is the reference).
6. Drag and collapse the operations panel; confirm no regression.

---

## PART 3 - NOT VERIFIED / LIMITATIONS

- **No model has been trained or fine-tuned.** Recognition of watch, spectacles,
  charger, headphones and shaker is **unchanged** - they remain outside the
  model's vocabulary and this phase does not add them.
- **Tier assignments are environment-specific judgement, not measured results.**
  `primary` (14) / `secondary` (26) / `implausible` (40) is a partition chosen
  for an indoor cabin scene; it is cheap to revise once labelled data exists.
- **All thresholds remain unfitted.** `per_class_thresholds` is empty;
  `default_threshold` 0.35 and `margin_min` 0.10 are placeholders. The UI now
  says so (`policy.thresholds_fitted: false` -> a `warn-note` in Analysis ->
  Detection and an `UNFITTED` label in Diagnostics), and
  `scripts/fit_thresholds.py` refuses to run against the unlabelled split. The
  "baseline vs policy on `val`" table, fitted per-class thresholds, and the
  labelled model comparison are all still pending the developer's labels.
- **The policy audit reports rejection counts on unlabelled footage - these are
  frequencies, not accuracy.** One clip, one recording session, 481 raw
  detections. `surfboard`/`couch`/`umbrella` are the model's calls, not verified
  against ground truth.
- **End-to-end latency is the server-side capture -> snapshot number.** The
  browser's own capture-encode and final paint are not measured here (no camera);
  they are a Block 13 developer check. The `frame age p50 177 ms` figure is with
  perception ON over the developer's footage at `analysis_fps` 10 with
  `intra_op_threads` 6.
- **Studio camera quality is verified server-side only.** The
  `test_studio_capture_quality.py` assertions cover the store (full-resolution
  original, separate thumbnail, provenance). The actual browser capture path
  (`createImageBitmap` of the live track, `getCapabilities` probe) has not run
  against a real camera in this environment.
- **No tracking, temporal reasoning, scene relations, risk inference,
  recommendations or voice exist** - none started, none placeheld.
  `StateSnapshot.tracks` is still always `[]`. Instance *recognition* is in the
  schema (`kind: "instance"`) but no instance inference exists.
- **The Scene Snapshot Studio does not exist yet** - that is Phase 6.

---

## What changed (map)

New: `predictivesense/perception/vocabulary.py`,
`predictivesense/api/static/studio/studio-state.js`, `scripts/policy_audit.py`,
`tests/unit/test_vocabulary_tiers.py`, `tests/unit/test_policy_states.py`,
`tests/integration/test_studio_state_machine.py`,
`tests/integration/test_studio_capture_quality.py`, this report.

Modified: `predictivesense/perception/policy.py` (six states, tiers,
suppressed != unknown), `predictivesense/config/settings.py`
(`PolicyVocabularyConfig`, `thresholds_fitted`, `show_suppressed_in_diagnostics`,
`domain_classes` -> computed_field), `config/profiles/dev.yaml` +
`config/profiles/eval.yaml` (three-tier `vocabulary` replacing `domain_classes`),
`predictivesense/core/types.py` (`Detection.tier`),
`predictivesense/objects/samples.py` + `predictivesense/api/objects.py` (capture
provenance, verbatim JPEG original, thumbnail != original guard),
`predictivesense/api/static/features/overlay.js` (`Unknown` label, secondary
styling, suppressed hidden + Diagnostics reveal, click-to-select),
`predictivesense/api/static/features/policy.js` (six kinds, `thresholdsFitted`,
`ruleLabel`, `thresholdFor`),
`predictivesense/api/static/features/detection.js` (unfitted notice, tier
counts), `predictivesense/api/static/groups/diagnostics.js` (recognition-detail
table + selected-detection inspector, renamed policy counters, suppressed
toggle), `predictivesense/api/static/studio/studio.js` (rewritten on the state
machine; camera-path reuse; capture-quality), `predictivesense/api/static/studio/index.html`
(permanent `#inspect-actions`, `#pending-note`), `predictivesense/api/static/app.css`
(warn-note + recognition-detail table),
`scripts/fit_thresholds.py` (refuses unlabelled), `scripts/policy_effect.py`
(new state names), `scripts/benchmark_latency.py` (end-to-end loopback),
`tests/unit/test_policy_rules.py` + `tests/unit/test_ui_structure.py` (updated
for the contract change), `docs/architecture.md`, `docs/decisions.md`, `CLAUDE.md`.

Not touched (Block 8): `predictivesense/perception/detector.py` /
`perception/pose.py`, `camera/mailbox.py` / `camera/synthetic.py`, `telemetry/*`,
`analysis-worker.js`, `predictivesense/dataset/*`, `predictivesense/eval/*`,
`data/eval/*`, `static/ui/registry.js` / `group.js` / `resizer.js`. No tracker /
scene / temporal / risk / audio package created. No new dependency.
