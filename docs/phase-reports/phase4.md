# Phase 4 - Object Learning Studio & Operations Panel

Scope: environment-specific object **data collection**, plus operations-panel
usability. Delivered: a dedicated `/studio` screen that stops monitoring while
open and lets the developer teach PredictiveSense the objects in this cabin by
capturing / uploading multiple images per object, each with **one bounding box**
and condition tags, with measurable quality checks and hard-negative support,
persisted into a versioned, training-ready store kept strictly separate from the
Phase 2.5 evaluation set; a model **version registry** (read/validate/resolve
only); and a **resizable, keyboard-accessible** right-hand panel.

**No model was trained or fine-tuned. Recognition is unchanged by this phase.**

Three separated parts, as required.

---

## PART 1 - MEASURED (commands run on this machine)

Machine: Windows 11, Intel Core Ultra 5 125H (14C / 18T), 16 GB LPDDR5, Intel Arc
integrated graphics, no NVIDIA / no CUDA. Python 3.11.9, `.venv`.

### Test suite

- `pytest -q` -> **292 passed, 2 skipped** (1 `hardware` - needs `--run-hardware`;
  1 `dataset` - skips cleanly, evaluation set still unlabelled). Baseline at the
  end of Phase 2.5 was 250 passed / 2 skipped; **+42 tests**, 0 regressions.
- `pytest -q -m models` -> **12 passed** (unchanged - Phase 4 touched no
  perception code).
- `pytest -q -m dataset` -> **1 skipped** with the "run `build_eval_frames.py`"
  message (never fails, never silently passes).
- New tests:
  - `tests/unit/test_object_registry.py` - profile CRUD, slug collisions
    disambiguated (`cup`, `cup-2`, `cup-3`), instance needs a `parent_class`,
    atomic write leaves no `.tmp` and valid JSON, corrupt `objects.json` /
    corrupt object `manifest.json` both raise, soft-delete moves the folder to
    `_deleted/` and drops it from the registry, sample counts, sample soft-delete
    moves image files to `_deleted/`, a box outside the image bounds is rejected.
  - `tests/unit/test_object_quality.py` - Laplacian variance orders a sharp
    checkerboard well above its blurred copy (>5x) and is tiny-crop-safe; dHash
    is stable under a small re-encode perturbation (<=4 bits) and separates a
    horizontal from a vertical gradient (>=8 bits); duplicate detection matches
    at the configured Hamming distance and not one below it; `compute_sample_quality`
    flags `blurry` / `near_duplicate`; coverage counts positives / views /
    occluded / hard-negatives on a constructed set and names the missing
    dimensions ("no far-distance samples", "no dim-lighting samples").
  - `tests/unit/test_model_registry.py` - the shipped `models/registry.json`
    loads with exactly one active version and its file SHA-256s match
    `models/manifest.json`; a two-version registry resolves the `active` one;
    zero or >1 active raises; a non-SHA-256 hash raises; a hash that mismatches
    `models/manifest.json` for a known filename raises.
  - `tests/unit/test_dataset_separation.py` - `check_dataset_separation` returns
    no violations for disjoint stores, flags a shared image **content hash**,
    flags a shared **source filename** vs an eval `source_clip`;
    `export_objects_coco.main` **exits 2 and writes nothing** on a violating
    export; a clean export produces train/val files whose `file_name` sets are
    disjoint and together cover every sample.
  - `tests/unit/test_panel_resize_contract.py` - an executable Python mirror of
    `clampPanelWidth` (kept byte-checked against the JS defaults) asserts
    clamping to `min_width_px` / `max_width_px`, the 40%-of-window cap, and that
    the viewport never drops below 45% for windows 600-1920 px; reset returns the
    configured default; static checks that `store.js` persists `panelWidth`,
    that the resizer wires `role="separator"`, `aria-valuenow`, arrow-key steps,
    `Home` / double-click reset, and a `requestAnimationFrame`-throttled drag,
    and that `shell.js` mounts it.
  - `tests/integration/test_objects_api.py` - full lifecycle through the app:
    create (`Watch` -> slug `watch`, `confusable_with` seeded with `clock`), post
    a camera sample with box + tags + role, post an upload sample (same record
    shape, `original_filename` kept, `hard_negative` -> `negative_for=[watch]`),
    list, PATCH the box, coverage reflects it, the stored image + thumbnail are
    served, sample DELETE is soft (files -> `_deleted/`), object DELETE is soft
    (folder moved, `moved_to` returned, 404 afterwards); missing box -> 422;
    box outside the image -> 400; oversize upload -> 413 naming
    `objects.max_image_mb`; instance without `parent_class` -> 400.
  - `tests/integration/test_studio_lifecycle.py` - with a counting stub
    perception engine wired into the live loop: perception is called while
    monitoring runs; **after `POST /api/studio/enter`, zero perception
    invocations over a 1.0 s interval** and `loop.paused is True`; re-entering
    returns the same token; a stale token to `/leave` -> 409; after `/leave`,
    `loop.paused is False` and perception resumes. With the browser-ingest
    config: `/ws/ingest` handshakes before the Studio, **is refused (close 4409)
    while the Studio is open**, and is accepted again after `/leave`; the
    preview-independence stall invariant still holds after returning
    (`producer` alive, `max_mailbox_depth <= 1`, `loop.error is None`).
    `GET /api/models/registry` returns `active == "v1"` with `detector` + `pose`
    file roles; `GET /studio` serves the page.

### Studio performance (server-side, this machine)

Measured with a `TestClient` against a worst-case **640x480 high-entropy**
(random) frame, JPEG ~164 KiB - real camera frames compress better and cost less:

| Measure | p50 | p95 | n |
|---|---|---|---|
| Quality computation per sample (Laplacian variance + dHash + duplicate scan), isolated | **3.87 ms** | 4.05 ms | 200 |
| `POST /api/objects/{id}/samples` round-trip (decode + quality + normalise-encode q92 + thumbnail + atomic manifest write) | **34.2 ms** | 38.8 ms | 50 |
| RSS delta over a 50-capture session | **+7.1 MB** (203 -> 210 MB) | - | 50 |

The **shutter-to-saved-sample** round-trip a developer experiences is this
server figure plus the browser's `canvas.drawImage` + `canvas.toBlob` (single
frame, not measured here - no camera in this environment; see Part 2). Memory is
bounded: the object store holds nothing in memory between requests, every
manifest write is `os.replace` of a temp file, and there are no new queues,
caches or listener lists.

### Panel resize

- `clampPanelWidth` boundary behaviour is asserted directly (see
  `test_panel_resize_contract.py`). In the browser pane (window ~1280 px): the
  handle reports `role="separator"`, `aria-valuenow="380"`, `aria-valuemax="512"`
  (40% of the window); `ArrowLeft` widened 380 -> 412 px, `Home` reset to 380 px,
  and `localStorage["ps.ui"].panelWidth` persisted as `380`. No console errors on
  `/` or `/studio`.
- The drag handler stores one `clientX`, schedules one `requestAnimationFrame`,
  and in that frame does exactly one read (`window.innerWidth`, inside
  `clampPanelWidth`) and one write (`--panel-w` custom property on `.app-shell`).
  It never touches `<video>`; `#preview` carries no `filter` / `transform` /
  `animation` / `transition` (still asserted by `test_ui_structure.py`).
- Collapse still keeps the always-visible `#panel-reopen` control; collapsed and
  expanded both restore from `localStorage` on reload. Overflow rules added to
  `app.css` (`overflow-x: hidden` on `html/body`, `min-width: 0` on the grid
  children, ellipsis + `title` on long device / object names).

### No perception while the Studio is open

`test_studio_lifecycle.py::test_enter_stops_perception_and_leave_restores`
asserts it directly with a call counter on the loop's perception engine: **0
calls** across a bounded 1.0 s window with the Studio open, and the loop's
`paused` flag is `True`. `POST /api/studio/enter` also causes `api/ingest.py` to
refuse new `/ws/ingest` connections (close code 4409), and the Studio page
attaches no analysis worker.

### Record Sample - root-cause findings

Investigated `static/features/recording.js` + `predictivesense/api/recorder.py`
end to end (both read in full, not guessed):

- **Where clips are saved:** `recorder.py:_output_dir` -> `config.recorder.output_dir`
  (`data/raw`), then `/<session_id>/<clip_id>.webm` with a sibling
  `<clip_id>.json` `ClipManifest`. The `dev` profile has `recorder.enabled: true`;
  `eval` has it `false` (a disabled recorder returns **403**, by design).
- **The workflow is functionally correct** and covered by
  `tests/integration/test_recorder_upload.py` (small blob stores with a full
  manifest, oversize -> 413, missing field -> 422, listing empty when the dir is
  absent) - all green. A `TestClient` upload round-trip works here.
- **The one real weakness is duration metadata, and it is environmental, not a
  bug in scope for this phase.** `recorder.py:_probe_duration_s` delegates to
  `camera/file_source.py:video_duration_s`, which is OpenCV's
  `CAP_PROP_FRAME_COUNT / CAP_PROP_FPS`. OpenCV's FFmpeg build on this machine
  frequently cannot read frame count / fps from a **VP8/VP9 WebM produced by the
  browser `MediaRecorder`** (Matroska/WebM with no explicit duration element),
  so `duration_s` is stored as `null`. This is handled gracefully everywhere
  (`_probe_duration_s` catches and logs at DEBUG; the clip list shows `?` for an
  unknown duration; `build_eval_frames.py` falls back to `nominal_fps`). It is a
  cosmetic metadata gap, the fix (bundling `ffprobe` or a WebM parser) is a new
  dependency and larger than this phase, and the prompt says to document and
  leave such cases. **No change was made to the save location or the manifest
  format.**
- Two small robustness notes left as-is (not broken, would be churn): the record
  button's `disabled` state is driven by `runtime.stream`, so it is only enabled
  once a preview stream is live (correct); `uploadRecording` reads
  `$("device-select")` which now lives under the Camera group - still resolves,
  and `track.label` is the primary source anyway.

The Studio (object **images** -> `data/objects/`) and Record Sample (scenario
**clips** -> `data/raw/`) are now visually and structurally separate: distinct
subheads in the Dataset group, distinct verbs, distinct destinations, distinct
routes.

---

## PART 2 - PHYSICALLY OBSERVED BY THE DEVELOPER

**Pending - not performed by this pass.** The Browser pane blocks camera
capture, so no live camera ran the Studio here.

Motivation for the phase, quoted from the developer's Phase 2 notes and marked
there as **observations of failure modes, not measured accuracy**: "watch ->
*donut* or undetected; spectacles, charger, shaker not detected; headphones ->
*person*; mug -> *phone*; bowl -> *wine glass*; keyboard -> *TV remote*; can ->
*phone*; bottles inconsistent. Person detection and pose generally work." These
classes (watch, spectacles, charger, headphones, shaker) are simply absent from
the COCO-80 vocabulary; the Phase 2.5 policy layer can only suppress wrong
labels, not add classes. This phase builds the machinery to collect the
environment-specific training data a custom model would need - it does not build
or train that model.

Developer checklist (Block 14), all pending:

1. Open `/studio`; confirm monitoring stops (no detections, no analysis
   activity) - the automated half is asserted; the by-eye half is the developer's.
2. Create an object; capture ~10 images across views / distances / lighting,
   adjusting the box each time; confirm the flow is quick (keyboard `c` to
   capture, arrow keys nudge the box, tags remembered between shots).
3. Upload an image to the same object; confirm it appears with the same fields.
4. Add hard negatives for one confusable object (the Studio prompts when
   `confusable_with` is set and there are 0 hard negatives).
5. Return to monitoring; confirm the camera and perception resume as before.
6. Drag the panel wider / narrower; confirm the video stays usable, nothing
   overflows, and the width survives a reload.
7. Collapse and reopen the panel.

---

## PART 3 - NOT VERIFIED / LIMITATIONS

- **No model has been trained or fine-tuned, and recognition is unchanged by
  this phase.** The Studio stores images and metadata; that is not training.
  Nothing in the UI implies otherwise (the model badge is read-only and labelled
  "no model trained in this phase"; `models/registry.json` v1 is the existing
  pre-exported YOLO11n).
- **Collected images are training data, not a trained model.** They live under
  `data/objects/` (git-ignored) and are exported to COCO by
  `scripts/export_objects_coco.py` for a *later* fine-tuning phase that is not
  started.
- **Quality thresholds are collection heuristics, not validated metrics.**
  `blur_var_min` (60.0), `min_box_area_frac` (0.01), `duplicate_hamming_max` (6)
  and every `coverage_targets` value are guidance for *what to collect next*, not
  a pass/fail gate and not a quality score. Documented as heuristics in
  `docs/decisions.md`. A flagged sample is marked, never auto-deleted.
- **Coverage targets are guidance, not evidence of sufficiency.** "meets_all"
  means the counts hit the configured guidance numbers, nothing more.
- **The object dataset is small and from one environment.** As of this phase it
  is empty (the developer collects it). One recording session exists under
  `data/raw/`.
- **Studio preview FPS and true shutter-to-saved-sample latency are not
  measured** - no camera ran here. The server-side save figure is measured; the
  browser `drawImage`+`toBlob` term is not.
- **`models/registry.json` v1 `metrics_ref` points at a Phase 2.5 eval file that
  does not exist yet** (`results/eval_yolo11n_policy-on_val.json`) - the
  evaluation set is still unlabelled. The registry module loads and validates
  the entry; the metrics file is the developer's Phase 2.5 step.
- **No tracker, temporal state, scene relations, risk inference, recommendations
  or voice exist** - none started, none placeheld. `StateSnapshot.tracks` is
  still always `[]`. Instance *recognition* is in the schema (`kind: "instance"`)
  but no instance inference exists.
- The Record Sample WebM `duration_s: null` metadata gap is environmental
  (OpenCV/FFmpeg + browser WebM) and left documented, not fixed.

---

## What changed (map)

New package `predictivesense/objects/` (`vocab`, `registry`, `samples`,
`quality` - stdlib + numpy only, no new dependency); new package
`predictivesense/models/` (`registry` - read/validate/resolve, no activation
code); `predictivesense/api/objects.py` + `predictivesense/api/studio.py`;
`scripts/export_objects_coco.py`; `models/registry.json` (tracked - `.gitignore`
exception). `AnalysisLoop` gained `pause()` / `resume()` / `paused` (Studio
lifecycle). `api/ingest.py` refuses connections while the Studio is active.
`camera/_opencv.py` gained `decode_image_bgr` / `encode_jpeg` / `thumbnail_jpeg`
so `cv2` stays out of `api/` (same precedent as Phase 2.5's `read_image_bgr`).
Config: `objects.*`, `studio.*`, `ui.panel.*` sections. UI: `static/ui/resizer.js`
(new), `shell.js` mounts it, `store.js` persists `panelWidth`, `app.css` resizer
+ overflow rules, `groups/dataset.js` links to the Studio and separates the two
workflows, `index.html` adds the topbar Studio link + the resize handle,
`static/studio/` (new standalone page). Contracts in `core/types.py` unchanged.
