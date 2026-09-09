# Phase 7 - Bulk Image Upload for the Object Learning Studio

Scope: **one feature** - bulk image upload with proposed bounding boxes, inside
the existing Object Learning Studio. A data-collection accelerator, nothing more.
**No training, no Scene Snapshot Studio, no tracking, no temporal reasoning, no
risk, no voice.** No new runtime dependency. No change to recognition behaviour,
the recognition policy, the vocabulary tiers, the evaluation dataset or the model
registry.

Three separated parts, as required.

---

## PART 1 - MEASURED (inspection + commands run on this machine)

Machine: Windows 11, Intel Core Ultra 5 125H (14C / 18T), 16 GB LPDDR5, Intel Arc
integrated graphics, no NVIDIA / no CUDA. Python 3.11.9, `.venv`.

### What the inspection found about the existing architecture

Names in this prompt were checked against the real code; where they differ the
real one is used.

- **A sample is created and persisted** by `SampleStore.add(...)` in
  `predictivesense/objects/samples.py` (stdlib + numpy only - the caller hands in
  decoded/encoded bytes). The HTTP entry is `POST /api/objects/{id}/samples`
  (`predictivesense/api/objects.py::add_sample`), which: reads the multipart
  image, size-checks it against `objects.max_image_mb`, `decode_image_bgr` (from
  `predictivesense/camera/_opencv.py` - `cv2` is confined to `camera/` +
  `perception/`), runs `compute_sample_quality(...)`
  (`objects/quality.py`: Laplacian-variance blur, dHash pHash, near-duplicate,
  box-area fraction), stores the **full-resolution original** verbatim if it is
  already JPEG (else re-encodes q92) plus a **separate** `thumbnail_jpeg(...)`,
  then `store.add(...)` and `_sync_counts(...)` (writes the registry's
  `sample_count`/`positive_count`/`negative_count`).
- **Exact stored sample fields** (`ObjectSample`): `sample_id`, `object_id`,
  `path`, `role` (`positive`|`negative`|`hard_negative`), `negative_for`,
  `box` (`[x,y,w,h]` px, required, `_validate_box` allows a 1px tolerance and
  clamps), `width`, `height`, `conditions` (7 fixed dimensions from
  `objects/vocab.py`, missing keys filled with the dimension default),
  `quality`, `source` (`camera`|`upload`), `device_label`, `original_filename`,
  `captured_utc`, `git_commit` (resolved once, lazily), `consent_ack`,
  `thumb_path`, and the Phase 5 capture-quality provenance (`capture_path`,
  `requested_resolution`, `achieved_resolution`, `encoded_quality`,
  `original_bytes`). Manifest at `data/objects/<id>/manifest.json`, written
  atomically; sample deletion is soft (`_deleted/`).
- **The box editor** lived inside `static/studio/studio.js` as `mediaDims` /
  `contentRect` / `centreBox` / `clampBox` / `layoutBox` / `wireBoxEditor`,
  operating on the module-scoped `state.boxImg` and the fixed `#stage` /
  `#preview` / `#upload-preview` / `#sample-box` elements (drag to move, 4 corner
  handles to resize, arrow keys to nudge, Shift = larger step). It is now
  **extracted verbatim** to `static/studio/box-editor.js` as
  `createBoxEditor({ stage, box, media, onChange })`; `studio.js` builds one
  instance (`state.boxImg` aliases `editor.box`) and the bulk-upload reviewer
  builds a second on its own elements - **one implementation, not two.**
- **Quality checks** are invoked only through `compute_sample_quality(...)`; the
  bulk path calls it identically, with `existing_hashes` = the object's
  committed sample pHashes **plus** the pHashes of earlier-processed items in the
  same batch (within-batch + against-existing near-duplicate detection).
- **The Studio state machine** (`static/studio/studio-state.js`,
  `browsing|object_selected|capturing|reviewing`, six transitions) drives
  `render()` as a pure function of `(state, context)`. **Phase 7 does not touch
  it** - the bulk-upload panel is its own view owned by `batch.js`, shown/hidden
  independently, closed on object switch. The Python transition-table mirror in
  `test_studio_state_machine.py` is unchanged.
- **Browser tests** (`tests/browser/`): a hand-rolled Playwright harness (base
  `playwright` only) - package-scoped `live_server` (real uvicorn in a thread,
  seeded temp objects store, perception off), function-scoped `page` + a
  `ConsoleWatch` that fails on any non-benign `console.error` / `pageerror`.
  Chromium fake media device for capture. Skips cleanly when Playwright or
  Chromium is absent.
- **The detector**: `PerceptionEngine` (`perception/engine.py`) holds the ORT
  detector + pose sessions; **the engine does not apply the recognition policy**
  (that is done downstream in `pipeline/loop.py` / `pipeline/recorded.py`). So
  `engine.infer(frame).detections` is already the raw detector output. Phase 7
  adds a lean `engine.detect(frame)` (detector only, no pose, no counters) for
  the proposer. `app.state.perception` is the loop's shared engine; the Studio
  pauses the loop, so using it for one-shot annotation reuses the warm session
  and never resumes monitoring.

### What was implemented

- **`predictivesense/objects/batches.py`** - `BatchStore`: staging under
  `data/objects/<id>/_staging/<batch_id>/` with a `batch.json` manifest +
  `<item>.jpg` / `<item>.thumb.jpg` files. `create` / `add_staged_image` /
  `update_item` / `set_status` / `set_progress` / `overwrite` / `list_batches` /
  `discard` / `cleanup_stale(ttl_hours)`. `BatchItem` dataclass is the BLOCK 6
  contract. Stdlib + numpy only; atomic writes; unsafe batch ids rejected.
- **`predictivesense/objects/proposals.py`** - `propose_box(raw_detections,
  w, h, min_score)` -> `BoxProposal(box, source, status, raw_class, score,
  alternates)`. Highest score above `min_score`; ties broken by **larger area,
  then box centre nearest image centre**. Empty/low -> `manual_required` +
  `centred_default_box` (the same 30%/40% centred box the camera path seeds).
  Class-agnostic: the raw class is a **hint only** (`raw_class`), never a field
  that could be read as the sample class. Stdlib + numpy only; never imports the
  detector, `onnxruntime` or `cv2`.
- **`predictivesense/api/object_batches.py`** - the batch router (mounted in
  `api/app.py`):

  | Method | Path | Behaviour |
  |---|---|---|
  | POST | `/api/objects/{id}/batches` | multipart `files` + `role`; per-file + total-size validation; stages accepted images (normalised JPEG + server thumbnail); starts proposals on a 1-worker `ThreadPoolExecutor`; returns `{batch_id, accepted, rejected:[{filename,reason}], status:"processing", total}` |
  | GET | `/api/objects/{id}/batches/{bid}` | `{status, processed, total, role, items:[BatchItem]}` - the poll target |
  | PATCH | `/api/objects/{id}/batches/{bid}/items/{iid}` | update `box` (bounds/zero-area checked -> 400; `null` = delete box), `role` (+ `negative_for` auto-fill), `conditions`, `box_confirmed_by_human`; a human touch sets `box_confirmed_by_human=true` + status `edited` |
  | POST | `/api/objects/{id}/batches/{bid}/save` | commit every item with a valid box **through `persist_sample(...)`** (the shared camera path); `{saved, remaining, skipped:[{item_id,reason}], batch_cleared}`; unresolved items stay staged; all saved -> staging discarded; counts synced |
  | DELETE | `/api/objects/{id}/batches/{bid}` | discard the batch + staging (404 if unknown) |
  | GET | `/api/objects/{id}/batches/{bid}/items/{iid}/image` | staged image (`?thumb=1`), path-confined |

  Background proposal worker: builds a `Frame`, calls `engine.detect(frame)`
  (raw, **policy bypassed** - recorded in `docs/decisions.md`), `propose_box`,
  `compute_sample_quality`; one image failing is caught and that item becomes
  `manual_required` with `error` recorded - the batch continues. All manifest
  read-modify-writes are serialised under one module lock.
- **`predictivesense/api/objects.py`** - extracted `persist_sample(store,
  objects_cfg, *, data, box, conditions, role, negative_for, source, ...,
  extra_provenance)` - the single decode -> quality -> normalise -> thumbnail ->
  `store.add` path. `add_sample` (camera route) now calls it; **its behaviour is
  unchanged** (same 400/413/422 for empty / oversize / missing box;
  `test_objects_api.py` unchanged and green).
- **`ObjectSample` / `SampleStore.add`** - five additive fields (`batch_id`,
  `proposal_source`, `proposal_raw_class`, `proposal_score`,
  `box_confirmed_by_human`), all `None` on the camera path. `source` for a
  bulk-committed sample is `"upload_batch"`.
- **`predictivesense/config/settings.py`** - `objects.batch` section
  (`max_images` 60, `max_total_mb` 400, `staging_ttl_hours` 24,
  `proposal_min_score` 0.10, `thumbnail_px` 240, `min_image_px` 32); mirrored
  into `config/profiles/dev.yaml` + `eval.yaml`.
- **`predictivesense/api/studio.py`** - `POST /api/studio/enter` now calls
  `cleanup_stale_batches(app)` so an abandoned batch older than the TTL is
  removed at Studio entry (never blocks entry).
- **`scripts/export_objects_coco.py`** - `_iter_object_samples` now skips any
  sample path under `_staging/` explicitly (staging was already excluded because
  it never enters `manifest.json`; the guard + docstring make it enforced).
- **Frontend** - `static/studio/box-editor.js` (extracted editor);
  `static/studio/batch.js` (`initBatch(deps)` - the whole bulk-upload panel:
  multi-file `#bulk-upload-input`, grid with server thumbnails + status badges,
  progressive polling with a `N uploaded · P proposed · M need a box · R
  reviewed` count, a Review All reviewer with Prev/Next + the shared box editor +
  delete-box + add-box + per-item role + condition tags with remembered
  defaults + keyboard `n`/`p`/`a`/`x`, "jump to items needing a box", Save all,
  Discard batch); `studio.js` refactored to the extracted editor + wires
  `initBatch`; `index.html` + `studio.css` carry the panel markup/styles (no
  second stylesheet needed).

### Proposal behaviour

- Boxes come from the **existing** detector's **raw** output, before the
  recognition policy. A `watch` proposed as `donut` / `clock` still yields its
  rectangle; the policy's `implausible` tier is not consulted. Verified by
  `tests/unit/test_box_proposal.py::test_policy_is_bypassed_*` and the
  `models`-marked `tests/integration/test_object_batches_api.py::
  test_real_detector_proposes_a_box_bypassing_the_policy`.
- The predicted label is stored as `proposal_raw_class` / `proposal_score`
  **only**, shown in the reviewer as "detector hint: donut (0.42), not the
  label". The sample's class is always the selected object.
- Detector-empty -> `manual_required` + centred default box, `proposal_source =
  default_centred`. One image failing -> `manual_required` + the error string,
  batch continues.
- `box_confirmed_by_human`: `false` for an untouched proposal (even after a bulk
  Save all); `true` once the developer moved/resized/drew the box or pressed
  "Accept box" in the reviewer.

### Save / staging

- **Save all** commits every item with a valid box through `persist_sample`
  (`source="upload_batch"`, `capture_path="upload_batch"`, provenance:
  `batch_id`, `proposal_source`, `proposal_raw_class`, `proposal_score`,
  `box_confirmed_by_human`, plus every field the camera path writes). Items
  without a valid box are **neither saved nor discarded** - they stay staged and
  the result says *"N samples saved · training data updated. M still need a
  bounding box."* When every item is saved the staging directory is removed.
- Staging never enters `manifest.json`, so it is excluded from `SampleStore.
  counts()`, `coverage_summary(...)` and `scripts/export_objects_coco.py` by
  construction (test:
  `test_object_batches_api.py::test_staging_is_excluded_from_the_coco_export`,
  `test_batch_store.py::test_staged_items_are_excluded_from_sample_counts_and_coverage`).
- Discard removes the batch directory; `cleanup_stale` (TTL, default 24 h) runs
  at Studio entry.
- The save message never says or implies the model learned or was trained -
  asserted by `tests/browser/test_studio_bulk_upload.py::
  test_no_text_claims_the_model_learned_or_was_trained`.

### Preserved (verified by test)

- Camera capture: `tests/integration/test_objects_api.py` unchanged and green;
  `tests/integration/test_studio_capture_quality.py` green;
  `tests/browser/test_studio_flow.py::test_fake_camera_capture_*` green.
- Studio lifecycle: `test_studio_lifecycle.py` (enter pauses the loop with zero
  perception invocations from the loop; leave restores it; ingest socket closed
  while active), `test_studio_state_machine.py` (transition table + mirror
  unchanged), `test_studio_flow.py` (object selection, Back to camera, Save and
  return) - all green.
- Dataset separation: `test_dataset_separation.py` +
  `test_object_batches_api.py::test_staging_is_excluded_from_the_coco_export` -
  staged and saved batch images never reach `data/eval`.
- Recognition behaviour: `perception/detector.py`, `pose.py`, `policy.py`,
  `vocabulary.py` untouched; `pytest -q -m models` = 12 passed (unchanged) plus
  the one new `models` proposal test.
- Preview independence / single-slot mailbox / no outbound network:
  `test_preview_independence.py`, `test_no_outbound_network.py` green. No
  tracker, temporal state, risk, or voice code added.
- Model registry untouched; no model trained, fine-tuned or activated.

### Tests

- `pytest -q -m "not browser"`: **355 passed, 2 skipped** (baseline at `3408cce`
  = 327 passed / 2 skipped non-browser; **+28**, 0 regressions).
- `pytest -q -m browser`: **19 passed** (baseline 13; **+6**), run twice, stable.
- Total: **374 passed, 2 skipped** (baseline 340 / 2).
- `pytest -q -m models`: **13 passed** (12 unchanged + 1 new bulk-upload
  proposal test).
- New files: `tests/unit/test_batch_store.py` (8), `tests/unit/test_box_proposal.py`
  (8), `tests/integration/test_object_batches_api.py` (13, one `models`),
  `tests/browser/test_studio_bulk_upload.py` (7).
- `python scripts/export_objects_coco.py --out data/objects/coco_train.json`
  exits 2 ("no object profiles to export") because `data/objects/` is still
  empty on this machine - unchanged pre-existing behaviour; the staging-exclusion
  path is covered by the integration test above.

### Performance (measured, this machine)

Real `yolo11n` detector, 20 frames decoded at 1920x1080 from the one clip in
`data/raw/`, `intra_op_threads=6`, `provider=cpu`, warm session
(`scratchpad/bench_batch.py`):

| Metric | Value |
|---|---|
| Proposal latency per image (detect + `propose_box`) | **p50 62 ms, p95 78 ms** |
| 20-image batch wall-clock, serial | **~1.25 s** |
| RSS delta over 60 proposals | **+33 MB** (the ORT detector session, already resident while the Studio's loop is paused; reuses `app.state.perception`, no new session) |
| Staging disk per image (normalised JPEG q92 + 240px thumb) | **~0.25 MB** -> ~5 MB / 20-image batch, ~15 MB / 60-image batch |
| Thumbnails | generated once, server-side; the grid uses `?thumb=1`, never full-res scaled in CSS |
| UI responsiveness while processing | the Review All reviewer opens while a batch is still processing - asserted by `test_progress_indicator_and_page_stays_interactive`; the request thread returns immediately with the `batch_id`, proposals run on a 1-worker executor, the client polls every 400 ms and the grid rebuilds only when an item's status actually changed |

---

## PART 2 - PHYSICALLY OBSERVED BY THE DEVELOPER

**Pending.** Not performed by the assistant (BLOCK 12):

1. Open the Studio, select `Watch`, upload ~20 real photographs in one operation.
2. Confirm proposed boxes appear and are usable, and that failures are clearly
   marked.
3. Correct a few boxes in Review All; add one manually.
4. **Save all**; confirm the sample count rises by the expected number and
   unresolved items remain.
5. Upload a hard-negative batch (clocks, a bracelet) and confirm the role and
   `negative_for`.
6. Confirm camera capture still works on the same object afterwards.

---

## PART 3 - NOT VERIFIED / LIMITATIONS

- **No model is trained, fine-tuned or activated.** Uploading images changes
  recognition **not at all** - it only grows `data/objects/`. "training data
  updated" in the UI means exactly that.
- **Proposal quality on objects outside the detector's COCO-80 vocabulary is
  unmeasured.** A `watch` / spectacles / charger / headphones / shaker has no
  class in the model, so proposals for those will often be `manual_required` or a
  loose `donut`/`clock` box. The measured latency above is on cabin-scene frames
  where the detector fires; it says nothing about proposal *accuracy* on the
  custom objects.
- **`box_confirmed_by_human` distinguishes proposed from confirmed boxes, but
  proposal bias in the resulting dataset is unquantified** until a training phase
  can report what fraction of the training set was detector-proposed and test
  whether that biased results.
- **Browser tests use generated fixture images** (random noise + a drawn
  rectangle) with perception **off** in the browser fixture, so every item is
  `manual_required` there. Real-photo behaviour - proposal hit rate, the grid
  with a mix of `ready`/`flagged`/`manual_required`, the reviewer on a real
  detector box - is a developer check (Part 2).
- The proposal detector call shares the one ORT session with the monitoring
  loop. The Studio pauses that loop, so in normal use there is no contention; if
  a batch were somehow processed with monitoring live, detector calls would
  serialise and briefly slow the loop. Not exercised.
- `data/objects/` is empty on this machine, so the end-to-end COCO export with
  bulk-collected samples is a developer step.
