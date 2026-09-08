# PredictiveSense — Phase 4: Object Learning Studio & Operations Panel

> Paste this entire file into Claude Code in a **fresh session**. Do not summarise it.
> **Scope: environment-specific object data collection, plus panel usability.** No training, no fine-tuning, no tracking, no temporal reasoning, no risk, no voice, no environment scan.

---

## BLOCK 1 — CONTEXT

PredictiveSense is a local-first research prototype on a one-month deadline, supporting a research paper. Its long-term pipeline is camera/video → perception → recognition → tracking → temporal state → relations → risk → recommendation → alert, across two input modes (real-time and recorded video) that share one core.

Physical testing shows the pretrained detector is not sufficient in this environment: watch → *donut* or undetected; spectacles, charger, shaker not detected; headphones → *person*; mug → *phone*; bowl → *wine glass*; keyboard → *TV remote*; can → *phone*; bottles inconsistent. Person detection and pose generally work. **These are developer observations of failure modes, not measured accuracy.**

Phase 2.5 built the measurement foundation — a labelled evaluation set, an evaluation harness, and a recognition policy layer that rejects out-of-domain and weakly-supported predictions as `unknown`. That reduces confident wrong labels but **adds no new class**: watch, spectacles, charger, headphones and shaker are simply absent from the model's vocabulary. Only a custom-trained model fixes that, and a custom model needs environment-specific training data that does not exist yet.

**This phase builds the machinery that collects that data**, and fixes the operations panel usability that will otherwise get worse with every phase.

**Target machine:** Windows, Intel Core Ultra 5 125H, 16 GB LPDDR5, Intel Arc integrated graphics + NPU, **no NVIDIA GPU, no CUDA**. Python 3.11.9 in `.venv`.

---

## BLOCK 2 — CURRENT REPOSITORY STATE

Committed through `df31a72` (Phase 2.5). Present and working: `predictivesense/{core,config,camera,pipeline,perception,dataset,eval,telemetry,api}`; browser-owned camera with independent native preview; Worker analysis path over `WS /ws/ingest`; single-slot mailbox; recorded-video driver (deterministic); clip recorder; application shell with a group registry (`static/ui/`), groups `input, camera, video, dataset, analysis, research, diagnostics`, features `analysis-client, camera-capture, detection, overlay, policy, pose, recording, videos, metrics, runtime, analysis-prefs`; label editor at `/label`; scripts for model fetch, provider/latency/camera benchmarks, eval-frame building, splits, threshold fitting, detection eval, policy effect, class-coverage audit.

`data/` currently holds `raw/` (one recording session), `eval/`, `videos/`.

**Inspect before modifying:** `CLAUDE.md`, `docs/architecture.md`, `docs/decisions.md`, `docs/phase-reports/phase2_5.md`, `predictivesense/core/types.py`, `predictivesense/dataset/*`, `predictivesense/api/app.py`, `predictivesense/api/recorder.py`, `predictivesense/api/labels.py`, `static/ui/shell.js`, `static/ui/registry.js`, `static/ui/store.js`, `static/groups/dataset.js`, `static/features/camera-capture.js`, `static/features/recording.js`, `static/app.css`.

---

## BLOCK 3 — OBJECTIVE

Deliver an **Object Learning Studio**: a dedicated screen where the developer teaches PredictiveSense the objects in this environment by capturing or uploading multiple images per object, each with one bounding box and condition tags, with measurable quality checks and support for hard negatives — persisted into a versioned, training-ready dataset that is strictly separated from the Phase 2.5 evaluation data.

Alongside it, make the right-hand operations panel **resizable and cleanly collapsible**, and investigate the reported problem with the existing Record Sample workflow.

**No model is trained in this phase.** Storing images is not training, and nothing in the UI may imply otherwise.

---

## BLOCK 4 — REQUIREMENTS

### 4.1 Studio lifecycle — monitoring must actually stop

1. The Studio is a **separate top-level screen**, not a group inside the monitoring panel. Route: `/studio`, reachable from a clear entry in the Dataset group and from the top bar.
2. On entering the Studio the monitoring pipeline **stops**: the analysis Worker is terminated, the ingest socket is closed, perception is disabled, and the analysis loop is paused. Nothing may keep detecting, and no frames may reach `/ws/ingest` while the Studio is open.
3. The Studio opens its own camera preview through the existing camera abstraction, with **no analysis worker attached**. Preview only.
4. On leaving the Studio ("Save and return" or "Return"), the previous monitoring state is restored exactly as it was, including the selected device and the perception toggles.
5. This is testable and must be tested: with the Studio active, assert zero ingest messages and zero perception invocations over a bounded interval.

### 4.2 Object registry

6. An object is a persisted profile with a stable id. Support both **class** and **instance** kinds now in the schema, but implement no instance-recognition inference — this phase stores data only.
7. Full CRUD: create, rename, edit metadata, list, delete (delete moves the folder to `data/objects/_deleted/` rather than destroying it, and says so).
8. Nothing about the object vocabulary is hard-coded. The developer can add any new object later without a code change.

### 4.3 Capture and upload

9. Camera capture: choose camera, live preview, capture a still. Also accept file upload of one or more images. Both paths produce the same sample record.
10. **Every sample carries exactly one bounding box for the object.** Show an adjustable box over the preview — default a centred rectangle the user drags and resizes — and store it with the image. This makes the data training-ready with no second labelling pass, which is the difference between a dataset that is usable in two weeks and one that is not.
11. **Condition tags per sample**, from fixed vocabularies: `view` (front/back/left/right/top/tilted), `distance` (close/medium/far), `lighting` (bright/normal/dim), `background` (plain/cluttered), `occlusion` (none/partial), `held` (on-surface/in-hand), `frame_position` (centre/edge). Defaults are remembered between captures so a burst of similar shots is fast.
12. Review strip: thumbnails of the session's captures, click to inspect, adjust box, retag, or discard before saving.
13. Keyboard shortcuts for capture and discard; the developer will take dozens of images per object.

### 4.4 Quality checks — measurable, and honestly labelled

14. Per sample, compute and store: **blur** as the variance of the Laplacian over the box crop; **box area fraction** of the frame; **perceptual hash** (implement dHash in numpy — do not add a dependency) and the Hamming distance to existing samples of the same object, flagging near-duplicates.
15. Per object, compute **coverage**: counts per condition dimension, number of distinct views, distance spread, lighting spread, occluded count, negative count, total positives.
16. Surface these as **collection guidance**, not as a score: "3 views, no far-distance samples, no occluded samples". Thresholds for "enough" come from config and must be documented in `docs/decisions.md` as **heuristics chosen for data-collection guidance, not scientific quality metrics**. Do not invent a composite 0–100 quality number.
17. A sample flagged blurry or duplicate is not deleted automatically — it is marked and the developer decides.

### 4.5 Negatives and hard negatives

18. A sample has a `role`: `positive`, `negative`, or `hard_negative`. A negative sample records `negative_for: [object_id, ...]`.
19. Each object profile carries a `confusable_with` list — the classes it is actually mistaken for. Seed it from the observed failures: watch ↔ clock/bracelet/hand/donut; headphones ↔ person; mug ↔ phone/bowl/glass; bottle/shaker ↔ can/cylinder; keyboard ↔ remote; can ↔ phone. The Studio uses this list to prompt for hard negatives when an object has none.
20. Hard-negative capture is a first-class action in the Studio, not an afterthought.

### 4.6 Persistence and — critically — dataset separation

21. Layout:

```
data/objects/objects.json                  # registry: profiles, versions, counts
data/objects/<object_id>/images/*.jpg
data/objects/<object_id>/manifest.json     # samples with boxes, tags, quality, provenance
data/objects/_deleted/                     # soft-deleted profiles
```

22. **The object dataset is training data. The Phase 2.5 evaluation set is test data. They must never merge.** Add a test asserting no image path, image hash or source clip appears in both. Any future export that would violate this must fail loudly. Record this rule in `docs/decisions.md`.
23. `scripts/export_objects_coco.py` converts object samples into a COCO detection file at `data/objects/coco_train.json`, ready for a later fine-tuning phase, with a session/object-disjoint train/val split of its own. It does not touch `data/eval/`.
24. Every sample records provenance: capture time, camera device label, source (`camera` | `upload`), original filename if uploaded, app git commit, and `consent_ack` where a person may be in frame.

### 4.7 Model registry — versioning without training

25. Create `models/registry.json`: a list of model versions, each with `version_id`, files and SHA-256s, source, creation time, licence, `metrics_ref` pointing at a `results/` evaluation file, and an `active` flag. Register the current detector as the **baseline v1** with its Phase 2.5 metrics reference.
26. Provide `predictivesense/models/registry.py` to read, list and validate the registry, and to resolve the active model for the perception layer. **Activation is a deliberate operation, never an implicit overwrite.** No training, no new model, no activation logic beyond reading the flag.

### 4.8 Operations panel — resizable and collapsible

27. A drag handle on the panel's left edge resizes it live. Constraints: minimum panel width 300 px; maximum the smaller of 560 px and 40% of the window; the viewport never below 45% of window width. Cursor changes on hover; the handle is visible enough to find and quiet enough not to intrude.
28. Width **persists** through the existing UI state mechanism, alongside the collapsed flag. Double-click the handle resets to default.
29. Keyboard accessible: the handle is focusable, left/right arrows resize in steps, `Home` resets, with a visible focus ring and an appropriate ARIA role and value.
30. Collapse keeps a small, obvious reopen control; the app must never reach a state where the panel cannot be recovered. Collapsed and expanded both restore correctly after reload.
31. No layout breakage at any width: no horizontal page overflow, no text escaping its container, long device and object names ellipsise with a tooltip. **Fix the layout, never shrink the font to fit.**

### 4.9 Record Sample investigation

32. Inspect `static/features/recording.js` and `predictivesense/api/recorder.py`, reproduce the reported failure, and determine the root cause. **Do not guess where recordings are saved — read the code.**
33. Fix it only if it is genuinely broken and the fix is contained. If the cause is environmental or larger than this phase, document it precisely in the report and leave it. Do not change the save location or the manifest format without recording the reason.
34. Keep Record Sample (scenario clips) and Object Learning (object images) clearly separate in the UI and in storage. They are different workflows with different verbs and different destinations.

---

## BLOCK 5 — ARCHITECTURE CONSTRAINTS

- Preview independence, single-slot mailbox, newest-wins, worker backpressure: unchanged.
- Detection and tracking remain separate; **no tracker exists and none may be created**.
- Real-time and recorded modes stay distinct; both will later consume the same object dataset and model registry.
- Recorded processing stays deterministic.
- Contracts extend additively only, each addition recorded in `docs/decisions.md`.
- No CUDA, no cloud runtime, no runtime downloads.
- **No new Python dependency.** dHash, Laplacian variance and thumbnailing are numpy/OpenCV, both already present.
- No unbounded queues, caches or listener lists.
- Reuse the existing camera abstraction, shell, registry, controls and dataset store. Do not fork them.
- Images and manifests never enter git.

---

## BLOCK 6 — FILES TO CREATE

```
predictivesense/objects/__init__.py
predictivesense/objects/registry.py         # ObjectProfile CRUD, ids, soft delete
predictivesense/objects/samples.py          # sample records, storage, provenance
predictivesense/objects/quality.py          # laplacian variance, dHash, duplicates, coverage
predictivesense/objects/vocab.py            # condition vocabularies, confusable_with seeds
predictivesense/models/__init__.py
predictivesense/models/registry.py          # model version registry read/validate/resolve
predictivesense/api/objects.py              # object + sample endpoints
predictivesense/api/studio.py               # studio page, lifecycle enter/leave
scripts/export_objects_coco.py
predictivesense/api/static/studio/index.html
predictivesense/api/static/studio/studio.js
predictivesense/api/static/studio/studio.css
predictivesense/api/static/ui/resizer.js    # panel drag handle, keyboard, persistence
tests/unit/test_object_registry.py
tests/unit/test_object_quality.py
tests/unit/test_model_registry.py
tests/unit/test_dataset_separation.py
tests/integration/test_objects_api.py
tests/integration/test_studio_lifecycle.py
tests/unit/test_panel_resize_contract.py
docs/phase-reports/phase4.md
models/registry.json
```

## BLOCK 7 — FILES TO MODIFY

```
predictivesense/config/settings.py          # objects.*, studio.*, ui.panel.* sections
config/profiles/dev.yaml, eval.yaml
predictivesense/api/app.py                  # mount objects + studio routers, /studio page
predictivesense/api/static/ui/shell.js      # mount the resizer, width state
predictivesense/api/static/ui/store.js      # panel width in persisted UI state
predictivesense/api/static/app.css          # resizer styling, overflow rules
predictivesense/api/static/groups/dataset.js  # Object Learning entry; keep Record Sample separate
predictivesense/api/static/features/recording.js  # only if a real fix is found
predictivesense/api/recorder.py             # only if a real fix is found
docs/architecture.md, docs/decisions.md, CLAUDE.md
```

## BLOCK 8 — MUST NOT BE CREATED OR MODIFIED

- `predictivesense/tracking/`, `scene/`, `temporal/`, `risk/`, `policy/` (alert policy), `audio/` — none exist; none may be created, not even empty.
- `predictivesense/camera/mailbox.py`, `camera/synthetic.py`, `telemetry/*`, `analysis-worker.js`, `perception/detector.py`, `perception/pose.py`, `perception/policy.py`.
- `predictivesense/dataset/*` and `data/eval/*` — the evaluation dataset is closed. The object dataset is a **separate** store.
- `predictivesense/eval/*` and the eval scripts.
- `static/ui/registry.js`, `static/ui/group.js` — register into them; do not edit.
- Any prior prompt file or existing phase report other than adding a new one.

If one of these genuinely blocks you, **stop and ask**.

---

## BLOCK 9 — CONTRACTS

**`ObjectProfile`**

| Field | Type |
|---|---|
| `object_id` | `str` (slug, stable) |
| `name` | `str` |
| `kind` | `"class" \| "instance"` |
| `parent_class` | `str \| None` (instances only) |
| `category` | `str \| None` |
| `description` | `str \| None` |
| `confusable_with` | `list[str]` |
| `created_utc`, `updated_utc` | `str` |
| `sample_count`, `positive_count`, `negative_count` | `int` |
| `status` | `"collecting" \| "ready_for_training" \| "archived"` — set by the developer or by coverage rules, never implying a trained model exists |

**`ObjectSample`**

| Field | Type |
|---|---|
| `sample_id` | `str` |
| `object_id` | `str` |
| `path` | `str` (relative) |
| `role` | `"positive" \| "negative" \| "hard_negative"` |
| `negative_for` | `list[str]` |
| `box` | `[x, y, w, h]` in pixels, required |
| `width`, `height` | `int` |
| `conditions` | `{view, distance, lighting, background, occlusion, held, frame_position}` |
| `quality` | `{blur_var: float, box_area_frac: float, phash: str, duplicate_of: str \| None, flags: list[str]}` |
| `source` | `"camera" \| "upload"` |
| `device_label`, `original_filename` | `str \| None` |
| `captured_utc`, `git_commit`, `consent_ack` | `str` / `str` / `bool` |

**API**

| Method | Path | Behaviour |
|---|---|---|
| GET | `/api/objects` | List profiles with counts and coverage summary |
| POST | `/api/objects` | Create profile |
| GET/PATCH/DELETE | `/api/objects/{id}` | Read / update metadata / soft-delete |
| GET | `/api/objects/{id}/samples` | List samples |
| POST | `/api/objects/{id}/samples` | Multipart: image + box + conditions + role → sample record |
| PATCH | `/api/objects/{id}/samples/{sid}` | Adjust box, tags, role |
| DELETE | `/api/objects/{id}/samples/{sid}` | Remove a sample |
| GET | `/api/objects/{id}/coverage` | Coverage summary and guidance strings |
| POST | `/api/studio/enter` | Stop monitoring, return prior state token |
| POST | `/api/studio/leave` | Restore monitoring from the token |
| GET | `/api/models/registry` | Model versions and the active one |
| GET | `/studio` | The Studio page |

**Config**

```yaml
objects:
  root: data/objects
  max_image_mb: 12
  thumbnail_px: 240
  blur_var_min: 60.0          # heuristic guidance threshold, see decisions.md
  min_box_area_frac: 0.01
  duplicate_hamming_max: 6
  coverage_targets:
    min_positives: 20
    min_views: 4
    min_distances: 2
    min_lighting: 2
    min_occluded: 2
    min_hard_negatives: 5
studio:
  stop_monitoring_on_enter: true
ui:
  panel:
    default_width_px: 380
    min_width_px: 300
    max_width_px: 560
    max_width_frac: 0.40
```

---

## BLOCK 10 — ERROR HANDLING

**Fail loudly, exit non-zero:** a corrupt `objects.json` or object manifest; a sample whose image file is missing; a box outside image bounds; an export that would place an evaluation image into the object training set.

**Degrade gracefully, log once:** an oversize or unreadable upload (reject with a clear message, keep the session); a camera unavailable in the Studio (show the empty state with next steps); a missing model registry entry (report, do not crash).

**Never:** write a sample without a box; delete a sample or profile without soft-delete; leave monitoring running when the Studio is open; hold a camera device open after leaving the Studio; overwrite an object manifest without an atomic write.

---

## BLOCK 11 — PERFORMANCE

Measure and report: Studio capture round-trip (shutter to saved sample, p50/p95); quality computation cost per sample; Studio preview FPS; panel resize interaction — assert resizing does not force a video re-layout stutter, and that no layout thrash occurs (batch reads and writes, use a rAF-throttled handler); memory across a 50-capture session; confirmation that no perception ran while the Studio was open.

Re-run the preview-independence stall test after returning from the Studio to monitoring. Do not claim any figure not produced on this machine.

---

## BLOCK 12 — TESTS

Existing suite must stay green — that is the primary regression gate. Markers as existing.

**Unit**

1. `test_object_registry` — create/read/update/soft-delete; slug collisions handled; counts update; atomic manifest writes; corrupt manifest rejected.
2. `test_object_quality` — Laplacian variance ordering on synthetic sharp vs blurred crops; dHash stable under re-encode and differing for different images; duplicate detection at the configured Hamming distance; coverage counts correct on a constructed sample set.
3. `test_model_registry` — schema validation; exactly one active version; unknown-hash entry rejected.
4. `test_dataset_separation` — no image path, content hash or source clip appears in both `data/objects` and `data/eval`; the exporter refuses a violating export.
5. `test_panel_resize_contract` — pure logic: clamping to min/max and the 40% rule, persistence round-trip, reset behaviour.

**Integration**

6. `test_objects_api` — full lifecycle: create object, post a sample with box and tags, list, patch box, coverage reflects it, delete is soft; oversize upload rejected; missing box rejected.
7. `test_studio_lifecycle` — `/api/studio/enter` stops ingest and perception (assert zero ingest messages and zero perception calls over a bounded interval); `/api/studio/leave` restores the prior state exactly.

**Regression** — `test_preview_independence`, `test_ingest_socket`, `test_recorded_driver`, `test_recorder_upload`, `test_camera_recovery`, `test_no_outbound_network`, Phase 1.6 UI structure tests, Phase 2 perception tests, Phase 2.5 policy/eval tests.

---

## BLOCK 13 — COMMANDS

```powershell
cd C:\Users\mummi\Documents\Projects\PredictiveSense
.\.venv\Scripts\Activate.ps1

pip install -e ".[dev]"
pip freeze --exclude-editable | Out-File -Encoding utf8NoBOM requirements.lock.txt

pytest -q
pytest -q -m "not slow"

python scripts\run_app.py --profile dev --source-kind browser
# developer works at /studio

python scripts\export_objects_coco.py --out data\objects\coco_train.json
```

Run everything yourself, fix your own failures, re-run. The only steps the developer performs are the physical checks in Block 14.

---

## BLOCK 14 — PHYSICAL VERIFICATION (developer, not you)

Short. Mark as pending and list:

1. Open the Studio; confirm monitoring stops (no detections running, no analysis activity).
2. Create an object, capture ~10 images across different views, distances and lighting, adjusting the box each time; confirm the flow is quick and obvious.
3. Upload an image to the same object; confirm it appears with the same fields.
4. Add hard negatives for one confusable object.
5. Return to monitoring; confirm the camera and perception resume as before.
6. Drag the panel wider and narrower; confirm the video stays usable, nothing overflows, and the width survives a reload.
7. Collapse and reopen the panel.

---

## BLOCK 15 — DEFINITION OF DONE

- [ ] Studio reachable at `/studio`; entering provably stops monitoring; leaving restores it exactly; test asserts both.
- [ ] Object CRUD with soft delete; class/instance schema present; vocabulary not hard-coded.
- [ ] Capture and upload both produce samples with exactly one box, condition tags, quality fields and full provenance.
- [ ] Quality checks implemented with no new dependency; guidance strings shown; thresholds documented as heuristics in `docs/decisions.md`.
- [ ] Hard negatives supported; `confusable_with` seeded from the observed failures.
- [ ] Object dataset stored separately from `data/eval`; separation test passes; COCO exporter produces a disjoint train/val split.
- [ ] `models/registry.json` created with the current detector as baseline v1 and its metrics reference; registry module reads and validates it; no training performed.
- [ ] Panel resizes by drag and by keyboard, clamps correctly, persists, resets on double-click; collapse and reopen work; no overflow at any width.
- [ ] Record Sample investigated; root cause reported; fixed only if genuinely broken and contained.
- [ ] Full suite green; new tests pass.
- [ ] `docs/phase-reports/phase4.md` in the three-part format; tree committed; no remote.

---

## BLOCK 16 — REPORT

`docs/phase-reports/phase4.md`, three separated parts:

**Measured** — capture round-trip p50/p95, quality computation cost, Studio preview FPS, memory over a 50-capture session, evidence that no perception ran while the Studio was open, panel resize behaviour under test, full test results, Record Sample root-cause findings.

**Physically observed by the developer** — pending until performed; quote the Phase 2 failure-mode observations as motivation, attributed and marked as observations rather than measurements.

**Not verified / limitations** — must state plainly: **no model has been trained and recognition is unchanged by this phase**; collected images are training data, not a trained model; quality thresholds are collection heuristics, not validated metrics; coverage targets are guidance, not evidence of sufficiency; no tracking, temporal reasoning, risk inference or voice exists; the object dataset is small and from one environment.

Reply with only:

```
IMPLEMENTED: ...
STUDIO LIFECYCLE: ...
DATASET SEPARATION: ...
PANEL: ...
RECORD SAMPLE FINDINGS: ...
TESTS: ...
LIMITATIONS: ...
NEXT PHASE NOT STARTED: confirmed
```

---

## BLOCK 17 — EXPLICITLY NOT IN THIS PHASE

Training or fine-tuning of any model; model activation or replacement; instance-recognition inference; environment scan; marking a wrong live prediction as an error sample (the data model supports it, the flow is a later phase); tracking, temporal state, scene relations, risk inference, recommendations, voice; any change to the recognition policy or detector behaviour; any change to the evaluation dataset or harness.

---

## BLOCK 18 — EXECUTION INSTRUCTIONS

Inspect before modifying. Reuse the existing camera abstraction, shell, registry, controls, dataset store and telemetry — do not fork or rewrite them. Implement only this phase. Run all tests and measurements yourself and fix your own routine failures without asking. Do not ask for approval on routine steps; stop only for a genuinely blocking ambiguity or a missing prerequisite, and say precisely what is blocking. Do not add dependencies. Do not spend effort on cosmetics outside the Studio and the panel. Document anything genuinely unresolved rather than papering over it. Never claim a measurement you did not run or a physical check you did not perform.

When Phase 4 is finished, **stop**. Do not begin the next phase.
