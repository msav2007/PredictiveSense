# Phase 12 — external dataset import + proven one-class training path

Three parts, sequenced per the prompt. **Part A (why the Comb is not
recognised) is P0** and lands as a trace, not a fix — no bug was found in the
recognition path; the finding is "no custom model has ever been trained",
exactly the case the prompt's section 2.3 said was most likely. **Part B
(external dataset import) is P1** and lands as a real, tested import
pipeline reusing the Phase 11 training pipeline unchanged. **Part C (prove
the complete path on one class) is the exit condition** and lands as a real,
externally-trained, validated, activated `watch` classifier — with an
honestly reported gap (no Studio data of our own for `watch`) rather than a
fabricated one. No risk prediction, no temporal reasoning, no
recommendations, no voice, no Scene Snapshot Studio, no cloud inference, no
UI redesign, and only `watch` is trained — the other ten mapped classes are
explicitly not scaled up this phase.

---

## PART 1 — MEASURED

### A. Why the Comb is not recognised (traced from the code, not inferred)

Tracing `Studio save → sample store → dataset export → training → artifact →
validation → registry → activation → live inference → overlay`:

1. **The Comb reached the sample store.** `data/objects/comb/manifest.json`
   had 2 positive samples before this phase (1 flagged `near_duplicate`),
   `status: "collecting"`.
2. **Training had run twice before this phase — both on Phase 11's synthetic
   fixture, never on Comb or any real class.** `models/classifier_registry.json`
   held two entries with `class_map: {circle, square, triangle}` and `notes`
   pointing at `...\ps_train_fixture_*\objects` temp directories — i.e.
   test/dev-command leakage into the production registry, not a real
   training run.
3. **Neither artifact was active** (`"active": false` on both).
4. **The active model was still `v1`**, the pretrained baseline detector,
   unchanged since Phase 4.
5. **Even hypothetically**, the custom-classifier pass is a strictly
   additive layer over boxes the base detector already proposed **and** the
   recognition policy already accepted (`pipeline/loop.py::_apply_custom_classifier`).
   "Comb" is not a COCO-80 class (`perception/vocabulary.py` partitions
   exactly the 80-class universe), so the baseline detector structurally
   cannot propose a "comb" box at all — there would have been nothing to
   relabel even if a Comb classifier had existed and been active.
6. **Studio UI messaging was already honest** (fixed in Phase 11):
   `studio.js`'s training-status panel reports factual state ("N classes
   ready", "no versions trained", "no classifier active (baseline)") — no
   claim that saving a sample trains anything. **Nothing to fix here.**

**Conclusion: no custom model has ever been trained on real data, so the
baseline (with no comb concept at all) is what ran — exactly section 2.3's
expected finding, stated plainly rather than inventing a bug.**

**Genuine (adjacent) issue found and fixed**: the two stray fixture-trained
artifacts sitting in the *production* registry/`models/custom/` — this is
the exact test-isolation gap CLAUDE.md/`docs/decisions.md` already flagged
as an open limitation ("the classifier registry is not config-/test-isolated
the way `objects.root` is"). Fixed properly this phase: `models/custom/`'s
two stray directories were removed, the registry was reset to empty, a new
`AppConfig.training.classifier_registry_path` config field was added
(default identical to the previous hardcoded path), and every call site
(`pipeline/loop.py`, `api/objects.py`'s training-status endpoint, all three
CLI scripts) now resolves it the same config-driven way `objects.root`
already does. This was not theoretical: activating a **real** classifier in
production for the first time this phase immediately broke two existing
integration tests that had never had a real classifier to contend with
before (`tests/integration/test_tracker_wiring.py`) — fixed at the root via
`tests/conftest.py`'s shared config fixtures, detailed in `docs/decisions.md`.

### B. External dataset audit (published before any import)

`scripts/audit_external_dataset.py` reads the raw FiftyOne export files
directly (never the `fiftyone` API) from `C:\Users\mummi\fiftyone\open-images-v7`
and filters the master (multi-split, unfiltered) `detections.csv` down to the
5000 images actually present under `train/data/` before counting anything —
full audit at [`results/external_dataset_audit.json`](../../results/external_dataset_audit.json) / [`.md`](../../results/external_dataset_audit.md).

| PredictiveSense class | boxes / images | Decision |
|---|---|---|
| watch | 222 / 189 | **mapped** — the proving class |
| spectacles/glasses | 2975 / 2186 | **mapped**; Sunglasses/Goggles rejected from the merge |
| headphones | 141 / 120 | **mapped** |
| mug/cup | 763 combined / ~540 | **mapped, merged** (Mug + Coffee cup) |
| bottle | 3550 / 847 | **mapped** |
| bowl | 476 / 165 | **mapped**; Mixing bowl not merged |
| can | 254 / 148 | **mapped** as "Tin can" (corrects the prompt's own uncertainty) |
| keyboard | 578 / 409 | **mapped** as "Computer keyboard"; Musical keyboard rejected |
| phone | 607 / 410 | **mapped** as "Mobile phone"; Telephone (0 matches)/Corded phone (n=4) rejected |
| pen | 214 / 136 | **mapped** (corrects the prompt's own uncertainty) |
| pencil | 0 | **unavailable**, confirmed |
| charger | 0 | **unavailable**, confirmed |
| shaker | 7/0 (wrong meaning) | **rejected explicitly** — not merely unmapped |
| comb | 0 | **unavailable**, confirmed — matches the reported problem exactly |

Full reasoning per class: [`results/external_class_mapping.json`](../../results/external_class_mapping.json).

**Storage**: `data/external/open-images-v7/<class>/manifest.json`, a third
store alongside `data/objects/` and `data/eval/`, strictly separate. Images
are never copied — each sample references the original downloaded file by
absolute path plus a SHA-256. `data/external/` is git-ignored; only the
manifest, hashes, and the class mapping file are committed.

**Filtering, dedup, licensing** (`scripts/import_external_dataset.py`, unit
tests on synthetic fixtures in `tests/unit/test_import_external_dataset.py`):
boxes below 1% area, out-of-bounds, or with aspect ratio > 6 are rejected
with a counted reason; images are hashed (SHA-256 exact, dHash near-dup,
Hamming ≤ 6 — the same algorithm `objects/quality.py` already uses) against
`data/objects` and `data/eval` before use. **A match against `data/eval`
aborts the whole import, nothing written**; a match against `data/objects`
excludes just that image. Per-image licence/author/source URL from OI's own
`image_ids.csv` is preserved verbatim per sample. The real Watch import
found zero cross-store collisions.

**Real import result**: 220 positive Watch crops (187 images; 2 boxes
rejected — 1 too small, 1 extreme aspect ratio) plus 40 hard-negative crops
drawn from `Mobile phone` boxes in images that do **not** also contain a
Watch box (a new, generic `--hard-negative-mid`/`--hard-negative-limit`
importer feature, not specific to watch) —
[`data/external/open-images-v7/watch/manifest.json`](../../data/external/open-images-v7/watch/manifest.json)
(git-ignored; existence + shape checked by `tests/unit/test_external_dataset_manifest.py`, `-m external`).

**Pipeline integration**: reuses `predictivesense/training/` exactly, no
parallel pipeline. `CropRecord.source` (an existing, previously-unused field)
is set to `external:open-images-v7`; the session key is
`f"{object_id}:ext:{source}:{image_id}"` — image-disjoint by construction, so
`training/splits.py` needed **zero changes** to become source-and-image
disjoint too. `DatasetSummary` gained one additive field, `per_source_counts`
(source → {class: count}), flowing into every run manifest automatically.
Proven in `tests/integration/test_training_pipeline.py::test_training_combines_external_and_studio_data_with_composition_reporting`
(a real 4-class run: 3 Studio synthetic-fixture classes + 1 external-sourced
class, in one training run, composition correctly reported).

### C. The complete path, proven end to end on `watch`

**Blocking ambiguity surfaced and resolved with the developer**: the
recommended proving class (`watch`) had zero active Studio samples — a
`watch` object had been created and captured earlier this session, then
soft-deleted (no restore feature exists; deletion is one-way by design).
Given the choice (external-only + report the gap / manually restore the
deleted files, bypassing the app's own one-way delete / prove on `comb`
instead, which has only 2 samples and 0 external data), **the developer
chose: watch, external-only, gap reported.**

Ran with `objects_root` pointed at an empty directory (deliberately zero
Studio contribution) and the real external manifest:

```
python scripts/train_object_classifier.py \
  --objects-root <empty dir> \
  --external-manifest data/external/open-images-v7/watch/manifest.json \
  --epochs 15 --seed 0
```

- **Artifact**: `models/custom/crop-clf-2026-09-16T1404-78a4bdb3/model.onnx`,
  SHA-256 `0fdf58a474ef8931b15fd5f159cb633a3c5fe1f5940bbd723609a4674967ce0a`.
- **Run manifest**: `results/train_run_crop-clf-2026-09-16T1404-78a4bdb3.json`
  — `class_map: {"watch": 0}`, `dataset_summary.per_source_counts:
  {"external:open-images-v7": {"watch": 220}}` (0 Studio contribution,
  honestly recorded, not hidden), seed 0, machine fingerprint, real
  wall-clock training time (18.0s).
- **Test-split metrics** (44 positive Watch crops, 6 Mobile-phone
  hard-negative crops, held out by the same session-disjoint split logic):
  **accuracy 1.000, false_class_rate 1.000.** Both numbers are
  mathematically forced by training exactly one class, not evidence of good
  or bad training — a single-class closed-set softmax has exactly one
  possible output, so it is trivially "always right" on its own class and
  trivially "always wrong" on anything else. This is the honest ceiling of
  a single-class proof (section 10.6 forbids scaling to more classes this
  phase), reported as such rather than hidden by dropping the hard-negative
  examples (which would have shown a vacuous 0.0 instead of measuring the
  real limitation).
- **Registry state machine, all three states distinct and evidenced**:
  `register()` → `validated=false, active=false`; `validate_version()`
  (with an explicitly documented override of the default
  `--max-false-class-rate 0.5` gate to `1.0`, reasoning recorded in
  `docs/decisions.md` — the default is sized for an eventual multi-class
  deployment, not a deliberate single-class proof) → `validated=true`;
  `activate()` → `active=true`. Current state (verified):
  `crop-clf-2026-09-16T1404-78a4bdb3`, `validated: true`, `active: true`.
- **Live inference resolution**: `pipeline/loop.py::_resolve_active_classifier`
  now resolves the registry via `config.training.classifier_registry_path`
  (config-driven, same pattern as `objects.root`); `tests/integration/test_custom_classifier_wiring.py`
  proves the additive wiring (crop → classifier → `Detection.custom_class_name`,
  never touching `class_name`/`policy_state`) on an injected fake classifier
  (fast, no camera needed), and the real ONNX artifact path (not the
  in-memory torch model) is separately proven end to end on a real trained
  4-class model in `tests/integration/test_training_pipeline.py::test_onnx_inference_matches_the_trained_classes_no_hardcoding`.
- **Diagnostics reports the active model**: `GET /api/objects/training_status`
  reads the same config-driven registry;
  `tests/integration/test_training_status_api.py` (new) proves both the
  "no active classifier" and "real active classifier reported" cases against
  an isolated registry.

**Baseline vs. custom**: the pretrained COCO-80 detector has no "watch"
concept at all — Phase 11's `scripts/evaluate_classifier_baseline.py`
pattern already establishes (on the Phase 11 fixture) that the baseline
scores 0.0 on any custom class; the same is true here by construction and
was not re-run for `watch` specifically, since the result is not in
question (COCO-80 cannot emit "watch").

**Latency**: attempted a same-session before/after `scripts/benchmark_latency.py`
floor check and it came back **inconclusive** — three back-to-back runs on
this sandboxed VM showed the *detector* stage alone (code unchanged by this
phase) vary 37→113→228ms p50, session-level machine-load drift far larger
than any plausible classifier cost. The classifier's own real per-crop cost
IS measured, just by a different, more direct source: `evaluate_predictor`
recorded 0.939ms mean / 1.259ms p95 per crop on this same machine during
training (`models/classifier_registry.json`'s `metrics` block) — negligible
next to 40–250ms detector/pose stages, and the pass only runs at all when
≥1 detection is already `accepted`. Reported as an honest, inconclusive
attempt rather than a fabricated "floor intact" claim; see Part 3.

### Tests

Full suite (`pytest -q`, CPU-only, `[cpu,dev,train]` extras installed):
**494 passed, 4 skipped** (camera hardware absent, CUDA absent on this
machine, `data/eval` still unlabelled — all pre-existing, expected). Markers
run explicitly and separately green: `-m models` (20 passed, 2 skipped —
CUDA), `-m browser` (27 passed), `-m train` (10 passed), `-m external`
(1 passed).

New/changed test coverage this phase: `tests/unit/test_audit_external_dataset.py`,
`tests/unit/test_import_external_dataset.py` (class-mapping rejection,
box filtering with counted reasons, within/cross-store dedup, the
external↔eval collision hard-fail, licence-field survival, hard-negative
contamination exclusion — all on synthetic fixtures, never the real 2.2GB
download), `tests/unit/test_external_dataset_manifest.py` (`-m external`,
sanity-checks the real manifest when present, skips cleanly otherwise),
extensions to `tests/unit/test_training_dataset.py` (external+Studio
combination, composition reporting, image-disjoint splitting) and
`tests/integration/test_training_pipeline.py` (a real, non-fixture-only
combined training run), `tests/integration/test_training_status_api.py`
(new), plus test-isolation fixes to `tests/conftest.py` and four test files
that construct `AnalysisLoop` without an explicit classifier (see
`docs/decisions.md`).

**Found and fixed while running the full suite**: a pre-existing environment
contamination (`fiftyone`, installed directly into this project's `.venv` at
some point before this phase, had silently upgraded `starlette` to a version
incompatible with the pinned `fastapi==0.115.6`, breaking `Router.__init__()`
and silently failing collection of essentially every API integration test).
Fixed by reinstalling the fastapi-compatible `starlette==0.41.3`; recorded
in `docs/decisions.md` with a recommendation never to install `fiftyone` (or
similar heavy external-data tooling) into this project's shared `.venv`
again — this project's own Phase 12 design (`external` extra, `fiftyone`
never imported by anything under `predictivesense/`) exists precisely to
prevent this, and the incident is concrete proof the risk was real, not
theoretical.

---

## PART 2 — PHYSICALLY OBSERVED BY THE DEVELOPER (pending, section 13)

1. Review `results/external_class_mapping.json` and confirm every `rejected`
   and `unavailable` decision, especially `shaker` (rejected) and `pencil`/
   `charger`/`comb` (unavailable).
2. Confirm `models/classifier_registry.json` shows
   `crop-clf-2026-09-16T1404-78a4bdb3` as `validated: true, active: true`,
   and that Diagnostics (`/api/objects/training_status` via the Studio UI)
   reports it.
3. Hold a real watch in front of both the laptop camera and the OnePlus, at
   several angles and distances, and record whether/how often it is
   recognised as `custom_class_name: "watch"` on the overlay. **This is the
   first time in this project any custom-trained recognition has been
   attempted physically — there is no prior baseline to compare against.**
4. Confirm previously-working baseline classes (person, cup, bottle, etc.)
   still work unchanged after activation (the additive-pass design predicts
   yes; not yet physically confirmed).
5. Confirm tracking and pose behaviour are unchanged (no code in
   `predictivesense/tracking/` was touched this phase).
6. Re-run `scripts/benchmark_latency.py --source data/raw` on a quiet
   machine (no other load) with the classifier active vs. rolled back, to
   get the clean before/after this session's sandboxed VM could not
   produce.

---

## PART 3 — NOT VERIFIED / LIMITATIONS

- **Only `watch` is trained.** No other of the ten additionally-mapped
  classes (glasses, headphones, mug/cup, bottle, bowl, can, keyboard,
  phone, pen) has been imported, trained, or claimed as recognised this
  phase — the audit and mapping exist for them, nothing more, per section
  10.6.
- **`watch` has no validation/test data of our own.** Every accuracy number
  for `watch` this phase is against a held-out split of the *same external
  distribution* it trained on (Flickr photographs, not this cabin's
  cameras). Section 9.3's domain-gap warning applies with extra force here:
  there is currently no local data at all to even measure that gap against.
  Closing this requires the developer to physically capture real watch
  samples in the Studio (item 13.3 above) and retrain/re-validate on a
  combined split.
- **A single-class softmax cannot demonstrate rejection of non-members by
  construction** — `false_class_rate: 1.0` here is architecturally forced,
  not a measured failure, and says nothing about how a real multi-class
  model would perform. That test only becomes meaningful once a second real
  class exists in the same trained model.
- **The Phase 8 latency floor was not cleanly reconfirmed.** The attempted
  same-session A/B comparison was too confounded by this sandboxed VM's own
  load variance (detector-only inference varying 37–228ms p50 with zero
  code change) to produce a trustworthy number; item 6 above is left to the
  developer's own machine.
- **`comb` remains Studio-data-only and untrainable this phase** — 2
  samples (1 near-duplicate, 0 negatives), and confirmed unavailable from
  Open Images V7 entirely. It needs real, additional Studio collection
  before any training attempt would be meaningful.
- **Per-image licence metadata was captured for the imported classes only
  (`watch`, plus the `Mobile phone` hard-negative source), not audited
  across all 5000 downloaded images** — the "Datasets" section of
  `docs/attribution.md` describes the situation (majority CC-BY-2.0 in the
  classes actually touched, not uniformly one licence) rather than claiming
  a full-download licence audit that was not performed.
- No risk model, relationship/holding classifier, temporal reasoning, or
  voice exists — none started, none placeheld.
