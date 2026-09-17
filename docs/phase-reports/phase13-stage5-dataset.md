# Phase 13 — Stage 5: Detection Dataset Build

Builds the detection-format dataset and split machinery only. No training, no detector
changes, no live-pipeline changes. Full prompt: `PredictiveSense-Stage5-Prompt.md`.

---

## PART A — MEASURED

### 0. Housekeeping

**Stage 2.5 test numbers**, previously only reported in chat, are now recorded in
`docs/decisions.md` under "Stage 2.5 verification (2026-09-17)": full suite 502 passed /
4 skipped / 0 failed (206.56s); browser suite run twice, 27 passed / 0 failed each time.
Re-verified at the start of this stage (below) against a re-fetched `origin/main` (`0 0`)
and the deactivated classifier entry's current on-disk content — the repo matched every
claim in the Stage 2.5 report.

**`CLAUDE.md`'s "Current phase status"** was still describing Phase 12 as current; it now
lists Phase 13 Stages 1/2/2.5/5 and describes `training/` as one parked experiment (its
one artifact deactivated) rather than "the project's training story," per this stage's
instruction.

### 1. Repository vs. prompt — where they disagreed

The prompt's §1 factual claims were re-verified against the repository before anything
was built, not trusted:

| Claim | Verified |
|---|---|
| `data/external/open-images-v7/watch/manifest.json`: 260 samples (220 positive/187 images + 40 hard negatives, 2 rejected, all CC-BY-2.0, all human-confirmed) | **Matches exactly** — re-read the manifest directly. |
| `results/external_dataset_audit.json`: Watch 222/189, Bottle 3550/847, Glasses 2975/2186, Mug 250/187, Coffee cup 513/379 | **Matches exactly.** |
| `data/objects/comb/`: 1 image | **Matches** — 1 sample in the manifest (the second file in `images/` is its own thumbnail, not a second photo). |
| `data/eval/annotations.json`: 34 images, 0 annotations | **Matches exactly.** |

One thing the prompt did not claim but the repository revealed: **`results/external_class_mapping.json` and `results/external_dataset_audit.json` — Phase 12's own committed-looking artifacts — were never actually tracked by git.** `.gitignore`'s blanket `results/*` rule silently excluded them from every commit, including all of Stage 2.5's, without ever showing up in `git status` (untracked-and-ignored files are invisible to a plain status check). Fixed in this stage's `.gitignore` change, alongside allow-listing this stage's own `results/detection_class_map.json` and `results/detection_dataset_audit.{json,md}` for the same reason — the class map is required to be "a single committed JSON file" (§3.1), and it would silently not have been.

### 2. Class map — published before any split

`results/detection_class_map.json`, versioned, one row per target class carrying every
contributing source class's dataset/name/MID, merge decision + reason, or
rejection/unavailability + reason, plus an `own_capture` pointer to the matching Studio
object where one exists. Every figure and every "zero matches"/"class absent" claim
carried forward from Phase 12's `external_class_mapping.json` was **re-verified against
the current audit JSON and the raw `classes.csv`** on 2026-09-17 (not copied forward
blind) — all matched, including the four classes with zero matches (`Can opener`,
`Pencil sharpener`, `Salt and pepper shakers`, `Telephone`) and comb/pencil/charger's
total absence from the 601-class vocabulary (re-confirmed by `grep` on the raw file).

Proven by `tests/unit/test_detection_class_map.py` (9 tests): watch mapped with a real
MID; mug/cup is an explicit two-source merge with a recorded reason; shaker is
**rejected**, not merely unmapped, with its reason recorded; pencil/charger/comb are
`unavailable` with empty source lists; every mapped entry has a real MID; every entry has
a recognised status.

### 3. The manifest-shape problem (§2) — exhaustive multi-class requery

`scripts/external_detection_source.py::query_active_classes()` is the fix: **one pass**
over the raw `detections.csv`, filtered to `(ImageID in image_ids) & (LabelName in
active_mids)` where `image_ids` is the *entire* download population and `active_mids` is
*every* mid across *every* class this build is producing — never one class's mids against
that class's own image subset. There is no per-class pre-filtering step for the function
to accidentally silo classes into, by construction.

Proven directly, as the prompt required: `tests/unit/test_external_detection_source.py`
builds a synthetic image with a watch box AND a mug box and asserts both survive
(`test_an_image_with_two_target_classes_yields_both_classes_boxes`), plus a same-class
two-boxes-on-one-image case (mirroring the real watch manifest's 220-box/187-image
shape), an empty-active-set guard, and an out-of-scope-image exclusion check. 4/4 pass.

**Zero-box images** (§2.2) are preserved as legitimate zero-annotation records —
`predictivesense/dataset/detection_convert.py::add_detection_record` calls
`CocoStore.set_frame_boxes(iid, [], seeded=False, labelled=True)` for them, never drops
them. Proven by `test_zero_box_image_survives_as_a_labelled_zero_annotation_record`.

### 4. Converter — extends the existing importer + COCO store, no third path

`CocoStore.set_frame_boxes()` gained one additive, optional field: a per-box `"extra"`
dict merged verbatim into the stored annotation, so `source_class` (the raw OI class name,
e.g. "Mug" vs. "Coffee cup" for the merged `mug/cup` target) and `box_confirmed_by_human`
survive per box without inventing a parallel annotation format. Existing
`test_coco_store.py`/`test_splits_no_leakage.py`/`test_label_api.py` (18 tests) still pass
unchanged — the extension is backward compatible.

Image-level provenance (source dataset + version, licence, author, source URL, original
image id, `image_sha256`, target classes present, filtering notes, and — for own-capture
records — session role/provenance/quality) lives in `ps_provenance`, the field the store
already provided for exactly this purpose.

Own-capture conversion (`predictivesense/dataset/detection_convert.py::own_capture_records`)
reuses `predictivesense.training.dataset.session_key_for` (imported, not reimplemented) so
own-capture session derivation is byte-identical to the crop-classifier path's.

### 5. Duplicate / blur / geometry / licence audit — before any split

`scripts/audit_detection_dataset.py` reuses `predictivesense/objects/quality.py`'s
existing dHash/Hamming/Laplacian-variance heuristics (no new dependency, no second
implementation). Duplicates are grouped and reported, never deleted, except exact
byte-duplicates (reported under their own key). Proven by
`tests/unit/test_audit_detection_dataset.py` (3 tests: per-class totals/zero-box count,
exact-duplicate grouping without deletion, below-min-area counting).

### 6. Splits — image-, duplicate-group-, session-disjoint, stratified, hashed

`predictivesense/dataset/detection_splits.py` extends
`predictivesense/training/splits.py::build_crop_splits`'s exact algorithm (per-class
independent seeded shuffle-and-cut of whole *groups*, then merge) to a generalised
"group" that covers both own-capture sessions and external near-duplicate-merged groups.
A `UnionFind` merges any two groups connected by a duplicate edge the audit found — a
near-duplicate group can never straddle a split, even across sources.
`tests/unit/test_detection_splits.py` (10 tests) proves: image/group disjointness,
same-seed determinism (identical content hash), a merged duplicate pair always landing in
one split, realized per-class counts, and — the explicit §3.5 requirement — a class with
1 image refuses a training split while a class with 50 does not.

**Train/val ↔ `data/eval` collision hard-fails**, never warns:
`assert_no_eval_collision` checks both exact `image_sha256` match and near-duplicate
(Hamming ≤ 6) against eval frame phashes and raises `EvalCollisionError`. Extended (not
duplicated) into the end-to-end build via `scripts/build_detection_dataset.py`, proven by
`test_a_training_image_matching_an_eval_frame_hard_fails_the_whole_build` — the build
writes a store + audit, then aborts before `splits.json` exists.

**Own-capture session-disjointness** proven end to end:
`test_own_capture_images_in_different_sessions_can_land_in_different_splits` builds 62
Studio samples (2 in one batch/session, 60 each in their own), runs the real orchestrator,
and confirms the 2 same-session images always land in the same split.

### 7. Own-capture path — plumbing smoke test, not a dataset

`scripts/build_detection_dataset.py --own-capture-classes comb:comb` was run for real
against the actual `data/objects/comb/` (1 image). It correctly:

- Converts the 1 sample into a detection record (1 image / 1 annotation, session
  `comb:cam:OnePlus Nord 4 (Windows Virtual Camera):1988413`) and writes a store + audit —
  the plumbing works.
- **Refuses to emit a training split**: `SPLIT REFUSED: class 'comb' has 1 image(s)
  available for a train split - below the configured minimum of 50.` Exit code 3.
  `splits.json` was never written.

**This is a plumbing smoke test, not a dataset** — stated here explicitly per the
prompt's instruction. One comb image did not, and structurally cannot, become a training
set by accident.

### 8. The real run — `watch` + `mug/cup`, chosen classes justified

Command:

```
.venv\Scripts\python scripts\build_detection_dataset.py --external-classes watch "mug/cup" --out-dir data\external\detection --seed 0 --min-images-per-class 50
```

**Second-class choice: `mug/cup`, not `Bottle`.** Bottle has far more raw boxes
(3550/847 vs. mug/cup's combined 763/446), but its **median box_area_frac is only
0.0199** against mug/cup's 0.107–0.150 (Mug) / 0.107 (Coffee cup) — an order of magnitude
smaller, i.e. most Bottle boxes are small/distant objects in cluttered scenes, a real data-
quality concern for a first multi-class proof. Mug/cup still gives ample data (763 boxes
pre-filter, well above Watch's 222) with a far healthier size distribution. Both overlap
COCO classes; per the prompt this is a Stage 8 question, not a reason to avoid them here.

**Honest finding: the exhaustive-requery mechanism does not change any box count for
*this specific* real run.** A direct query of the raw CSV found **zero images** where a
Watch box and a Mug/Coffee-cup box co-occur in this particular 5,000-image download (watch
essentially never appears in-frame with anything else audited — only 1 of its 189 images
overlaps *any* other target class, `Glasses`). The mechanism *is* genuinely exercised
within the merged `mug/cup` class itself, though: **120 of 446 candidate images carry
both a Mug box and a Coffee-cup box** — a per-class importer that queried Mug and Coffee
cup as two separate passes without merging first could plausibly have mishandled these;
querying the full merged mid set in one pass does not. The general risk this stage warns
about (693 images with 2+ target classes across all 19 candidate OI classes) remains real
for any future import of additional classes — it is simply not exercised by this
particular two-class pair's real numbers.

**Result** (`data/external/detection/build_summary.json`, `results/detection_dataset_audit.{json,md}`):

| class | images | boxes | zero-box images |
|---|---:|---:|---:|
| watch | 187 | 220 | — |
| mug/cup | 437 | 717 | — |
| (neither, hard negatives/all-boxes-rejected) | — | — | 11 |
| **total** | **635 images** | **937 boxes** | |

Box rejections: `box_too_small`=46, `box_extreme_aspect_ratio`=2 (raw 985 boxes → 937
kept; every rejected box is counted, none silently dropped). **`watch`'s 220/187 exactly
reproduces Phase 12's independently-built crop-classifier import** (same 2 rejections,
same reasons) — a strong cross-check that this stage's from-scratch, exhaustive-requery
converter agrees with the earlier single-class importer where their scope overlaps.

Box area fraction (post-filter): watch median 0.328, p95 0.950; mug/cup median 0.144,
p95 0.728 — mug/cup's boxes remain meaningfully smaller than watch's even after
filtering, consistent with the pre-import audit signal.

Blur variance (post-filter): watch median 496, p95 3170; mug/cup median 127, p95 1175
(both well above the 60.0 "blurry" heuristic floor at the median; mug/cup has more
lower-tail blur than watch).

Licence: 100% `https://creativecommons.org/licenses/by/2.0/` (635/635).

Duplicates: 0 exact byte-duplicates; 0 within-class near-duplicate groups; **1
across-class near-duplicate group** — images `0283104d36470fd6.jpg` (mug/cup, a
Google-branded travel mug) and `06f04931d107573a.jpg` (watch, a black smartwatch on a
splash background). **Visually inspected both — they are genuinely different objects.**
This is a dHash false positive: two plain-white-background product-photo compositions
with similar coarse dark/light silhouette gradients hashed within the Hamming-6 threshold
by coincidence. The duplicate-group-disjoint split policy merged them into one atomic
group regardless (a safe direction to err — a wrongly-merged pair costs a
negligible amount of split independence; a missed *real* duplicate would leak). 0
duplicates against `data/eval`.

**[Stage 5.1 correction]** The two tables immediately below replace the Stage 5 originals,
which were wrong in two ways, both fixed in Stage 5.1 (see
`docs/PROJECT-STATE.md` §6 and the "Stage 5.1" entry in `docs/decisions.md`):

1. **The 11 zero-box images were never in any split** (`predictivesense/dataset/detection_splits.py::build_detection_splits`
   stratified per class, and a zero-box image has no class, so its group was silently
   absent from every stratum). Fixed: zero-box groups now get their own pseudo-class
   stratum (`__zero_box__`), split by the identical seeded shuffle-and-cut, exempt from
   `min_images_per_class` (hard negatives are not a trainable positive class). A new
   invariant check inside `build_detection_splits` itself now hard-fails if this ever
   regresses (`sum(len(split)) != len(store.image_ids())`), and
   `tests/unit/test_detection_splits.py::test_every_image_lands_in_exactly_one_split` /
   `test_zero_box_images_are_assigned_to_a_split_not_dropped` guard it directly.
2. **The original table's "groups" and box tallies were mislabelled/miscounted**: it said
   "634 groups (635 images minus the one merged duplicate pair)" — the actual group count
   (independently re-derived by direct id-set inspection, both before and after the
   zero-box fix) was **623**, not 634; 623 + the 11 previously-unassigned zero-box groups
   = **634 groups now** (the zero-box fix, not the duplicate-pair merge, accounts for the
   difference). Separately, the "watch boxes"/"mug/cup boxes" columns were **box counts**,
   not image counts, and were not labelled as such — the split JSON now stores
   `realized_per_class_box_counts` and `realized_per_class_image_counts` as two distinct,
   explicitly-named fields rather than one ambiguous `realized_per_class_counts`.

Rebuilt with the fix (same command, same seed 0): 635 images now land in exactly one
split (verified directly: `sum(len(split)) == 635`, `union(splits) == image_ids`, 0
missing).

Splits (seed 0, target 70/15/15, `min_images_per_class=50`; **634 groups total**: 445
train + 96 val + 93 test):

| split | groups | watch boxes | mug/cup boxes | zero-box images | watch images | mug/cup images | total images |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 445 | 152 | 488 | 8 | 131 | 307 | 446 |
| val | 96 | 37 | 126 | 2 | 28 | 66 | 96 |
| test | 93 | 31 | 103 | 1 | 28 | 64 | 93 |
| **total** | **634** | **220** | **717** | **11** | **187** | **437** | **635** |

Realized fractions by group count: train 0.7019, val 0.1514, test 0.1467. Content hash
(new, post-fix): `65f5b55c2ca0bc77b7ef3fc33a71b58a1d96c2507325d4c0634032eef4b11f50` — this
necessarily differs from Stage 5's original hash since the split assignment itself
changed (the zero-box images now have a destination). No eval collision (build
completed; `assert_no_eval_collision` did not raise).

### 9. Full suite

Before this stage's changes (Stage 2.5 baseline, `docs/decisions.md`): **502 passed, 4
skipped, 0 failed** (206.56s).

After (this stage's 38 new tests: `test_detection_convert.py` ×6,
`test_detection_splits.py` ×10, `test_external_detection_source.py` ×5,
`test_audit_detection_dataset.py` ×3, `test_detection_class_map.py` ×9,
`test_build_detection_dataset.py` ×5):

```
.venv\Scripts\python -m pytest -q
540 passed, 4 skipped, 0 failed, 49 warnings in 239.44s (0:03:59)
```

Delta: **+38 passed, 0 failed, 0 skipped change** — every new test is additive; nothing
existing broke. The `CocoStore.set_frame_boxes` extension was verified not to regress the
existing 18 coco-store/split/label-API tests before any new code was written on top of it.

**One transient failure observed and investigated, not hidden**: a third full-suite run
(after the `.gitignore` fix, before writing this report) showed
`tests/integration/test_object_batches_api.py::test_upload_associates_all_images_with_the_selected_object`
FAILED (539 passed, 1 failed, 4 skipped). Re-run alone: passed. Re-run as its whole file
(13 tests): passed. A fourth full-suite run: **540 passed, 0 failed** again. Classified as
a flake, not a regression — it did not reproduce in isolation, at file scope, or on a
second full run, and nothing in this stage's changes touches `api/object_batches.py` or
its dependencies. Consistent with CLAUDE.md's documented history of cross-test
executor-sharing flakes in this suite (Phase 8 follow-up). Reported here rather than
silently re-run until green.

**[Stage 5.1 update]** Timeboxed investigation (§3 of `PredictiveSense-Stage5.1-Prompt.md`)
did reproduce it — 2 failures across 60 repeated runs of this file (~3.3%), full detail
and root-cause finding recorded in `docs/PROJECT-STATE.md` and `docs/decisions.md`
("Stage 5.1"). Short version: it is **not** a recurrence of Phase 8's module-level shared
`ThreadPoolExecutor` bug (`batch_proposals_executor` is correctly scoped to `app.state`,
confirmed by reading `api/object_batches.py`/`api/app.py`) — it is `_poll()`'s fixed
2.5s budget (50 × 50ms) occasionally too tight for the background proposal thread under
heavy concurrent CPU load. Recorded as a known non-blocking intermittent, not fixed, per
the stage's own scope boundary.

---

## PART B — PHYSICAL VERIFICATION (PENDING — developer-performed)

Not performed by this stage; nothing here should be treated as observed.

- **Spot-check twenty converted boxes by eye** against their source images
  (`data/external/detection/coco_detection.json`'s `images`/`annotations`, cross-referenced
  by `ps_provenance.image_path`) and confirm the boxes are correct and tight. A starting
  list of 10 watch + 10 mug/cup image ids for this check can be pulled from
  `splits.json`'s `train` group (any id's `ps_provenance.image_path` is the original file
  on disk, `.venv\Scripts\python -c "..."` or any image viewer opens it directly).
- Confirm the one flagged across-class "duplicate" (images `0283104d36470fd6.jpg` /
  `06f04931d107573a.jpg`) still reads as two genuinely different objects when viewed
  full-size (this stage's own inspection used the Read tool's rendered preview, not a
  pixel-level compare).
- Confirm `results/detection_class_map.json`'s merge/rejection reasoning still reads as
  sound on a fresh read, independent of this session's own reasoning.

---

## PART C — KNOWN LIMITATIONS

- **The partial-annotation risk (§2.1) is handled by construction, but not exercised by
  this run's real numbers.** Watch and mug/cup happen not to co-occur in this particular
  5,000-image download (0 of 187/437 images). The machinery is proven correct by a
  dedicated synthetic-fixture test and by real internal exercise within the merged
  `mug/cup` class (120 dual-box images) — but nothing in this run's own output visibly
  demonstrates a box surviving that a naive per-class importer would have dropped from a
  *different* target class. A future import of a third class (Bottle, Glasses, etc.)
  alongside these two is far more likely to exercise it for real, per the 693-image
  multi-target-class figure in the underlying audit.
- **Realized split sizes are small in absolute terms for `watch`** — 131/28/28 **images**
  (152/37/31 boxes) — despite clearing the configured minimum by 2.6×. `mug/cup` fares
  better at 307/66/64 images (488/126/103 boxes). Neither `watch`'s total (187 images) nor
  `mug/cup`'s (437 images) is remotely close to the Data Collection Protocol's own stated
  "~200-300 images per class, below ~150 expect a poor result" guidance for a *detector*
  once split three ways, since these counts came entirely from the external download, not
  from a fine-tuning run (out of this stage's scope) — this is a dataset-size caveat for
  whoever runs training next, not a
  claim about this stage's own correctness.
- **No comb data exists beyond the 1-image smoke test.** Confirmed structurally
  unbuildable into a training split by the same machinery that will refuse it again
  automatically once real comb captures exist and are re-run through this exact path.
- **No own-capture eval data and no own-capture watch/mug-cup training data exist yet.**
  Every number in Part A comes from the external Open Images source only, exactly as
  Phase 12 also had to report for `watch`. The Data Collection Protocol
  (`docs/PredictiveSense-Data-Collection-Protocol.md`) remains the developer's, not this
  session's, next step.
- **The one across-class "duplicate" pair is a measured false positive of the dHash
  heuristic**, not a defect in the merge logic — flagged transparently above rather than
  silently accepted or silently dropped, consistent with `objects/quality.py`'s own
  documented caveat that these are collection-guidance heuristics, not scientific quality
  metrics.
- **A pre-existing Stage 2.5 gap was found and fixed, not merely noted**: Phase 12's
  `results/external_class_mapping.json` and `results/external_dataset_audit.json` were
  never actually committed (silently excluded by `.gitignore`'s `results/*`, invisible to
  `git status`). Fixed in this stage's `.gitignore` change alongside allow-listing this
  stage's own three new `results/` artifacts. This stage's changes, including this fix,
  are **not yet committed** — Stage 5's prompt did not request a commit, so none was made;
  `git status` after this stage shows the new/modified files as pending.
- **The `min_images_per_class` refusal aborts the WHOLE build on the first
  below-threshold class**, not just that one class's split, when external and own-capture
  classes are requested together in one invocation. This stage's real run kept the
  successful watch/mug-cup build and the refused comb smoke test as two separate
  invocations specifically to avoid this ambiguity; a future multi-class run mixing a
  well-supplied and an under-supplied class in one call would currently lose the
  well-supplied class's split too. Worth a partial-success mode if that combination is
  ever actually wanted.
