# PredictiveSense — PROJECT STATE (living snapshot)

**Single source of "where we actually are".** Intent lives in
`PredictiveSense_Project_Description.docx` (the official Word proposal). Implementation history
lives in `projectContext.md` and `docs/phase-reports/`. Decisions live in `docs/decisions.md`.
This file is the current position only — update it whenever the state materially changes, and
keep the "Previous → Current → Next" shape rather than erasing history.

**Last updated:** 2026-09-17 (Stage 5.1), from a direct repository audit (git, file contents,
dataset files, the Word document, and the phase reports). Not from conversation memory.

**Status vocabulary:** `[VERIFIED WORKING]` `[IMPLEMENTED — NOT YET VERIFIED]`
`[PARTIALLY IMPLEMENTED]` `[IN PROGRESS]` `[BLOCKED]` `[PLANNED]` `[NOT IMPLEMENTED]`
`[FAILED — NEEDS FIX]` `[DEPRECATED]`

---

## 0. CONFLICTS THAT MUST BE RESOLVED

### CONFLICT 1 — two incompatible "Phase" numbering schemes (HIGH PRIORITY)

**SOURCE A — the Word document, §Phase table (the official research plan, 9 phases):**
1 Literature & scope · 2 Data · 3 Baseline · 4 Proposed model · 5 Evaluation ·
6 Real-time system · 7 Dashboard & accessibility · 8 Testing · 9 Documentation

**SOURCE B — the repository, `docs/phase-reports/` (the build history, phases 0–13):**
0 scaffold · 1 camera · 1.5/1.6 camera+UI · 2 perception · 2.5 recognition · 4 Studio ·
5 recognition trust · 6 Studio repair · 7 bulk upload · 8 latency/portability · 9 tracker ·
10 tracking continuity · 11 freshness + crop training · 12 external dataset · 13 perception rebuild

**CONFLICT:** the same word and the same numbers mean entirely different things. The repo's
"Phase 8 complete" means latency instrumentation; the Word document's Phase 8 is *Testing*.
Anyone reading the two together will conclude testing is done. It is not.

**CURRENT DECISION:** keep both, but never write a bare "Phase N" again. Use **R1–R9** for the
Word document's research phases and **B0–B13** for the repository's build phases.

**REASON:** the Word document is the authoritative statement of intent and cannot be renumbered;
the phase reports are a historical record and must not be rewritten.

### CONFLICT 2 — where the custom detector sits in the order

**BLUEPRINT / perception-rebuild prompt says:** perception must be reliable *before* temporal,
context and risk work begins. `CLAUDE.md` carries the matching prohibition ("Do not start risk
prediction without the user's explicit go-ahead").

**MASTER HANDOFF §39 says:** continue the recorded-video pipeline, build the temporal/context
foundation (5) and risk-state pipeline (6) *before* custom object detection (8).

**DIFFERENCE:** whether temporal/risk may be built on the baseline COCO detector while the
custom detector is developed in parallel.

**WHY IT MATTERS:** the Word document's own scenarios decide this. TC01–TC10 and TC13–TC14 use
**mug, bottle, can, laptop, bag, chair, cable, person** — with the single exception of the cable,
every one is already a COCO class the baseline detects today. **Watch and comb appear in no
official test case.** They are architecture proofs, not scenario requirements.

**CURRENT DECISION (recommended, pending the developer's confirmation):** §39's order is
**correct and better than the earlier blueprint**. The temporal/context/risk layer can be built
and evaluated on the baseline detector using recorded video, because the objects the official
test cases need are already detectable. The custom detector proceeds in parallel as the
architecture proof, and is not on the critical path to the first review.

**REASON:** it removes the data-collection blocker from the critical path without weakening any
claim, and it matches the Word document's own scenario list.

### CONFLICT 3 — smartphone-first vs recorded-video-first

**Word document §11/§18:** the smartphone camera is the sensor and the first review demonstrates
a live feed. **Master handoff §10:** recorded video first; live camera afterwards.
**CURRENT DECISION:** recorded-video-first for development, live smartphone for demonstration.
No conflict in substance — the live path already works (B1–B8) and is not being removed.

---

## 1. CURRENT POSITION

```
Research phase (Word doc):   R2 Data  /  R3 Baseline   — both incomplete
Build phase (repo):          B13 perception rebuild, Stage 5.1 complete
Current objective:           temporal layer on the baseline detector, recorded video
Last verified success:       Stage 5.1 — 542 passed / 4 skipped / 0 failed; 635 images, 937 boxes,
                             every image in exactly one split (zero-box defect fixed and
                             guarded by test), committed and pushed (origin/main...main = 0 0)
Current task:                none — Stage 5.1 closed out
Next task:                   Stage 6 — temporal layer, BLOCKED on the developer's TC video
                             recordings (data/videos/ is empty)
```

**Not started, correctly:** temporal layer, context/interaction, risk estimation, early warning,
recommendation, voice/TTS, TC01–TC14 harness. `Relation`/`RiskState`/`Alert` exist as frozen
contract types (`core/types.py:280,289,297`) and are deliberately unpopulated.

---

## 2. COMPONENT STATUS

| Component | Status | Where | Next step |
|---|---|---|---|
| Camera / frame transport | `[VERIFIED WORKING]` | `camera/`, `api/ingest.py` | none |
| Preview independence, mailbox, staleness | `[VERIFIED WORKING]` | `camera/mailbox.py` | none |
| Recorded pipeline (deterministic) | `[VERIFIED WORKING]` | `pipeline/recorded.py` | becomes the primary dev path |
| Baseline detector | `[VERIFIED WORKING]` | `models/yolo11n.onnx`, registry `v1` | preserve; never overwrite |
| Recognition policy (3 tiers) | `[VERIFIED WORKING]` | `perception/policy.py` | none |
| Pose | `[VERIFIED WORKING]` | `perception/pose.py` | feeds R4 posture work |
| Tracker | `[VERIFIED WORKING]` | `tracking/` | feeds temporal layer |
| Provider resolution (cpu/cuda/dml) | `[VERIFIED WORKING]` | `perception/runtime.py` | none |
| Object Learning Studio | `[DEPRECATED — isolated, preserved]` | `objects/`, flags off | leave alone |
| Crop classifier | `[DEPRECATED]` | `training/`, artifact deactivated | leave alone |
| Detection dataset + splits | `[VERIFIED WORKING]` | `dataset/detection_*.py` | none — proceed to Stage 6 on the baseline detector |
| Custom detector | `[NOT IMPLEMENTED]` | — | blocked on licence decision |
| Labelled eval set | `[BLOCKED]` | `data/eval/` 34 images, **0 annotations** | developer labels at `/label` |
| Temporal / context | `[NOT IMPLEMENTED]` | — | next major build stage |
| Risk engine / early warning | `[NOT IMPLEMENTED]` | — | after temporal |
| Recommendation | `[NOT IMPLEMENTED]` | — | after risk |
| Dashboard | `[PARTIALLY IMPLEMENTED]` | `api/static/` — shows detections/tracks, not risk | extend at R7 |
| Voice / accessibility | `[NOT IMPLEMENTED]` | — | R7 |
| Detection eval harness | `[IMPLEMENTED — NOT YET VERIFIED]` | `eval/`, `scripts/eval_detection.py` | unusable until labels exist |
| Early-warning time, false-alarm rate | `[NOT IMPLEMENTED]` | — | needs temporal + event timings |
| Literature review / research gap | `[NOT IMPLEMENTED]` | — | **R1, never started** |
| TC01–TC14 traceability | `[NOT IMPLEMENTED]` | zero mentions repo-wide | create the matrix |

---

## 3. OFFICIAL TEST CASES — TC01–TC14

Authoritative source: the Word document's test-case table. Text must not be altered.
**Every row is `[NOT IMPLEMENTED]` today.** No TC identifier appears anywhere in the repository.

| TC | Scenario | Objects needed | In COCO today? | Status |
|---|---|---|---|---|
| TC01 | Good posture | person + pose | yes | `[NOT IMPLEMENTED]` |
| TC02 | Developing poor posture | person + pose, over time | yes | `[NOT IMPLEMENTED]` |
| TC03 | Mug near active hand | cup, person/hand | yes | `[NOT IMPLEMENTED]` |
| TC04 | Stable bottle/can | bottle | yes | `[NOT IMPLEMENTED]` |
| TC05 | Developing fall | bottle + edge + tilt | yes | `[NOT IMPLEMENTED]` |
| TC06 | Object falling | bottle, motion | yes | `[NOT IMPLEMENTED]` |
| TC07 | Laptop + drink | laptop, bottle/cup | yes | `[NOT IMPLEMENTED]` |
| TC08 | Cable pulls device | **cable**, phone | **no — not a COCO class** | `[BLOCKED]` on a custom class |
| TC09 | Path obstacle | backpack, chair | yes | `[NOT IMPLEMENTED]` |
| TC10 | Spill | liquid region | **no** | `[BLOCKED]` — needs its own approach |
| TC11 | Hole / pothole | outdoor surface hazard | **no** | `[BLOCKED]` — needs data |
| TC12 | Accessibility voice | obstacle + TTS | partial | `[NOT IMPLEMENTED]` |
| TC13 | Risk removed | state transition | yes | `[NOT IMPLEMENTED]` |
| TC14 | False-risk situation | normal activity | yes | `[NOT IMPLEMENTED]` |

**Key finding: 10 of 14 test cases need only COCO classes the baseline already detects.** The
blockers are TC08 (cable), TC10 (spill), TC11 (pothole) — none of which is `watch` or `comb`.

---

## 4. DATA STATE (counted from disk, 2026-09-17)

| Store | Contents |
|---|---|
| `data/eval/` | 34 images, **0 annotations**, 35 frame files — the single biggest blocker |
| `data/external/open-images-v7/watch/` | 260 samples: 220 boxes / 187 images + 40 hard negatives, all CC-BY-2.0, all human-confirmed |
| `data/external/detection/` | Stage 5 output, corrected in Stage 5.1: 635 images, 937 boxes, classes `watch` + `mug/cup`; splits 446/96/93 images (634 groups: 445/96/93), verified disjoint AND every image accounted for (0 missing, was 11 missing pre-5.1) |
| `data/objects/` | `comb`: **1 image**. `_deleted/`: comb, watch (soft-deleted, no restore path) |
| `data/raw/` | 1 capture session |
| `data/videos/` | empty — **and recorded-video-first needs this filled** |

Open Images V7 audit (5,000-image subset): `Comb`, `Pencil`, `Charger` are **absent from the
601-class vocabulary entirely**. `Cocktail shaker` was explicitly **rejected**, not merely
unmapped — it is a different object.

---

## 5. MODEL STATE

| Model | State |
|---|---|
| `yolo11n.onnx` (detector) | **active**, registry `v1`, sha `634279b4…`, COCO-80, **AGPL-3.0** |
| `yolo11n-pose.onnx` | active, sha `93e2866b…`, **AGPL-3.0** |
| `yolox_tiny.onnx` | present, unused, **Apache-2.0**, input 416, YOLOX decode |
| `crop-clf-…-78a4bdb3` | **deactivated**, `validated:false`, `active:false`, `superseded_reason` recorded |
| Custom detector | **does not exist** |

**OPEN DECISION — the developer's, not the implementer's:** AGPL-3.0 (YOLO11n) vs Apache-2.0
(YOLOX-tiny). It becomes binding the moment a derived model is trained. `models/manifest.json`
already records it as "an OPEN, UNRESOLVED decision".

---

## 6. VERIFIED FAILURES AND FIXES (do not repeat)

| Failure | Root cause | Fix | Status |
|---|---|---|---|
| Comb → `Toothbrush 51%` | comb is not a COCO class; the crop classifier cannot invent a box | architecture change (B13) | root cause understood, not yet fixed |
| `learned: watch 100%` | single-class softmax is forced to 1.0 for any input; applied to every accepted detection with no class check or confidence floor | overlay deleted; classifier off by default; registry entry deactivated | `[VERIFIED WORKING]` |
| Cross-test flake | module-level `ThreadPoolExecutor` shared across app instances | moved to `app.state` | fixed B8; **`test_object_batches_api.py` re-investigated in Stage 5.1 — see below, NOT a recurrence of this bug** |
| Phases 9–13 uncommitted | never committed after B8 | Stage 2.5 committed + pushed | `[VERIFIED WORKING]` |
| Phase 12 audit artifacts never committed | `.gitignore` `results/*` swallowed them silently | allow-listed in Stage 5, committed in Stage 5.1 | `[VERIFIED WORKING]` |
| Stage 5: 11 zero-box hard negatives in no split | per-class stratifier (`groups_by_class`) has no stratum for a zero-class image, so its group never entered `assign` | Stage 5.1: zero-box groups get their own `__zero_box__` pseudo-class stratum, split identically, exempt from `min_images_per_class`; `build_detection_splits` now asserts `sum(len(split)) == len(images)` internally | `[VERIFIED WORKING]` — guarded by `tests/unit/test_detection_splits.py::test_every_image_lands_in_exactly_one_split` / `test_zero_box_images_are_assigned_to_a_split_not_dropped` |
| `test_object_batches_api.py` intermittent failure | `_poll()`'s fixed 50×50ms=2.5s budget occasionally too tight for the background proposal thread (`_run_proposals`) under heavy concurrent CPU load | **not fixed — timeboxed per Stage 5.1 scope.** Confirmed NOT a shared-executor-state bug: `app.state.batch_proposals_executor` is created/shut down per app instance (read directly in `api/app.py`), nothing shared across tests | `[KNOWN NON-BLOCKING INTERMITTENT]` — reproduced 2/60 (~3.3%) runs of the file under artificially heavy repeated-invocation load (two distinct symptoms: a `KeyError: 'status'` from a momentarily-incomplete poll response, and a plain 2.5s poll timeout with `processed==total` but `status` still `"processing"`); did not reproduce in a normal single full-suite run in this stage. Full repro detail: `docs/decisions.md` "Stage 5.1". Revisit only if it starts appearing in normal (non-stress-tested) runs. |

---

## 7. STANDING DECISIONS (do not reverse without recording a conflict)

- Open Images V7 is the external dataset; it stays separate from own data and from `data/eval/`.
- `watch` is the first custom **architecture-proof** class — not a test-case requirement.
- The baseline detector is preserved and never overwritten.
- Custom detection must work independently of the COCO vocabulary.
- Recorded video is prioritised over live smartphone for development.
- CPU must work; CUDA is optional and never assumed.
- Detector / tracker / temporal / risk stay modular and independently replaceable.
- Storing an image is not training; training is not validation; validation is not activation.
- Classification is not detection, and must never be presented as detection.

---

## 8. IMMEDIATE ROADMAP

| # | Step | Blocked by | Owner |
|---|---|---|---|
| ~~1~~ | ~~**Stage 5.1** — commit Stage 5; fix the zero-box split defect; chase the object_batches failure~~ | — | done 2026-09-17 |
| 2 | **Label the 34 eval frames** at `/label` | — | **developer** |
| 3 | **Record TC videos** — one clip per scenario for TC01–TC07, TC09, TC13, TC14 | — | **developer** |
| 4 | **R1 literature review** — the research gap has never been defined | — | developer + Claude |
| 5 | **Temporal layer** on the baseline detector, recorded video | 3 | Claude Code |
| 6 | Context/interaction, then risk state, then warning + recommendation | 5 | Claude Code |
| 7 | Licence decision, then custom detector training | developer | **developer** |
| 8 | TC01–TC14 traceability matrix + harness | 3 | Claude Code |

**Step 3 is the critical path.** `data/videos/` is empty, and recorded-video-first (Stage 6,
the temporal layer) cannot start without clips. Stage 5.1 is done; nothing on the Claude Code
side is currently blocked on anything but this.

---

## 9. HISTORY MARKER

- **Previously:** the project tried to teach custom objects through a crop classifier attached to
  the COCO detector. This was proven structurally incapable of detecting a class the base detector
  never proposes.
- **Currently:** that experiment is isolated and preserved; the detection-format dataset path
  (Stage 5) is complete and verified — 635 images / 937 boxes across `watch` + `mug/cup`, every
  image in exactly one split (Stage 5.1 fixed the 11-image zero-box gap), committed and pushed;
  no custom detector has been trained.
- **Next:** build the temporal/context/risk layers on the baseline detector using recorded video
  (Stage 6 — blocked on the developer's TC video recordings, `data/videos/` is empty), while the
  custom detector proceeds in parallel as an architecture proof once the developer decides the
  detector licence.
