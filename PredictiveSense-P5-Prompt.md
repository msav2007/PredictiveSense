# PredictiveSense — Phase 5: Recognition Trust & Studio Repair

> Paste this entire file into Claude Code in a **fresh session**. Do not summarise it.
> **Scope: fix what is wrong now.** No Scene Snapshot Studio (that is Phase 6), no training, no tracking, no temporal reasoning, no risk, no voice.

---

## BLOCK 1 — CONTEXT

PredictiveSense is a local-first research prototype on a one-month deadline, supporting a research paper. Pipeline: camera/video → perception → recognition → tracking → temporal state → relations → risk → recommendation → alert, across a real-time mode and a recorded-video mode that share one core.

Phases 0, 1, 1.5, 1.6, 2, 2.5 and 4 are complete (`500ad08`). Working: camera abstraction, browser-owned preview with independent analysis, single-slot mailbox, recorded driver, ONNX detector + pose, recognition policy, COCO dataset store, label editor, evaluation harness, Object Learning Studio, object/model registries, dataset separation, resizable/collapsible operations panel, ~292 tests.

**Three things are wrong, and all three block the developer from collecting data:**

1. **The recognition policy is rejecting valid baseline classes.** `domain_classes` is a 14-entry whitelist (`person, cup, bottle, laptop, keyboard, mouse, chair, book, cell phone, backpack, handbag, scissors, bowl, remote`). Everything else the detector emits — `clock`, `bed`, `tv`, `dining table`, `couch`, `sink`, `potted plant`, `vase`, `wine glass` — is rejected as out-of-domain. The product requirement is that PredictiveSense **works out of the box on the general model**; the user must not have to enrol `person`, `laptop` or `keyboard` before the system is useful. A narrow whitelist breaks that.
2. **Out-of-domain is being displayed as "Unknown (was Bed)".** That is a category error. The model *did* recognise it; we chose not to show it. Presenting a suppressed known class as an unknown object is misleading, and it is confusing on screen.
3. **The thresholds were never fitted.** `per_class_thresholds` is empty and `default_threshold: 0.35`, `margin_min: 0.10` are placeholders, because the evaluation set is not labelled yet. The UI presents them as if they were fitted. They must be shown as unfitted.

Additionally, the Object Learning Studio has lifecycle bugs (Back to camera, re-selecting an object, Save & Return) and its capture is visibly blurrier than the main monitoring preview.

**Observed recognition failures** (developer observations, **not** measured accuracy): watch → *donut* or undetected; spectacles, charger not detected; headphones → *person*; mug → *phone*; bowl → *wine glass*; keyboard → *TV remote*; can → *phone*; bottles inconsistent; recognition feels slow. Person and pose generally work.

**Target machine:** Windows, Intel Core Ultra 5 125H, 16 GB LPDDR5, Intel Arc integrated graphics + NPU, **no NVIDIA GPU, no CUDA**. Python 3.11.9 in `.venv`.

---

## BLOCK 2 — OBJECTIVE

Make recognition behaviour trustworthy and honest, and make the Object Learning Studio actually usable, so the developer can begin collecting data immediately.

Specifically: redesign how the policy treats classes outside the scenario vocabulary so the general model stays useful out of the box; separate *suppressed known class* from *genuinely unknown object* in both logic and display; label unfitted parameters as unfitted; repair the Studio's state lifecycle; and bring Studio capture quality up to the main camera path.

---

## BLOCK 3 — REQUIREMENTS

### 3.1 Policy audit — evidence first, then change

1. Read `predictivesense/perception/policy.py` and `config/settings.py` in full. For **each** rule — domain restriction, per-class threshold, top-2 margin, size/aspect — write down in `docs/decisions.md`: what it is intended to reject, what it is actually rejecting, and whether its parameter is supported by labelled data (none of them currently are).
2. Create `scripts/policy_audit.py --source <clip|dir> [--frames N]`. It runs the detector over real footage and reports, **without needing labels**: total detections; per-rule rejection counts; per-raw-class counts of accepted / unknown / rejected-out-of-domain / rejected-size; the confidence distribution of what each rule rejected; and the top raw classes suppressed by domain restriction. Output `results/policy_audit_<label>.json` + a markdown table.
3. Run it over the existing footage in `data/raw` and over a synthetic-frame control. The audit is the evidence for every change made in 3.2. **Do not change a threshold without a number from this audit or a stated, documented reason.**

### 3.2 Three-tier class vocabulary — replaces the whitelist

4. Replace the binary whitelist with three tiers in config:
   - **`primary`** — classes the MVP scenarios depend on. Shown normally; eligible for scenario logic later.
   - **`secondary`** — other classes plausible in an indoor room (`clock`, `tv`, `bed`, `couch`, `dining table`, `sink`, `potted plant`, `vase`, `refrigerator`, `microwave`, `oven`, `toilet`, `teddy bear`, and similar). **Shown normally**, visually de-emphasised, and marked `secondary` in the snapshot so later phases can ignore them without the user losing them.
   - **`implausible`** — classes that cannot occur here and are almost always misfires: `donut`, `banana`, `apple`, `sandwich`, `cake`, `pizza`, `airplane`, `train`, `elephant`, `zebra`, `giraffe`, `surfboard`, `skis` and the rest of the outdoor/animal/food set. Suppressed, with the reason recorded.
5. Every class the shipped detector can emit must fall into exactly one tier. A startup validation asserts the partition is total and disjoint, and fails loudly on a typo or a missing class.
6. **This preserves out-of-the-box usefulness.** `person`, `laptop`, `keyboard`, `chair`, `bottle`, `cup`, `clock`, `tv` and the rest work immediately with no enrolment. Custom data extends the system later; it is not a precondition for it working.
7. Record the tier assignment rationale in `docs/decisions.md` as an **environment-specific judgement, not a measured result**, and note that tiers are cheap to revise once labelled data exists.

### 3.3 Suppressed ≠ unknown

8. Introduce a clean distinction in `policy_state` and in the contract:
   - **`accepted`** — shown with its class.
   - **`accepted_secondary`** — shown with its class, de-emphasised.
   - **`unknown_low_confidence` / `unknown_margin`** — the model is genuinely unsure *what* this is. Box kept, labelled `Unknown`.
   - **`suppressed_implausible`** — a confident prediction of a class that cannot be here. **Not** shown as `Unknown` on the main overlay; hidden by default with a Diagnostics-only toggle to reveal it, because showing it would put a false object on screen and calling it "Unknown" would misrepresent a known-class misfire.
   - **`rejected_size`** — box too small or aspect implausible; hidden.
9. The counts must still reconcile exactly across all states, and the existing reconciliation test must be extended, not weakened.

### 3.4 Unknown display

10. **Main overlay:** the label is exactly `Unknown`. No `(was Clock)`, no raw class, no rule name.
11. **Diagnostics:** for the most recent frame, and for a selected detection, show `decision`, `raw class`, `confidence`, `runner-up`, `rule that fired`, `rejection reason`, and the effective threshold used. **Delete no information — relocate it.**
12. Hovering or clicking an `Unknown` box on the overlay may reveal the same detail in a small inspector, but the resting label stays `Unknown`.

### 3.5 Honesty about unfitted parameters

13. While `per_class_thresholds` is empty, the Analysis → Detection panel and the Diagnostics group must state plainly that thresholds are **unfitted defaults**, not values fitted on validation data, and name the script that will fit them once the evaluation set is labelled.
14. `scripts/fit_thresholds.py` must refuse to run against an unlabelled or empty split with a message saying exactly what is missing.
15. Nothing in the UI may present an unfitted heuristic as a validated setting.

### 3.6 Object Learning Studio — lifecycle repair

16. Inspect `static/studio/studio.js`, `api/objects.py`, `api/studio.py` and the store before changing anything. Reproduce each fault, identify the root cause, and record it in the report. **Do not paper over state bugs with page reloads or forced re-fetches.**
17. Required behaviour:
   - **Back to camera** returns to the capture view with the selected object, its profile, its saved samples, the box editor and all controls intact. No reload.
   - **Selecting an existing object** restores the full editor: samples, metadata, camera, box editor, coverage guidance, edit and delete controls.
   - **Save & Return** persists everything, leaves the Studio, restores monitoring, and loses no newly captured sample. No stale state on re-entry.
   - Controls never disappear until a navigation away and back.
18. Model the Studio's states explicitly — `browsing` → `object_selected` → `capturing` → `reviewing` — with one owner of that state and no duplicated flags. Guessing at implicit state is what produced these bugs.
19. Unsaved captures must not be silently discarded: warn before leaving with pending samples.
20. Add regression tests for each fault, so these do not return.

### 3.7 Studio camera quality

21. Studio capture must reuse the **same** camera acquisition path and constraints as the main monitoring preview. Do not maintain an inferior second capture path.
22. Request the best practical resolution the selected device supports: inspect `track.getCapabilities()` where available, apply constraints, then read `track.getSettings()` and **record requested versus achieved**. Never present a requested resolution as achieved.
23. Save the captured frame at **full achieved resolution**, encoded at high quality (JPEG quality ≥ 0.92 or PNG), preserving aspect ratio. Thumbnails are generated separately and stored separately; the original is never overwritten by a downscaled copy.
24. Capture from a `VideoFrame` or an `ImageBitmap` of the live track — not from a scaled preview element — and record in the sample's provenance which path was used, plus the achieved capture resolution.
25. Report the before/after: the resolution and file size the Studio was saving previously, and what it saves now. If some of the blur is the camera itself rather than the pipeline, say so rather than claiming a fix.

### 3.8 Performance measurement

26. Measure and report **end-to-end capture-to-overlay latency** (p50/p95) with perception on — this is what the developer experiences as "recognition feels slow", and it is not the same as detector milliseconds. Also report detector ms, pose ms, analysis FPS, drop rate, frame age, CPU and RSS.
27. Apply **only** optimizations justified by these measurements and cheap to verify; if none is justified, report that and change nothing. No model swaps, no quantisation, no threading redesign in this phase.
28. Re-verify preview independence with the stall hook after all changes.

### 3.9 Panel

29. The resizable/collapsible panel shipped in Phase 4. **Verify only** — drag, keyboard, clamping, persistence, collapse and reopen, no horizontal overflow, long names ellipsised. Fix a regression if one exists; do not redesign.

---

## BLOCK 4 — ARCHITECTURE CONSTRAINTS

- Preview independence, single-slot mailbox, newest-wins, worker backpressure: unchanged.
- Detection and tracking remain separate; **no tracker exists and none may be created**.
- Real-time and recorded modes share the identical perception and policy code; both use the same object dataset and model registry.
- Recorded processing stays deterministic.
- The general model remains the baseline and must work without any enrolment.
- Contracts extend additively only; each addition recorded in `docs/decisions.md`.
- Dataset separation between `data/objects` (training) and `data/eval` (test) is inviolable.
- No new dependency, no CUDA, no cloud runtime, no runtime downloads.
- No unbounded queues, caches or listener lists.
- **Storing an image is not training.** Nothing in the UI may imply an object is recognised because it was captured.

---

## BLOCK 5 — FILES TO INSPECT FIRST

`CLAUDE.md`, `docs/architecture.md`, `docs/decisions.md`, `docs/phase-reports/phase2_5.md`, `docs/phase-reports/phase4.md`, `predictivesense/perception/policy.py`, `predictivesense/perception/classes.py`, `predictivesense/config/settings.py`, `config/profiles/dev.yaml`, `predictivesense/api/studio.py`, `predictivesense/api/objects.py`, `predictivesense/objects/*`, `static/studio/studio.js`, `static/features/overlay.js`, `static/features/detection.js`, `static/features/camera-capture.js`, `static/groups/diagnostics.js`, `static/ui/store.js`, `scripts/fit_thresholds.py`.

## BLOCK 6 — FILES TO CREATE

```
scripts/policy_audit.py
predictivesense/perception/vocabulary.py          # tier definitions, validation, tier lookup
predictivesense/api/static/studio/studio-state.js # explicit Studio state machine
tests/unit/test_vocabulary_tiers.py
tests/unit/test_policy_states.py
tests/integration/test_studio_state_machine.py
tests/integration/test_studio_capture_quality.py
docs/phase-reports/phase5.md
```

## BLOCK 7 — FILES TO MODIFY

```
predictivesense/perception/policy.py         # tiers, suppressed vs unknown, states
predictivesense/config/settings.py           # policy vocabulary tiers, unfitted flag
config/profiles/dev.yaml, eval.yaml          # three tiers replacing domain_classes
predictivesense/core/types.py                # additive: policy_state values, tier field
predictivesense/api/static/features/overlay.js      # Unknown label, secondary styling, suppressed hidden
predictivesense/api/static/features/detection.js    # unfitted-threshold notice
predictivesense/api/static/groups/diagnostics.js    # decision/raw/confidence/rule/reason, suppressed toggle
predictivesense/api/static/studio/studio.js         # use the state machine; fix lifecycle
predictivesense/api/static/studio/index.html        # if the state machine needs mount points
predictivesense/api/objects.py, api/studio.py       # only where a real lifecycle bug lives
predictivesense/objects/samples.py                  # full-resolution original + separate thumbnail
scripts/fit_thresholds.py                    # refuse unlabelled splits with a clear message
docs/architecture.md, docs/decisions.md, CLAUDE.md
```

## BLOCK 8 — MUST NOT BE CREATED OR MODIFIED

- `predictivesense/tracking/`, `scene/`, `temporal/`, `risk/`, `policy/` (alert policy), `audio/` — none exist; none may be created, not even empty.
- `predictivesense/camera/mailbox.py`, `camera/synthetic.py`, `telemetry/*`, `analysis-worker.js`, `perception/detector.py`, `perception/pose.py`.
- `predictivesense/dataset/*`, `predictivesense/eval/*`, `data/eval/*` — the evaluation dataset and harness are closed.
- `static/ui/registry.js`, `static/ui/group.js`, `static/ui/resizer.js` — verify, do not redesign.
- Any prior prompt file; any existing phase report other than adding a new one.
- **No Scene Snapshot Studio** — that is Phase 6.

If one of these genuinely blocks you, **stop and ask**.

---

## BLOCK 9 — CONTRACTS

**Config (replaces `domain_classes`)**

```yaml
policy:
  enabled: true
  vocabulary:
    primary: [person, cup, bottle, laptop, keyboard, mouse, chair, book,
              cell phone, backpack, handbag, scissors, bowl, remote]
    secondary: [clock, tv, bed, couch, dining table, sink, potted plant, vase,
                refrigerator, microwave, oven, toilet, teddy bear, ...]
    implausible: [donut, banana, apple, sandwich, cake, pizza, airplane, train,
                  elephant, zebra, giraffe, surfboard, skis, ...]
  thresholds_fitted: false        # set true only by scripts/fit_thresholds.py
  per_class_thresholds: {}
  default_threshold: 0.35         # unfitted default - see docs/decisions.md
  margin_min: 0.10                # unfitted default
  min_box_area_frac: 0.0005
  show_suppressed_in_diagnostics: true
```

**`policy_state` values:** `accepted`, `accepted_secondary`, `unknown_low_confidence`, `unknown_margin`, `suppressed_implausible`, `rejected_size`. Counts reconcile: every input detection maps to exactly one state.

**`Detection` additions (additive only):** `tier: "primary" | "secondary" | "implausible" | "unlisted"`. `raw_class_name`, `policy_state` and `runner_up` already exist.

**Studio state machine:** states `browsing | object_selected | capturing | reviewing`; transitions `select`, `back_to_camera`, `capture`, `review`, `save_and_return`, `discard`. One owner, no duplicated flags, every transition testable without a browser.

**Sample provenance additions:** `capture_path` (`videoframe` | `imagebitmap` | `element`), `requested_resolution`, `achieved_resolution`, `encoded_quality`, `original_bytes`, `thumbnail_path`.

---

## BLOCK 10 — ERROR HANDLING

**Fail loudly, exit non-zero:** a vocabulary partition that is not total and disjoint; a class name in any tier the detector cannot emit; `fit_thresholds.py` on an unlabelled split; a corrupt object manifest.

**Degrade gracefully, log once:** `getCapabilities()` unsupported (fall back to constraints, record that capabilities were unavailable); a camera that will not deliver the requested resolution (record achieved, continue); a policy exception (count the frame, pass detections through unchanged, never kill the loop).

**Never:** display a suppressed known class as `Unknown`; present an unfitted threshold as fitted; discard a captured sample without warning; hold a camera device open after leaving the Studio; save only a downscaled image as the original.

---

## BLOCK 11 — TESTS

Existing suite must stay green — that is the primary regression gate.

**Unit**

1. `test_vocabulary_tiers` — partition is total and disjoint over the detector's full class list; unknown class name in a tier fails loudly; tier lookup correct.
2. `test_policy_states` — each state produced by a constructed detection; counts reconcile exactly across all six states; `suppressed_implausible` is not `unknown`; policy disabled is a pass-through; a raising rule is caught and counted.
3. Extend the existing reconciliation test rather than replacing it.

**Integration**

4. `test_studio_state_machine` — every transition, including the three reported faults: back-to-camera preserves selection and samples; re-selecting an object restores the editor; save-and-return persists samples and restores monitoring. Assert no state is lost and no reload is required.
5. `test_studio_capture_quality` — a captured sample stores an original at the achieved resolution with a separate thumbnail; requested and achieved are both recorded; the original is not the thumbnail.
6. Regression: `test_preview_independence`, `test_studio_lifecycle`, `test_objects_api`, `test_dataset_separation`, `test_ingest_socket`, `test_recorded_driver`, `test_camera_recovery`, `test_no_outbound_network`, Phase 1.6 UI structure tests, Phase 2 and 2.5 tests.

7. A UI-text test asserting the overlay label for an unknown detection is exactly `Unknown` and that no `(was ` string appears in overlay code paths.

---

## BLOCK 12 — COMMANDS

```powershell
cd C:\Users\mummi\Documents\Projects\PredictiveSense
.\.venv\Scripts\Activate.ps1

pytest -q
pytest -q -m "not slow"
pytest -q -m models

python scripts\policy_audit.py --source data\raw --frames 400
python scripts\benchmark_latency.py            # end-to-end capture-to-overlay
python scripts\run_app.py --profile dev --source-kind browser
```

Run everything yourself, fix your own failures, re-run. The only steps the developer performs are the Block 13 checks.

---

## BLOCK 13 — PHYSICAL VERIFICATION (developer, not you)

Short. Mark as pending and list:

1. On the live overlay, confirm `person`, `laptop`, `keyboard`, `chair`, `bottle` and `clock` are recognised **without any enrolment**, and that no label reads `Unknown (was …)`.
2. Confirm a genuinely unsupported object (watch, charger, spectacles) shows as `Unknown` or nothing — not as a confident wrong class.
3. Open Diagnostics and confirm the raw class, confidence and rule are visible for a selected detection.
4. In the Studio: select an object, capture, use **Back to camera**, re-select the object, confirm nothing is lost; then **Save & Return** and confirm monitoring resumes.
5. Compare a Studio capture with the main preview and confirm the sharpness gap is gone or explain what remains.
6. Drag and collapse the panel; confirm no regression.

---

## BLOCK 14 — DEFINITION OF DONE

- [ ] `policy_audit.py` run on real footage; results written; every policy change traced to a number or a documented reason.
- [ ] Three-tier vocabulary implemented; partition total and disjoint; validated at startup.
- [ ] General model works out of the box — no enrolment needed for baseline classes.
- [ ] `suppressed_implausible` distinct from `unknown` in logic, contract and display; counts reconcile.
- [ ] Overlay shows exactly `Unknown`; full detail available in Diagnostics; nothing deleted.
- [ ] Unfitted thresholds labelled as unfitted in the UI; `fit_thresholds.py` refuses unlabelled splits.
- [ ] Studio state machine implemented; all three reported faults fixed with root causes documented and regression tests added.
- [ ] Studio captures store a full-resolution original with requested vs achieved recorded, plus a separate thumbnail; before/after reported.
- [ ] End-to-end capture-to-overlay latency measured p50/p95; any optimization justified by measurement or none applied.
- [ ] Preview independence re-verified; panel verified.
- [ ] Full suite green; `docs/phase-reports/phase5.md` in the three-part format; tree committed; no remote.

---

## BLOCK 15 — KNOWN LIMITATIONS TO STATE IN THE REPORT

No model has been trained; recognition of watch, spectacles, charger, headphones and shaker is **unchanged** — they remain outside the model's vocabulary and this phase does not add them. Tier assignments are environment-specific judgement, not measured results. All thresholds remain unfitted until the evaluation set is labelled. The policy audit reports rejection counts on unlabelled footage — these are frequencies, not accuracy. No tracking, temporal reasoning, scene relations, risk inference or voice exists. The Scene Snapshot Studio does not exist yet.

## BLOCK 16 — REPORT FORMAT

`docs/phase-reports/phase5.md`, three separated parts — **Measured** (audit tables, per-rule counts, latency before/after, capture resolution before/after, test results), **Physically observed by the developer** (pending until performed; quote the failure-mode observations as motivation, attributed), **Not verified / limitations** (Block 15).

Reply with only:

```
POLICY AUDIT FINDINGS: ...
VOCABULARY CHANGE: ...
UNKNOWN DISPLAY: ...
STUDIO FIXES (root causes): ...
CAPTURE QUALITY BEFORE/AFTER: ...
LATENCY: ...
TESTS: ...
LIMITATIONS: ...
PHASE 6 NOT STARTED: confirmed
```

---

## BLOCK 17 — EXECUTION INSTRUCTIONS

Inspect before modifying. Reuse the existing camera abstraction, shell, registry, controls, dataset store and telemetry — do not fork or rewrite them. Make the minimum change that satisfies each requirement. Implement only this phase. Run all tests and measurements yourself and fix your own routine failures without asking. Do not ask for approval on routine steps; stop only for a genuinely blocking ambiguity, and say precisely what is blocking. Add no dependencies. Document anything genuinely unresolved rather than papering over it. Never claim a measurement you did not run or a physical check you did not perform.

**Do not start Phase 6 (Scene Snapshot Studio).** When Phase 5's exit criteria are met, stop.
