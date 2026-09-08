# PredictiveSense — Phase 2.5: Recognition Reliability & Measurement

> Paste this entire file into Claude Code in a **fresh session**. Do not summarise it.
> **Scope: make perception trustworthy and measurable.** No tracking, no object enrollment UI, no training or fine-tuning, no temporal reasoning, no risk, no voice.

---

## BLOCK 1 — CONTEXT

PredictiveSense is a local-first research prototype on a one-month deadline, supporting a research paper. Phases 0, 1, 1.5, 1.6 and 2 are complete and committed: contracts, config, telemetry, single-slot mailbox, browser-owned camera with independent native preview, Worker analysis path, recorded-video driver, clip recorder, application shell with a group registry, and ONNX detection + pose with an overlay and a class-coverage audit.

**Existing Phase 2 measurements — use these, do not replace them with invented figures:** CPU detector p50 ≈ 46.8 ms, pose p50 ≈ 49.7 ms, combined ≈ 96.5 ms; CPU provider chosen (DirectML measured slower on this machine); smaller input sizes measured faster; analysis throughput ≈ 10 FPS with perception on versus ≈ 20 FPS off; preview independence held; mailbox bounded; recorded inference deterministic.

**The problem this phase exists to solve.** Physical testing shows the pretrained detector is not trustworthy in this environment. Observed failure modes (developer observations, **not** measured accuracy): watch → *donut*; headphones → *person*; mug → *phone*; bowl → *wine glass*; keyboard → *TV remote*; can → *phone*; bottle inconsistent; spectacles, charger, shaker not detected; small handheld objects unreliable; labels slow to update.

A confidently wrong label is worse than no label in a system that will later infer risk. But **nothing can be improved that is not first measured**, and right now the project has no labelled data, therefore no accuracy baseline, therefore no way to prove any change helped. That is what this phase fixes.

**Target machine:** Windows, Intel Core Ultra 5 125H, 16 GB LPDDR5, Intel Arc integrated graphics + NPU, **no NVIDIA GPU, no CUDA**. Python 3.11.9 in `.venv`.

---

## BLOCK 2 — OBJECTIVE

Produce the project's first **labelled evaluation set from its own footage**, an evaluation harness that reports real detection metrics, and a **recognition policy layer** that suppresses out-of-domain classes and rejects weakly-supported predictions as `unknown` — with the policy's effect proven by a before/after comparison on that set. Also evaluate one permissively-licensed alternative detector on the identical set, so the outstanding AGPL-3.0 decision is settled by evidence rather than assumption.

At the end of this phase the project can state, with numbers from its own data, how well perception works in this room and how much the policy layer improved it.

---

## BLOCK 3 — REQUIREMENTS

### 3.1 Labelled evaluation set — the foundation

1. Create `predictivesense/dataset/` with an annotation store in **COCO detection JSON format** (`images`, `annotations`, `categories`). COCO is chosen so the same files feed standard fine-tuning tooling in a later phase without conversion. Record that reason in `docs/decisions.md`.
2. `scripts/build_eval_frames.py --source data/raw --out data/eval/frames --every-n N --max-per-clip M` samples frames from the developer's recorded clips, writes them as JPEGs, and records provenance per frame: source clip, timestamp, session id, camera device, and the clip manifest's condition tags.
3. **A minimal labelling tool**, served by the existing app under a new `Research` group entry (`GET /label`): shows one frame at a time on a canvas with the ability to **draw, move, resize, relabel and delete** boxes, choose a class from the domain vocabulary, mark a frame `done`, and navigate next/previous. Keyboard shortcuts for next/previous and for the most common classes. Saves to the COCO store through `POST /api/labels`.
4. **Detector-assisted seeding is optional and must be recorded.** A frame may be pre-seeded with current detector output to speed labelling, but each image record must carry `seeded: true|false`. Seeded labelling biases recall upward — a missed object the detector never proposed is easy to overlook — so:
   - the tool must make adding a missed box as easy as accepting a proposed one, and
   - **at least 30% of the evaluation frames must be labelled with seeding off**, and the harness must report metrics for the unseeded subset separately as a bias check.
5. **Splits are session-disjoint, never frame-disjoint.** Adjacent frames are near-duplicates; splitting by frame is leakage. `scripts/build_splits.py` assigns whole recording sessions to `val` and `test`, writes `data/eval/splits.json` with a content hash, and a test asserts no session and no image id appears in both.
6. **Thresholds and policy parameters are fitted on `val` only.** `test` is opened once, at the end, and every opening is recorded in `results/test_set_openings.md` with the date and reason.
7. Target scale for this phase: **roughly 200–400 labelled frames** spanning at least 4 recording sessions and both cameras. Fewer is acceptable if honestly reported; the harness must print the sample count beside every metric so no number is read as more solid than it is.

### 3.2 Evaluation harness

8. `scripts/eval_detection.py --split val|test --model <name> --policy on|off` computes, at IoU 0.5 with greedy score-ordered matching: per-class precision, recall, F1, support; overall mAP@0.5 if it can be computed correctly from this data, otherwise omit it rather than approximate it; a **confusion matrix including `background` and `unknown` rows and columns**; and a list of the most frequent confusions with example image ids.
9. Report a **false-class rate**: predictions that overlap a ground-truth box (IoU ≥ 0.5) but carry the wrong class. This is the metric that directly represents the watch → donut failure, and it is the headline number of this phase.
10. Output `results/eval_<model>_<policy>_<split>.json` plus a readable markdown table. Every run records: model name and hash, policy config, split hash, git commit, machine, seed.
11. The harness must refuse to run on an empty or partially-labelled split and say what is missing.

### 3.3 Recognition policy layer

12. Create `predictivesense/perception/policy.py`. It sits **after** the detector and **before** the snapshot. The detector itself is not modified.
13. Rules, each independently switchable and each counted:
   - **Domain restriction.** A configured `domain_classes` whitelist of classes plausible in this environment. Anything outside it is rejected. This alone should eliminate *donut*, *wine glass*, *TV remote* and similar; whether it does is an empirical question the harness answers.
   - **Per-class confidence thresholds**, fitted on `val` from the precision/recall curve. A single global threshold is not acceptable.
   - **Top-2 margin rule.** If the best and second-best class scores are within `margin_min`, the detection is not confident about *which* class it is — emit `unknown` rather than the top-1 guess.
   - **Minimum box area** as a fraction of the frame, and an aspect-ratio sanity bound, both per class where justified.
14. **`unknown` is a first-class outcome, not a discard.** A rejected-but-real detection keeps its box and renders as `unknown`, visually distinct from both a confident detection and from nothing at all. Suppressing the box entirely would hide a real object from later phases.
15. Extend the `Detection` contract **additively** (permitted by the Phase 0 rule, requires a `docs/decisions.md` line): `raw_class_name: str` (what the model said), `class_name` (what the policy decided, possibly `"unknown"`), `policy_state: str` (`accepted` | `unknown_low_confidence` | `unknown_margin` | `rejected_out_of_domain` | `rejected_size`), and `runner_up: tuple[str, float] | None`. Do not rename or reshape existing fields.
16. **Do not invent a threshold and call it principled.** Every parameter's default must come from a curve computed on `val`, and the file that justified it must be named in the config comment. Where a parameter is a judgement call, say so in `docs/decisions.md` and report its sensitivity.
17. The policy runs identically in real-time and recorded modes.

### 3.4 Alternative-model comparison — settles the licence question

18. Evaluate exactly **one** permissively-licensed alternative detector (Apache-2.0 or MIT; YOLOX-tiny/small ONNX is the obvious candidate) through the existing model-agnostic wrapper, adding a decode variant if needed. Same input size, same policy, same splits, no tuning of either model.
19. Produce `results/model_comparison.md`: per-class F1, false-class rate, latency p50/p95, model size, and licence for each model.
20. Record the outcome in `docs/decisions.md` as a licence decision with evidence: continue with the AGPL model and accept its release implications, or switch. **Do not decide silently, and do not claim the licence question is resolved if the alternative was not actually evaluated.**

### 3.5 Bounded latency work

21. Only these, each measured before and after: adopt the measured-best detector input size from the Phase 2 sweep; **gate pose** so it runs only when at least one `person` detection is present, or every `pose_every_n` frames, whichever the measurement favours; ensure ORT session options (intra-op thread count) are set explicitly and swept over a small range on this 14-core machine.
22. Report the effect on end-to-end label latency: capture timestamp to overlay update, p50 and p95. That number, not detector milliseconds alone, is what the developer experiences as "labels are slow".
23. No other optimization in this phase. No model surgery, no quantisation, no threading redesign.

### 3.6 UI

24. `unknown` detections render distinctly — dashed outline, muted colour, label `Unknown`. Never styled like a confident detection, never invisible.
25. Under **Analysis → Detection**, add policy controls: policy on/off, domain restriction on/off, margin rule on/off, and a read-only summary of the active thresholds. Under **Diagnostics**, add per-rule rejection counters and the raw-vs-decided label for the most recent frame.
26. Create the **Research** group (reserved in Phase 1.6, first content now): links to the labelling tool, the labelling progress summary (frames labelled, per class, per session, seeded vs unseeded), and the latest evaluation summary read from `results/`.
27. Do not modify the shell, the registry, or the group taxonomy. Register into them.

---

## BLOCK 4 — ARCHITECTURE CONSTRAINTS

- Preview independence is inviolable; re-verify with the stall test.
- Single-slot mailbox, newest-wins, worker backpressure: unchanged.
- **Detection and tracking stay separate.** This phase adds no identity, no association, no cross-frame history, and no temporal smoothing of labels. Label stabilisation belongs to the tracking phase.
- Real-time and recorded modes share the identical perception and policy code.
- No CUDA, no runtime network access, no cloud inference, no new heavyweight dependency.
- No unbounded queues, caches, or accumulating per-frame lists.
- Model weights and dataset images never enter git; manifests and hashes do.
- Contracts extend additively only.

---

## BLOCK 5 — FILES TO INSPECT FIRST

`CLAUDE.md`, `docs/architecture.md`, `docs/decisions.md`, `docs/attribution.md`, `docs/phase-reports/phase2.md`, `results/class_coverage.md`, `predictivesense/core/types.py`, `predictivesense/perception/*`, `predictivesense/pipeline/loop.py`, `predictivesense/pipeline/recorded.py`, `predictivesense/api/static/ui/registry.js`, `predictivesense/api/static/groups/*`, `config/profiles/dev.yaml`.

## BLOCK 6 — FILES TO CREATE

```
predictivesense/dataset/__init__.py
predictivesense/dataset/coco_store.py        # read/write COCO JSON, id allocation, validation
predictivesense/dataset/splits.py            # session-disjoint split builder + hashing
predictivesense/dataset/quality.py           # per-frame provenance, seeded flag, counts
predictivesense/perception/policy.py         # the recognition policy layer
predictivesense/eval/__init__.py
predictivesense/eval/matching.py             # IoU matching, greedy assignment
predictivesense/eval/metrics.py              # P/R/F1, confusion incl. background & unknown, false-class rate
predictivesense/eval/report.py               # json + markdown writers
predictivesense/api/labels.py                # GET/POST label endpoints, frame listing
scripts/build_eval_frames.py
scripts/build_splits.py
scripts/eval_detection.py
scripts/fit_thresholds.py                    # per-class thresholds + margin from val curves
predictivesense/api/static/label/index.html  # labelling tool page
predictivesense/api/static/label/label.js
predictivesense/api/static/groups/research.js
predictivesense/api/static/features/policy.js
tests/unit/test_coco_store.py
tests/unit/test_splits_no_leakage.py
tests/unit/test_matching.py
tests/unit/test_metrics.py
tests/unit/test_policy_rules.py
tests/integration/test_label_api.py
tests/integration/test_eval_harness.py
docs/phase-reports/phase2_5.md
```

## BLOCK 7 — FILES TO MODIFY

```
predictivesense/core/types.py                # additive Detection fields only
predictivesense/config/settings.py           # policy.*, dataset.*, eval.* sections
config/profiles/dev.yaml, eval.yaml          # defaults with comments naming the justifying result file
predictivesense/perception/runtime.py        # explicit ORT thread options
predictivesense/pipeline/loop.py             # apply policy; pose gating
predictivesense/pipeline/recorded.py         # same
predictivesense/api/app.py                   # mount labels router + /label page
predictivesense/api/static/features/overlay.js       # unknown rendering
predictivesense/api/static/features/detection.js     # policy controls
predictivesense/api/static/groups/diagnostics.js     # rejection counters, raw vs decided
predictivesense/api/static/app.js            # register Research group
tests/unit/test_no_forbidden_imports.py      # keep bans; allow new modules where legitimate
docs/architecture.md, docs/decisions.md, docs/attribution.md, CLAUDE.md
```

## BLOCK 8 — MUST NOT BE CREATED OR MODIFIED

- `predictivesense/tracking/`, `scene/`, `temporal/`, `risk/`, `policy/` (the alert policy — distinct from `perception/policy.py`), `audio/` — none exist; none may be created, not even empty.
- `predictivesense/camera/mailbox.py`, `camera/synthetic.py`, `telemetry/*`, `analysis-worker.js`.
- `static/ui/*` — register into the registry, do not edit it.
- Any prior prompt file, or any existing phase report other than adding to it.
- Do not modify `perception/detector.py`'s inference behaviour; the policy is a separate layer.

If one of these genuinely blocks you, **stop and ask**.

---

## BLOCK 9 — CONTRACTS

**Config**

```yaml
policy:
  enabled: true
  domain_restriction: true
  domain_classes: [person, cup, bottle, laptop, keyboard, mouse, chair, book,
                   cell phone, backpack, handbag, scissors, bowl, remote]
  per_class_thresholds: {}        # fitted on val by scripts/fit_thresholds.py
  default_threshold: 0.35         # fallback only; justify in decisions.md
  margin_min: 0.10                # top1 - top2 below this -> unknown
  min_box_area_frac: 0.0005
  emit_unknown: true              # unknown keeps its box
dataset:
  root: data/eval
  coco_path: data/eval/annotations.json
  splits_path: data/eval/splits.json
  min_unseeded_fraction: 0.30
eval:
  iou_threshold: 0.5
  results_dir: results
```

**`Detection` additions (additive only):** `raw_class_name: str`, `policy_state: str`, `runner_up: tuple[str, float] | None`.

**Label API**

| Method | Path | Behaviour |
|---|---|---|
| GET | `/api/labels/frames` | Paged frame list with `labelled`, `seeded`, session id |
| GET | `/api/labels/frame/{id}` | Frame metadata + existing boxes |
| POST | `/api/labels/frame/{id}` | Replace that frame's boxes; sets `labelled`, records `seeded` |
| GET | `/api/labels/progress` | Counts by class, session, seeded/unseeded |
| GET | `/label` | The labelling page |

**Policy output invariant:** every input detection produces exactly one output detection — accepted, or `unknown`, or rejected — and the counts must reconcile. A test asserts `accepted + unknown + rejected == input count`.

---

## BLOCK 10 — ERROR HANDLING

**Fail loudly, exit non-zero:** a corrupt or schema-invalid COCO file; a split file whose hash does not match its content; an evaluation run against an unlabelled or partially-labelled split; a `domain_classes` entry that is not a class the model can emit.

**Degrade gracefully, log once:** a single unreadable frame during labelling (skip, report); a missing results file when the Research group renders (show "not yet run"); policy disabled by config (behave exactly as Phase 2 did).

**Never:** silently drop a detection without counting it; write labels without provenance; overwrite an existing annotation file without a backup; let a policy exception kill the analysis loop.

---

## BLOCK 11 — PERFORMANCE REQUIREMENTS

Measure and report: policy layer cost per frame (must be negligible — assert it is under 1 ms p95); detector and pose latency before and after the input-size and thread-count changes; pose gating effect on combined per-frame cost and on analysis FPS; **end-to-end label latency, capture to overlay, p50 and p95, before and after**; drop rate and frame age under load; peak RSS.

Re-run the preview-independence stall test with policy and perception active. Do not claim any figure not produced by a run on this machine.

---

## BLOCK 12 — TESTS

Markers as existing, plus `models` for weight-dependent tests and a new `dataset` marker for tests needing labelled data — both skip cleanly with a clear message when their inputs are absent.

**Unit**

1. `test_coco_store` — round-trip; schema validation rejects malformed records; id allocation never collides; `seeded` preserved.
2. `test_splits_no_leakage` — no session and no image id in two splits; hash changes when content changes; a deliberately leaky split is rejected.
3. `test_matching` — greedy IoU matching on hand-built boxes: correct assignment, duplicates counted as false positives, unmatched ground truth counted as misses.
4. `test_metrics` — precision/recall/F1 on a hand-computed example; confusion matrix totals reconcile; false-class rate correct on a constructed case; empty input safe.
5. `test_policy_rules` — each rule in isolation on synthetic detections: out-of-domain rejected; below-threshold → `unknown_low_confidence`; small top-2 margin → `unknown_margin`; tiny box rejected; **counts reconcile exactly**; policy disabled is a pass-through.

**Integration**

6. `test_label_api` — post boxes, read them back, progress counts update, provenance preserved.
7. `test_eval_harness` (`dataset` marker) — on a tiny fixture annotation set, produces the expected metric values and writes both output files.
8. Perception with policy on vs off over the same fixture clip: deterministic in both cases.

**Regression — must stay green:** `test_preview_independence`, `test_ingest_socket`, `test_recorded_driver`, `test_recorder_upload`, `test_camera_recovery`, `test_no_outbound_network`, all Phase 1.6 UI structure tests, all Phase 2 perception tests.

---

## BLOCK 13 — COMMANDS

```powershell
cd C:\Users\mummi\Documents\Projects\PredictiveSense
.\.venv\Scripts\Activate.ps1

pip install -e ".[dev]"
pip freeze --exclude-editable | Out-File -Encoding utf8NoBOM requirements.lock.txt

pytest -q

python scripts\build_eval_frames.py --source data\raw --out data\eval\frames --every-n 15 --max-per-clip 40
python scripts\run_app.py --profile dev --source-kind browser     # developer labels at /label

python scripts\build_splits.py
python scripts\fit_thresholds.py --split val
python scripts\eval_detection.py --split val --model yolo11n --policy off
python scripts\eval_detection.py --split val --model yolo11n --policy on
python scripts\eval_detection.py --split val --model <alternative> --policy on
python scripts\eval_detection.py --split test --model <chosen> --policy on   # once, at the end

pytest -q -m dataset
```

Run everything yourself and fix your own failures. The only step the developer performs is the labelling itself.

---

## BLOCK 14 — PHYSICAL VERIFICATION (developer, not you)

Keep it short. Mark as pending and list:

1. Label the sampled frames at `/label` — including at least 30% with seeding off.
2. Confirm on the live overlay that `unknown` is visually obvious and distinct from a confident detection.
3. Confirm the previously observed failures (watch → donut, bowl → wine glass, keyboard → remote) now render as `unknown` or disappear rather than showing a confident wrong label.
4. Confirm labels update faster than before, and that the preview stays smooth.

---

## BLOCK 15 — REPORT

`docs/phase-reports/phase2_5.md`, three separated parts:

**Measured** — labelled set size, sessions, per-class support, seeded/unseeded split; baseline metrics (policy off) and post-policy metrics on `val`, per class, with the false-class rate as the headline; the unseeded-subset bias check; the model comparison table with licences; latency before/after including end-to-end label latency; the single `test` run at the end; test suite results.

**Physically observed by the developer** — pending until performed. Quote the Phase 2 failure-mode observations as the motivation, attributed, and clearly marked as observations rather than measurements.

**Not verified / limitations** — must state: the evaluation set is small and from one environment, so metrics do not generalise; detector-assisted labelling may bias recall upward and the unseeded subset is the check on that; `unknown` handling reduces confident errors but does not add any new class — objects absent from the model's vocabulary (watch, spectacles, charger, headphones, shaker) remain unrecognised and will require a custom-trained model in a later phase; no tracking, temporal reasoning, risk inference or voice exists.

Reply with only:

```
IMPLEMENTED: ...
LABELLED SET: ...
BASELINE vs POLICY (val): ...
MODEL COMPARISON / LICENCE: ...
LATENCY BEFORE/AFTER: ...
TESTS: ...
LIMITATIONS: ...
NEXT PHASE NOT STARTED: confirmed
```

---

## BLOCK 16 — DEFINITION OF DONE

- [ ] ≥200 frames labelled across ≥4 sessions and both cameras, ≥30% unseeded, stored as valid COCO with provenance.
- [ ] Session-disjoint `val`/`test` splits with hashes; leakage test passes.
- [ ] Evaluation harness produces per-class P/R/F1, confusion including `background` and `unknown`, and false-class rate, with sample counts beside every metric.
- [ ] Baseline (policy off) and policy-on results on `val`, reported side by side.
- [ ] Per-class thresholds and margin fitted on `val` only; `test` opened exactly once and the opening logged.
- [ ] Policy layer implemented with all four rules independently switchable and counts reconciling exactly.
- [ ] `unknown` renders distinctly and keeps its box.
- [ ] One permissively-licensed alternative model evaluated on the identical set; `results/model_comparison.md` written; licence decision recorded with evidence in `docs/decisions.md`.
- [ ] Latency work measured before and after, including end-to-end label latency.
- [ ] Preview independence re-verified with policy and perception active.
- [ ] Research group registered with labelling links and progress; Analysis → Detection carries policy controls; Diagnostics carries rejection counters.
- [ ] Full suite green; `dataset` and `models` tests skip cleanly when inputs are absent.
- [ ] `docs/phase-reports/phase2_5.md` complete; tree committed; no remote.

---

## BLOCK 17 — PROHIBITIONS

1. Do not implement tracking, identity, association, cross-frame label smoothing, temporal state, risk, alert policy, or voice.
2. Do not train, fine-tune or modify any model. Do not build the object-enrollment UI — that is the next phase.
3. Do not report a confidence value as accuracy, or a detection frequency as precision or recall.
4. Do not fit any threshold on `test`, or open `test` more than once, or omit the opening log.
5. Do not split by frame; splits are session-disjoint.
6. Do not claim the licence question is settled unless the alternative model was actually evaluated on the same data.
7. Do not invent a threshold and present it as principled — every default cites the curve that produced it.
8. Do not suppress a rejected detection's box; `unknown` keeps it.
9. Do not modify the detector's inference behaviour; the policy is a separate layer.
10. Do not modify the Phase 1.6 shell, registry or group taxonomy — register into them.
11. Do not break preview independence, the mailbox, or worker backpressure.
12. Do not download anything at runtime; commit no weights and no dataset images.
13. Do not claim a measurement you did not run on this machine, or a physical check you did not perform.
14. Do not stop for routine errors — diagnose, fix, re-run. Do not ask for approval on routine steps.
15. Do not mark a done item complete without running the check.

When Phase 2.5 is finished, **stop**. Do not begin the next phase.
