# PredictiveSense — Phase 6: Studio Repair (Browser-Verified) & Recognition Responsiveness

> Paste this entire file into Claude Code in a **fresh session**. Do not summarise it.
> **Scope: make the Object Learning Studio actually work in a browser, and prove it with a browser test.** No Scene Snapshot Studio (Phase 7), no training, no tracking, no temporal reasoning, no risk, no voice.

---

## BLOCK 1 — CONTEXT

PredictiveSense is a local-first research prototype on a one-month deadline, supporting a research paper. Phases 0, 1, 1.5, 1.6, 2, 2.5, 4 and 5 are complete (`b40ac47`). 321 tests pass.

**Phase 5 claimed to fix the Object Learning Studio's lifecycle bugs and its tests pass. In the browser, the Studio is more broken than before.** That gap is the most important fact in this phase: the existing tests exercise `studio-state.js` as a module and the Python API directly, and **no test loads the page and clicks anything**. An entire class of defect — unbound handlers, render-order faults, module load errors — is currently invisible to the suite. Fixing the bugs without closing that hole guarantees they come back.

**Target machine:** Windows, Intel Core Ultra 5 125H, 16 GB LPDDR5, Intel Arc integrated graphics + NPU, **no NVIDIA GPU, no CUDA**. Python 3.11.9 in `.venv`. Chrome/Edge.

---

## BLOCK 2 — EXACT OBSERVED BUGS

Reported by the developer against the running app at `/studio`:

1. `/studio` opens. A `Watch` object exists in the Objects list. **Clicking the Watch row does nothing.**
2. The centre pane stays on *"Select or create an object to start collecting images."*
3. Collection Guidance stays on *"Select an object."*
4. The object editor, camera preview and sample controls never appear.
5. **"Save and return" does nothing.** The developer cannot leave the Studio normally.
6. Consequently no Watch images can be collected.

---

## BLOCK 3 — ROOT-CAUSE INVESTIGATION (do this first)

**Reproduce before fixing. Do not start editing until you can state the cause in one sentence.**

1. Start the app and open `/studio`. **Read the browser console first** — two independent handlers being dead at once usually means a module-load exception or a failed import, not two separate bugs.
2. Two concrete leads from a static read of the current code. Confirm or refute each; do not assume either is correct:
   - **`static/studio/studio.js` binds click listeners for `btn-return`, `btn-new-object`, `no-kind`, `no-cancel`, the create form, `btn-rename`, `btn-delete-object`, the condition selects, `consent-ack`, `camera-select`, the video, the box, `upload-input`, the sample-thumbnail `li` elements (line ~754), the inspect buttons, `btn-capture` and `btn-reset-box` — but there appears to be no listener bound to the rows of the objects list.** If the list is re-rendered by `render()`, any binding attached to a replaced node is lost. Delegate from the stable list container instead of binding per row.
   - **`studio-state.js` `can()` returns false for `save_and_return` whenever `hasPending()` is true, and `dispatch()` throws on that transition when pending.** If the state is stuck in `browsing` because object selection never fires, and `browsing` has no `save_and_return` entry in the transition table, the return button is dead for a second, independent reason. Check whether `hasPending()` can latch true (for example `dirtyInspect` never cleared) and whether a thrown transition is being swallowed.
3. Write the confirmed root cause of **each** symptom into the report. If the real cause differs from the leads above, say so plainly — the leads are hypotheses, not instructions.
4. Check for a silent failure path: a `try/catch` that swallows a transition error, a `can()` guard that returns without feedback, or a handler that throws before binding the rest.

---

## BLOCK 4 — THE STRUCTURAL FIX: BROWSER TESTS

5. Add **Playwright** as a `dev`-only dependency with a new `browser` pytest marker. Tests marked `browser` **skip cleanly with a clear message** when Playwright or its browser binary is absent, so the default suite and any machine without it are unaffected. Install the Chromium binary once via `playwright install chromium`.
6. This is the one new dependency permitted in this phase, and it is justified: two consecutive phases have shipped frontend defects that the suite reported as fixed. A smoke test that loads the page and clicks is cheaper than a third repetition.
7. The browser suite must cover, against the real app:
   - `/studio` loads with **zero console errors** (assert on collected console messages, not just on the DOM).
   - Clicking an existing object row transitions to the editor: object name, sample count, camera area, capture controls and guidance are all present.
   - Clicking the same row repeatedly is stable.
   - Clicking a different object replaces the editor state correctly.
   - **Back to camera** returns to capture state with the object still selected and the box reset.
   - **Save and return** navigates back to `/` and leaves no pending state.
   - The main page `/` loads with zero console errors and the panel resizes.
8. Camera-dependent steps use Chromium's fake media device (`--use-fake-device-for-media-stream`, `--use-fake-ui-for-media-stream`) so capture flows are testable headlessly. Anything that genuinely needs real hardware stays a developer check.

---

## BLOCK 5 — OBJECT LEARNING STUDIO — REQUIRED BEHAVIOUR

9. Purpose, unchanged: *"I know what this object is and want to collect high-quality labelled training data for it."* One object profile holds many samples. The user must never create a new `Watch` for each image.
10. **Selecting an existing object** must reliably enter the editor and show: object name, kind, sample count, confusable-with guidance, camera preview, capture controls, the sample editor when a sample is selected, positive / negative / hard-negative role, the bounding-box editor, view, distance, lighting, background, occlusion, held state, frame position, consent, save changes, discard, back to camera, sample thumbnails, coverage guidance, edit metadata, delete object. **No page reload.**
11. **Back to camera** leaves review/edit state, returns to capture, preserves the selected object and its saved samples, clears stale editing state, and **resets the box to a fresh centred default rather than reusing the inspected sample's coordinates**.
12. **Save and return** persists any staged sample, prevents silent loss, leaves the Studio, resumes monitoring if the Studio paused it, restores the main app, and preserves new data. **Confirm only when there is genuinely something unsaved** — never on a normal clean return, and never via a full browser reload as the normal path.
13. If a transition is refused, the UI must say why. A dead button with no feedback is the defect that produced this phase.

## BLOCK 6 — STATE MACHINE

14. Keep **one** owner of Studio state. Do not add a second machine beside `studio-state.js`; fix the one that exists.
15. States `browsing | object_selected | capturing | reviewing`; transitions deterministic; every transition reachable from the UI and covered by a test.
16. Required paths: object click → editor; capture → review; save → object still available; back to camera → capture; select another object → new editor; save and return → main app.
17. Guard conditions that can block a transition (`hasPending`) must be observable in Diagnostics and must not be able to latch permanently.

## BLOCK 7 — STUDIO CAMERA QUALITY

18. Phase 5 rebuilt this path (same constraints as monitoring, `getCapabilities` → `applyConstraints` → `getSettings`, `createImageBitmap` from the live track, JPEG stored verbatim, separate thumbnail, requested vs achieved recorded). **Verify it end to end in the browser now that the Studio is reachable** — assert a captured sample's stored original matches the achieved resolution and is not the thumbnail.
19. Do not rebuild it. Fix only what the verification shows is broken, and report the achieved resolution actually observed.

---

## BLOCK 8 — RECOGNITION RESPONSIVENESS (bounded)

Phase 5 measured end-to-end frame age p50 **177 ms** / p95 **231 ms**, analysis ~8.5 FPS, detector p50 ~65–72 ms, pose p50 ~52–57 ms, combined ~129 ms, and correctly applied no optimization because none was justified under its constraints. This phase permits exactly three bounded, measured experiments — **no model swap, no quantisation, no threading redesign, no architecture change.**

20. **Pose gating.** `perception.pose_requires_person` exists and is `false`, never measured. Measure `false` vs `true` on the same footage: combined per-frame cost, frame age p50/p95, analysis FPS, and any change in pose availability when a person *is* present. Adopt only if it helps and costs nothing in person frames.
21. **Detector input size, re-decided.** Phase 2 chose 640 "to keep small objects". The small objects that motivated it — watch, spectacles, charger — are outside the model's vocabulary and are not detected at any input size. Re-measure 480 vs 640 on current footage for latency **and** detection counts of the classes actually in the primary tier. Change the default only if 480 loses nothing that matters.
22. **One canonical latency protocol.** Phase 2 reports detector p50 46.8 ms; Phase 5 reports 65–72 ms; the config is identical, so these are two different measurement conditions. Define one protocol (isolated vs in-app, warm-up policy, frame source, thread settings), document it in `docs/decisions.md`, re-report both numbers under it, and note in the earlier reports which protocol produced them. **Two contradictory latencies in one project's reports is a defect in the paper, not just in the docs.**
23. Report the outcome of each experiment including "no change was justified" where that is the answer.

---

## BLOCK 9 — WHAT MUST NOT REGRESS

24. **Unknown behaviour from Phase 5 stands.** Main overlay label is exactly `Unknown`; `suppressed_implausible` is hidden on the overlay, never shown as `Unknown`; raw class, confidence, runner-up, tier, rule and threshold stay in Diagnostics. Do not hide raw information to make output look cleaner.
25. **Baseline general recognition stands.** `person`, `laptop`, `keyboard`, `chair`, `bottle`, `cup`, `clock`, `tv` and the rest of the primary and secondary tiers work with no enrolment. Never make the system depend on the user teaching it common objects.
26. **Collected ≠ trained ≠ activated.** Saving an image must never be presented as the model having learned the object. The Studio's honesty language stays.
27. Dataset separation between `data/objects` (training) and `data/eval` (test) is inviolable; the separation test stays green.
28. Preview independence, single-slot mailbox, newest-wins, worker backpressure, deterministic recorded processing, detection/tracking separation: all unchanged. **No tracker exists and none may be created.**
29. Real-time and recorded modes keep sharing one perception path, one object dataset and one model registry. No second learning system.
30. Panel resize/collapse: verify, do not redesign.

---

## BLOCK 10 — SCENE SNAPSHOT STUDIO

31. **Not implemented in this phase.** Phase 7 will build it: one high-quality scene frame → detector proposes many boxes → the user corrects, names and saves many labelled samples at once, including naming an `Unknown`. It stays a **separate** Studio from Object Learning, permanently: Object Learning is *"I know this object, collect data for it"*; Scene Snapshot is *"capture one frame, name many things quickly"*.
32. Leave clean extension points only where they cost nothing — the capture path, the box editor and the sample-writing API should be reusable without modification. Do not build UI, routes or endpoints for it now.

## BLOCK 11 — SESSION MEMORY

33. Not implemented. A named object may later carry a session-scoped instance id to assist tracking within one session, but **session memory is not permanent model knowledge** and no re-identification system is built here.

---

## BLOCK 12 — FILES

**Inspect first:** `CLAUDE.md`, `docs/architecture.md`, `docs/decisions.md`, `docs/phase-reports/phase5.md`, `static/studio/studio.js`, `static/studio/studio-state.js`, `static/studio/index.html`, `static/ui/controls.js`, `static/ui/store.js`, `predictivesense/api/studio.py`, `predictivesense/api/objects.py`, `predictivesense/objects/samples.py`, `config/profiles/dev.yaml`.

**Create**

```
tests/browser/__init__.py
tests/browser/conftest.py                  # playwright fixtures, fake media flags, console collector
tests/browser/test_studio_flow.py
tests/browser/test_main_page.py
scripts/benchmark_recognition_paths.py     # pose gating + input size experiments, one protocol
docs/phase-reports/phase6.md
```

**Modify**

```
static/studio/studio.js                    # event delegation, render/binding order, feedback on refused transitions
static/studio/studio-state.js              # only if a guard can latch or a transition is missing
static/studio/index.html                   # only if a stable container is needed for delegation
pyproject.toml                             # playwright in [dev]; browser marker
requirements.lock.txt                      # regenerate, UTF-8 no BOM
config/profiles/dev.yaml, eval.yaml        # only if an experiment justifies a default change
docs/decisions.md, docs/architecture.md, CLAUDE.md
docs/phase-reports/phase2.md, phase5.md    # add the canonical-protocol note to the latency figures
```

**Must NOT be created or modified**

- `predictivesense/tracking/`, `scene/`, `temporal/`, `risk/`, `policy/` (alert policy), `audio/` — none exist; none may be created, not even empty.
- `perception/detector.py`, `perception/pose.py`, `perception/policy.py`, `perception/vocabulary.py` — Phase 5's recognition behaviour is settled; touch only if an experiment in Block 8 requires a config-level change.
- `camera/mailbox.py`, `camera/synthetic.py`, `telemetry/*`, `analysis-worker.js`.
- `predictivesense/dataset/*`, `predictivesense/eval/*`, `data/eval/*`.
- `static/ui/registry.js`, `static/ui/group.js`, `static/ui/resizer.js`.
- Any prior prompt file; any existing phase report except the two latency notes above.

If one of these genuinely blocks you, **stop and ask**.

---

## BLOCK 13 — CONTRACTS

No new API. No contract reshaping. Permitted additive changes only, each with a `docs/decisions.md` line:

- A Diagnostics-visible Studio state readout: `{state, selected_object_id, has_pending, last_refused_transition}`.
- `perception.pose_requires_person` default may change if Block 8.20 justifies it.
- `perception.detector.input_size` default may change if Block 8.21 justifies it.

Existing endpoints (`/api/objects*`, `/api/studio/enter|leave|status`, `/api/models/registry`) keep their shapes.

## BLOCK 14 — ERROR HANDLING

**Fail loudly:** a module-load error in the Studio page must surface as a visible in-page error banner, not a silent dead UI. A refused transition must tell the user why.

**Degrade gracefully:** Playwright or its browser missing → `browser` tests skip with a clear message, never fail; no camera in the test environment → fake device; a camera that will not deliver the requested resolution → record achieved and continue.

**Never:** swallow a transition exception; leave a button bound to a handler that can silently no-op; rely on a page reload as the normal return path; hold a camera device open after leaving the Studio.

---

## BLOCK 15 — TESTS

**Existing 321 tests must stay green.** Markers as existing plus `browser`.

1. `tests/browser/test_studio_flow.py` — the full Block 4.7 list, against the real app, with console-error assertions.
2. `tests/browser/test_main_page.py` — `/` loads clean; panel drag and collapse work.
3. Extend `test_studio_state_machine.py` with a regression test per confirmed root cause from Block 3.
4. A test asserting object-list rows are handled by delegation from a stable container, so a re-render cannot orphan the binding.
5. `test_studio_capture_quality` — extend to assert the stored original's dimensions equal the achieved resolution and differ from the thumbnail.
6. Regression: preview independence, studio lifecycle, objects API, dataset separation, ingest socket, recorded driver, camera recovery, no-outbound-network, Phase 1.6 UI structure, Phase 2 / 2.5 / 5 perception and policy tests.

## BLOCK 16 — COMMANDS

```powershell
cd C:\Users\mummi\Documents\Projects\PredictiveSense
.\.venv\Scripts\Activate.ps1

pip install -e ".[dev]"
playwright install chromium
pip freeze --exclude-editable | Out-File -Encoding utf8NoBOM requirements.lock.txt

pytest -q
pytest -q -m browser
pytest -q -m "not slow"

python scripts\benchmark_recognition_paths.py --source data\raw --frames 300
python scripts\run_app.py --profile dev --source-kind browser
```

Run all of these yourself. Fix your own failures and re-run.

---

## BLOCK 17 — PHYSICAL VERIFICATION (developer, not you)

Short. Mark as pending and list:

1. Open `/studio`, click **Watch** — the editor, camera and controls appear.
2. Capture 2–3 images; use **Back to camera** between them; confirm the box resets and nothing is lost.
3. **Save and return** — lands on the main page with monitoring running.
4. Compare a Studio capture with the main preview for sharpness.
5. On the main overlay confirm baseline classes still work and no label reads `Unknown (was …)`.

## BLOCK 18 — DEFINITION OF DONE

- [ ] Each of the six reported symptoms reproduced, root-caused in one sentence, and fixed.
- [ ] Object row clicks work by delegation from a stable container; repeated and alternating selection stable.
- [ ] Back to camera preserves the object, clears editing state, resets the box.
- [ ] Save and return works with no reload, confirming only when something is genuinely unsaved.
- [ ] A refused transition always produces visible feedback.
- [ ] Playwright browser suite added, passing, and skipping cleanly where unavailable; zero console errors on `/` and `/studio`.
- [ ] Studio capture quality verified in-browser; achieved resolution reported.
- [ ] Pose-gating and input-size experiments run and reported, with "no change justified" as an acceptable outcome.
- [ ] One canonical latency protocol defined; Phase 2 and Phase 5 figures reconciled under it.
- [ ] Unknown behaviour, baseline recognition, dataset separation, preview independence, panel: all verified unregressed.
- [ ] Full suite green; `docs/phase-reports/phase6.md` in the three-part format; tree committed and clean; no remote.

## BLOCK 19 — LIMITATIONS TO STATE IN THE REPORT

No model trained; watch, spectacles, charger, headphones and shaker remain unrecognised. Thresholds remain unfitted until the evaluation set is labelled. The policy audit's evidence base is one unrepresentative clip. Browser tests use a fake media device, so real camera image quality remains a developer check. No Scene Snapshot Studio, no tracking, temporal reasoning, scene relations, risk, recommendations or voice. `StateSnapshot.tracks` is still empty.

## BLOCK 20 — REPORT FORMAT

`docs/phase-reports/phase6.md`, three separated parts — **Measured** (root causes, browser-test results, capture resolution observed, experiment tables, reconciled latency protocol, full test results), **Physically observed by the developer** (pending), **Not verified / limitations** (Block 19).

Reply with only:

```
ROOT CAUSES: ...
FIXES: ...
BROWSER TESTS: ...
CAPTURE QUALITY VERIFIED: ...
RESPONSIVENESS EXPERIMENTS: ...
LATENCY PROTOCOL: ...
TESTS: ...
LIMITATIONS: ...
PHASE 7 NOT STARTED: confirmed
```

---

## BLOCK 21 — EXECUTION INSTRUCTIONS

Inspect first. **Reproduce every symptom before changing a line** — a fix without a reproduction is a guess, and the last phase's guesses are why this one exists. Reuse existing infrastructure; do not fork or rewrite it. Make the minimum change that satisfies each requirement. Add no dependency beyond Playwright in `[dev]`. Run all tests and measurements yourself and fix your own routine failures without asking. Do not ask for approval on routine steps; stop only for a genuinely blocking ambiguity and say precisely what is blocking. Document real limitations rather than papering over them. Never claim a measurement you did not run or a physical check you did not perform.

**DO NOT START PHASE 7 (Scene Snapshot Studio).** When Phase 6's exit criteria are met, stop.
