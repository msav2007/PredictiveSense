# PredictiveSense — Phase 1.6: Interface Architecture

> Paste this entire file into Claude Code in a **fresh session**. Do not summarise it.
> **This is a presentation-layer phase.** Do NOT start Phase 2. Do NOT implement detection, pose, tracking, temporal, risk, or voice.

---

## BLOCK 1 — CONTEXT

PredictiveSense is a local-only research prototype supporting a research paper. Phases 0, 1 and 1.5 are complete and committed: typed contracts, strict config, telemetry, single-slot mailbox, browser-owned camera with a native `srcObject` preview, a Web Worker analysis path over `WS /ws/ingest`, a recorded-video driver, a clip recorder, camera benchmarking and recovery.

Read first, and work from what is actually present: `CLAUDE.md`, `docs/architecture.md`, `docs/decisions.md`, `docs/phase-reports/phase1.md`, then `predictivesense/api/static/index.html` (~90 lines), `app.js` (~693 lines), `app.css` (~103 lines), `analysis-worker.js` (~233 lines).

**Physically verified and not to be disturbed:** the integrated camera and the OnePlus Nord 4 Windows virtual camera both work through the browser; switching works; the integrated camera feels faster, the OnePlus has a small but usable delay.

**The problem this phase fixes.** The current page is a flat stack of `<section class="panel">` blocks — preview, metrics, browser measurement, clip metadata, recorded video — with every control visible at once and engineering metrics competing with the video. That was right for building the input layer and is wrong for the six phases that follow. This phase establishes the interface architecture **once** so later phases add capability into an existing structure instead of redesigning the page.

**Target machine:** Windows, Intel Core Ultra 5 125H, 16 GB RAM, Intel Arc integrated graphics, no CUDA. Python 3.11.9 in `.venv`. Chrome/Edge.

---

## BLOCK 2 — OBJECTIVE

Reorganise the dashboard into a video-first application shell with a collapsible right-hand control panel of collapsible groups, two first-class input modes, a Normal / Diagnostics distinction, and a reusable component layer with **one documented registration call** that every future phase uses to add its controls.

Behaviour below the view layer does not change. No camera, transport, mailbox, telemetry, API or test behaviour is altered except where this prompt explicitly says so.

---

## BLOCK 3 — REQUIREMENTS

### 3.1 Application shell

1. Three regions: **top bar**, **viewport**, **right control panel**. CSS grid; no framework; no build step; ES modules served statically as they are today.
2. **Top bar** shows: product name; active mode (`Real-time` / `Recorded video`); system status pill (`Ready` / `Running` / `Degraded` / `Error`); a **Diagnostics** toggle; a **panel collapse** toggle. It never scrolls out of view.
3. **Viewport** fills all remaining space and holds the `<video>` element, the recording indicator, and a transparent overlay container (`#overlay-layer`) that later phases draw into. It must letterbox any aspect ratio without distortion and never shrink below 45% of the window width when the panel is open.
4. **Control panel** is collapsible to a narrow rail with an obvious reopen control. Collapsed state persists in `localStorage`. Collapsing enlarges the viewport; the top bar retains mode and status so nothing essential is lost.

### 3.2 Component layer

5. Create a small module set under `predictivesense/api/static/ui/`. Names are yours; responsibilities are not:
   - **shell** — builds top bar, viewport and panel; owns collapse state.
   - **group** — renders one collapsible group: title, one-line summary, open/close control, body. Identical spacing and behaviour for every group.
   - **registry** — holds group definitions, sorts by `order`, filters by mode and by normal/diagnostics visibility, mounts and re-renders.
   - **controls** — shared primitives: setting row (label + control + optional hint), action button with a `variant` of `primary | secondary | danger`, status indicator, metric row, compact summary line.
   - **store** — a tiny observable holding UI state: `mode`, `diagnosticsVisible`, `panelCollapsed`, `openGroups`, plus the latest `StateSnapshot`. Subscribers re-render; no global mutable leakage.
6. **The extension contract.** Exactly one call registers a group:

```js
registerGroup({
  id: "camera",              // stable, unique
  title: "Camera",
  order: 20,                 // sort key within the panel
  modes: ["realtime"],       // ["realtime"] | ["recorded"] | ["realtime","recorded"]
  view: "normal",            // "normal" | "diagnostics"
  summary: (state) => "Integrated Camera · 1280×720 · 30 FPS",
  render: (el, ctx) => { /* build the body once */ },
  update: (state) => { /* optional: cheap per-snapshot refresh */ },
});
```

  And one for Analysis sub-modules, which future phases use:

```js
registerAnalysisModule({ id: "detection", title: "Detection", order: 10, summary, render, update });
```

  Document both in `docs/architecture.md` with a worked example, because every later phase will be written against them.

7. A group's `render` runs once; `update` must be cheap and must not rebuild the DOM. `summary` must be a pure function of state.

### 3.3 The eight groups

8. Register exactly these, in this order. Only the ones with content today have bodies; do not create empty placeholder groups for future phases.

| id | Title | Modes | Contains now |
|---|---|---|---|
| `input` | Input | both | The Real-time / Recorded video mode selector, nothing else |
| `camera` | Camera | realtime | Device select, resolution, frame rate, camera status; advanced capture settings behind an "Advanced" disclosure |
| `video` | Recorded video | recorded | Video file list and selection, selected file name and duration when known, replay speed, Analyse action, analysis status and output |
| `dataset` | Dataset & recording | both | Record research sample, scenario tag, notes, consent checkbox, recorded clip list |
| `diagnostics` | Diagnostics | both | Every existing metric, relocated (see 3.5) |

  `analysis`, `alerts`, `research` are **not created in this phase** — they have no content yet. Their ids, titles and order values are reserved in a documented constant so later phases register them without renegotiating the taxonomy.

9. **One logical home.** Camera selection appears only under Camera; video selection only under Recorded video; recording only under Dataset & recording. No duplicate control anywhere.

### 3.4 Input modes

10. Mode is **client-side view state**, persisted in `localStorage`. It selects which groups are shown and which viewport source is displayed. It does **not** mutate the server's `mode` config field; recorded analysis continues to run through the existing `POST /api/analyze`. State this explicitly in `docs/architecture.md` so it is not mistaken for a backend switch.
11. Selecting **Real-time** shows the live preview and the Camera group; the Recorded video group is hidden. Selecting **Recorded video** shows the selected file in the viewport with playback controls; the Camera group is hidden and live capture is released (stop the tracks — do not hold the device).
12. Switching modes must never leave a camera device open, a worker running, or a socket half-closed. Cover this with a test.

### 3.5 Normal vs Diagnostics

13. Every metric currently on the page moves into the Diagnostics group: preview FPS, capture FPS, analysis FPS, dropped analysis frames, drop rate, mailbox depth, frame age, decode ms, ingest bytes/s, reconnects, clock RTT, switch ms, stale, worker skip counters, browser measurement block and its sample-capture action, provider/backend info. **Delete nothing.**
14. The Diagnostics toggle in the top bar reveals the Diagnostics group and any `view: "diagnostics"` rows inside other groups. Default is off. The state persists.
15. Normal view keeps only: mode, system status, the current-configuration summaries, primary actions, and any warning that requires user action.

### 3.6 Text, layout and hierarchy

16. Rename user-facing strings; keep raw keys in Diagnostics: `asfast` → **Fastest**, `realtime` (replay) → **Real-time speed**; `worker_skips_t/b/p` → **Worker skips** (with the breakdown shown in Diagnostics); "backend owns the camera" → **"Live preview unavailable in backend-camera mode."**
17. Action hierarchy is visible in styling: **primary** (Start, Analyse, Record sample, Select video), **secondary** (Settings, Advanced, Diagnostics, Capture sample), **danger** (Stop, Delete, Clear).
18. No text overflows. Long device names and long file names ellipsise with a `title` tooltip carrying the full string. Fix layout, never shrink font size to fit.
19. Correct spelling, sentence case for labels, no exposed internal identifiers in Normal view.
20. Responsive from 1280×720 up. At narrower widths the panel may overlay the viewport rather than compressing it below the 45% floor.

### 3.7 Preservation — non-negotiable

21. Preserve exactly: browser-owned capture, native `srcObject` preview (never read, drawn, or replaced for display), the analysis Worker path, newest-wins and backpressure behaviour, the single-slot mailbox, camera switching and recovery, resolution and frame-rate selection, all metrics collection, the clip recorder, the recorded-video driver, every API route and payload shape, telemetry, and the existing `assertPreviewUncomposited` guard.
22. **Move logic, do not rewrite it.** The camera, worker, recording, video and metrics functions in `app.js` are relocated into modules essentially as they are. `app.js` becomes the composition root: import modules, register groups, mount the shell. Any behavioural change must be listed in the report.

---

## BLOCK 4 — CONSTRAINTS

- No UI framework, no bundler, no build step, no new runtime dependency.
- Preview independence is inviolable: a stalled analysis path must not affect the preview. Re-run `test_preview_independence` after every change.
- No unbounded queues, caches or listener lists — unsubscribe on unmount.
- No CUDA, no cloud calls, no runtime downloads, no external fonts or CDN assets. Everything ships from `static/`.
- No fake or placeholder intelligence. Nothing unimplemented may appear as a working feature.
- Do not refactor Python beyond what item 3.5 relocation genuinely requires.
- Accessibility basics: keyboard-reachable controls, visible focus, `aria-expanded` on group toggles, labels tied to inputs.

---

## BLOCK 5 — FILES

**Create**

```
predictivesense/api/static/ui/shell.js
predictivesense/api/static/ui/group.js
predictivesense/api/static/ui/registry.js
predictivesense/api/static/ui/controls.js
predictivesense/api/static/ui/store.js
predictivesense/api/static/ui/format.js          # label maps, ellipsis + tooltip helpers, number formatting
predictivesense/api/static/groups/input.js
predictivesense/api/static/groups/camera.js
predictivesense/api/static/groups/video.js
predictivesense/api/static/groups/dataset.js
predictivesense/api/static/groups/diagnostics.js
predictivesense/api/static/features/camera-capture.js   # logic moved out of app.js
predictivesense/api/static/features/analysis-client.js  # worker + ingest socket, moved
predictivesense/api/static/features/recording.js        # MediaRecorder + upload, moved
predictivesense/api/static/features/videos.js           # list + analyse, moved
predictivesense/api/static/features/metrics.js          # sampling + browser metrics, moved
tests/unit/test_ui_structure.py
tests/unit/test_ui_text.py
tests/integration/test_static_assets.py
```

**Modify**

```
predictivesense/api/static/index.html    # becomes a shell mount point, not a panel stack
predictivesense/api/static/app.js        # composition root only
predictivesense/api/static/app.css       # shell grid, group styling, button variants, overflow rules
predictivesense/api/app.py               # only if a static route must be added for ui/ and groups/
docs/architecture.md                     # UI architecture, group taxonomy, extension contract with example
docs/decisions.md                        # this phase's decisions
docs/phase-reports/phase1.md             # add a Phase 1.6 section
CLAUDE.md                                # UI rules + "where does a new control go" table
```

**Must NOT be created or modified**

- Anything under `perception/`, `tracking/`, `scene/`, `temporal/`, `risk/`, `policy/`, `audio/`, `research/` — none exist and none may be created, not even empty.
- `predictivesense/camera/*`, `predictivesense/pipeline/*`, `predictivesense/telemetry/*`, `predictivesense/core/*`, `predictivesense/config/*` — unless a change is unavoidable, in which case **stop and ask**.
- `analysis-worker.js` — its logic is correct and measured; only touch it if a moved import path requires it.
- The Phase 0 / P1 / P1.5 prompt files.

---

## BLOCK 6 — ERROR HANDLING

- A group whose `render` throws must be caught, logged, and shown as a single failed-group message; it must not blank the panel or stop other groups.
- Mode switch with no camera permission, no devices, or no videos present shows a clear empty state that says what to do next — never a blank panel.
- A missing or unreadable video file shows a specific error, not a silent failure.
- Camera loss keeps the shell alive: status goes `Degraded`, the viewport shows a clear message, recovery re-enters `Running` without a page reload.
- No `console.log` left in shipped code paths; keep a single debug logger gated by the Diagnostics toggle.

---

## BLOCK 7 — TESTS

**Existing suite must stay green — that is the primary regression gate.**

New Python tests (static analysis of the shipped assets; no browser automation, no new dependency):

1. `test_ui_structure` — `index.html` contains the shell mount points and no legacy panel stack; every registered group id in `groups/` appears exactly once; group modules each export `id`, `title`, `order`, `modes`, `summary`, `render`; reserved ids (`analysis`, `alerts`, `research`) are declared in the constants file and **not** registered; no group id is registered twice.
2. `test_ui_text` — forbidden user-facing strings are absent from Normal-view markup and group modules: `asfast`, `backend owns the camera`, `worker_skips_t`; required replacements are present; no `console.log(` in `static/**/*.js` outside the gated logger.
3. `test_static_assets` — every module referenced by `index.html` and by any `import` in `static/**/*.js` resolves to a file that exists and is served by the app; a `GET` for each returns 200 with a JavaScript content type.
4. Preservation tests — `test_preview_independence`, `test_ingest_socket`, `test_recorded_driver`, `test_recorder_upload`, `test_camera_recovery` all still pass unchanged.
5. Add an assertion that `index.html` still carries `autoplay muted playsinline` on the video element and applies no CSS filter, transform or opacity animation to it.

**Optional, only if `node` is already on PATH** (check with `where node`; skip cleanly if absent, never install it): a small unit test of `registry.js` — registration, ordering, mode filtering, duplicate-id rejection.

---

## BLOCK 8 — COMMANDS

```powershell
cd C:\Users\mummi\Documents\Projects\PredictiveSense
.\.venv\Scripts\Activate.ps1

pytest -q
pytest -q -m "not slow"
python scripts\run_app.py --profile dev --source-kind browser
```

Run everything yourself, fix your own failures, re-run. Do not ask the developer to run routine commands.

---

## BLOCK 9 — ACCEPTANCE CRITERIA

- [ ] Full existing test suite passes; new tests pass.
- [ ] Shell renders: top bar with mode, status, Diagnostics toggle, panel toggle; viewport; collapsible right panel.
- [ ] Panel collapse enlarges the viewport; state persists across reload.
- [ ] Five groups registered (`input`, `camera`, `video`, `dataset`, `diagnostics`), each with a working summary and open/close; `analysis`, `alerts`, `research` reserved but not registered.
- [ ] Mode switching shows and hides the correct groups, releases the camera when leaving Real-time, and leaves no worker or socket running.
- [ ] Every metric that existed before is present under Diagnostics; none deleted.
- [ ] Normal view free of engineering metrics; Diagnostics toggle reveals them; both states persist.
- [ ] Text replacements applied; no forbidden strings; action variants styled distinctly.
- [ ] Long device and file names ellipsise with a tooltip and do not overflow.
- [ ] Viewport never below 45% of window width with the panel open at 1280 wide.
- [ ] Camera selection, switching, resolution/FPS control, recording and recorded-video analysis all still work.
- [ ] Extension contract documented in `docs/architecture.md` with a worked "add a group" example.
- [ ] `CLAUDE.md` carries the permanent UI rules and a "where does a new control go" table.
- [ ] Committed, working tree clean, no remote. **Phase 2 not started.**

---

## BLOCK 10 — PHYSICAL VERIFICATION (developer, not you)

Mark these "pending developer verification" in the report and list them:

1. Real-time mode: integrated camera opens and previews.
2. Real-time mode: OnePlus virtual camera opens and previews.
3. Switching between the two works and the previous device is released.
4. Video is clearly the primary visual area, panel open and closed.
5. Panel collapses and reopens cleanly.
6. Recorded video mode can be selected and a file chosen.
7. No text overlaps, clips, or overflows at the working window size.
8. The interface is understandable without documentation — the developer can find camera selection, video selection, recording and diagnostics without searching.

Do not ask the developer to repeat camera tests that already passed unless a change you made genuinely invalidates one; if so, say which and why.

---

## BLOCK 11 — REPORT

Add a **Phase 1.6** section to `docs/phase-reports/phase1.md` with three separated parts: **Measured** (test results, asset checks, any before/after), **Physically observed by the developer** (existing observations, quoted and attributed; new checks marked pending), **Not verified / limitations**.

Reply with only:

```
CHANGED: ...
GROUPS REGISTERED: ...
PRESERVED (verified by test): ...
TESTS: ...
LIMITATIONS: ...
PHASE 2 NOT STARTED: confirmed
```

---

## BLOCK 12 — PROHIBITIONS

1. Do not start Phase 2 or create any perception, tracking, temporal, risk, alert or voice module, even empty.
2. Do not create placeholder controls for unimplemented features, or show anything as working that is not.
3. Do not break preview independence, the newest-wins mailbox, or the worker backpressure behaviour.
4. Do not rewrite working camera, worker, recording or recorded-video logic — move it.
5. Do not change any API route, payload shape, config schema or contract.
6. Do not add a UI framework, bundler, build step, CDN asset, external font, or any runtime dependency.
7. Do not delete any existing metric — relocate it.
8. Do not duplicate a control in two groups.
9. Do not leave `console.log` in shipped paths or an unbounded listener list.
10. Do not shrink fonts to solve overflow; fix the layout.
11. Do not claim a physical check you did not perform.
12. Do not refactor unrelated Python.
13. Do not stop for routine errors — diagnose, fix, re-run.
14. Do not mark an acceptance item complete without running the check.

When Phase 1.6 is finished, **stop**.
