/* Object Learning Studio (Phase 4, lifecycle + capture-quality repair Phase 5).
 *
 * A standalone screen (not the dashboard shell). On entry it calls
 * POST /api/studio/enter, which pauses the monitoring pipeline; on exit it calls
 * POST /api/studio/leave, which restores it. The camera preview here is
 * getUserMedia -> <video>.srcObject only, with no analysis worker attached.
 *
 * Phase 5:
 *  - an explicit state machine (studio-state.js) owns browsing / object_selected
 *    / capturing / reviewing. The DOM is a pure function of it; no reloads.
 *  - camera acquisition reuses the SAME constraints as the main monitoring
 *    preview and asks the device for its best practical resolution
 *    (getCapabilities -> applyConstraints -> getSettings, requested vs achieved
 *    recorded).
 *  - capture is taken from an ImageBitmap of the live track at full achieved
 *    resolution and stored as the full-resolution original (JPEG q>=0.92); the
 *    thumbnail is separate.
 *  - Save & Return warns before discarding any pending (staged / unsaved) sample.
 *
 * Every saved sample carries exactly one bounding box, condition tags, a role,
 * and capture provenance. Storing images is not training.
 */
"use strict";

import { el } from "/static/ui/controls.js";
import { createStudioState } from "/static/studio/studio-state.js";
import { createBoxEditor } from "/static/studio/box-editor.js";
import { initBatch } from "/static/studio/batch.js";

const $ = (id) => document.getElementById(id);
const COND_KEY = "ps.studio.conditions";
const CONSENT_KEY = "ps.studio.consent";

// Same request as static/features/camera-capture.js (the main monitoring path).
const PREVIEW_REQUEST = { width: 1280, height: 720, frameRate: 30 };
// Ceiling for the "best practical resolution" probe (BLOCK 3.22).
const MAX_CAPTURE_DIM = 1920;

const sm = createStudioState();

// The one bounding-box editor, shared with the bulk-upload reviewer (batch.js).
// `state.boxImg` is a live alias of the editor's box for the rest of this file.
let editor = null;
let batch = null;

const state = {
  token: null,
  vocab: null,
  // mirrors of sm.context, written only by render()
  objectId: null,
  stageKind: "camera",
  editingSampleId: null,
  stream: null,
  track: null,
  requestedResolution: "",
  achievedResolution: "",
  capabilitiesAvailable: true,
  boxImg: { x: 0, y: 0, w: 0, h: 0 },
  uploadQueue: [],
  uploadUrl: null,
  left: false,
};

// ---------- fatal-error surface (BLOCK 14) ----------

/** A module-load fault or an unhandled rejection must show as a visible in-page
 *  banner, never a silent dead UI. Idempotent; creates the element if the page
 *  markup is missing it. */
function showFatal(message) {
  let bar = $("studio-error");
  if (!bar) {
    bar = el("div", { id: "studio-error", class: "studio-error-banner", role: "alert" });
    document.body.prepend(bar);
  }
  bar.textContent =
    `Studio error: ${message}. Reload the page; if it persists, open the browser console.`;
  bar.hidden = false;
}

window.addEventListener("error", (ev) => {
  showFatal(ev.message || String((ev.error && ev.error.message) || ev.error || "script error"));
});
window.addEventListener("unhandledrejection", (ev) => {
  const r = ev.reason;
  showFatal((r && (r.message || String(r))) || "unhandled promise rejection");
});

// ---------- lifecycle ----------

async function enter() {
  try {
    const r = await fetch("/api/studio/enter", { method: "POST" });
    const body = await r.json();
    state.token = body.token || null;
  } catch (err) {
    note(`could not stop monitoring: ${err}`);
  }
}

async function leave() {
  if (state.left) return;
  state.left = true;
  stopCamera();
  try {
    await fetch("/api/studio/leave", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ token: state.token }),
    });
  } catch {
    /* best effort; pagehide beacon below is the backstop */
  }
}

async function saveAndReturn() {
  // BLOCK 3.19 / 5.12: never silently discard a pending capture - confirm only
  // when there is genuinely something unsaved.
  if (sm.hasPending()) {
    const n = sm.context.pendingUploads;
    const what = n > 0 ? `${n} staged image${n === 1 ? "" : "s"}` : "unsaved edits to a sample";
    if (!window.confirm(`You have ${what} that have not been saved. Leave and discard them?`)) {
      note("Still in the Studio — save or discard the staged capture, then Save and return.");
      return;
    }
    sm.setPending({ uploads: 0, dirtyInspect: false });
  }
  // State-machine bookkeeping must never block the actual navigation. Phase 5
  // threw here from `browsing` (no `save_and_return` entry) and the async click
  // handler swallowed it, so the button went dead. `save_and_return` is now
  // valid from every state; the guard is belt-and-braces.
  try {
    if (sm.can("save_and_return")) sm.dispatch("save_and_return");
  } catch {
    /* never let a transition quirk trap the user in the Studio */
  }
  await leave();
  window.location.href = "/";
}

function wireExit() {
  $("btn-return").addEventListener("click", async (ev) => {
    ev.preventDefault();
    await saveAndReturn();
  });
  window.addEventListener("pagehide", () => {
    if (state.left) return;
    try {
      const blob = new Blob([JSON.stringify({ token: state.token })], { type: "application/json" });
      navigator.sendBeacon("/api/studio/leave", blob);
    } catch {
      /* ignore */
    }
  });
  window.addEventListener("beforeunload", (ev) => {
    if (!state.left && sm.hasPending()) {
      ev.preventDefault();
      ev.returnValue = "";
    }
  });
}

// ---------- render: DOM as a pure function of the state machine ----------

function render(snap) {
  const { state: st, context: ctx } = snap;
  state.objectId = ctx.objectId;
  state.editingSampleId = ctx.editingSampleId;
  state.stageKind = ctx.stage;

  $("empty-state").hidden = st !== "browsing";
  $("capture-pane").hidden = st === "browsing";

  $("preview").hidden = ctx.stage !== "camera";
  $("upload-preview").hidden = ctx.stage === "camera";
  $("btn-capture").textContent =
    ctx.stage === "camera" ? "Capture (c)" : ctx.stage === "upload" ? "Save upload (c)" : "Save changes (c)";
  $("btn-upload-label").hidden = ctx.stage === "inspect";

  const inspectBar = $("inspect-actions");
  if (inspectBar) inspectBar.hidden = st !== "reviewing";

  const pendingNote = $("pending-note");
  if (pendingNote) {
    pendingNote.hidden = !sm.hasPending();
    pendingNote.textContent = sm.hasPending()
      ? "Unsaved capture — save it or it will be discarded on leaving."
      : "";
  }

  // keep the active object highlighted in the sidebar without a network call
  for (const li of document.querySelectorAll(".object-item")) {
    li.classList.toggle("active", li.dataset.objectId === ctx.objectId);
  }

  // Returning to the live camera from an upload/inspect stage must start from a
  // fresh centred box, never the inspected sample's pixel coordinates
  // (BLOCK 5.11 / 3 root cause). This covers back_to_camera, discard, Escape and
  // the upload-queue-drained path uniformly.
  if (ctx.stage === "camera" && state.prevStage && state.prevStage !== "camera") {
    centreBox();
  }
  state.prevStage = ctx.stage;

  renderDiag(snap);
  layoutBox();
}

/** Diagnostics-visible Studio state readout (BLOCK 13): the one place the state
 *  machine's live state, selected object, pending guard and last refused
 *  transition are shown. `has_pending` cannot latch - a successful transition
 *  clears it and `lastRefused`. */
function renderDiag(snap) {
  const d = $("studio-diag");
  if (!d) return;
  d.textContent = [
    `state=${snap.state}`,
    `selected_object_id=${snap.context.objectId || "—"}`,
    `has_pending=${sm.hasPending() ? "yes" : "no"}`,
    `last_refused_transition=${snap.lastRefused || "—"}`,
  ].join("   ·   ");
}

// ---------- objects ----------

async function loadObjects() {
  const r = await fetch("/api/objects");
  const body = await r.json();
  state.objects = body.objects || [];
  const list = $("object-list");
  // Rows carry no per-node listener: `loadObjects()` calls `replaceChildren`
  // on every refresh, so a per-row binding would be orphaned on the next
  // render. Selection is handled by ONE delegated listener on this stable
  // container, wired once in `main()` (BLOCK 4.7 / 15.4).
  list.replaceChildren(
    ...state.objects.map((o) => {
      const li = el("li", {
        class: "object-item" + (o.object_id === state.objectId ? " active" : ""),
        dataset: { objectId: o.object_id },
      });
      li.append(
        el("span", { class: "oi-name ellipsis", text: o.name, title: `${o.name} (${o.object_id})` }),
        el("span", { class: "oi-count", text: `${o.sample_count}` }),
      );
      if (!o.coverage.meets_all && o.sample_count > 0) li.append(el("span", { class: "oi-flag", text: "!" }));
      return li;
    }),
  );
  if (!state.objects.length) list.append(el("li", { class: "muted", text: "No objects yet." }));
}

/** One delegated click listener on the stable #object-list container. A row is
 *  re-rendered on every `loadObjects()`; delegation from the parent means a
 *  re-render can never orphan the binding (BLOCK 3 root cause #1, 4.7, 15.4). */
function wireObjectList() {
  $("object-list").addEventListener("click", (ev) => {
    const row = ev.target.closest(".object-item");
    if (!row || !row.dataset.objectId) return;
    selectObject(row.dataset.objectId);
  });
}

function wireNewObject() {
  const form = $("new-object-form");
  $("btn-new-object").addEventListener("click", () => {
    form.hidden = !form.hidden;
    if (!form.hidden) $("no-name").focus();
  });
  $("no-kind").addEventListener("change", () => {
    $("no-parent").hidden = $("no-kind").value !== "instance";
  });
  $("no-cancel").addEventListener("click", () => {
    form.hidden = true;
  });
  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const payload = {
      name: $("no-name").value.trim(),
      kind: $("no-kind").value,
      parent_class: $("no-parent").value.trim() || null,
      category: $("no-category").value.trim() || null,
    };
    if (!payload.name) return;
    const r = await fetch("/api/objects", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!r.ok) {
      note(`create failed: ${await r.text()}`);
      return;
    }
    const prof = await r.json();
    form.hidden = true;
    $("no-name").value = "";
    $("no-category").value = "";
    $("no-parent").value = "";
    await loadObjects();
    selectObject(prof.object_id);
  });
}

async function selectObject(objectId) {
  // BLOCK 3.17: selecting an object restores the FULL editor - samples,
  // metadata, camera, box editor, coverage - with no reload and no stale
  // inspect controls.
  clearUploadQueue({ silent: true });
  batch?.onObjectChange();
  sm.dispatch("select", { objectId });
  await loadObjects();
  await refreshObject();
  await ensureCamera();
  centreBox();
}

async function refreshObject() {
  if (!state.objectId) return;
  const r = await fetch(`/api/objects/${state.objectId}`);
  if (!r.ok) {
    note(`object load failed: ${r.status}`);
    return;
  }
  const obj = await r.json();
  $("obj-title").textContent = obj.name;
  $("obj-kind").textContent = obj.kind + (obj.parent_class ? ` / ${obj.parent_class}` : "");
  $("obj-count").textContent = String(obj.sample_count);
  renderConfusablePrompt(obj);
  renderSamples(obj.samples || []);
  renderCoverage(obj.coverage);
}

function renderConfusablePrompt(obj) {
  const box = $("confusable-prompt");
  const hn = (obj.samples || []).filter((s) => s.role === "hard_negative").length;
  if (obj.confusable_with && obj.confusable_with.length && hn === 0) {
    box.hidden = false;
    box.textContent =
      `${obj.name} is often confused with ${obj.confusable_with.join(", ")}. ` +
      `Capture a few hard negatives (real photos of those, not ${obj.name}) with the "Hard negative" role.`;
  } else {
    box.hidden = true;
  }
}

function wireObjectMeta() {
  $("btn-rename").addEventListener("click", async () => {
    const cur = state.objects.find((o) => o.object_id === state.objectId);
    const name = window.prompt("Object name", cur ? cur.name : "");
    if (name == null) return;
    const conf = window.prompt(
      "Confused with (comma-separated classes)",
      cur ? (cur.confusable_with || []).join(", ") : "",
    );
    const patch = { name: name.trim() };
    if (conf != null) patch.confusable_with = conf.split(",").map((s) => s.trim()).filter(Boolean);
    const r = await fetch(`/api/objects/${state.objectId}`, {
      method: "PATCH",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(patch),
    });
    if (!r.ok) note(`update failed: ${await r.text()}`);
    await refreshObject();
    await loadObjects();
    render(sm.snapshot());
  });
  $("btn-delete-object").addEventListener("click", async () => {
    if (!window.confirm("Move this object's folder to data/objects/_deleted/? It is not destroyed.")) return;
    const r = await fetch(`/api/objects/${state.objectId}`, { method: "DELETE" });
    const body = await r.json().catch(() => ({}));
    note(body.moved_to ? `soft-deleted → ${body.moved_to}` : "soft-deleted");
    sm.setPending({ uploads: 0, dirtyInspect: false });
    sm.dispatch("save_and_return"); // back to browsing (object gone)
    state.objectId = null;
    await loadObjects();
  });
}

// ---------- conditions ----------

function loadConditionDefaults() {
  try {
    return { ...state.vocab.default_conditions, ...JSON.parse(localStorage.getItem(COND_KEY) || "{}") };
  } catch {
    return { ...state.vocab.default_conditions };
  }
}

function buildConditionSelects() {
  const host = $("conditions");
  const defaults = loadConditionDefaults();
  host.replaceChildren();
  for (const [dim, values] of Object.entries(state.vocab.conditions)) {
    const sel = el("select", { id: `cond-${dim}`, dataset: { dim } });
    for (const v of values) sel.append(el("option", { value: v, text: v, selected: v === defaults[dim] }));
    sel.addEventListener("change", persistConditionDefaults);
    host.append(el("label", { class: "cond-field" }, [el("span", { text: dim }), sel]));
  }
  try {
    $("consent-ack").checked = localStorage.getItem(CONSENT_KEY) === "1";
  } catch {
    /* ignore */
  }
  $("consent-ack").addEventListener("change", () => {
    try {
      localStorage.setItem(CONSENT_KEY, $("consent-ack").checked ? "1" : "0");
    } catch {
      /* ignore */
    }
  });
}

function readConditions() {
  const out = {};
  for (const dim of Object.keys(state.vocab.conditions)) out[dim] = $(`cond-${dim}`).value;
  return out;
}

function persistConditionDefaults() {
  try {
    localStorage.setItem(COND_KEY, JSON.stringify(readConditions()));
  } catch {
    /* ignore */
  }
  if (state.stageKind === "inspect") sm.setPending({ dirtyInspect: true });
}

function setConditions(values) {
  for (const [dim, v] of Object.entries(values || {})) {
    const sel = $(`cond-${dim}`);
    if (sel) sel.value = v;
  }
}

function currentRole() {
  const r = document.querySelector('input[name="role"]:checked');
  return r ? r.value : "positive";
}

// ---------- camera (reuses the main monitoring preview constraints) ----------

async function startCamera() {
  try {
    const probe = await navigator.mediaDevices.getUserMedia({ video: true });
    probe.getTracks().forEach((t) => t.stop());
  } catch (err) {
    stageMsg(`Camera unavailable (${err.name}). Upload images instead, then adjust the box.`);
    return;
  }
  await populateCameras();
  navigator.mediaDevices.addEventListener?.("devicechange", populateCameras);
  await openCamera($("camera-select").value || "");
  $("camera-select").addEventListener("change", () => openCamera($("camera-select").value));
}

/** Re-open the camera if it is not currently live (e.g. after returning from a
 *  sample inspect on a device that dropped). No-op when a stream is healthy. */
async function ensureCamera() {
  if (state.track && state.track.readyState === "live") return;
  await openCamera($("camera-select")?.value || "");
}

async function populateCameras() {
  const devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
  const sel = $("camera-select");
  const cur = sel.value;
  sel.replaceChildren(
    ...devices.map((d, i) => el("option", { value: d.deviceId, text: d.label || `Camera ${i}` })),
  );
  if (!devices.length) sel.append(el("option", { value: "", text: "No cameras" }));
  if (cur && devices.some((d) => d.deviceId === cur)) sel.value = cur;
}

async function openCamera(deviceId) {
  stopCamera();
  // Same request shape as static/features/camera-capture.js.
  const ideal = {
    width: { ideal: PREVIEW_REQUEST.width },
    height: { ideal: PREVIEW_REQUEST.height },
    frameRate: { ideal: PREVIEW_REQUEST.frameRate },
  };
  state.requestedResolution = `${PREVIEW_REQUEST.width}x${PREVIEW_REQUEST.height}`;
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: deviceId ? { deviceId: { exact: deviceId }, ...ideal } : ideal,
      audio: false,
    });
  } catch (err) {
    stageMsg(`Could not open camera (${err.name}).`);
    return;
  }
  state.track = state.stream.getVideoTracks()[0] || null;

  // BLOCK 3.22: ask the device for its best practical resolution.
  if (state.track && typeof state.track.getCapabilities === "function") {
    try {
      const caps = state.track.getCapabilities();
      const wMax = Math.min(caps.width?.max || PREVIEW_REQUEST.width, MAX_CAPTURE_DIM);
      const hMax = Math.min(caps.height?.max || PREVIEW_REQUEST.height, MAX_CAPTURE_DIM);
      if (wMax > PREVIEW_REQUEST.width || hMax > PREVIEW_REQUEST.height) {
        state.requestedResolution = `${wMax}x${hMax}`;
        await state.track.applyConstraints({ width: { ideal: wMax }, height: { ideal: hMax } });
      }
      state.capabilitiesAvailable = true;
    } catch {
      state.capabilitiesAvailable = true; // constraints applied best-effort
    }
  } else {
    state.capabilitiesAvailable = false;
    note("getCapabilities() unsupported — using constraints only; recording achieved resolution");
  }

  const s = state.track ? state.track.getSettings() : {};
  state.achievedResolution = s.width && s.height ? `${s.width}x${s.height}` : "";
  const video = $("preview");
  video.srcObject = state.stream; // srcObject only — never drawn for display
  stageMsg("");
  note(
    `camera: requested ${state.requestedResolution} → achieved ${
      state.achievedResolution || "?"
    } (${state.capabilitiesAvailable ? "getCapabilities ok" : "constraints only"})`,
  );
  video.addEventListener(
    "loadedmetadata",
    () => {
      if (state.stageKind === "camera") centreBox();
      layoutBox();
    },
    { once: true },
  );
}

function stopCamera() {
  if (state.stream) {
    state.stream.getTracks().forEach((t) => t.stop());
    state.stream = null;
  }
  state.track = null;
  const v = $("preview");
  if (v) v.srcObject = null;
}

function deviceLabel() {
  const sel = $("camera-select");
  return sel && sel.selectedOptions[0] ? sel.selectedOptions[0].textContent : "camera";
}

// ---------- stage / box editor ----------

function mediaDims() {
  if (state.stageKind === "camera") {
    const v = $("preview");
    return { w: v.videoWidth || 0, h: v.videoHeight || 0 };
  }
  const img = $("upload-preview");
  return { w: img.naturalWidth || 0, h: img.naturalHeight || 0 };
}

/** Build (once) the shared box editor bound to the capture stage. `state.boxImg`
 *  aliases the editor's live box object so the rest of this module is unchanged. */
function ensureEditor() {
  if (editor) return editor;
  editor = createBoxEditor({
    stage: $("stage"),
    box: $("sample-box"),
    media: mediaDims,
    onChange: () => {
      if (state.stageKind === "inspect") sm.setPending({ dirtyInspect: true });
    },
  });
  state.boxImg = editor.box;
  return editor;
}

function centreBox() {
  ensureEditor().centre();
}

function clampBox() {
  ensureEditor().clamp();
}

function layoutBox() {
  ensureEditor().layout();
}

function wireBoxEditor() {
  ensureEditor().wire();
}

function stageMsg(text) {
  const m = $("stage-msg");
  m.textContent = text;
  m.hidden = !text;
}

// ---------- capture / upload / save ----------

/** Grab the live track at full achieved resolution as an ImageBitmap (BLOCK
 *  3.24). Falls back to drawImage(<video>) if createImageBitmap is unavailable.
 *  Returns { blob, capturePath, achieved }. */
async function frameToBlob() {
  const v = $("preview");
  const w = v.videoWidth;
  const h = v.videoHeight;
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const gctx = canvas.getContext("2d");
  let capturePath = "element";
  try {
    if (typeof createImageBitmap === "function") {
      const bmp = await createImageBitmap(v);
      gctx.drawImage(bmp, 0, 0, w, h);
      bmp.close?.();
      capturePath = "imagebitmap";
    } else {
      gctx.drawImage(v, 0, 0, w, h);
    }
  } catch {
    gctx.drawImage(v, 0, 0, w, h);
    capturePath = "element";
  }
  const blob = await new Promise((res) => canvas.toBlob(res, "image/jpeg", 0.95));
  return { blob, capturePath, achieved: `${w}x${h}` };
}

async function postSample(blob, { source, originalFilename, capturePath, achieved }) {
  const box = ensureEditor().getBox();
  const form = new FormData();
  form.append("image", blob, "sample.jpg");
  form.append("box", JSON.stringify(box));
  form.append("conditions", JSON.stringify(readConditions()));
  form.append("role", currentRole());
  form.append("negative_for", JSON.stringify([]));
  form.append("source", source);
  form.append("device_label", source === "camera" ? deviceLabel() : "");
  if (originalFilename) form.append("original_filename", originalFilename);
  form.append("consent_ack", $("consent-ack").checked ? "true" : "false");
  form.append("capture_path", capturePath || (source === "upload" ? "upload" : "element"));
  form.append("requested_resolution", source === "camera" ? state.requestedResolution : "");
  form.append("encoded_quality", "0.95");

  const r = await fetch(`/api/objects/${state.objectId}/samples`, { method: "POST", body: form });
  if (!r.ok) {
    note(`save failed: ${r.status} ${await r.text()}`);
    return false;
  }
  const body = await r.json();
  const sm2 = body.sample || {};
  const flags = (sm2.quality && sm2.quality.flags) || [];
  const dims = sm2.achieved_resolution ? ` @ ${sm2.achieved_resolution}` : "";
  note(flags.length ? `saved${dims} — flagged: ${flags.join(", ")}` : `saved${dims}`);
  await refreshObject();
  await loadObjects();
  render(sm.snapshot());
  return true;
}

async function onCaptureClick() {
  if (!state.objectId) return;
  if (state.stageKind === "camera") {
    if (!state.stream) {
      note("no camera — upload images instead");
      return;
    }
    const { blob, capturePath, achieved } = await frameToBlob();
    // camera captures save immediately, so they are never "pending"
    await postSample(blob, { source: "camera", capturePath, achieved });
  } else if (state.stageKind === "upload") {
    const file = state.uploadQueue[0];
    if (!file) return;
    const ok = await postSample(file, {
      source: "upload",
      originalFilename: file.name,
      capturePath: "upload",
    });
    if (ok) nextUpload();
  } else if (state.stageKind === "inspect") {
    await saveInspect();
  }
}

function wireUpload() {
  $("upload-input").addEventListener("change", (ev) => {
    const files = [...ev.target.files].filter((f) => f.type.startsWith("image/"));
    ev.target.value = "";
    if (!files.length) return;
    state.uploadQueue = files;
    showUpload();
  });
}

function showUpload() {
  const file = state.uploadQueue[0];
  if (!file) {
    clearUploadQueue();
    return;
  }
  if (state.uploadUrl) URL.revokeObjectURL(state.uploadUrl);
  state.uploadUrl = URL.createObjectURL(file);
  const img = $("upload-preview");
  img.onload = () => {
    sm.dispatch("capture", { count: state.uploadQueue.length });
    centreBox();
    note(`upload ${state.uploadQueue.length} queued — adjust the box, then Save upload`);
  };
  img.src = state.uploadUrl;
}

function nextUpload() {
  state.uploadQueue.shift();
  if (state.uploadQueue.length) {
    sm.setPending({ uploads: state.uploadQueue.length });
    showUpload();
  } else {
    clearUploadQueue();
  }
}

function clearUploadQueue({ silent = false } = {}) {
  state.uploadQueue = [];
  if (state.uploadUrl) {
    URL.revokeObjectURL(state.uploadUrl);
    state.uploadUrl = null;
  }
  sm.setPending({ uploads: 0 });
  if (!silent && (sm.state === "capturing")) sm.dispatch("back_to_camera");
}

// ---------- review strip + inspector ----------

function renderSamples(samples) {
  const strip = $("review-strip");
  $("review-count").textContent = String(samples.length);
  strip.replaceChildren(
    ...samples
      .slice()
      .reverse()
      .map((s) => {
        const li = el("li", {
          class: "review-item" + (s.sample_id === state.editingSampleId ? " editing" : ""),
        });
        const img = el("img", {
          alt: s.role,
          loading: "lazy",
          src: `/api/objects/${state.objectId}/samples/${s.sample_id}/image?thumb=1`,
        });
        const badge = el("span", { class: `ri-role ${s.role}`, text: s.role.replace("_", " ") });
        const flags = (s.quality && s.quality.flags) || [];
        li.append(img, badge);
        if (flags.length) li.append(el("span", { class: "ri-flag", text: flags.join(",") }));
        li.addEventListener("click", () => inspectSample(s));
        return li;
      }),
  );
}

async function inspectSample(s) {
  sm.dispatch("review", { sampleId: s.sample_id });
  if (state.uploadUrl) {
    URL.revokeObjectURL(state.uploadUrl);
    state.uploadUrl = null;
  }
  const img = $("upload-preview");
  img.onload = () => {
    ensureEditor().setBox(s.box);
  };
  img.src = `/api/objects/${state.objectId}/samples/${s.sample_id}/image`;
  setConditions(s.conditions);
  const roleInput = document.querySelector(`input[name="role"][value="${s.role}"]`);
  if (roleInput) roleInput.checked = true;
  await loadObjects();
  render(sm.snapshot());
}

async function saveInspect() {
  const sid = state.editingSampleId;
  if (!sid) return;
  const box = ensureEditor().getBox();
  const r = await fetch(`/api/objects/${state.objectId}/samples/${sid}`, {
    method: "PATCH",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      box,
      conditions: readConditions(),
      role: currentRole(),
    }),
  });
  if (!r.ok) {
    note(`update failed: ${await r.text()}`);
    return;
  }
  note("sample updated");
  sm.setPending({ dirtyInspect: false });
  await refreshObject();
}

async function discardInspect() {
  const sid = state.editingSampleId;
  if (!sid || !window.confirm("Discard this sample? (soft delete — files move to _deleted/)")) return;
  const r = await fetch(`/api/objects/${state.objectId}/samples/${sid}`, { method: "DELETE" });
  if (!r.ok) note(`delete failed: ${await r.text()}`);
  sm.dispatch("discard");
  await refreshObject();
  await loadObjects();
}

// ---------- coverage ----------

function renderCoverage(cov) {
  if (!cov) return;
  const kv = $("coverage-kv");
  const rows = [
    ["positives", `${cov.total_positives} / ${cov.targets.min_positives}`],
    ["distinct views", `${cov.distinct_views} / ${cov.targets.min_views}`],
    ["distances", cov.distances_present.join(", ") || "—"],
    ["lighting", cov.lighting_present.join(", ") || "—"],
    ["occluded", `${cov.occluded_count} / ${cov.targets.min_occluded}`],
    ["hard negatives", `${cov.total_hard_negatives} / ${cov.targets.min_hard_negatives}`],
    ["negatives", String(cov.total_negatives)],
    ["blurry flagged", String(cov.blurry_count)],
    ["near-duplicates", String(cov.duplicate_count)],
  ];
  kv.replaceChildren(
    ...rows.map(([k, v]) => el("div", {}, [el("dt", { text: k }), el("dd", { text: v })])),
  );
  const list = $("guidance-list");
  const items = cov.guidance && cov.guidance.length ? cov.guidance : ["Coverage targets met — good to go."];
  list.replaceChildren(...items.map((g) => el("li", { text: g })));
}

// ---------- misc ----------

function note(text) {
  $("capture-note").textContent = String(text);
}

/** A refused transition must always tell the user why - a dead button with no
 *  feedback is the defect that produced this phase (BLOCK 5.13 / 14). */
function refuseNote(transition) {
  sm.refuse(transition);
  const why = sm.hasPending()
    ? "there is an unsaved capture — save or discard it first"
    : `not available from “${sm.state}”`;
  note(`Can't ${transition.replace(/_/g, " ")} now: ${why}.`);
}

function wireInspectActions() {
  // The inspect action bar is a permanent element in index.html; show/hide is
  // driven by render() only (BLOCK 3.17 - controls do not vanish mid-session).
  $("btn-inspect-discard").addEventListener("click", discardInspect);
  $("btn-inspect-back").addEventListener("click", () => {
    if (sm.can("back_to_camera")) sm.dispatch("back_to_camera");
    else refuseNote("back_to_camera");
  });
}

function wireKeys() {
  window.addEventListener("keydown", (ev) => {
    if (ev.target.matches("input, select, textarea")) return;
    if (ev.key === "c" && state.objectId) {
      ev.preventDefault();
      onCaptureClick();
    } else if (ev.key === "d" && state.editingSampleId) {
      ev.preventDefault();
      discardInspect();
    } else if (ev.key === "Escape" && sm.state !== "browsing" && sm.state !== "object_selected") {
      ev.preventDefault();
      sm.dispatch("back_to_camera");
    }
  });
}

async function loadModelBadge() {
  try {
    const r = await fetch("/api/models/registry");
    const body = await r.json();
    $("active-model").textContent = body.available ? `model ${body.active}` : "model —";
    $("active-model").title = body.available
      ? `Active model version ${body.active} (read-only; no model trained in this phase)`
      : `model registry unavailable: ${body.reason || "?"}`;
  } catch {
    /* ignore */
  }
}

async function main() {
  try {
    await enter();
    wireExit();
    sm.subscribe(render);
    state.vocab = await fetch("/api/objects/vocab").then((r) => r.json());
    buildConditionSelects();
    wireObjectList();
    wireNewObject();
    wireObjectMeta();
    wireBoxEditor();
    wireUpload();
    wireInspectActions();
    wireKeys();
    batch = initBatch({
      vocab: state.vocab,
      createBoxEditor,
      getObjectId: () => state.objectId,
      note,
      // commit landed: refresh counts, samples and coverage without a reload
      afterSave: async () => {
        await refreshObject();
        await loadObjects();
        render(sm.snapshot());
      },
    });
    $("btn-capture").addEventListener("click", onCaptureClick);
    $("btn-reset-box").addEventListener("click", centreBox);
    await loadModelBadge();
    await loadObjects();
    await startCamera();
    render(sm.snapshot());
  } catch (err) {
    // A failure anywhere in init leaves a visible banner, never a silent dead UI.
    showFatal((err && (err.message || String(err))) || "initialisation failed");
    throw err;
  }
}

main();
