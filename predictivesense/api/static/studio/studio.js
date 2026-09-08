/* Object Learning Studio (Phase 4).
 *
 * A standalone screen (not the dashboard shell). On entry it calls
 * POST /api/studio/enter, which pauses the monitoring pipeline; on exit it calls
 * POST /api/studio/leave, which restores it. The camera preview here is
 * getUserMedia -> <video>.srcObject only, with no analysis worker attached.
 *
 * Every saved sample carries exactly one bounding box (in source-image pixels),
 * condition tags from the fixed vocabularies, and a role. Storing images is not
 * training.
 */
"use strict";

import { el } from "/static/ui/controls.js";

const $ = (id) => document.getElementById(id);
const COND_KEY = "ps.studio.conditions";
const CONSENT_KEY = "ps.studio.consent";

const state = {
  token: null,
  vocab: null,
  objectId: null,
  objects: [],
  stream: null,
  stageKind: "camera", // "camera" | "upload" | "inspect"
  boxImg: { x: 0, y: 0, w: 0, h: 0 }, // source-image pixel coords
  uploadQueue: [], // File[]
  uploadUrl: null,
  editingSampleId: null,
  left: false,
};

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

function wireExit() {
  $("btn-return").addEventListener("click", async (ev) => {
    ev.preventDefault();
    await leave();
    window.location.href = "/";
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
}

// ---------- objects ----------

async function loadObjects() {
  const r = await fetch("/api/objects");
  const body = await r.json();
  state.objects = body.objects || [];
  const list = $("object-list");
  list.replaceChildren(
    ...state.objects.map((o) => {
      const li = el("li", {
        class: "object-item" + (o.object_id === state.objectId ? " active" : ""),
        onClick: () => selectObject(o.object_id),
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
  state.objectId = objectId;
  state.editingSampleId = null;
  clearUploadQueue();
  setStageKind("camera");
  $("empty-state").hidden = true;
  $("capture-pane").hidden = false;
  await loadObjects();
  await refreshObject();
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
  });
  $("btn-delete-object").addEventListener("click", async () => {
    if (!window.confirm("Move this object's folder to data/objects/_deleted/? It is not destroyed.")) return;
    const r = await fetch(`/api/objects/${state.objectId}`, { method: "DELETE" });
    const body = await r.json().catch(() => ({}));
    note(body.moved_to ? `soft-deleted → ${body.moved_to}` : "soft-deleted");
    state.objectId = null;
    $("capture-pane").hidden = true;
    $("empty-state").hidden = false;
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

// ---------- camera ----------

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
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: deviceId ? { deviceId: { exact: deviceId } } : true,
      audio: false,
    });
  } catch (err) {
    stageMsg(`Could not open camera (${err.name}).`);
    return;
  }
  const video = $("preview");
  video.srcObject = state.stream; // srcObject only — never drawn for display
  stageMsg("");
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
  const v = $("preview");
  if (v) v.srcObject = null;
}

function deviceLabel() {
  const sel = $("camera-select");
  return sel && sel.selectedOptions[0] ? sel.selectedOptions[0].textContent : "camera";
}

// ---------- stage / box editor ----------

function setStageKind(kind) {
  state.stageKind = kind;
  $("preview").hidden = kind !== "camera";
  $("upload-preview").hidden = kind === "camera";
  $("btn-capture").textContent =
    kind === "camera" ? "Capture (c)" : kind === "upload" ? "Save upload (c)" : "Save changes (c)";
  $("btn-upload-label").hidden = kind === "inspect";
  layoutBox();
}

function mediaDims() {
  if (state.stageKind === "camera") {
    const v = $("preview");
    return { w: v.videoWidth || 0, h: v.videoHeight || 0 };
  }
  const img = $("upload-preview");
  return { w: img.naturalWidth || 0, h: img.naturalHeight || 0 };
}

function contentRect() {
  const stage = $("stage").getBoundingClientRect();
  const { w: mw, h: mh } = mediaDims();
  if (!mw || !mh) return { x: 0, y: 0, w: stage.width, h: stage.height, scale: 1 };
  const scale = Math.min(stage.width / mw, stage.height / mh);
  return {
    x: (stage.width - mw * scale) / 2,
    y: (stage.height - mh * scale) / 2,
    w: mw * scale,
    h: mh * scale,
    scale,
  };
}

function centreBox() {
  const { w, h } = mediaDims();
  if (!w || !h) {
    state.boxImg = { x: 0, y: 0, w: 0, h: 0 };
    layoutBox();
    return;
  }
  state.boxImg = { x: Math.round(w * 0.3), y: Math.round(h * 0.3), w: Math.round(w * 0.4), h: Math.round(h * 0.4) };
  layoutBox();
}

function clampBox() {
  const { w, h } = mediaDims();
  if (!w || !h) return;
  const b = state.boxImg;
  b.w = Math.max(8, Math.min(b.w, w));
  b.h = Math.max(8, Math.min(b.h, h));
  b.x = Math.max(0, Math.min(b.x, w - b.w));
  b.y = Math.max(0, Math.min(b.y, h - b.h));
}

function layoutBox() {
  clampBox();
  const c = contentRect();
  const b = state.boxImg;
  const box = $("sample-box");
  box.style.left = `${c.x + b.x * c.scale}px`;
  box.style.top = `${c.y + b.y * c.scale}px`;
  box.style.width = `${b.w * c.scale}px`;
  box.style.height = `${b.h * c.scale}px`;
}

function wireBoxEditor() {
  const box = $("sample-box");
  let mode = null; // "move" | "nw" | "ne" | "sw" | "se"
  let startX = 0;
  let startY = 0;
  let orig = null;

  function toImgDelta(dx, dy) {
    const s = contentRect().scale || 1;
    return { dx: dx / s, dy: dy / s };
  }

  function onDown(ev) {
    mode = ev.target.classList.contains("bh")
      ? [...ev.target.classList].find((c) => ["nw", "ne", "sw", "se"].includes(c))
      : "move";
    startX = ev.clientX;
    startY = ev.clientY;
    orig = { ...state.boxImg };
    box.setPointerCapture?.(ev.pointerId);
    ev.preventDefault();
  }
  function onMove(ev) {
    if (!mode) return;
    const { dx, dy } = toImgDelta(ev.clientX - startX, ev.clientY - startY);
    const b = state.boxImg;
    if (mode === "move") {
      b.x = orig.x + dx;
      b.y = orig.y + dy;
    } else {
      if (mode.includes("w")) {
        b.x = orig.x + dx;
        b.w = orig.w - dx;
      }
      if (mode.includes("e")) b.w = orig.w + dx;
      if (mode.includes("n")) {
        b.y = orig.y + dy;
        b.h = orig.h - dy;
      }
      if (mode.includes("s")) b.h = orig.h + dy;
    }
    layoutBox();
  }
  function onUp() {
    mode = null;
  }
  box.addEventListener("pointerdown", onDown);
  window.addEventListener("pointermove", onMove);
  window.addEventListener("pointerup", onUp);

  box.addEventListener("keydown", (ev) => {
    const step = ev.shiftKey ? 10 : 2;
    const b = state.boxImg;
    if (ev.key === "ArrowLeft") b.x -= step;
    else if (ev.key === "ArrowRight") b.x += step;
    else if (ev.key === "ArrowUp") b.y -= step;
    else if (ev.key === "ArrowDown") b.y += step;
    else return;
    ev.preventDefault();
    layoutBox();
  });

  window.addEventListener("resize", layoutBox);
}

function stageMsg(text) {
  const m = $("stage-msg");
  m.textContent = text;
  m.hidden = !text;
}

// ---------- capture / upload / save ----------

function frameToBlob() {
  const v = $("preview");
  const canvas = document.createElement("canvas");
  canvas.width = v.videoWidth;
  canvas.height = v.videoHeight;
  canvas.getContext("2d").drawImage(v, 0, 0, canvas.width, canvas.height);
  return new Promise((res) => canvas.toBlob(res, "image/jpeg", 0.92));
}

async function postSample(blob, { source, originalFilename }) {
  clampBox();
  const b = state.boxImg;
  const form = new FormData();
  form.append("image", blob, "sample.jpg");
  form.append("box", JSON.stringify([Math.round(b.x), Math.round(b.y), Math.round(b.w), Math.round(b.h)]));
  form.append("conditions", JSON.stringify(readConditions()));
  form.append("role", currentRole());
  form.append("negative_for", JSON.stringify([]));
  form.append("source", source);
  form.append("device_label", source === "camera" ? deviceLabel() : "");
  if (originalFilename) form.append("original_filename", originalFilename);
  form.append("consent_ack", $("consent-ack").checked ? "true" : "false");

  const r = await fetch(`/api/objects/${state.objectId}/samples`, { method: "POST", body: form });
  if (!r.ok) {
    note(`save failed: ${r.status} ${await r.text()}`);
    return false;
  }
  const body = await r.json();
  const flags = (body.sample.quality && body.sample.quality.flags) || [];
  note(flags.length ? `saved — flagged: ${flags.join(", ")}` : "saved");
  await refreshObject();
  await loadObjects();
  return true;
}

async function onCaptureClick() {
  if (!state.objectId) return;
  if (state.stageKind === "camera") {
    if (!state.stream) {
      note("no camera — upload images instead");
      return;
    }
    const blob = await frameToBlob();
    await postSample(blob, { source: "camera" });
  } else if (state.stageKind === "upload") {
    const file = state.uploadQueue[0];
    if (!file) return;
    const ok = await postSample(file, { source: "upload", originalFilename: file.name });
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
    setStageKind("upload");
    centreBox();
    note(`upload ${state.uploadQueue.length} queued — adjust the box, then Save upload`);
  };
  img.src = state.uploadUrl;
}

function nextUpload() {
  state.uploadQueue.shift();
  if (state.uploadQueue.length) showUpload();
  else clearUploadQueue();
}

function clearUploadQueue() {
  state.uploadQueue = [];
  if (state.uploadUrl) {
    URL.revokeObjectURL(state.uploadUrl);
    state.uploadUrl = null;
  }
  if (state.stageKind !== "camera") setStageKind("camera");
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
  state.editingSampleId = s.sample_id;
  if (state.uploadUrl) {
    URL.revokeObjectURL(state.uploadUrl);
    state.uploadUrl = null;
  }
  const img = $("upload-preview");
  img.onload = () => {
    setStageKind("inspect");
    state.boxImg = { x: s.box[0], y: s.box[1], w: s.box[2], h: s.box[3] };
    layoutBox();
  };
  img.src = `/api/objects/${state.objectId}/samples/${s.sample_id}/image`;
  setConditions(s.conditions);
  const roleInput = document.querySelector(`input[name="role"][value="${s.role}"]`);
  if (roleInput) roleInput.checked = true;
  ensureInspectActions();
  await loadObjects();
}

function ensureInspectActions() {
  if ($("inspect-actions")) {
    $("inspect-actions").hidden = false;
    return;
  }
  const bar = el("div", { class: "inspect-actions", id: "inspect-actions" }, [
    el("button", {
      class: "btn btn-danger",
      type: "button",
      text: "Discard sample",
      onClick: discardInspect,
    }),
    el("button", {
      class: "btn btn-secondary",
      type: "button",
      text: "Back to camera",
      onClick: () => {
        state.editingSampleId = null;
        $("inspect-actions").hidden = true;
        setStageKind("camera");
        loadObjects();
      },
    }),
  ]);
  $("capture-note").before(bar);
}

async function saveInspect() {
  const sid = state.editingSampleId;
  if (!sid) return;
  clampBox();
  const b = state.boxImg;
  const r = await fetch(`/api/objects/${state.objectId}/samples/${sid}`, {
    method: "PATCH",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      box: [Math.round(b.x), Math.round(b.y), Math.round(b.w), Math.round(b.h)],
      conditions: readConditions(),
      role: currentRole(),
    }),
  });
  if (!r.ok) {
    note(`update failed: ${await r.text()}`);
    return;
  }
  note("sample updated");
  await refreshObject();
}

async function discardInspect() {
  const sid = state.editingSampleId;
  if (!sid || !window.confirm("Discard this sample? (soft delete — files move to _deleted/)")) return;
  const r = await fetch(`/api/objects/${state.objectId}/samples/${sid}`, { method: "DELETE" });
  if (!r.ok) note(`delete failed: ${await r.text()}`);
  state.editingSampleId = null;
  $("inspect-actions").hidden = true;
  setStageKind("camera");
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

function wireKeys() {
  window.addEventListener("keydown", (ev) => {
    if (ev.target.matches("input, select, textarea")) return;
    if (ev.key === "c" && state.objectId) {
      ev.preventDefault();
      onCaptureClick();
    } else if (ev.key === "d" && state.editingSampleId) {
      ev.preventDefault();
      discardInspect();
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
  await enter();
  wireExit();
  state.vocab = await fetch("/api/objects/vocab").then((r) => r.json());
  buildConditionSelects();
  wireNewObject();
  wireObjectMeta();
  wireBoxEditor();
  wireUpload();
  wireKeys();
  $("btn-capture").addEventListener("click", onCaptureClick);
  $("btn-reset-box").addEventListener("click", centreBox);
  await loadModelBadge();
  await loadObjects();
  await startCamera();
}

main();
