/* Bulk image upload for the Object Learning Studio (Phase 7).
 *
 * Upload many images for the selected object in one operation. The server
 * proposes one bounding box per image from the existing detector's RAW output
 * (before the recognition policy - the policy would suppress exactly the box we
 * want). The developer reviews and corrects them in a grid + a fast
 * previous/next reviewer, then "Save all" commits the reviewed samples through
 * the same creation path the camera capture uses.
 *
 * A proposed box is NOT ground truth. `box_confirmed_by_human` records whether a
 * human moved, resized, drew or explicitly accepted the box. Storing images is
 * not training - nothing here trains, fine-tunes or activates a model.
 *
 * This module owns only the bulk-upload panel. It reuses the shared box editor
 * (box-editor.js) and never touches the Studio state machine.
 */
"use strict";

import { el } from "/static/ui/controls.js";

const $ = (id) => document.getElementById(id);
const BR_COND_KEY = "ps.studio.batch.conditions";
const POLL_MS = 400;

const NEEDS_BOX = new Set(["manual_required", "error"]);

const b = {
  deps: null,
  batchId: null,
  items: [],
  role: "positive",
  pollTimer: null,
  reviewIndex: -1,
  editor: null,
  dirty: false,
  _gridSig: null,
};

export function initBatch(deps) {
  b.deps = deps;

  $("bulk-upload-input").addEventListener("change", (ev) => {
    const files = [...ev.target.files].filter((f) => f.type.startsWith("image/"));
    ev.target.value = "";
    if (files.length) uploadBatch(files);
  });

  $("btn-batch-close").addEventListener("click", () => hidePanel());
  $("btn-batch-discard").addEventListener("click", discardBatch);
  $("btn-batch-review").addEventListener("click", () => openReviewer(0));
  $("btn-batch-save").addEventListener("click", saveAll);
  $("btn-batch-attention").addEventListener("click", jumpToAttention);
  $("btn-br-prev").addEventListener("click", () => step(-1));
  $("btn-br-next").addEventListener("click", () => step(1));
  $("btn-br-accept").addEventListener("click", acceptBox);
  $("btn-br-add").addEventListener("click", addBox);
  $("btn-br-delete").addEventListener("click", deleteBox);

  buildConditionSelects();

  b.editor = b.deps.createBoxEditor({
    stage: $("batch-stage"),
    box: $("batch-box"),
    media: () => {
      const img = $("batch-img");
      return { w: img.naturalWidth || 0, h: img.naturalHeight || 0 };
    },
    onChange: () => {
      b.dirty = true;
    },
  });
  b.editor.wire();

  for (const inp of document.querySelectorAll('input[name="br-role"]')) {
    inp.addEventListener("change", () => {
      b.dirty = true;
    });
  }

  window.addEventListener("keydown", (ev) => {
    if ($("batch-reviewer").hidden) return;
    if (ev.target.matches("input, select, textarea")) return;
    if (ev.key === "n") step(1);
    else if (ev.key === "p") step(-1);
    else if (ev.key === "a") acceptBox();
    else if (ev.key === "x") deleteBox();
    else return;
    ev.preventDefault();
  });

  // Switching object (or returning to browse) closes any open batch view; the
  // staged batch itself is left on disk for the TTL sweep / an explicit Discard.
  return { onObjectChange: () => hidePanel() };
}

// ---------- upload ----------

async function uploadBatch(files) {
  const oid = b.deps.getObjectId();
  if (!oid) {
    b.deps.note("Select an object first.");
    return;
  }
  b.role = $("batch-role").value || "positive";
  const form = new FormData();
  for (const f of files) form.append("files", f, f.name);
  form.append("role", b.role);

  batchNote(`Uploading ${files.length} image${files.length === 1 ? "" : "s"}…`);
  let r;
  try {
    r = await fetch(`/api/objects/${oid}/batches`, { method: "POST", body: form });
  } catch (err) {
    batchNote(`upload failed: ${err}`);
    return;
  }
  const body = await r.json().catch(() => ({}));
  if (!r.ok) {
    const detail = body && body.detail;
    batchNote(`upload rejected: ${typeof detail === "string" ? detail : JSON.stringify(detail || body)}`);
    return;
  }
  b.batchId = body.batch_id;
  b.items = [];
  showPanel();
  const rej = (body.rejected || [])
    .map((d) => `${d.filename} (${d.reason})`)
    .join("; ");
  batchNote(
    `${body.accepted.length} image${body.accepted.length === 1 ? "" : "s"} staged` +
      (rej ? ` · rejected: ${rej}` : "") +
      " · proposing boxes…",
  );
  renderSkeleton(body.accepted.length);
  poll();
}

// ---------- poll ----------

async function poll() {
  const oid = b.deps.getObjectId();
  if (!oid || !b.batchId) return;
  let body;
  try {
    const r = await fetch(`/api/objects/${oid}/batches/${b.batchId}`);
    if (r.status === 404) {
      hidePanel();
      return;
    }
    body = await r.json();
  } catch {
    b.pollTimer = setTimeout(poll, POLL_MS);
    return;
  }
  b.items = body.items || [];
  renderGrid();
  updateCounts(body);
  if (body.status !== "ready") {
    batchNote(`Processing ${body.processed} / ${body.total}…`);
    b.pollTimer = setTimeout(poll, POLL_MS);
  } else {
    const need = b.items.filter(needsBox).length;
    batchNote(
      need
        ? `Proposals ready. ${need} image${need === 1 ? "" : "s"} still need a box - Review all.`
        : "Proposals ready. Review all, then Save all.",
    );
  }
}

// ---------- grid ----------

function renderSkeleton(n) {
  const grid = $("batch-grid");
  grid.replaceChildren(
    ...Array.from({ length: n }, () =>
      el("li", { class: "batch-cell pending" }, [el("span", { class: "bc-badge", text: "…" })]),
    ),
  );
}

function needsBox(it) {
  return NEEDS_BOX.has(it.status) || !it.box;
}

function gridSignature() {
  return b.items
    .map(
      (it) =>
        `${it.item_id}:${it.status}:${it.box_confirmed_by_human ? 1 : 0}:` +
        `${(it.box || []).map((n) => Math.round(n)).join(",")}:${(it.quality?.flags || []).join(",")}`,
    )
    .join("|");
}

function renderGrid() {
  const oid = b.deps.getObjectId();
  const grid = $("batch-grid");
  // Rebuild only when something actually changed - polling every 400ms must not
  // churn the <img> elements (re-requesting thumbnails on every tick).
  const sig = gridSignature();
  if (sig === b._gridSig && grid.children.length === b.items.length) return;
  b._gridSig = sig;
  grid.replaceChildren(
    ...b.items.map((it, i) => {
      const cell = el("li", {
        class: `batch-cell status-${it.status}` + (needsBox(it) ? " needs-box" : ""),
        dataset: { index: String(i) },
        title: `${it.filename} — ${it.status}`,
      });
      const img = el("img", {
        alt: it.filename,
        src: `/api/objects/${oid}/batches/${b.batchId}/items/${it.item_id}/image?thumb=1`,
      });
      cell.append(img);
      // the current box, drawn on the thumbnail (same aspect mapping as the img)
      if (it.box && it.box.length === 4 && it.width && it.height) {
        const [bx, by, bw, bh] = it.box;
        const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        svg.setAttribute("class", "bc-boxlayer");
        svg.setAttribute("viewBox", `0 0 ${it.width} ${it.height}`);
        svg.setAttribute("preserveAspectRatio", "xMidYMid slice");
        const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        rect.setAttribute("x", bx);
        rect.setAttribute("y", by);
        rect.setAttribute("width", bw);
        rect.setAttribute("height", bh);
        svg.append(rect);
        cell.append(svg);
      }
      const badge = el("span", { class: "bc-badge", text: badgeText(it) });
      cell.append(badge);
      if (it.box_confirmed_by_human) cell.append(el("span", { class: "bc-ok", text: "✓" }));
      const flags = (it.quality && it.quality.flags) || [];
      if (flags.length) cell.append(el("span", { class: "bc-flag", text: flags.join(",") }));
      cell.addEventListener("click", () => openReviewer(i));
      return cell;
    }),
  );
}

function badgeText(it) {
  if (it.status === "manual_required") return "needs box";
  if (it.status === "error") return "error";
  if (it.status === "flagged") return "check";
  if (it.status === "edited") return "edited";
  return "ready";
}

function updateCounts(body) {
  const total = body.total ?? b.items.length;
  const proposed = b.items.filter((it) => it.box && !needsBox(it)).length;
  const need = b.items.filter(needsBox).length;
  const reviewed = b.items.filter((it) => it.box_confirmed_by_human).length;
  $("batch-counts").textContent =
    `${total} uploaded · ${proposed} proposed · ${need} need a box · ${reviewed} reviewed`;
}

// ---------- reviewer ----------

function buildConditionSelects() {
  const host = $("br-conditions");
  const vocab = b.deps.vocab || { conditions: {}, default_conditions: {} };
  let defaults = { ...vocab.default_conditions };
  try {
    defaults = { ...defaults, ...JSON.parse(localStorage.getItem(BR_COND_KEY) || "{}") };
  } catch {
    /* ignore */
  }
  host.replaceChildren();
  for (const [dim, values] of Object.entries(vocab.conditions || {})) {
    const sel = el("select", { id: `br-cond-${dim}`, dataset: { dim } });
    for (const v of values) sel.append(el("option", { value: v, text: v, selected: v === defaults[dim] }));
    sel.addEventListener("change", () => {
      b.dirty = true;
      persistConditionDefaults();
    });
    host.append(el("label", { class: "cond-field" }, [el("span", { text: dim }), sel]));
  }
}

function readConditions() {
  const out = {};
  for (const sel of document.querySelectorAll('#br-conditions select')) out[sel.dataset.dim] = sel.value;
  return out;
}

function setConditions(values) {
  for (const [dim, v] of Object.entries(values || {})) {
    const sel = $(`br-cond-${dim}`);
    if (sel) sel.value = v;
  }
}

function persistConditionDefaults() {
  try {
    localStorage.setItem(BR_COND_KEY, JSON.stringify(readConditions()));
  } catch {
    /* ignore */
  }
}

function currentBrRole() {
  const r = document.querySelector('input[name="br-role"]:checked');
  return r ? r.value : "positive";
}

async function openReviewer(index) {
  if (!b.items.length) {
    batchNote("Nothing to review yet.");
    return;
  }
  await persistCurrent();
  b.reviewIndex = Math.max(0, Math.min(index, b.items.length - 1));
  $("batch-reviewer").hidden = false;
  loadReviewItem();
}

function loadReviewItem() {
  const oid = b.deps.getObjectId();
  const it = b.items[b.reviewIndex];
  if (!it) return;
  b.dirty = false;
  $("br-pos").textContent = `${b.reviewIndex + 1} / ${b.items.length}`;
  const flags = (it.quality && it.quality.flags) || [];
  $("br-status").textContent =
    `${it.filename} — ${it.status}` +
    (it.proposal_source === "detector" && it.proposal_raw_class
      ? ` · detector hint: ${it.proposal_raw_class} (${(it.proposal_score ?? 0).toFixed(2)}), not the label`
      : it.proposal_source === "default_centred"
        ? " · no detection — drag the centred box"
        : "") +
    (it.error ? ` · ${it.error}` : "") +
    (flags.length ? ` · flags: ${flags.join(", ")}` : "");
  const roleInput = document.querySelector(`input[name="br-role"][value="${it.role || b.role}"]`);
  if (roleInput) roleInput.checked = true;
  setConditions(Object.keys(it.conditions || {}).length ? it.conditions : undefined);

  const img = $("batch-img");
  img.onload = () => {
    if (it.box && it.box.length === 4) b.editor.setBox(it.box);
    else b.editor.centre();
  };
  img.src = `/api/objects/${oid}/batches/${b.batchId}/items/${it.item_id}/image`;
}

async function step(delta) {
  if ($("batch-reviewer").hidden) return;
  await persistCurrent();
  b.reviewIndex = Math.max(0, Math.min(b.reviewIndex + delta, b.items.length - 1));
  loadReviewItem();
}

async function acceptBox() {
  if ($("batch-reviewer").hidden) return;
  b.dirty = true; // explicit human confirmation
  await persistCurrent({ confirm: true });
  if (b.reviewIndex < b.items.length - 1) step(1);
  else batchNote("Last item accepted.");
}

function addBox() {
  if ($("batch-reviewer").hidden) return;
  b.editor.centre();
  b.dirty = true;
}

async function deleteBox() {
  if ($("batch-reviewer").hidden) return;
  const it = b.items[b.reviewIndex];
  if (!it) return;
  const oid = b.deps.getObjectId();
  const r = await fetch(
    `/api/objects/${oid}/batches/${b.batchId}/items/${it.item_id}`,
    {
      method: "PATCH",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ box: null }),
    },
  );
  if (r.ok) {
    it.box = null;
    it.status = "manual_required";
    it.box_confirmed_by_human = false;
    b.dirty = false;
    b.editor.centre();
    renderGrid();
    updateCounts({ total: b.items.length });
    $("br-status").textContent = `${it.filename} — box removed, draw a new one`;
  } else {
    batchNote(`could not delete box: ${await r.text()}`);
  }
}

/** PATCH the currently-open reviewer item if the developer touched it. */
async function persistCurrent({ confirm = false } = {}) {
  const it = b.items[b.reviewIndex];
  if (!it || (!b.dirty && !confirm)) return;
  const oid = b.deps.getObjectId();
  const box = b.editor.getBox();
  const payload = {
    box,
    role: currentBrRole(),
    conditions: readConditions(),
    box_confirmed_by_human: true,
  };
  const r = await fetch(
    `/api/objects/${oid}/batches/${b.batchId}/items/${it.item_id}`,
    {
      method: "PATCH",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    },
  );
  if (r.ok) {
    Object.assign(it, (await r.json()).item || {});
    b.dirty = false;
    renderGrid();
    updateCounts({ total: b.items.length });
  } else {
    batchNote(`save failed: ${await r.text()}`);
  }
}

function jumpToAttention() {
  const idx = b.items.findIndex(needsBox);
  if (idx < 0) {
    batchNote("Every image has a bounding box.");
    return;
  }
  openReviewer(idx);
}

// ---------- save / discard ----------

async function saveAll() {
  const oid = b.deps.getObjectId();
  if (!oid || !b.batchId) return;
  await persistCurrent();
  const r = await fetch(`/api/objects/${oid}/batches/${b.batchId}/save`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ consent_ack: false }),
  });
  if (!r.ok) {
    batchNote(`save failed: ${await r.text()}`);
    return;
  }
  const body = await r.json();
  // Collected != trained != activated. Never imply the model learned anything.
  const tail = body.remaining
    ? ` ${body.remaining} still need${body.remaining === 1 ? "s" : ""} a bounding box.`
    : "";
  await b.deps.afterSave();
  if (body.batch_cleared) {
    hidePanel();
  } else {
    await refreshOnce();
    $("batch-reviewer").hidden = true;
  }
  // set the result note last so a refresh cannot clobber it
  batchNote(`${body.saved} sample${body.saved === 1 ? "" : "s"} saved · training data updated.${tail}`);
}

/** One GET to refresh the grid + counts without touching #batch-note. */
async function refreshOnce() {
  const oid = b.deps.getObjectId();
  if (!oid || !b.batchId) return;
  try {
    const r = await fetch(`/api/objects/${oid}/batches/${b.batchId}`);
    if (r.status === 404) {
      hidePanel();
      return;
    }
    const body = await r.json();
    b.items = body.items || [];
    b.reviewIndex = Math.min(b.reviewIndex, b.items.length - 1);
    renderGrid();
    updateCounts(body);
  } catch {
    /* leave the grid as-is */
  }
}

async function discardBatch() {
  const oid = b.deps.getObjectId();
  if (!oid || !b.batchId) {
    hidePanel();
    return;
  }
  if (!window.confirm("Discard this upload batch? Staged images are deleted; nothing is saved.")) return;
  await fetch(`/api/objects/${oid}/batches/${b.batchId}`, { method: "DELETE" }).catch(() => {});
  hidePanel();
  b.deps.note("Upload batch discarded.");
}

// ---------- panel visibility ----------

function showPanel() {
  $("batch-panel").hidden = false;
}

function hidePanel() {
  if (b.pollTimer) clearTimeout(b.pollTimer);
  b.pollTimer = null;
  b.batchId = null;
  b.items = [];
  b.reviewIndex = -1;
  b._gridSig = null;
  $("batch-reviewer").hidden = true;
  $("batch-panel").hidden = true;
  $("batch-grid").replaceChildren();
}

function batchNote(text) {
  $("batch-note").textContent = String(text);
}
