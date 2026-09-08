/* PredictiveSense Phase 2.5 labelling tool.
 *
 * One frame at a time on a canvas: draw / move / resize / relabel / delete
 * boxes, pick a class from the domain vocabulary, mark the frame done, navigate.
 * Saves to POST /api/labels/frame/{id} (records `seeded`). Adding a missed box
 * is exactly as cheap as accepting a seeded one (BLOCK 3.1.4).
 *
 * Not part of the dashboard shell - a standalone research tool served at /label.
 */
"use strict";

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");

const HANDLE = 8; // px hit radius for resize handles

const state = {
  frames: [],
  pos: 0,
  frame: null, // current frame payload
  img: new Image(),
  boxes: [], // {x, y, w, h, category}  (image-pixel space)
  categories: [],
  activeClass: null,
  selected: -1,
  seeded: false,
  done: false,
  drag: null, // {mode, boxIndex, handle, startX, startY, orig}
};

function colorFor(i) {
  const h = (i * 0.61803398875) % 1.0;
  return `hsl(${Math.round(h * 360)}, 70%, 60%)`;
}

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    const body = await r.text();
    throw new Error(`${r.status} ${path}: ${body.slice(0, 300)}`);
  }
  return r.json();
}

// -- load -----------------------------------------------------------

async function boot() {
  try {
    const list = await api("/api/labels/frames?limit=1000");
    state.frames = list.frames;
    if (!state.frames.length) {
      statusEl.textContent =
        "No frames. Run scripts/build_eval_frames.py, then reload.";
      return;
    }
    // resume at the first unlabelled frame
    const firstUnlabelled = state.frames.findIndex((f) => !f.labelled);
    state.pos = firstUnlabelled >= 0 ? firstUnlabelled : 0;
    await loadCurrent();
    wireKeys();
    wireMouse();
    document.getElementById("prev").onclick = () => nav(-1, false);
    document.getElementById("next").onclick = () => nav(1, false);
    document.getElementById("save").onclick = () => nav(1, true);
    document.getElementById("seed").onclick = seedFromDetector;
    document.getElementById("clear").onclick = () => {
      state.boxes = [];
      state.selected = -1;
      render();
    };
    document.getElementById("seededchk").onchange = (e) => {
      state.seeded = e.target.checked;
      renderPills();
    };
  } catch (err) {
    statusEl.textContent = String(err);
  }
}

async function loadCurrent(withSeed = false) {
  const meta = state.frames[state.pos];
  const url = `/api/labels/frame/${meta.image_id}${withSeed ? "?seed=1" : ""}`;
  const frame = await api(url);
  state.frame = frame;
  state.categories = frame.categories.map((c) => c.name);
  if (!state.activeClass) state.activeClass = state.categories[0];
  state.seeded = !!frame.seeded || withSeed;
  state.done = !!frame.labelled;
  state.selected = -1;

  const src = withSeed && frame.seed_boxes ? frame.seed_boxes : frame.boxes;
  state.boxes = src.map((b) => ({
    x: b.bbox[0], y: b.bbox[1], w: b.bbox[2], h: b.bbox[3], category: b.category,
  }));

  state.img = new Image();
  state.img.onload = render;
  state.img.src = `/api/labels/image/${meta.image_id}`;
  canvas.width = frame.width;
  canvas.height = frame.height;
  document.getElementById("seededchk").checked = state.seeded;
  buildPalette();
  renderPills();
  render();
}

async function nav(dir, save) {
  try {
    if (save) await saveCurrent();
    const nextPos = state.pos + dir;
    if (nextPos < 0 || nextPos >= state.frames.length) {
      statusEl.textContent = dir > 0 ? "At the last frame." : "At the first frame.";
      return;
    }
    state.pos = nextPos;
    await loadCurrent();
  } catch (err) {
    statusEl.textContent = String(err);
  }
}

async function saveCurrent() {
  const meta = state.frames[state.pos];
  const body = {
    boxes: state.boxes
      .filter((b) => b.w > 1 && b.h > 1)
      .map((b) => ({ category: b.category, bbox: [b.x, b.y, b.w, b.h] })),
    seeded: state.seeded,
    done: true,
  };
  const updated = await api(`/api/labels/frame/${meta.image_id}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  state.frames[state.pos] = {
    ...meta, labelled: true, seeded: updated.seeded, boxes: updated.boxes.length,
  };
  statusEl.textContent = `Saved ${body.boxes.length} box(es) for frame ${meta.image_id}.`;
}

async function seedFromDetector() {
  statusEl.textContent = "Seeding from the detector…";
  try {
    await loadCurrent(true);
    statusEl.textContent = `Seeded ${state.boxes.length} proposal(s). Edit / add / delete, then Save.`;
  } catch (err) {
    statusEl.textContent = `Seed unavailable: ${err}`;
  }
}

// -- palette ------------------------------------------------------

function buildPalette() {
  const wrap = document.getElementById("palette");
  wrap.replaceChildren();
  state.categories.forEach((name, i) => {
    const key = i < 9 ? String(i + 1) : name[0].toLowerCase();
    const row = document.createElement("div");
    row.className = "cls" + (name === state.activeClass ? " active" : "");
    row.dataset.name = name;
    row.innerHTML =
      `<span class="key">${key}</span>` +
      `<span class="swatch" style="background:${colorFor(i)}"></span>` +
      `<span>${name}</span>`;
    row.onclick = () => setActiveClass(name);
    wrap.append(row);
  });
}

function setActiveClass(name) {
  state.activeClass = name;
  if (state.selected >= 0) state.boxes[state.selected].category = name;
  buildPalette();
  render();
}

function renderPills() {
  document.getElementById("pos").textContent = `${state.pos + 1} / ${state.frames.length}`;
  document.getElementById("sess").textContent = "session " + (state.frame?.session_id || "—").slice(0, 8);
  const sp = document.getElementById("seededpill");
  sp.textContent = state.seeded ? "seeded" : "unseeded";
  sp.classList.toggle("on", state.seeded);
  const dp = document.getElementById("donepill");
  dp.textContent = state.done ? "done" : "not done";
  dp.classList.toggle("on", state.done);
}

// -- render -----------------------------------------------------

function render() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (state.img && state.img.complete && state.img.naturalWidth) {
    ctx.drawImage(state.img, 0, 0, canvas.width, canvas.height);
  }
  state.boxes.forEach((b, i) => {
    const ci = Math.max(0, state.categories.indexOf(b.category));
    ctx.lineWidth = i === state.selected ? 3 : 2;
    ctx.strokeStyle = colorFor(ci);
    ctx.strokeRect(b.x, b.y, b.w, b.h);
    ctx.fillStyle = colorFor(ci);
    ctx.font = "16px system-ui, sans-serif";
    const label = b.category;
    const tw = ctx.measureText(label).width + 10;
    ctx.fillRect(b.x, Math.max(0, b.y - 20), tw, 20);
    ctx.fillStyle = "#06101f";
    ctx.fillText(label, b.x + 5, Math.max(14, b.y - 5));
    if (i === state.selected) {
      ctx.fillStyle = "#fff";
      for (const [hx, hy] of corners(b)) ctx.fillRect(hx - 4, hy - 4, 8, 8);
    }
  });
  renderBoxList();
}

function renderBoxList() {
  const el = document.getElementById("boxlist");
  el.replaceChildren();
  state.boxes.forEach((b, i) => {
    const row = document.createElement("div");
    row.className = "row" + (i === state.selected ? " sel" : "");
    row.innerHTML = `<span>${b.category}</span><span>${Math.round(b.w)}×${Math.round(b.h)}</span>`;
    row.onclick = () => { state.selected = i; render(); };
    el.append(row);
  });
}

function corners(b) {
  return [
    [b.x, b.y], [b.x + b.w, b.y], [b.x, b.y + b.h], [b.x + b.w, b.y + b.h],
  ];
}

// -- mouse ------------------------------------------------------

function toImg(evt) {
  const r = canvas.getBoundingClientRect();
  return {
    x: ((evt.clientX - r.left) / r.width) * canvas.width,
    y: ((evt.clientY - r.top) / r.height) * canvas.height,
  };
}

function hitHandle(b, p) {
  const cs = corners(b);
  for (let h = 0; h < cs.length; h++) {
    if (Math.hypot(cs[h][0] - p.x, cs[h][1] - p.y) <= HANDLE) return h;
  }
  return -1;
}

function hitBox(p) {
  for (let i = state.boxes.length - 1; i >= 0; i--) {
    const b = state.boxes[i];
    if (p.x >= b.x && p.x <= b.x + b.w && p.y >= b.y && p.y <= b.y + b.h) return i;
  }
  return -1;
}

function wireMouse() {
  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId);
    const p = toImg(e);
    if (state.selected >= 0) {
      const h = hitHandle(state.boxes[state.selected], p);
      if (h >= 0) {
        state.drag = { mode: "resize", boxIndex: state.selected, handle: h };
        return;
      }
    }
    const bi = hitBox(p);
    if (bi >= 0) {
      state.selected = bi;
      const b = state.boxes[bi];
      state.drag = { mode: "move", boxIndex: bi, dx: p.x - b.x, dy: p.y - b.y };
      render();
      return;
    }
    // new box
    state.boxes.push({ x: p.x, y: p.y, w: 0, h: 0, category: state.activeClass });
    state.selected = state.boxes.length - 1;
    state.drag = { mode: "resize", boxIndex: state.selected, handle: 3, startX: p.x, startY: p.y };
    render();
  });

  canvas.addEventListener("pointermove", (e) => {
    if (!state.drag) return;
    const p = toImg(e);
    const b = state.boxes[state.drag.boxIndex];
    if (state.drag.mode === "move") {
      b.x = p.x - state.drag.dx;
      b.y = p.y - state.drag.dy;
    } else {
      const anchor = state.drag.handle === 3 && state.drag.startX !== undefined
        ? { x: state.drag.startX, y: state.drag.startY }
        : oppositeCorner(b, state.drag.handle);
      b.x = Math.min(anchor.x, p.x);
      b.y = Math.min(anchor.y, p.y);
      b.w = Math.abs(p.x - anchor.x);
      b.h = Math.abs(p.y - anchor.y);
    }
    render();
  });

  const end = () => {
    if (state.drag) {
      const b = state.boxes[state.drag.boxIndex];
      if (b && (b.w < 3 || b.h < 3) && state.drag.mode === "resize" && state.drag.startX !== undefined) {
        state.boxes.splice(state.drag.boxIndex, 1);
        state.selected = -1;
      }
      state.drag = null;
      render();
    }
  };
  canvas.addEventListener("pointerup", end);
  canvas.addEventListener("pointercancel", end);
}

function oppositeCorner(b, handle) {
  const cs = corners(b);
  return { x: cs[3 - handle][0], y: cs[3 - handle][1] };
}

// -- keyboard ---------------------------------------------------

function wireKeys() {
  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT") return;
    if (e.key === "n" || e.key === "ArrowRight") { e.preventDefault(); nav(1, false); }
    else if (e.key === "p" || e.key === "ArrowLeft") { e.preventDefault(); nav(-1, false); }
    else if (e.key === "s" || e.key === "S") { e.preventDefault(); nav(1, true); }
    else if (e.key === "e") { e.preventDefault(); seedFromDetector(); }
    else if (e.key === "d") { state.done = !state.done; renderPills(); }
    else if ((e.key === "Delete" || e.key === "Backspace") && state.selected >= 0) {
      e.preventDefault();
      state.boxes.splice(state.selected, 1);
      state.selected = -1;
      render();
    } else if (/^[1-9]$/.test(e.key)) {
      const idx = Number(e.key) - 1;
      if (idx < state.categories.length) setActiveClass(state.categories[idx]);
    } else {
      const byLetter = state.categories.find((c) => c[0].toLowerCase() === e.key);
      if (byLetter) setActiveClass(byLetter);
    }
  });
}

boot();
