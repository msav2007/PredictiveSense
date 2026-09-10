/* Perception overlay: draws detection boxes and pose skeletons on the
 * #overlay-layer canvas ABOVE the <video>. It never touches the <video> element
 * (no read, no draw-into, no replace) - it only sizes a sibling <canvas> to the
 * displayed video box and paints into it.
 *
 * Coordinate space: live analysis frames are downscaled by the worker to
 * capture.analysis_width x analysis_height, so detection/pose coordinates are in
 * that space. They are mapped onto the letterboxed <video> display rect. (The
 * backend-owned and recorded paths do not produce live /ws/state snapshots for
 * the browser, so this is the only overlay scenario - see docs/architecture.md.)
 *
 * Honest uncertainty: detections inside the configured low-confidence band draw
 * dashed and dimmed; low-visibility keypoints are de-emphasised; a stale
 * snapshot dims the whole overlay and is labelled "STALE".
 */
"use strict";

import { runtime, emit, MAX_AGE_SAMPLES } from "/static/features/runtime.js";
import { store } from "/static/ui/store.js";
import { isLayerEnabled } from "/static/features/analysis-prefs.js";
import { effectiveDetection, isPolicyView, onPolicyViewChange } from "/static/features/policy.js";

// Muted styling for a policy `unknown` detection - visually distinct from a
// confident detection and from nothing at all (BLOCK 3.14 / 3.24).
const UNKNOWN_COLOUR = "rgb(148, 163, 184)";
// Even more muted for a `secondary`-tier detection: shown, de-emphasised (BLOCK 3.3).
const SECONDARY_COLOUR = "rgb(120, 133, 148)";
// Faint dotted for a revealed `suppressed_implausible` box (Diagnostics only).
const SUPPRESSED_COLOUR = "rgb(90, 100, 116)";

// COCO 17-keypoint skeleton (mirrors perception/classes.py SKELETON_EDGES).
const SKELETON_EDGES = [
  [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12],
  [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
  [1, 3], [2, 4], [3, 5], [4, 6],
];

let canvas = null;
let ctx = null;
let ro = null;
let started = false;
// Boxes drawn on the last frame, for click hit-testing (display px).
let lastBoxes = [];
// Selected detection's bbox (analysis-space), for the Diagnostics inspector.
let selectedBBox = null;

function iou(a, b) {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]);
  const y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return ua > 0 ? inter / ua : 0;
}

/** Deterministic per-class colour - mirrors perception/classes.py class_color. */
function classColor(id) {
  const h = (id * 0.61803398875) % 1.0;
  const s = 0.65;
  const v = 0.95;
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  const [r, g, b] = [
    [v, t, p], [q, v, p], [p, v, t], [p, q, v], [t, p, v], [v, p, q],
  ][i % 6];
  return `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)})`;
}

function aliasFor(name) {
  const map = {
    "cell phone": "Phone",
    tv: "TV / monitor",
    couch: "Sofa",
    "potted plant": "Plant",
    "dining table": "Table",
    "wine glass": "Wine glass",
    "sports ball": "Ball",
    "hair drier": "Hair dryer",
    remote: "Remote control",
    mouse: "Computer mouse",
  };
  if (map[name]) return map[name];
  return name.charAt(0).toUpperCase() + name.slice(1);
}

export function initOverlay() {
  if (started) return;
  started = true;
  const layer = document.getElementById("overlay-layer");
  if (!layer) return;
  canvas = document.createElement("canvas");
  canvas.id = "overlay-canvas";
  canvas.style.position = "absolute";
  canvas.style.inset = "0";
  canvas.style.width = "100%";
  canvas.style.height = "100%";
  canvas.style.pointerEvents = "none";
  layer.appendChild(canvas);
  ctx = canvas.getContext("2d");

  const wrap = layer.parentElement || layer;
  ro = new ResizeObserver(() => resize(wrap));
  ro.observe(wrap);
  window.addEventListener("resize", () => resize(wrap));
  resize(wrap);

  store.subscribe(() => draw());
  onPolicyViewChange(() => draw());
  const video = document.getElementById("preview");
  if (video) {
    video.addEventListener("loadedmetadata", () => draw());
    video.addEventListener("resize", () => draw());
  }

  // Click-to-select a detection: the resting overlay label stays "Unknown", but
  // clicking a box reveals its full detail in the Diagnostics inspector
  // (BLOCK 3.11 / 3.12). The canvas keeps pointer-events:none; we hit-test the
  // last-drawn boxes on a listener attached to the layer's wrapper.
  wrap.addEventListener("click", (ev) => {
    const r = wrap.getBoundingClientRect();
    const cx = ev.clientX - r.left;
    const cy = ev.clientY - r.top;
    let hit = null;
    for (const b of lastBoxes) {
      if (cx >= b.x && cx <= b.x + b.w && cy >= b.y && cy <= b.y + b.h) {
        if (!hit || b.w * b.h < hit.w * hit.h) hit = b; // smallest box wins
      }
    }
    selectedBBox = hit ? hit.det.bbox.slice() : null;
    emit("detection-selected", { detection: hit ? hit.det : null });
    draw();
  });

  draw();
}

function resize(wrap) {
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = wrap.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  if (ctx) ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}

/** The letterboxed rect (CSS px, relative to the wrap) the video paints into. */
function displayRect() {
  const wrap = canvas.parentElement.parentElement || canvas.parentElement;
  const cw = wrap.clientWidth;
  const ch = wrap.clientHeight;
  const video = document.getElementById("preview");
  const vw = video && video.videoWidth;
  const vh = video && video.videoHeight;
  if (!vw || !vh) return { x: 0, y: 0, w: cw, h: ch };
  const scale = Math.min(cw / vw, ch / vh);
  const w = vw * scale;
  const h = vh * scale;
  return { x: (cw - w) / 2, y: (ch - h) / 2, w, h };
}

function draw() {
  if (!ctx || !canvas) return;
  const paintT0 = performance.now();
  const wrap = canvas.parentElement.parentElement || canvas.parentElement;
  ctx.clearRect(0, 0, wrap.clientWidth, wrap.clientHeight);

  if (store.get().mode !== "realtime") return;
  const snap = runtime.lastSnapshot || store.get().snapshot;
  if (!snap) return;

  const cap = runtime.config?.capture || {};
  const refW = cap.analysis_width || 640;
  const refH = cap.analysis_height || 480;
  const rect = displayRect();
  const sx = rect.w / refW;
  const sy = rect.h / refH;
  const mapX = (x) => rect.x + x * sx;
  const mapY = (y) => rect.y + y * sy;

  ctx.save();
  if (snap.stale) ctx.globalAlpha = 0.32;

  lastBoxes = [];
  const band = runtime.config?.perception?.detector?.low_confidence_band || [0.35, 0.5];
  const revealSuppressed = isPolicyView("suppressed");
  if (isLayerEnabled("detection")) {
    for (const d of snap.detections || []) drawDetection(d, mapX, mapY, band, revealSuppressed);
  }
  if (isLayerEnabled("pose")) {
    const visThr =
      runtime.config?.perception?.pose?.keypoint_visibility_threshold ?? 0.3;
    for (const p of snap.poses || []) drawPose(p, mapX, mapY, visThr);
  }
  ctx.restore();

  if (snap.stale) {
    ctx.save();
    ctx.font = "12px ui-monospace, monospace";
    ctx.fillStyle = "#ffb454";
    ctx.fillText("STALE — last analysis shown, not current", rect.x + 8, rect.y + 18);
    ctx.restore();
  }

  // Phase 8 capture->paint attribution. `capture_client_ts_ms` is the worker's
  // own drawImage epoch-ms clock echoed back through the snapshot, so this
  // subtraction is a pure client-clock delta with no cross-clock error. Only
  // meaningful for a fresh (non-stale) frame.
  runtime.lastPaintMs = performance.now() - paintT0;
  const capMs = snap.metrics && snap.metrics.capture_client_ts_ms;
  if (!snap.stale && typeof capMs === "number" && capMs > 0) {
    const paintAge = performance.timeOrigin + performance.now() - capMs;
    if (paintAge >= 0 && paintAge < 60000) {
      runtime.paintAgeSamples.push(paintAge);
      if (runtime.paintAgeSamples.length > MAX_AGE_SAMPLES) {
        runtime.paintAgeSamples.shift();
      }
    }
  }
}

function drawDetection(d, mapX, mapY, band, revealSuppressed) {
  const eff = effectiveDetection(d);

  // rejected_size is always hidden; suppressed_implausible is hidden on the main
  // overlay and only revealed by the Diagnostics-only toggle (BLOCK 3.8).
  if (eff.kind === "rejected") return;
  if (eff.kind === "suppressed" && !revealSuppressed) return;

  const [x1, y1, x2, y2] = d.bbox;
  const px = mapX(x1);
  const py = mapY(y1);
  const pw = mapX(x2) - px;
  const ph = mapY(y2) - py;

  lastBoxes.push({ det: d, x: px, y: py, w: pw, h: ph });
  const selected = selectedBBox && iou(selectedBBox, d.bbox) > 0.5;

  let colour;
  let alpha;
  let dashPattern;
  let resting;
  if (eff.kind === "accepted") {
    colour = classColor(d.class_id);
    const lowConf = d.score >= band[0] && d.score < band[1];
    alpha = lowConf ? 0.5 : 1.0;
    dashPattern = lowConf ? [6, 4] : [];
    resting = `${aliasFor(eff.label || d.class_name)} ${(d.score * 100).toFixed(0)}%`;
  } else if (eff.kind === "secondary") {
    colour = SECONDARY_COLOUR;
    alpha = 0.62;
    dashPattern = [5, 4];
    resting = aliasFor(eff.label || d.raw); // its real class, de-emphasised
  } else if (eff.kind === "suppressed") {
    colour = SUPPRESSED_COLOUR;
    alpha = 0.5;
    dashPattern = [2, 3];
    resting = `${aliasFor(eff.raw)} · suppressed`; // Diagnostics reveal only
  } else {
    // unknown_low_confidence / unknown_margin - the resting label is exactly
    // "Unknown": no former-class annotation, no raw class, no rule name on the
    // main overlay (BLOCK 3.10). Full detail lives in Diagnostics.
    colour = UNKNOWN_COLOUR;
    alpha = 0.55;
    dashPattern = [6, 4];
    resting = "Unknown";
  }

  ctx.save();
  ctx.strokeStyle = selected ? "#ffd166" : colour;
  ctx.lineWidth = selected ? 3 : eff.kind === "accepted" && !dashPattern.length ? 2.5 : 1.5;
  ctx.globalAlpha *= selected ? 1.0 : alpha;
  ctx.setLineDash(dashPattern);
  ctx.strokeRect(px, py, pw, ph);
  ctx.setLineDash([]);

  ctx.font = "12px ui-monospace, monospace";
  const tw = ctx.measureText(resting).width + 10;
  const th = 16;
  ctx.fillStyle = selected ? "#ffd166" : colour;
  ctx.globalAlpha *= 0.9;
  ctx.fillRect(px, Math.max(0, py - th), tw, th);
  ctx.fillStyle = "#0b1220";
  ctx.globalAlpha = 1;
  ctx.fillText(resting, px + 5, Math.max(11, py - 4));

  // short confidence bar under the label
  const barW = 40;
  ctx.fillStyle = "rgba(0,0,0,0.35)";
  ctx.fillRect(px, py, barW, 3);
  ctx.fillStyle = colour;
  ctx.fillRect(px, py, barW * Math.min(1, Math.max(0, d.score)), 3);
  ctx.restore();
}

function drawPose(p, mapX, mapY, visThr) {
  const kp = p.keypoints || [];
  ctx.save();
  ctx.strokeStyle = "#4ade80";
  ctx.lineWidth = 2;
  for (const [a, b] of SKELETON_EDGES) {
    const ka = kp[a];
    const kb = kp[b];
    if (!ka || !kb) continue;
    const va = ka[2] >= visThr;
    const vb = kb[2] >= visThr;
    ctx.globalAlpha = va && vb ? 0.9 : 0.25;
    ctx.beginPath();
    ctx.moveTo(mapX(ka[0]), mapY(ka[1]));
    ctx.lineTo(mapX(kb[0]), mapY(kb[1]));
    ctx.stroke();
  }
  for (const k of kp) {
    const vis = k[2] >= visThr;
    ctx.globalAlpha = vis ? 1 : 0.3;
    ctx.fillStyle = vis ? "#bbf7d0" : "#6b7280";
    ctx.beginPath();
    ctx.arc(mapX(k[0]), mapY(k[1]), vis ? 3 : 2, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}
