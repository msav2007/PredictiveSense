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

import { runtime } from "/static/features/runtime.js";
import { store } from "/static/ui/store.js";
import { isLayerEnabled } from "/static/features/analysis-prefs.js";

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
  const video = document.getElementById("preview");
  if (video) {
    video.addEventListener("loadedmetadata", () => draw());
    video.addEventListener("resize", () => draw());
  }
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

  const band = runtime.config?.perception?.detector?.low_confidence_band || [0.35, 0.5];
  if (isLayerEnabled("detection")) {
    for (const d of snap.detections || []) drawDetection(d, mapX, mapY, band);
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
}

function drawDetection(d, mapX, mapY, band) {
  const [x1, y1, x2, y2] = d.bbox;
  const px = mapX(x1);
  const py = mapY(y1);
  const pw = mapX(x2) - px;
  const ph = mapY(y2) - py;
  const colour = classColor(d.class_id);
  const lowConf = d.score >= band[0] && d.score < band[1];

  ctx.save();
  ctx.strokeStyle = colour;
  ctx.lineWidth = lowConf ? 1.5 : 2.5;
  ctx.globalAlpha *= lowConf ? 0.5 : 1.0;
  ctx.setLineDash(lowConf ? [6, 4] : []);
  ctx.strokeRect(px, py, pw, ph);
  ctx.setLineDash([]);

  const label = `${aliasFor(d.class_name)} ${(d.score * 100).toFixed(0)}%`;
  ctx.font = "12px ui-monospace, monospace";
  const tw = ctx.measureText(label).width + 10;
  const th = 16;
  ctx.fillStyle = colour;
  ctx.globalAlpha *= lowConf ? 0.6 : 0.9;
  ctx.fillRect(px, Math.max(0, py - th), tw, th);
  ctx.fillStyle = "#0b1220";
  ctx.globalAlpha = 1;
  ctx.fillText(label, px + 5, Math.max(11, py - 4));

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
