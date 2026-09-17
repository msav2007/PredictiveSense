/* Perception overlay: draws track boxes and pose skeletons on the
 * #overlay-layer canvas ABOVE the <video>. It never touches the <video> element
 * (no read, no draw-into, no replace) - it only sizes a sibling <canvas> to the
 * displayed video box and paints into it.
 *
 * Coordinate space: live analysis frames are downscaled by the worker to
 * capture.analysis_width x analysis_height, so detection/pose/track coordinates
 * are in that space. They are mapped onto the letterboxed <video> display rect.
 * (The backend-owned and recorded paths do not produce live /ws/state snapshots
 * for the browser, so this is the only overlay scenario - see docs/architecture.md.)
 *
 * Phase 9 state/visual mapping (docs/architecture.md has the authoritative
 * table): each concept owns exactly one visual channel, so no two states render
 * identically -
 *   - track identity      -> hue, stable per track_id for the track's life
 *   - fresh vs coasting    -> stroke: solid (a detector observation landed this
 *                             frame) vs dashed (position is a motion prediction)
 *   - recognition uncertainty (Unknown) -> label text only, never a colour
 *   - vocabulary tier (secondary)       -> a small badge, not a colour change
 *   - snapshot staleness   -> whole-overlay dim + "STALE" banner (unchanged
 *                             from Phase 8) - a property of the frame, not any
 *                             one object, so the last real tracks stay visible
 *                             (dimmed) through a stale gap instead of vanishing
 *
 * `suppressed_implausible` detections are never tracked (the tracker's
 * TRACK_ELIGIBLE_STATES excludes them - they are not candidate objects), so the
 * Diagnostics-only "reveal suppressed" toggle draws them separately, straight
 * from `snap.detections`, with no identity/freshness concept (there is none).
 */
"use strict";

import { runtime, emit, MAX_AGE_SAMPLES } from "/static/features/runtime.js";
import { store } from "/static/ui/store.js";
import { isLayerEnabled } from "/static/features/analysis-prefs.js";
import { effectiveDetection, isPolicyView, onPolicyViewChange } from "/static/features/policy.js";

// Faint dotted grey for a revealed `suppressed_implausible` box (Diagnostics
// only) - these are never tracked, so no identity hue applies to them.
const SUPPRESSED_COLOUR = "rgb(90, 100, 116)";
// Tier badge background - a small marker, never the whole box's colour, so it
// cannot be confused with the (unrelated) Unknown-uncertainty rendering.
const TIER_BADGE_BG = "rgb(71, 85, 105)";
// A track's displayed *kind* (accepted/secondary/unknown) only changes once the
// new kind has persisted for this long - single-cycle policy_state flicker
// (root-cause table: results/grey_state_frequency.md) must not blink the label
// every frame. Presentation-only: Diagnostics still reads the raw per-frame
// policy_state untouched. ~3 analysis cycles at the measured p50 interval
// (results/track_thresholds.md, ~47ms) without being so slow it reads as lag.
const KIND_DWELL_MS = 150;

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
// Boxes drawn on the last frame, for click hit-testing (display px). Each is
// either { kind: "track", track, x, y, w, h } or (suppressed-reveal only)
// { kind: "detection", det, x, y, w, h }.
let lastBoxes = [];
// Selected track's id (stable across frames - no bbox re-matching needed),
// for the Diagnostics per-track inspector (section 9.5).
let selectedTrackId = null;

/** Deterministic per-id colour (golden-ratio hue step, mirrors
 * perception/classes.py class_color's algorithm). Phase 9: keyed by
 * `track_id` so a track's colour is its stable visual identity, independent
 * of its class - two Unknown objects tracked at once render in two different,
 * individually stable hues. */
function identityColor(id) {
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

// Per-track displayed-kind hysteresis (section 9.4). Bounded by construction:
// pruned to exactly this frame's live track ids on every draw(), so it can
// never outgrow the backend's own bounded track count.
const kindState = new Map(); // track_id -> { kind, pendingKind, pendingSinceMs }

function stableKind(trackId, rawKind, nowMs) {
  let s = kindState.get(trackId);
  if (!s) {
    s = { kind: rawKind, pendingKind: null, pendingSinceMs: 0 };
    kindState.set(trackId, s);
    return s.kind;
  }
  if (rawKind === s.kind) {
    s.pendingKind = null;
    return s.kind;
  }
  if (s.pendingKind !== rawKind) {
    s.pendingKind = rawKind;
    s.pendingSinceMs = nowMs;
    return s.kind; // not yet material - keep showing the current kind
  }
  if (nowMs - s.pendingSinceMs >= KIND_DWELL_MS) {
    s.kind = rawKind;
    s.pendingKind = null;
    return s.kind;
  }
  return s.kind;
}

function pruneKindState(liveTrackIds) {
  for (const id of kindState.keys()) {
    if (!liveTrackIds.has(id)) kindState.delete(id);
  }
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

  // Click-to-select a track (or, for a revealed suppressed box, a raw
  // detection): the resting overlay label stays "Unknown", but clicking a box
  // reveals its full detail in the Diagnostics inspector (BLOCK 3.11 / 3.12;
  // section 9.5 per-track detail). The canvas keeps pointer-events:none; we
  // hit-test the last-drawn boxes on a listener attached to the layer's wrapper.
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
    selectedTrackId = hit && hit.kind === "track" ? hit.track.track_id : null;
    emit("detection-selected", {
      track: hit && hit.kind === "track" ? hit.track : null,
      detection: hit && hit.kind === "detection" ? hit.det : null,
    });
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
  const revealSuppressed = isPolicyView("suppressed");
  const nowMs = paintT0;
  const poseLayerOn = isLayerEnabled("pose");
  const detectionLayerOn = isLayerEnabled("detection");
  const visThr = runtime.config?.perception?.pose?.keypoint_visibility_threshold ?? 0.3;
  const poseMaxAgeMs = runtime.config?.tracking?.pose_max_age_ms ?? 900;
  if (detectionLayerOn || poseLayerOn) {
    const tracks = snap.tracks || [];
    pruneKindState(new Set(tracks.map((t) => t.track_id)));
    for (const t of tracks) {
      if (detectionLayerOn) drawTrack(t, mapX, mapY, nowMs);
      // Phase 10 section 5: pose is bound to the person TRACK (persists
      // through a gap in the engine's own per-frame cadence/reuse
      // bookkeeping - results/pose_continuity_raw.md measured that gap
      // vanishing the skeleton on 36.4% of live snapshots), not drawn
      // straight off snap.poses per frame. Independent of the "detection"
      // box-layer toggle, same as the old frame-level pose draw was.
      if (poseLayerOn && t.pose_keypoints && t.pose_keypoints.length) {
        const ageFrac = Math.max(0, Math.min(1, (t.pose_age_ms || 0) / poseMaxAgeMs));
        drawPose({ keypoints: t.pose_keypoints }, mapX, mapY, visThr, ageFrac);
      }
    }
    if (detectionLayerOn && revealSuppressed) {
      for (const d of snap.detections || []) {
        if (d.policy_state === "suppressed_implausible") drawSuppressedDetection(d, mapX, mapY);
      }
    }
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
  const drawEndTs = performance.now();
  runtime.lastPaintMs = drawEndTs - paintT0;
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

  // Phase 8 post-emission investigation: draw() only issues canvas commands -
  // it does not prove the compositor has actually presented them. Schedule a
  // requestAnimationFrame right after the synchronous draw work returns; a rAF
  // callback runs immediately before the browser composites the next frame, so
  // this delta is a standard proxy for "how long until this paint reaches the
  // screen" (not the true compositor cost, which DevTools cannot expose to page
  // script either). Skipped for a stale snapshot (nothing new to compose).
  if (!snap.stale) {
    requestAnimationFrame(() => {
      const compositorMs = performance.now() - drawEndTs;
      runtime.lastCompositorMs = compositorMs;
      runtime.compositorMsSamples.push(compositorMs);
      if (runtime.compositorMsSamples.length > MAX_AGE_SAMPLES) {
        runtime.compositorMsSamples.shift();
      }
    });
  }
}

/** One tracked object. Five independent visual channels (docs/architecture.md):
 * identity=hue, freshness=stroke, uncertainty=label text, tier=badge,
 * staleness=whole-overlay (applied by the caller via ctx.globalAlpha). */
function drawTrack(t, mapX, mapY, nowMs) {
  // Adapt the Track's last-fresh-observation fields into the shape
  // effectiveDetection() already knows how to read (Detection-shaped) - reuses
  // the one place the policy-view toggles and the accepted/secondary/unknown
  // decision logic live, instead of duplicating it here.
  const adapted = {
    raw_class_name: t.observed_class || t.class_name,
    class_name: t.class_name,
    policy_state: t.policy_state,
    tier: t.tier,
  };
  const eff = effectiveDetection(adapted);
  // rejected_size / suppressed_implausible are never tracked at all (the
  // tracker's TRACK_ELIGIBLE_STATES excludes them) - nothing to filter here.

  const kind = stableKind(t.track_id, eff.kind, nowMs);

  const [x1, y1, x2, y2] = t.bbox;
  const px = mapX(x1);
  const py = mapY(y1);
  const pw = mapX(x2) - px;
  const ph = mapY(y2) - py;

  lastBoxes.push({ kind: "track", track: t, x: px, y: py, w: pw, h: ph });
  const selected = selectedTrackId === t.track_id;

  // Identity: hue is a pure function of track_id, independent of class or
  // state, so the SAME object reads as the same colour across frames even
  // while coasting or before recognition settles - and two simultaneously
  // Unknown objects read as visibly different identities.
  const colour = identityColor(t.track_id);
  // Freshness: solid = a detector observation landed this frame; dashed = the
  // box is this cycle's motion prediction (coasting). This is the ONLY thing
  // stroke style communicates now - it no longer doubles as an uncertainty or
  // confidence-band signal (root-cause table: those were two more
  // independent, uncoordinated grey-producing conditions layered on the same
  // dash channel - see docs/decisions.md).
  const dashPattern = t.fresh ? [] : [6, 4];

  const label = labelFor(t, kind);

  ctx.save();
  ctx.strokeStyle = selected ? "#ffd166" : colour;
  ctx.lineWidth = selected ? 3 : t.fresh ? 2.5 : 1.5;
  ctx.setLineDash(dashPattern);
  ctx.strokeRect(px, py, pw, ph);
  ctx.setLineDash([]);

  ctx.font = "12px ui-monospace, monospace";
  const tw = ctx.measureText(label).width + 10;
  const th = 16;
  ctx.fillStyle = selected ? "#ffd166" : colour;
  ctx.globalAlpha *= 0.9;
  ctx.fillRect(px, Math.max(0, py - th), tw, th);
  ctx.fillStyle = "#0b1220";
  ctx.globalAlpha = 1;
  ctx.fillText(label, px + 5, Math.max(11, py - 4));

  // Tier: a small badge, never a colour change - kept visually distinct from
  // the Unknown-uncertainty rendering (which only ever changes label text).
  let badgeStackY = Math.max(0, py - th);
  if (kind === "secondary") {
    const badge = "tier: secondary";
    ctx.font = "10px ui-monospace, monospace";
    const bw = ctx.measureText(badge).width + 8;
    const bh = 13;
    badgeStackY -= bh + 1;
    ctx.fillStyle = TIER_BADGE_BG;
    ctx.globalAlpha = 0.85;
    ctx.fillRect(px, badgeStackY, bw, bh);
    ctx.fillStyle = "#e2e8f0";
    ctx.globalAlpha = 1;
    ctx.fillText(badge, px + 4, badgeStackY + 10);
  }

  // Phase 13 section 3: the additive custom-classifier badge previously drawn
  // here is removed entirely, not just gated - it was rendered straight from
  // the track's custom-classifier name/confidence fields with no confidence
  // floor and no check that the value was compatible with the detector's own class,
  // and a single-output-class classifier's softmax is mathematically forced
  // toward full confidence regardless of the crop (docs/phase-reports/
  // phase13-stage1-inspection.md section 4; docs/decisions.md Phase 13). The
  // underlying fields still exist on the wire contract (frozen, additive)
  // but the live pipeline no longer populates them unless
  // training.classifier_enabled is explicitly turned back on
  // (predictivesense/pipeline/loop.py), and this overlay never renders them
  // even if it is.

  // Confidence bar: the last detector confidence this track actually carries
  // (never fabricated - section 8), dimmer while coasting since it is a
  // carried-forward value, not this frame's.
  const conf = t.last_detector_confidence;
  if (conf != null) {
    const barW = 40;
    ctx.globalAlpha *= t.fresh ? 1.0 : 0.5;
    ctx.fillStyle = "rgba(0,0,0,0.35)";
    ctx.fillRect(px, py, barW, 3);
    ctx.fillStyle = colour;
    ctx.fillRect(px, py, barW * Math.min(1, Math.max(0, conf)), 3);
  }
  ctx.restore();
}

/** The resting label text. Uncertainty (Unknown) and tier are TEXT-ONLY
 * concerns here - colour and stroke are handled entirely by the caller. */
function labelFor(t, kind) {
  if (kind === "unknown") {
    // No former-class annotation, no raw class, no rule name on the main
    // overlay (BLOCK 3.10) - and deliberately no confidence percentage
    // (section 8.4 is about absent confidence; this is a separate, existing
    // design choice to keep "Unknown" textually clean). Full detail in
    // Diagnostics.
    return "Unknown";
  }
  const name = aliasFor(t.class_name || t.observed_class);
  const conf = t.last_detector_confidence;
  if (conf == null) return name;
  const pct = `${(conf * 100).toFixed(0)}%`;
  // Section 8.3: the viewer must always be able to tell whether the number is
  // this frame's or carried forward - and, if carried forward, how old.
  if (t.fresh) return `${name} ${pct}`;
  const ageS = Math.max(0, t.last_detector_confidence_age_ms || 0) / 1000;
  return `${name} ${pct} (${ageS.toFixed(1)}s old)`;
}

/** A revealed `suppressed_implausible` detection (Diagnostics-only toggle).
 * Never tracked - no identity or freshness concept applies, so this keeps the
 * old flat grey/dotted rendering rather than pretending it has either. */
function drawSuppressedDetection(d, mapX, mapY) {
  const [x1, y1, x2, y2] = d.bbox;
  const px = mapX(x1);
  const py = mapY(y1);
  const pw = mapX(x2) - px;
  const ph = mapY(y2) - py;
  lastBoxes.push({ kind: "detection", det: d, x: px, y: py, w: pw, h: ph });

  const raw = d.raw_class_name || d.class_name;
  const resting = `${aliasFor(raw)} · suppressed`;

  ctx.save();
  ctx.strokeStyle = SUPPRESSED_COLOUR;
  ctx.lineWidth = 1.5;
  ctx.globalAlpha *= 0.5;
  ctx.setLineDash([2, 3]);
  ctx.strokeRect(px, py, pw, ph);
  ctx.setLineDash([]);

  ctx.font = "12px ui-monospace, monospace";
  const tw = ctx.measureText(resting).width + 10;
  const th = 16;
  ctx.fillStyle = SUPPRESSED_COLOUR;
  ctx.globalAlpha *= 0.9;
  ctx.fillRect(px, Math.max(0, py - th), tw, th);
  ctx.fillStyle = "#0b1220";
  ctx.globalAlpha = 1;
  ctx.fillText(resting, px + 5, Math.max(11, py - 4));
  ctx.restore();
}

/** Draws one skeleton. `ageFrac` in [0,1] - 0 = just bound this cycle, 1 = at
 * the track's pose_max_age_ms - so emphasis fades continuously as the pose
 * ages, rather than flipping between two fixed looks (section 5.2: "an aged
 * pose must never look like a fresh measurement", and must not blink). */
function drawPose(p, mapX, mapY, visThr, ageFrac = 0) {
  const kp = p.keypoints || [];
  const aged = ageFrac > 0.02;
  ctx.save();
  // Fresh: solid bright green, fading toward dim amber-grey and a dashed
  // stroke as ageFrac approaches 1 (the track's pose_max_age_ms bound).
  const edgeColour = aged ? "#a1a1aa" : "#4ade80";
  const dimFresh = 0.9 - 0.5 * ageFrac;
  const dimLow = 0.25 - 0.15 * ageFrac;
  ctx.strokeStyle = edgeColour;
  ctx.lineWidth = 2;
  if (aged) ctx.setLineDash([5, 4]);
  for (const [a, b] of SKELETON_EDGES) {
    const ka = kp[a];
    const kb = kp[b];
    if (!ka || !kb) continue;
    const va = ka[2] >= visThr;
    const vb = kb[2] >= visThr;
    ctx.globalAlpha = va && vb ? dimFresh : dimLow;
    ctx.beginPath();
    ctx.moveTo(mapX(ka[0]), mapY(ka[1]));
    ctx.lineTo(mapX(kb[0]), mapY(kb[1]));
    ctx.stroke();
  }
  ctx.setLineDash([]);
  for (const k of kp) {
    const vis = k[2] >= visThr;
    ctx.globalAlpha = (vis ? 1 : 0.3) * (1 - 0.5 * ageFrac);
    ctx.fillStyle = vis ? (aged ? "#d4d4d8" : "#bbf7d0") : "#6b7280";
    ctx.beginPath();
    ctx.arc(mapX(k[0]), mapY(k[1]), vis ? 3 : 2, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}
