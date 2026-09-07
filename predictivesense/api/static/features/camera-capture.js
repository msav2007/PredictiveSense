/* Browser-owned capture: getUserMedia -> <video>.srcObject preview, device
 * switching, resolution / frame-rate selection, preview-FPS measurement, and the
 * uncomposited-preview guard.
 *
 * Relocated from app.js essentially verbatim (Phase 1.6). Behavioural deltas,
 * both to decouple modules:
 *   - starting/stopping the analysis Worker is now an event ("stream" /
 *     "stream-stopped") consumed by features/analysis-client.js, instead of a
 *     direct startAnalysisWorker() call.
 *   - the browser-metrics panel refresh is an event ("sample-refresh") consumed
 *     by features/metrics.js.
 * The preview <video> is still only ever assigned srcObject - never read, drawn,
 * or replaced for display. The single permitted read is createImageBitmap() in
 * the rVFC analysis fallback (pumpBitmaps lives in analysis-client.js).
 */
"use strict";

import { runtime, emit } from "/static/features/runtime.js";
import { store } from "/static/ui/store.js";
import { note } from "/static/ui/log.js";
import { PREVIEW_UNAVAILABLE_TEXT } from "/static/ui/format.js";

const $ = (id) => document.getElementById(id);
const DEVICE_KEY = "ps.deviceId";
const SIZE_KEY = "ps.previewSize";

const PREVIEW_SIZES = {
  "640x480": { width: 640, height: 480 },
  "1280x720": { width: 1280, height: 720 },
  "1920x1080": { width: 1920, height: 1080 },
};
const REQUESTED_FPS = 30;

let statusEl = null; // camera-group status indicator, set by attachControls()
let started = false;

function setCamStatus(level, text) {
  if (statusEl && typeof statusEl.set === "function") statusEl.set(level, text);
}

function showViewportMessage(text) {
  const el = $("viewport-msg");
  if (!el) return;
  el.textContent = text;
  el.hidden = !text;
}

function hideViewportMessage() {
  const el = $("viewport-msg");
  if (el) el.hidden = true;
}

/* ---- wiring called once by groups/camera.js ---- */

export function attachControls({ deviceSelect, sizeSelect, status }) {
  statusEl = status || statusEl;

  if (deviceSelect) {
    deviceSelect.addEventListener("change", () => {
      const id = deviceSelect.value;
      try {
        localStorage.setItem(DEVICE_KEY, id);
      } catch {
        /* ignore */
      }
      runtime.switchStartMs = performance.now();
      runtime.switchFrom = deviceLabel(runtime.currentDeviceId);
      openStream(id);
    });
  }

  if (sizeSelect) wirePreviewSize(sizeSelect);
}

/* ---- lifecycle ---- */

export async function initCameraCapture() {
  if (runtime.owner === "backend") {
    setCamStatus("error", "Backend owns the camera");
    showViewportMessage(PREVIEW_UNAVAILABLE_TEXT);
    const ds = $("device-select");
    const ps = $("preview-size");
    if (ds) ds.disabled = true;
    if (ps) ps.disabled = true;
    note("preview + recording disabled: capture.owner = backend");
    store.setCameraInfo({ label: "Backend camera", width: 0, height: 0, fps: 0, unavailable: true });
    return;
  }
  if (store.get().mode === "realtime") await startBrowserCapture();
}

/** Leaving Real-time: stop every track, drop the worker, release the device. */
export function releaseCapture() {
  stopCurrentStream();
  hideViewportMessage();
  setCamStatus("", "Released");
}

/** Returning to Real-time: re-acquire the last device. */
export async function resumeCapture() {
  if (runtime.owner === "backend") return;
  hideViewportMessage();
  if (started) await openStream($("device-select")?.value || runtime.currentDeviceId || "");
  else await startBrowserCapture();
}

async function startBrowserCapture() {
  started = true;
  setCamStatus("degraded", "Requesting permission…");
  try {
    const probe = await navigator.mediaDevices.getUserMedia({ video: true });
    probe.getTracks().forEach((t) => t.stop());
  } catch (err) {
    setCamStatus("error", `Permission denied (${err.name})`);
    showViewportMessage("Camera permission denied. Allow camera access, then reload.");
    note(`camera permission denied: ${err.name}`);
    return;
  }

  await populateDevices();
  navigator.mediaDevices.addEventListener?.("devicechange", onDeviceChange);

  let savedSize = null;
  try {
    savedSize = localStorage.getItem(SIZE_KEY);
  } catch {
    /* ignore */
  }
  if (savedSize && PREVIEW_SIZES[savedSize]) {
    runtime.currentSizeKey = savedSize;
    const ps = $("preview-size");
    if (ps) ps.value = savedSize;
  }
  let saved = "";
  try {
    saved = localStorage.getItem(DEVICE_KEY) || "";
  } catch {
    /* ignore */
  }
  const ds = $("device-select");
  if (saved && ds) ds.value = saved;
  await openStream(ds ? ds.value : "");
}

function deviceLabel(id) {
  const sel = $("device-select");
  if (sel) for (const opt of sel.options) if (opt.value === id) return opt.textContent;
  return id ? "selected" : "default";
}

async function populateDevices() {
  const devices = (await navigator.mediaDevices.enumerateDevices()).filter(
    (d) => d.kind === "videoinput",
  );
  const sel = $("device-select");
  if (!sel) return;
  const current = sel.value;
  sel.innerHTML = "";
  devices.forEach((d, i) => {
    const opt = document.createElement("option");
    opt.value = d.deviceId;
    opt.textContent = d.label || `Camera ${i}`;
    opt.title = opt.textContent;
    sel.appendChild(opt);
  });
  if (!devices.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "No cameras found";
    sel.appendChild(opt);
  }
  if (current && devices.some((d) => d.deviceId === current)) sel.value = current;
}

async function onDeviceChange() {
  await populateDevices();
  const sel = $("device-select");
  const stillThere = sel
    ? Array.from(sel.options).some((o) => o.value === runtime.currentDeviceId)
    : false;
  if (runtime.currentDeviceId && !stillThere) {
    note("active camera disappeared — falling back to default");
    setCamStatus("degraded", "Camera lost — recovering…");
    runtime.switchStartMs = performance.now();
    runtime.switchFrom = "(lost)";
    await openStream("");
  }
}

function stopCurrentStream() {
  emit("stream-stopped");
  const video = $("preview");
  if (runtime.stream) {
    runtime.stream.getTracks().forEach((t) => t.stop());
    runtime.stream = null;
  }
  if (video) video.srcObject = null;
}

async function openStream(deviceId) {
  stopCurrentStream();
  setCamStatus("degraded", "Opening…");

  const size = PREVIEW_SIZES[runtime.currentSizeKey] || PREVIEW_SIZES["1280x720"];
  const ideal = {
    width: { ideal: size.width },
    height: { ideal: size.height },
    frameRate: { ideal: REQUESTED_FPS },
  };
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: deviceId ? { deviceId: { exact: deviceId }, ...ideal } : ideal,
      audio: false,
    });
  } catch (err) {
    note(`getUserMedia(${deviceId ? "selected" : "default"}) failed: ${err.name}; trying default`);
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: ideal, audio: false });
    } catch (err2) {
      setCamStatus("error", `No camera available (${err2.name})`);
      showViewportMessage("No camera available. Connect a camera and reload.");
      note(`no camera available: ${err2.name}`);
      populateDevices();
      return;
    }
  }
  runtime.stream = stream;
  runtime.currentDeviceId = deviceId;

  const video = $("preview");
  if (video) {
    video.srcObject = stream; // the only thing we ever do with <video> for display
    measurePreviewFps(video);
    markSwitchComplete(video);
  }
  hideViewportMessage();
  populateDevices();

  const track = stream.getVideoTracks()[0];
  const s = track ? track.getSettings() : {};
  setCamStatus("running", `${s.width || "?"}×${s.height || "?"} @ ${Math.round(s.frameRate || 0)} FPS`);
  publishCameraInfo(s);
  note(
    `preview: requested ${size.width}x${size.height}@${REQUESTED_FPS} → got ` +
      `${s.width || "?"}x${s.height || "?"} @ ${Math.round(s.frameRate || 0)} fps`,
  );

  emit("stream", { stream });
  emit("sample-refresh");
}

function publishCameraInfo(settings) {
  const sel = $("device-select");
  const label =
    sel?.selectedOptions?.[0]?.textContent ||
    runtime.stream?.getVideoTracks()?.[0]?.label ||
    "Camera";
  store.setCameraInfo({
    label,
    width: settings.width || 0,
    height: settings.height || 0,
    fps: settings.frameRate || 0,
  });
}

function wirePreviewSize(sizeSelect) {
  sizeSelect.addEventListener("change", async () => {
    const key = sizeSelect.value;
    if (!PREVIEW_SIZES[key]) return;
    runtime.currentSizeKey = key;
    try {
      localStorage.setItem(SIZE_KEY, key);
    } catch {
      /* ignore */
    }
    const size = PREVIEW_SIZES[key];
    const track = runtime.stream?.getVideoTracks()[0];
    if (!track) {
      await openStream(runtime.currentDeviceId);
      return;
    }
    runtime.switchStartMs = performance.now();
    runtime.switchFrom = `${key} (applyConstraints)`;
    try {
      await track.applyConstraints({
        width: { ideal: size.width },
        height: { ideal: size.height },
        frameRate: { ideal: REQUESTED_FPS },
      });
      markSwitchComplete($("preview"));
      const s = track.getSettings();
      setCamStatus("running", `${s.width}×${s.height} @ ${Math.round(s.frameRate || 0)} FPS`);
      publishCameraInfo(s);
      note(`applyConstraints ${key}: got ${s.width}x${s.height} @ ${Math.round(s.frameRate || 0)} fps`);
      emit("sample-refresh");
    } catch (err) {
      note(`applyConstraints failed (${err.name}); re-acquiring`);
      await openStream(runtime.currentDeviceId);
    }
  });
}

function markSwitchComplete(video) {
  if (!video || !runtime.switchStartMs) return;
  if (!("requestVideoFrameCallback" in HTMLVideoElement.prototype)) {
    runtime.switchStartMs = 0;
    return;
  }
  const startedAt = runtime.switchStartMs;
  const from = runtime.switchFrom;
  runtime.switchStartMs = 0;
  video.requestVideoFrameCallback(() => {
    const ms = performance.now() - startedAt;
    const to = deviceLabel(runtime.currentDeviceId);
    runtime.switchTimes.push({ from, to, ms: Math.round(ms) });
    if (runtime.switchTimes.length > 12) runtime.switchTimes.shift();
    const cell = $("m-switch_ms");
    if (cell) cell.textContent = Math.round(ms).toString();
    note(`switch ${from} → ${to}: ${Math.round(ms)} ms to first frame`);
    emit("sample-refresh");
  });
}

function measurePreviewFps(video) {
  if (!("requestVideoFrameCallback" in HTMLVideoElement.prototype)) {
    note("requestVideoFrameCallback unavailable: preview fps not measured");
    return;
  }
  runtime.previewFrames = 0;
  let windowStart = performance.now();
  const tick = () => {
    runtime.previewFrames += 1;
    const now = performance.now();
    if (now - windowStart >= 1000) {
      runtime.previewFps = (runtime.previewFrames * 1000) / (now - windowStart);
      runtime.previewFrames = 0;
      windowStart = now;
      const cell = $("m-preview_fps");
      if (cell) cell.textContent = runtime.previewFps.toFixed(1);
    }
    if (video.srcObject) video.requestVideoFrameCallback(tick);
  };
  video.requestVideoFrameCallback(tick);
}

/* Guard: no compositor work on the <video> surface (Phase 1 Requirement 8). */
export function assertPreviewUncomposited() {
  const video = $("preview");
  if (!video || !window.getComputedStyle) return;
  const cs = getComputedStyle(video);
  const offenders = [];
  if (cs.filter && cs.filter !== "none") offenders.push(`filter:${cs.filter}`);
  if (cs.transform && cs.transform !== "none") offenders.push(`transform:${cs.transform}`);
  if (cs.boxShadow && cs.boxShadow !== "none") offenders.push("box-shadow");
  if (cs.animationName && cs.animationName !== "none") offenders.push(`animation:${cs.animationName}`);
  if (offenders.length) {
    console.warn("[preview] compositor work on <video>:", offenders.join(", "));
    note(`preview WARNING: ${offenders.join(", ")}`);
  }
}
