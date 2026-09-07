/* Camera group (Real-time only): device select, resolution, frame rate, camera
 * status; advanced capture settings behind an "Advanced" disclosure. The camera
 * logic itself lives in features/camera-capture.js - this module only builds the
 * controls and hands them over.
 */
"use strict";

import { runtime } from "/static/features/runtime.js";
import { el, settingRow, statusIndicator } from "/static/ui/controls.js";
import { GROUP_ORDER } from "/static/groups/constants.js";
import { attachControls } from "/static/features/camera-capture.js";

export const id = "camera";
export const title = "Camera";
export const order = GROUP_ORDER.camera;
export const modes = ["realtime"];
export const view = "normal";

export function summary(state) {
  const c = state.cameraInfo;
  if (!c) return "No camera selected";
  if (c.unavailable) return "Backend-owned";
  return `${c.label} · ${c.width}×${c.height} · ${Math.round(c.fps)} FPS`;
}

const RESOLUTIONS = [
  { value: "640x480", label: "640 × 480" },
  { value: "1280x720", label: "1280 × 720" },
  { value: "1920x1080", label: "1920 × 1080" },
];

const ADVANCED_KEYS = [
  ["device_backend", "Backend"],
  ["fourcc", "FOURCC"],
  ["buffer_size", "Buffer size"],
  ["warmup_frames", "Warm-up frames"],
  ["open_timeout_s", "Open timeout (s)"],
  ["reconnect_initial_s", "Reconnect initial (s)"],
  ["reconnect_max_s", "Reconnect max (s)"],
  ["max_ws_buffered_bytes", "WS backpressure ceiling (B)"],
];

export function render(body) {
  const deviceSelect = el("select", { id: "device-select" });
  deviceSelect.append(el("option", { value: "", text: "—" }));

  const sizeSelect = el("select", { id: "preview-size" });
  for (const r of RESOLUTIONS) {
    sizeSelect.append(el("option", { value: r.value, text: r.label, selected: r.value === "1280x720" }));
  }

  const fpsOut = el("output", { id: "cam-fps-read", class: "readout", text: "30 FPS requested" });
  const status = statusIndicator("cam-status");

  body.append(
    settingRow("Camera", deviceSelect, "Long names truncate; hover for the full name"),
    settingRow("Resolution", sizeSelect),
    settingRow("Frame rate", fpsOut),
    settingRow("Status", status),
  );

  const cap = (runtime.config && runtime.config.capture) || {};
  const advBody = el("dl", { class: "kv-list" });
  for (const [key, label] of ADVANCED_KEYS) {
    if (cap[key] === undefined) continue;
    advBody.append(el("div", {}, [el("dt", { text: label }), el("dd", { text: String(cap[key]) })]));
  }
  const advanced = el("details", { class: "advanced" }, [
    el("summary", { class: "btn btn-secondary", text: "Advanced" }),
    advBody,
  ]);
  body.append(advanced);

  attachControls({ deviceSelect, sizeSelect, status });
}

export function update(state) {
  const out = document.getElementById("cam-fps-read");
  if (out && state.cameraInfo && !state.cameraInfo.unavailable) {
    out.textContent = `${Math.round(state.cameraInfo.fps || 0)} FPS · 30 requested`;
  }
}
