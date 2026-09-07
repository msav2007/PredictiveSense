/* Label maps, user-facing strings, and small formatting helpers.
 *
 * Raw engineering keys are renamed for Normal view here; the Diagnostics group
 * still shows the raw keys alongside. Keeping every rename in one module means a
 * group never hard-codes an internal identifier.
 */
"use strict";

/** Client-side view mode -> top-bar label. */
export const MODE_LABELS = {
  realtime: "Real-time",
  recorded: "Recorded video",
};

/** System status -> pill label. */
export const STATUS_LABELS = {
  ready: "Ready",
  running: "Running",
  degraded: "Degraded",
  error: "Error",
};

/** Replay speed: wire value (sent to POST /api/analyze) -> user-facing label. */
export const REPLAY_MODES = [
  { value: "asfast", label: "Fastest" },
  { value: "realtime", label: "Real-time speed" },
];

/** Shown in the viewport when capture.owner = backend. */
export const PREVIEW_UNAVAILABLE_TEXT = "Live preview unavailable in backend-camera mode.";

/** Raw metric key -> Normal-view label. Diagnostics keeps the raw key too. */
export const METRIC_LABELS = {
  preview_fps: "Preview FPS",
  capture_fps: "Capture FPS",
  analysis_fps: "Analysis FPS",
  dropped_analysis_frames: "Dropped analysis frames",
  drop_rate: "Drop rate",
  mailbox_depth: "Mailbox depth",
  frame_age_ms: "Frame age (ms)",
  decode_ms: "Decode (ms)",
  ingest_bytes_per_s: "Ingest (B/s)",
  reconnects: "Reconnects",
  clock_offset_rtt_ms: "Clock RTT (ms)",
  switch_ms: "Switch (ms)",
  stale: "Stale",
  worker_skips: "Worker skips",
};

export function fmtNum(value, digits = 1) {
  if (value === undefined || value === null || Number.isNaN(value) || value < 0) return "—";
  return Number(value).toFixed(digits);
}

export function fmtMB(bytes) {
  if (!bytes && bytes !== 0) return "—";
  return `${(bytes / 1e6).toFixed(1)} MB`;
}

/** Set text and a full-string tooltip; CSS class truncates with an ellipsis. */
export function ellipsize(el, text) {
  const s = String(text ?? "");
  el.textContent = s;
  el.title = s;
  el.classList.add("ellipsis");
}
