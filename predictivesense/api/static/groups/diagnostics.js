/* Diagnostics group (both modes, diagnostics view): every engineering metric
 * that used to be on the page, relocated here. Nothing is deleted. The metric
 * row ids (m-*) and the browser-metrics markup match what features/metrics.js
 * writes to. Hidden unless the top-bar Diagnostics toggle is on.
 */
"use strict";

import { runtime } from "/static/features/runtime.js";
import { el, actionButton } from "/static/ui/controls.js";
import { fmtNum } from "/static/ui/format.js";
import { GROUP_ORDER } from "/static/groups/constants.js";
import { initMetrics } from "/static/features/metrics.js";

export const id = "diagnostics";
export const title = "Diagnostics";
export const order = GROUP_ORDER.diagnostics;
export const modes = ["realtime", "recorded"];
export const view = "diagnostics";

export function summary(state) {
  const snap = state.snapshot;
  if (!snap) return "Waiting for data";
  const m = snap.metrics || {};
  return `capture ${fmtNum(m.capture_fps)} FPS · age ${fmtNum(snap.frame_age_ms)} ms · ${
    snap.stale ? "stale" : "live"
  }`;
}

// [raw metric key, Normal-view label]. The raw key is shown next to the label.
const METRICS = [
  ["preview_fps", "Preview FPS"],
  ["capture_fps", "Capture FPS"],
  ["analysis_fps", "Analysis FPS"],
  ["dropped_analysis_frames", "Dropped analysis frames"],
  ["drop_rate", "Drop rate"],
  ["mailbox_depth", "Mailbox depth"],
  ["frame_age_ms", "Frame age (ms)"],
  ["decode_ms", "Decode (ms)"],
  ["ingest_bytes_per_s", "Ingest (B/s)"],
  ["reconnects", "Reconnects"],
  ["clock_offset_rtt_ms", "Clock RTT (ms)"],
  ["switch_ms", "Switch (ms)"],
  ["stale", "Stale"],
];

export function render(body) {
  const grid = el("dl", { class: "metric-grid", id: "metric-grid" });
  for (const [key, label] of METRICS) {
    grid.append(
      el("div", {}, [
        el("dt", {}, [label, el("span", { class: "raw-key", text: ` ${key}` })]),
        el("dd", { id: `m-${key}`, text: "—" }),
      ]),
    );
  }
  body.append(grid);
  body.append(el("p", { class: "note", id: "ingest-note", text: "analysis worker: idle" }));

  // ---- Phase 2 perception ----
  const p = (runtime.config && runtime.config.perception) || {};
  const det = p.detector || {};
  const pos = p.pose || {};
  body.append(el("p", { class: "subhead", text: "Perception" }));
  body.append(
    el("dl", { class: "kv-list" }, [
      kv("provider", String(p.provider ?? "—")),
      kv("detector model", baseName(det.model_path) + " @ " + (det.input_size ?? "—")),
      kv("pose model", baseName(pos.model_path) + " @ " + (pos.input_size ?? "—")),
      kv("pose_every_n", String(p.pose_every_n ?? 1)),
    ]),
  );
  const pgrid = el("dl", { class: "metric-grid", id: "perception-metric-grid" });
  for (const [key, label] of PERCEPTION_METRICS) {
    pgrid.append(
      el("div", {}, [
        el("dt", {}, [label, el("span", { class: "raw-key", text: ` ${key}` })]),
        el("dd", { id: `m-${key}`, text: "—" }),
      ]),
    );
  }
  body.append(pgrid);

  const cap = (runtime.config && runtime.config.capture) || {};
  body.append(
    el("p", { class: "subhead", text: "Provider" }),
    el("dl", { class: "kv-list" }, [
      el("div", {}, [el("dt", { text: "capture.owner" }), el("dd", { id: "owner-badge", text: runtime.owner })]),
      el("div", {}, [el("dt", { text: "device_backend" }), el("dd", { text: String(cap.device_backend ?? "auto") })]),
      el("div", {}, [
        el("dt", { text: "worker skips (throttle / busy / backpressure)" }),
        el("dd", { id: "m-worker_skips", text: "—" }),
      ]),
    ]),
  );

  body.append(el("p", { class: "subhead", text: "Browser measurement" }));
  body.append(el("dl", { class: "metric-grid", id: "browser-metrics" }, [
    el("div", {}, [el("dt", { text: "—" }), el("dd", { text: "waiting for stream" })]),
  ]));
  const labelInput = el("input", { type: "text", id: "metrics-label", value: "browser" });
  const sampleBtn = actionButton("Capture sample → results/", {
    variant: "secondary",
    id: "metrics-sample-btn",
  });
  const labelWrap = el("label", { class: "setting-row" }, [
    el("span", { class: "setting-label", text: "Sample label" }),
    labelInput,
  ]);
  body.append(labelWrap, sampleBtn);

  initMetrics();
}

// [raw key, label]. Populated from StateSnapshot.metrics by update(); -1 -> "—".
const PERCEPTION_METRICS = [
  ["detector_ms_p50", "Detector p50 (ms)"],
  ["detector_ms_p95", "Detector p95 (ms)"],
  ["pose_ms_p50", "Pose p50 (ms)"],
  ["pose_ms_p95", "Pose p95 (ms)"],
  ["perception_ms", "Perception per frame (ms)"],
  ["detections_per_frame", "Detections / frame"],
  ["poses_per_frame", "Poses / frame"],
  ["detector_warmup_ms", "Detector warm-up (ms)"],
  ["pose_warmup_ms", "Pose warm-up (ms)"],
  ["perception_frame_errors", "Perception frame errors"],
];

function kv(label, value) {
  return el("div", {}, [el("dt", { text: label }), el("dd", { text: value })]);
}

function baseName(p) {
  return typeof p === "string" ? p.split(/[\\/]/).pop() : "—";
}

export function update(state) {
  const cell = document.getElementById("m-worker_skips");
  if (cell) {
    const wm = runtime.workerMetrics || {};
    cell.textContent =
      wm.skipThrottle == null
        ? "—"
        : `${wm.skipThrottle} / ${wm.skipBusy} / ${wm.skipBackpressure}`;
  }
  const m = (state && state.snapshot && state.snapshot.metrics) || {};
  for (const [key] of PERCEPTION_METRICS) {
    const dd = document.getElementById(`m-${key}`);
    if (!dd) continue;
    const v = m[key];
    const digits = key.includes("per_frame") || key.includes("errors") ? 2 : 1;
    dd.textContent = v === undefined || v === null || v < 0 ? "—" : Number(v).toFixed(digits);
  }
}
