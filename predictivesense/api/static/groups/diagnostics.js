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

export function update() {
  const cell = document.getElementById("m-worker_skips");
  if (!cell) return;
  const wm = runtime.workerMetrics || {};
  cell.textContent =
    wm.skipThrottle == null ? "—" : `${wm.skipThrottle} / ${wm.skipBusy} / ${wm.skipBackpressure}`;
}
