/* Diagnostics group (both modes, diagnostics view): every engineering metric
 * that used to be on the page, relocated here. Nothing is deleted. The metric
 * row ids (m-*) and the browser-metrics markup match what features/metrics.js
 * writes to. Hidden unless the top-bar Diagnostics toggle is on.
 */
"use strict";

import { runtime, on } from "/static/features/runtime.js";
import { el, actionButton, settingRow } from "/static/ui/controls.js";
import { fmtNum } from "/static/ui/format.js";
import { GROUP_ORDER } from "/static/groups/constants.js";
import { initMetrics } from "/static/features/metrics.js";
import { isPolicyView, setPolicyView, ruleLabel, thresholdFor } from "/static/features/policy.js";

// Detection selected on the overlay (click) - full detail shown below.
let selectedDetection = null;
on("detection-selected", (ev) => {
  selectedDetection = (ev.detail && ev.detail.detection) || null;
  renderSelectedDetail(runtime.lastSnapshot);
});

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

// Phase 8: end-to-end latency stages + reconciling frame counters + pose cadence.
const PHASE8_METRICS = [
  ["stage_capture_to_snapshot_ms", "Capture→snapshot (ms)"],
  ["stage_detector_ms", "Stage: detector (ms)"],
  ["stage_pose_ms", "Stage: pose (ms)"],
  ["stage_mailbox_dwell_ms", "Stage: mailbox dwell (ms)"],
  ["stage_ws_out_ms_p50", "Stage: /ws/state out p50 (ms)"],
  ["frames_decoded", "Frames decoded"],
  ["frames_analysed", "Frames analysed"],
  ["dropped_browser_buffer", "Dropped: browser buffer"],
  ["dropped_mailbox", "Dropped: mailbox overwrite"],
  ["dropped_stale", "Dropped: too old (stale guard)"],
  ["frames_in_flight", "Frames in flight"],
  ["pose_reused", "Pose reused (stale) this frame"],
  ["pose_age_ms", "Reused pose age (ms)"],
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
      kv("provider (config)", String(p.provider ?? "—")),
      // Filled from GET /api/runtime below - the EP ACTUALLY in use, the active
      // model version, and the scheduling / staleness / pose-cadence policy.
      el("div", {}, [el("dt", {}, ["active EP"]), el("dd", { id: "rt-active-ep", text: "…" })]),
      el("div", {}, [el("dt", {}, ["model version"]), el("dd", { id: "rt-model-version", text: "…" })]),
      el("div", {}, [el("dt", {}, ["scheduler"]), el("dd", { id: "rt-scheduler", text: "…" })]),
      el("div", {}, [el("dt", {}, ["max frame age (ms)"]), el("dd", { id: "rt-max-age", text: "…" })]),
      el("div", {}, [el("dt", {}, ["pose cadence"]), el("dd", { id: "rt-pose-cadence", text: "…" })]),
      el("div", {}, [el("dt", {}, ["machine"]), el("dd", { id: "rt-machine", text: "…" })]),
      kv("detector model", baseName(det.model_path) + " @ " + (det.input_size ?? "—")),
      kv("pose model", baseName(pos.model_path) + " @ " + (pos.input_size ?? "—")),
    ]),
  );
  void fetchRuntime();
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

  // ---- Phase 8 latency stages + frame counters + pose cadence ----
  body.append(el("p", { class: "subhead", text: "Latency stages & frame counters (Phase 8)" }));
  const p8grid = el("dl", { class: "metric-grid", id: "phase8-metric-grid" });
  for (const [key, label] of PHASE8_METRICS) {
    p8grid.append(
      el("div", {}, [
        el("dt", {}, [label, el("span", { class: "raw-key", text: ` ${key}` })]),
        el("dd", { id: `m-${key}`, text: "—" }),
      ]),
    );
  }
  body.append(p8grid);

  // ---- Phase 2.5 recognition policy ----
  body.append(el("p", { class: "subhead", text: "Recognition policy" }));
  const polgrid = el("dl", { class: "metric-grid", id: "policy-metric-grid" });
  for (const [key, label] of POLICY_METRICS) {
    polgrid.append(
      el("div", {}, [
        el("dt", {}, [label, el("span", { class: "raw-key", text: ` ${key}` })]),
        el("dd", { id: `m-${key}`, text: "—" }),
      ]),
    );
  }
  body.append(polgrid);

  // Diagnostics-only reveal of `suppressed_implausible` boxes on the overlay
  // (BLOCK 3.8) - never shown on the main overlay.
  const supToggle = el("input", { type: "checkbox", id: "policy-show-suppressed" });
  supToggle.checked = isPolicyView("suppressed");
  supToggle.addEventListener("change", () => setPolicyView("suppressed", supToggle.checked));
  body.append(settingRow("Reveal suppressed (implausible) boxes", supToggle));

  // Full per-detection recognition detail (BLOCK 3.11): decision, raw class,
  // confidence, runner-up, rule that fired, effective threshold. Nothing is
  // deleted - it is relocated here from the overlay label.
  body.append(el("p", { class: "subhead", text: "Recognition detail" }));
  body.append(el("p", { class: "note", text: "Selected detection (click a box on the overlay):" }));
  body.append(el("div", { id: "rd-selected", class: "kv-list" }, [
    el("div", {}, [el("dt", { text: "—" }), el("dd", { text: "none selected" })]),
  ]));
  body.append(el("p", { class: "note", text: "Most recent frame — every detection:" }));
  body.append(el("div", { id: "rd-lastframe", class: "rd-table" }, [el("p", { class: "note", text: "—" })]));

  body.append(el("p", { class: "note", text: "raw → decided (last frame):" }));
  body.append(el("ul", { class: "line-list", id: "policy-raw-decided" }, [
    el("li", { text: "—" }),
  ]));

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

// Recognition policy counters (cumulative) + per-frame cost. Phase 5 six states.
const POLICY_METRICS = [
  ["policy_accepted", "Accepted (cum.)"],
  ["policy_accepted_secondary", "Accepted · secondary tier"],
  ["policy_unknown_low_confidence", "Unknown · low confidence"],
  ["policy_unknown_margin", "Unknown · margin"],
  ["policy_suppressed_implausible", "Suppressed · implausible tier"],
  ["policy_rejected_size", "Rejected · size"],
  ["policy_errors", "Policy errors"],
  ["policy_ms", "Policy cost (ms)"],
  ["policy_ms_p95", "Policy cost p95 (ms)"],
];

function kv(label, value) {
  return el("div", {}, [el("dt", { text: label }), el("dd", { text: value })]);
}

/* Phase 8: the EP actually in use, active model version, and the scheduling /
 * staleness / pose-cadence policy - fetched once on render from GET /api/runtime
 * (the live session, not the config). */
async function fetchRuntime() {
  const set = (id, v) => {
    const dd = document.getElementById(id);
    if (dd) dd.textContent = v == null || v === "" ? "—" : String(v);
  };
  try {
    const r = await fetch("/api/runtime");
    if (!r.ok) throw new Error(String(r.status));
    const d = await r.json();
    const ep = d.active_provider || "—";
    const resolved = d.provider_resolved ? ` (config ${d.requested_provider} → ${d.provider_resolved})` : "";
    set("rt-active-ep", ep + resolved);
    set("rt-model-version", d.model_version);
    set("rt-scheduler", d.scheduler);
    set("rt-max-age", d.max_frame_age_ms);
    set("rt-pose-cadence", d.pose_cadence_resolved || d.pose_cadence);
    const mm = d.machine || {};
    set("rt-machine", `${mm.cpu ?? "?"} · ${mm.logical_cores ?? "?"} cores · ${mm.ram_gb ?? "?"} GB · ORT ${mm.onnxruntime ?? "?"} · EPs ${(d.available_providers || []).join(", ")}`);
  } catch (e) {
    set("rt-active-ep", "unavailable");
  }
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
  for (const [key] of POLICY_METRICS) {
    const dd = document.getElementById(`m-${key}`);
    if (!dd) continue;
    const v = m[key];
    const digits = key.startsWith("policy_ms") ? 3 : 0;
    dd.textContent = v === undefined || v === null || v < 0 ? "—" : Number(v).toFixed(digits);
  }
  for (const [key] of PHASE8_METRICS) {
    const dd = document.getElementById(`m-${key}`);
    if (!dd) continue;
    const v = m[key];
    if (key === "pose_reused") {
      dd.textContent = v === undefined || v === null ? "—" : v >= 1 ? "yes (stale)" : "no";
      continue;
    }
    const digits = key.includes("_ms") ? 1 : 0;
    dd.textContent = v === undefined || v === null || v < 0 ? "—" : Number(v).toFixed(digits);
  }
  const rd = document.getElementById("policy-raw-decided");
  if (rd) {
    const dets = (state && state.snapshot && state.snapshot.detections) || [];
    const changed = dets.filter(
      (d) => (d.policy_state && d.policy_state !== "accepted") ||
             (d.raw_class_name && d.raw_class_name !== d.class_name),
    );
    rd.replaceChildren(
      ...(changed.length
        ? changed.slice(0, 8).map((d) =>
            el("li", {
              text: `${d.raw_class_name || d.class_name} → ${d.class_name} (${d.policy_state || "accepted"})`,
            }),
          )
        : [el("li", { text: "no changes this frame" })]),
    );
  }

  renderSelectedDetail(state && state.snapshot);
  renderRecognitionLastFrame(state && state.snapshot);
}

function iou(a, b) {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]);
  const y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return ua > 0 ? inter / ua : 0;
}

function detailRows(d) {
  const runner = d.runner_up ? `${d.runner_up[0]} ${(d.runner_up[1] * 100).toFixed(0)}%` : "—";
  const decision =
    d.policy_state === "accepted" || d.policy_state === "accepted_secondary"
      ? d.class_name
      : "Unknown / hidden";
  return [
    kv("decision", decision),
    kv("raw class", d.raw_class_name || d.class_name),
    kv("tier", d.tier || "primary"),
    kv("confidence", `${(d.score * 100).toFixed(1)}%`),
    kv("runner-up", runner),
    kv("policy_state", d.policy_state || "accepted"),
    kv("rule that fired", ruleLabel(d.policy_state)),
    kv("effective threshold", String(thresholdFor(d))),
  ];
}

function renderSelectedDetail(snapshot) {
  const host = document.getElementById("rd-selected");
  if (!host) return;
  let d = selectedDetection;
  // re-match the click against the current frame's detections by box overlap
  if (d && snapshot && Array.isArray(snapshot.detections)) {
    let best = null;
    let bestIoU = 0.4;
    for (const cand of snapshot.detections) {
      const s = iou(cand.bbox, d.bbox);
      if (s > bestIoU) {
        bestIoU = s;
        best = cand;
      }
    }
    if (best) d = best;
  }
  if (!d) {
    host.replaceChildren(el("div", {}, [el("dt", { text: "—" }), el("dd", { text: "none selected" })]));
    return;
  }
  host.replaceChildren(...detailRows(d));
}

function renderRecognitionLastFrame(snapshot) {
  const host = document.getElementById("rd-lastframe");
  if (!host) return;
  const dets = (snapshot && snapshot.detections) || [];
  if (!dets.length) {
    host.replaceChildren(el("p", { class: "note", text: "no detections this frame" }));
    return;
  }
  const rows = [
    el("div", { class: "rd-row rd-head" }, [
      el("span", { text: "decision" }),
      el("span", { text: "raw" }),
      el("span", { text: "conf" }),
      el("span", { text: "runner-up" }),
      el("span", { text: "rule" }),
      el("span", { text: "thr" }),
    ]),
  ];
  for (const d of dets.slice(0, 20)) {
    const decision =
      d.policy_state === "accepted" || d.policy_state === "accepted_secondary"
        ? d.class_name
        : "Unknown";
    rows.push(
      el("div", { class: "rd-row" }, [
        el("span", { text: decision }),
        el("span", { text: d.raw_class_name || d.class_name }),
        el("span", { text: `${(d.score * 100).toFixed(0)}%` }),
        el("span", { text: d.runner_up ? `${d.runner_up[0]} ${(d.runner_up[1] * 100).toFixed(0)}%` : "—" }),
        el("span", { text: ruleLabel(d.policy_state) }),
        el("span", { text: String(thresholdFor(d)) }),
      ]),
    );
  }
  host.replaceChildren(...rows);
}
