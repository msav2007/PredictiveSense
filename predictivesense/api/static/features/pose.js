/* Pose - an Analysis sub-module (registerAnalysisModule).
 *
 * Enable toggle (drives the overlay skeleton layer), read-only model name /
 * input size / person-confidence / keypoint-visibility threshold, and a live
 * summary line. The estimator runs in the backend analysis loop.
 */
"use strict";

import { runtime } from "/static/features/runtime.js";
import { el, settingRow } from "/static/ui/controls.js";
import { fmtNum } from "/static/ui/format.js";
import { isLayerEnabled, setLayerEnabled, onLayerChange } from "/static/features/analysis-prefs.js";

export const id = "pose";
export const title = "Pose";
export const order = 20;

export function summary(state) {
  const s = state?.snapshot;
  if (!s) return "Waiting for analysis";
  const n = (s.poses || []).length;
  const ms = s.metrics?.pose_ms;
  const on = isLayerEnabled("pose");
  return `${on ? "" : "(overlay off) "}${n} pose${n === 1 ? "" : "s"}${
    ms && ms >= 0 ? ` · ${fmtNum(ms)} ms` : ""
  }`;
}

export function render(body) {
  const cfg = runtime.config?.perception?.pose || {};
  const toggle = el("input", { type: "checkbox", id: "pose-enable" });
  toggle.checked = isLayerEnabled("pose");
  toggle.addEventListener("change", () => setLayerEnabled("pose", toggle.checked));
  onLayerChange((s) => {
    toggle.checked = !!s.pose;
  });

  body.append(
    settingRow("Overlay", toggle),
    roRow("Model", baseName(cfg.model_path) || "—"),
    roRow("Input size", cfg.input_size ? String(cfg.input_size) : "—"),
    roRow("Person confidence", cfg.conf != null ? String(cfg.conf) : "—"),
    roRow(
      "Keypoint visibility",
      cfg.keypoint_visibility_threshold != null
        ? String(cfg.keypoint_visibility_threshold)
        : "—",
    ),
    el("p", { class: "summary-line", id: "am-live-pose", text: "—" }),
  );
}

export function update(state) {
  const line = document.getElementById("am-live-pose");
  if (!line) return;
  const s = state?.snapshot;
  if (!s) {
    line.textContent = "—";
    return;
  }
  const n = (s.poses || []).length;
  const m = s.metrics || {};
  line.textContent =
    `${n} person${n === 1 ? "" : "s"} with a skeleton` +
    (m.pose_ms_p50 && m.pose_ms_p50 >= 0
      ? ` — p50 ${fmtNum(m.pose_ms_p50)} / p95 ${fmtNum(m.pose_ms_p95)} ms`
      : "");
}

function roRow(label, value) {
  return el("div", { class: "setting-row" }, [
    el("span", { class: "setting-label", text: label }),
    el("span", { class: "readout", text: value }),
  ]);
}

function baseName(p) {
  return typeof p === "string" ? p.split(/[\\/]/).pop() : "";
}
