/* Detection - an Analysis sub-module (registerAnalysisModule).
 *
 * Carries: an enable toggle (drives the overlay layer; see analysis-prefs.js),
 * read-only model name / input size / confidence threshold, and a live summary
 * line. The detector itself runs in the backend analysis loop; this module only
 * shows its output and toggles the overlay layer.
 */
"use strict";

import { runtime } from "/static/features/runtime.js";
import { el, settingRow } from "/static/ui/controls.js";
import { fmtNum } from "/static/ui/format.js";
import { isLayerEnabled, setLayerEnabled, onLayerChange } from "/static/features/analysis-prefs.js";

export const id = "detection";
export const title = "Detection";
export const order = 10;

function det(state) {
  return state?.snapshot?.metrics || {};
}

export function summary(state) {
  const s = state?.snapshot;
  if (!s) return "Waiting for analysis";
  const n = (s.detections || []).length;
  const m = det(state);
  const ms = m.detector_ms;
  const on = isLayerEnabled("detection");
  return `${on ? "" : "(overlay off) "}${n} object${n === 1 ? "" : "s"}${
    ms && ms >= 0 ? ` · ${fmtNum(ms)} ms` : ""
  }`;
}

export function render(body) {
  const cfg = runtime.config?.perception?.detector || {};
  const toggle = el("input", { type: "checkbox", id: "detection-enable" });
  toggle.checked = isLayerEnabled("detection");
  toggle.addEventListener("change", () => setLayerEnabled("detection", toggle.checked));
  onLayerChange((s) => {
    toggle.checked = !!s.detection;
  });

  body.append(
    settingRow("Overlay", toggle),
    roRow("Model", baseName(cfg.model_path) || "—"),
    roRow("Input size", cfg.input_size ? String(cfg.input_size) : "—"),
    roRow("Confidence", cfg.default_conf != null ? String(cfg.default_conf) : "—"),
    roRow(
      "Low-confidence band",
      Array.isArray(cfg.low_confidence_band) ? cfg.low_confidence_band.join(" – ") : "—",
    ),
    el("p", { class: "summary-line", id: "am-live-detection", text: "—" }),
  );
}

export function update(state) {
  const line = document.getElementById("am-live-detection");
  if (!line) return;
  const s = state?.snapshot;
  if (!s) {
    line.textContent = "—";
    return;
  }
  const dets = s.detections || [];
  const counts = {};
  for (const d of dets) counts[d.class_name] = (counts[d.class_name] || 0) + 1;
  const parts = Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([k, v]) => `${v}× ${k}`);
  const m = det(state);
  line.textContent =
    (parts.length ? parts.join(", ") : "no detections") +
    (m.detector_ms_p50 && m.detector_ms_p50 >= 0
      ? ` — p50 ${fmtNum(m.detector_ms_p50)} / p95 ${fmtNum(m.detector_ms_p95)} ms`
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
