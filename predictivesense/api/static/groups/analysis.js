/* Analysis group (Phase 2 fills the slot reserved in Phase 1.6).
 *
 * A host for the Detection and Pose sub-modules registered through the existing
 * registry (registerAnalysisModule). It renders each sub-module's body once,
 * refreshes their summaries per snapshot, and starts the perception overlay.
 * The shell, the registry and the group taxonomy are unchanged - this only
 * registers into them (constants.js just moved `analysis` out of the
 * still-reserved list).
 */
"use strict";

import { el } from "/static/ui/controls.js";
import { GROUP_ORDER } from "/static/groups/constants.js";
import { registerAnalysisModule, getAnalysisModules } from "/static/ui/registry.js";
import { initOverlay } from "/static/features/overlay.js";
import * as detectionModule from "/static/features/detection.js";
import * as poseModule from "/static/features/pose.js";

registerAnalysisModule(detectionModule);
registerAnalysisModule(poseModule);

export const id = "analysis";
export const title = "Analysis";
export const order = GROUP_ORDER.analysis;
export const modes = ["realtime", "recorded"];
export const view = "normal";

export function summary(state) {
  const s = state.snapshot;
  if (!s) return "Perception idle";
  const nd = (s.detections || []).length;
  const np = (s.poses || []).length;
  if (s.stale) return `stale · ${nd} obj / ${np} pose (last)`;
  return `${nd} object${nd === 1 ? "" : "s"} · ${np} pose${np === 1 ? "" : "s"}`;
}

export function render(body, ctx) {
  for (const mod of getAnalysisModules()) {
    const sumId = `am-sum-${mod.id}`;
    const head = el("div", { class: "analysis-module-head" }, [
      el("span", { class: "analysis-module-title", text: mod.title }),
      el("span", { class: "analysis-module-summary", id: sumId }),
    ]);
    const sub = el("div", { class: "analysis-module-body" });
    body.append(el("div", { class: "analysis-module", "data-module-id": mod.id }, [head, sub]));
    try {
      mod.render(sub, ctx);
    } catch (err) {
      sub.replaceChildren(el("p", { class: "group-error", text: `${mod.title} failed to load.` }));
      console.error(`[analysis:${mod.id}] render threw`, err);
    }
  }
  initOverlay();
}

export function update(state) {
  for (const mod of getAnalysisModules()) {
    const sumEl = document.getElementById(`am-sum-${mod.id}`);
    if (sumEl && typeof mod.summary === "function") {
      try {
        sumEl.textContent = mod.summary(state);
      } catch {
        sumEl.textContent = "—";
      }
    }
    if (typeof mod.update === "function") {
      try {
        mod.update(state);
      } catch (err) {
        console.error(`[analysis:${mod.id}] update threw`, err);
      }
    }
  }
}
