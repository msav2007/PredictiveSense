/* Dataset & recording group (both modes): record a research sample, scenario
 * tag, notes, consent, and the recorded-clip list. Logic lives in
 * features/recording.js.
 */
"use strict";

import { el, settingRow, actionButton } from "/static/ui/controls.js";
import { GROUP_ORDER } from "/static/groups/constants.js";
import { runtime } from "/static/features/runtime.js";
import { initRecording } from "/static/features/recording.js";

export const id = "dataset";
export const title = "Dataset & recording";
export const order = GROUP_ORDER.dataset;
export const modes = ["realtime", "recorded"];
export const view = "normal";

export function summary(state) {
  const d = state.datasetInfo;
  if (d && d.recording) return "Recording…";
  const n = d && typeof d.clips === "number" ? d.clips : 0;
  return `${n} clip${n === 1 ? "" : "s"} recorded`;
}

export function render(body) {
  const recordBtn = actionButton("Record sample", { variant: "primary", id: "record-btn" });
  recordBtn.disabled = !runtime.stream;

  const scenario = el("input", { type: "text", id: "scenario-tag", value: "cabin-unlabelled" });
  const notes = el("input", { type: "text", id: "clip-notes", placeholder: "optional" });

  const consent = el("input", { type: "checkbox", id: "consent-ack" });
  const consentRow = el("label", { class: "consent-row" }, [
    consent,
    el("span", { text: "Consent acknowledged for everyone on camera" }),
  ]);

  const clipList = el("ul", { id: "clip-list", class: "line-list" });

  body.append(
    recordBtn,
    settingRow("Scenario tag", scenario),
    settingRow("Notes", notes),
    consentRow,
    el("p", { class: "subhead", text: "Recorded clips" }),
    clipList,
  );

  initRecording();
}
