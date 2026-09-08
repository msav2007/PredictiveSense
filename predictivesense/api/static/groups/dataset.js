/* Dataset & recording group (both modes).
 *
 * Two separate workflows, deliberately kept apart (P4 Block 4.9.34):
 *   - Record sample: a scenario *video clip* -> data/raw/ (features/recording.js).
 *   - Object Learning Studio: object *still images* with one box each ->
 *     data/objects/ . It is a separate screen at /studio; entering it stops
 *     monitoring. This group only links to it.
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

  const studioLink = el("a", {
    class: "btn btn-secondary",
    href: "/studio",
    id: "open-studio",
    text: "Open Object Learning Studio →",
  });

  body.append(
    el("p", { class: "subhead", text: "Object Learning" }),
    el("p", {
      class: "setting-hint",
      text:
        "Teach PredictiveSense the objects in this environment by capturing images with one box each. Opening the Studio stops monitoring. Separate from Record sample and stored under data/objects.",
    }),
    studioLink,
    el("p", { class: "subhead", text: "Record sample (scenario clips)" }),
    recordBtn,
    settingRow("Scenario tag", scenario),
    settingRow("Notes", notes),
    consentRow,
    el("p", { class: "subhead", text: "Recorded clips" }),
    clipList,
  );

  initRecording();
}
