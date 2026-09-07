/* Input group: the Real-time / Recorded video mode selector, nothing else.
 * Mode is client-side view state (store), not a backend config switch.
 */
"use strict";

import { store } from "/static/ui/store.js";
import { el } from "/static/ui/controls.js";
import { MODE_LABELS } from "/static/ui/format.js";
import { GROUP_ORDER } from "/static/groups/constants.js";

export const id = "input";
export const title = "Input";
export const order = GROUP_ORDER.input;
export const modes = ["realtime", "recorded"];
export const view = "normal";

export function summary(state) {
  return MODE_LABELS[state.mode] || "—";
}

const OPTIONS = [
  { value: "realtime", label: "Real-time" },
  { value: "recorded", label: "Recorded video" },
];

let radios = [];

export function render(body) {
  const fieldset = el("fieldset", { class: "mode-select" }, [
    el("legend", { class: "sr-only", text: "Input mode" }),
  ]);
  radios = OPTIONS.map((opt) => {
    const input = el("input", {
      type: "radio",
      name: "ps-mode",
      value: opt.value,
      id: `mode-${opt.value}`,
    });
    input.addEventListener("change", () => {
      if (input.checked) store.setMode(opt.value);
    });
    const label = el("label", { class: "mode-option", for: `mode-${opt.value}` }, [
      input,
      el("span", { text: opt.label }),
    ]);
    return { input, label };
  });
  fieldset.append(...radios.map((r) => r.label));
  body.append(fieldset);
  syncRadios(store.get());
}

export function update(state) {
  syncRadios(state);
}

function syncRadios(state) {
  for (const r of radios) r.input.checked = r.input.value === state.mode;
}
