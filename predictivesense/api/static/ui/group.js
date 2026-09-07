/* One collapsible group: title, one-line summary, open/close control, body.
 * Identical spacing and behaviour for every group.
 *
 * `render` runs exactly once, when the group is mounted (not lazily on first
 * open) - feature modules wire to the elements it builds at init time. `update`
 * is a cheap per-state refresh that must not rebuild the DOM. `summary` must be
 * a pure function of state.
 */
"use strict";

import { store } from "/static/ui/store.js";
import { el } from "/static/ui/controls.js";

export function createGroup(def, ctx) {
  const bodyId = `group-body-${def.id}`;
  const titleEl = el("span", { class: "group-title", text: def.title });
  const summaryEl = el("span", { class: "group-summary" });
  const caretEl = el("span", { class: "group-caret", "aria-hidden": "true" });

  const head = el(
    "button",
    { class: "group-head", type: "button", "aria-controls": bodyId },
    [titleEl, summaryEl, caretEl],
  );
  const body = el("div", { class: "group-body", id: bodyId, role: "region" });
  const section = el("section", { class: "group", "data-group-id": def.id }, [head, body]);

  let bodyOk = true;
  try {
    def.render(body, ctx);
  } catch (err) {
    bodyOk = false;
    body.replaceChildren(el("p", { class: "group-error", text: `"${def.title}" failed to load.` }));
    console.error(`[group:${def.id}] render threw`, err);
  }

  function applyOpen() {
    const open = store.isGroupOpen(def.id);
    head.setAttribute("aria-expanded", open ? "true" : "false");
    section.classList.toggle("open", open);
    body.hidden = !open;
  }

  head.addEventListener("click", () => store.toggleGroup(def.id));

  function update(state) {
    try {
      summaryEl.textContent = def.summary(state);
    } catch (err) {
      summaryEl.textContent = "—";
      console.error(`[group:${def.id}] summary threw`, err);
    }
    if (bodyOk && typeof def.update === "function") {
      try {
        def.update(state);
      } catch (err) {
        console.error(`[group:${def.id}] update threw`, err);
      }
    }
  }

  function setVisible(visible) {
    section.hidden = !visible;
  }

  applyOpen();
  const unsub = store.subscribe(applyOpen);
  update(store.get());

  return {
    id: def.id,
    el: section,
    update,
    setVisible,
    destroy: () => unsub(),
  };
}
