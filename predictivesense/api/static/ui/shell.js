/* Binds the static shell skeleton in index.html: top bar (product name, mode,
 * status pill, Diagnostics toggle, panel-collapse toggle), viewport, and the
 * collapsible right control panel. Owns collapse + diagnostics state via the
 * store; never scrolls the top bar out of view (CSS).
 */
"use strict";

import { store } from "/static/ui/store.js";
import { setDebugEnabled } from "/static/ui/log.js";
import { MODE_LABELS, STATUS_LABELS } from "/static/ui/format.js";

export function mountShell(root) {
  const $ = (id) => root.querySelector(`#${id}`);

  const shell = $("app-shell") || root;
  const modeEl = $("tb-mode");
  const pill = $("status-pill");
  const diagBtn = $("btn-diagnostics");
  const collapseBtn = $("btn-collapse");
  const reopenBtn = $("panel-reopen");
  const viewportMsg = $("viewport-msg");

  diagBtn.addEventListener("click", () => store.setDiagnosticsVisible(!store.get().diagnosticsVisible));
  collapseBtn.addEventListener("click", () => store.togglePanel());
  reopenBtn.addEventListener("click", () => store.setPanelCollapsed(false));

  function render(state) {
    shell.dataset.collapsed = state.panelCollapsed ? "true" : "false";
    collapseBtn.setAttribute("aria-expanded", state.panelCollapsed ? "false" : "true");

    diagBtn.setAttribute("aria-pressed", state.diagnosticsVisible ? "true" : "false");
    diagBtn.classList.toggle("is-on", state.diagnosticsVisible);
    setDebugEnabled(state.diagnosticsVisible);

    modeEl.textContent = MODE_LABELS[state.mode] || "—";

    pill.dataset.level = state.status;
    pill.textContent = STATUS_LABELS[state.status] || "—";
  }

  render(store.get());
  store.subscribe(render);

  return {
    /** Show a full-viewport message (empty state / camera loss / backend mode). */
    showViewportMessage(text) {
      viewportMsg.textContent = text;
      viewportMsg.hidden = !text;
    },
    hideViewportMessage() {
      viewportMsg.hidden = true;
    },
  };
}
