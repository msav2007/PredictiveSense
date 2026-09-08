/* Operations-panel resize handle (Phase 4, P4 Block 4.8).
 *
 * A draggable handle on the panel's left edge. Width persists through the store
 * (alongside the collapsed flag), double-click resets to default, and the handle
 * is keyboard accessible (role="separator", left/right arrows step, Home resets,
 * ARIA value reported). The drag handler is rAF-throttled and batches its one
 * DOM read (window width) and one DOM write (the --panel-w custom property) so
 * resizing never forces a video re-layout thrash.
 */
"use strict";

import { store } from "/static/ui/store.js";

const STEP_PX = 16;

const DEFAULTS = {
  default_width_px: 380,
  min_width_px: 300,
  max_width_px: 560,
  max_width_frac: 0.4,
};

/** Pure: clamp a desired panel width against the constraints and the window.
 *  - never below `min_width_px`
 *  - never above the smaller of `max_width_px` and `max_width_frac * winWidth`
 *  - the viewport keeps at least (1 - max_width_frac) - 0.05 of the window,
 *    i.e. the panel never pushes the viewport below 45% at the default 40% cap.
 */
export function clampPanelWidth(width, winWidth, cfg = DEFAULTS) {
  const c = { ...DEFAULTS, ...(cfg || {}) };
  const w = Number.isFinite(winWidth) && winWidth > 0 ? winWidth : 1280;
  const desired = Number.isFinite(width) ? width : c.default_width_px;
  const fracCap = Math.floor(c.max_width_frac * w);
  const viewportFloor = Math.floor((1 - c.max_width_frac - 0.05) * w); // 45% at 40% cap
  const hardMax = Math.min(c.max_width_px, fracCap, Math.max(c.min_width_px, w - viewportFloor));
  const lo = Math.min(c.min_width_px, hardMax);
  return Math.round(Math.max(lo, Math.min(desired, hardMax)));
}

/** Resolve the effective config from the served /api/config `ui.panel` block. */
export function panelConfig(raw) {
  return { ...DEFAULTS, ...(raw || {}) };
}

export function mountResizer(shellEl, { panel } = {}) {
  const cfg = panelConfig(panel);
  const handle = shellEl.querySelector("#panel-resizer");
  if (!handle) return { destroy() {} };

  handle.setAttribute("role", "separator");
  handle.setAttribute("aria-orientation", "vertical");
  handle.setAttribute("aria-label", "Resize control panel");
  handle.setAttribute("tabindex", "0");
  handle.setAttribute("aria-valuemin", String(cfg.min_width_px));

  const winW = () => window.innerWidth || document.documentElement.clientWidth || 1280;

  function currentWidth() {
    const stored = store.get().panelWidth;
    return clampPanelWidth(
      Number.isFinite(stored) ? stored : cfg.default_width_px,
      winW(),
      cfg,
    );
  }

  function applyWidth(px, { persist = false } = {}) {
    const w = clampPanelWidth(px, winW(), cfg);
    if (store.get().panelCollapsed) {
      shellEl.style.removeProperty("--panel-w"); // let the collapsed CSS rule win
    } else {
      shellEl.style.setProperty("--panel-w", `${w}px`);
    }
    handle.setAttribute("aria-valuenow", String(w));
    handle.setAttribute("aria-valuemax", String(clampPanelWidth(1e6, winW(), cfg)));
    if (persist) store.setPanelWidth(w);
    return w;
  }

  // --- drag (pointer events, rAF-throttled) ---
  let dragging = false;
  let pendingClientX = 0;
  let rafId = 0;

  function onFrame() {
    rafId = 0;
    // one read (window width, inside clampPanelWidth) + one write (--panel-w).
    const next = winW() - pendingClientX;
    applyWidth(next);
  }

  function onPointerMove(ev) {
    if (!dragging) return;
    pendingClientX = ev.clientX;
    if (!rafId) rafId = requestAnimationFrame(onFrame);
  }

  function endDrag() {
    if (!dragging) return;
    dragging = false;
    document.body.style.removeProperty("user-select");
    document.body.style.removeProperty("cursor");
    window.removeEventListener("pointermove", onPointerMove);
    window.removeEventListener("pointerup", endDrag);
    if (rafId) {
      cancelAnimationFrame(rafId);
      rafId = 0;
    }
    const w = clampPanelWidth(winW() - pendingClientX, winW(), cfg);
    store.setPanelWidth(w);
  }

  function onPointerDown(ev) {
    if (store.get().panelCollapsed) return;
    dragging = true;
    pendingClientX = ev.clientX;
    document.body.style.userSelect = "none";
    document.body.style.cursor = "col-resize";
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", endDrag);
    ev.preventDefault();
  }

  function onDblClick() {
    applyWidth(cfg.default_width_px, { persist: true });
  }

  function onKeyDown(ev) {
    let handled = true;
    const w = currentWidth();
    if (ev.key === "ArrowLeft") applyWidth(w + STEP_PX, { persist: true });
    else if (ev.key === "ArrowRight") applyWidth(w - STEP_PX, { persist: true });
    else if (ev.key === "Home") applyWidth(cfg.default_width_px, { persist: true });
    else handled = false;
    if (handled) ev.preventDefault();
  }

  handle.addEventListener("pointerdown", onPointerDown);
  handle.addEventListener("dblclick", onDblClick);
  handle.addEventListener("keydown", onKeyDown);
  const unsub = store.subscribe(() => applyWidth(currentWidth()));
  window.addEventListener("resize", () => applyWidth(currentWidth()));

  applyWidth(currentWidth());

  return {
    destroy() {
      handle.removeEventListener("pointerdown", onPointerDown);
      handle.removeEventListener("dblclick", onDblClick);
      handle.removeEventListener("keydown", onKeyDown);
      unsub();
    },
  };
}
