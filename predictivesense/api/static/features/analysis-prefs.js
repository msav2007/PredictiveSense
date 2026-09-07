/* Per-layer overlay enable state for the Analysis sub-modules.
 *
 * "Switchable at runtime" (BLOCK 3.3): the server-side config flags
 * perception.detection_enabled / pose_enabled decide whether the backend runs
 * each model at all; these toggles decide whether the overlay DRAWS that layer.
 * Each toggle initialises from the matching config flag and is then persisted
 * per-viewer in localStorage. No API route is added.
 */
"use strict";

import { runtime } from "/static/features/runtime.js";

const KEY = "ps.overlay";
const listeners = new Set();
let state = null;

function defaults() {
  const p = runtime.config?.perception || {};
  return {
    detection: p.detection_enabled !== false,
    pose: p.pose_enabled !== false,
  };
}

function load() {
  const base = defaults();
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (typeof parsed.detection === "boolean") base.detection = parsed.detection;
      if (typeof parsed.pose === "boolean") base.pose = parsed.pose;
    }
  } catch {
    /* private mode - fall back to config defaults */
  }
  return base;
}

function ensure() {
  if (state === null) state = load();
  return state;
}

export function isLayerEnabled(id) {
  return !!ensure()[id];
}

export function setLayerEnabled(id, on) {
  const s = ensure();
  if (s[id] === !!on) return;
  s[id] = !!on;
  try {
    localStorage.setItem(KEY, JSON.stringify(s));
  } catch {
    /* ignore */
  }
  for (const fn of listeners) {
    try {
      fn(s);
    } catch {
      /* a bad listener must not break the rest */
    }
  }
}

export function onLayerChange(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}
