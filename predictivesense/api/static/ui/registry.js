/* Holds group definitions, sorts them by `order`, filters by mode and by
 * normal/diagnostics visibility, mounts once and re-renders on state change.
 *
 * The extension contract every later phase uses:
 *
 *   registerGroup({ id, title, order, modes, view, summary, render, update? })
 *   registerAnalysisModule({ id, title, order, summary, render, update? })
 *
 * See docs/architecture.md for a worked "add a group" example.
 */
"use strict";

import { store } from "/static/ui/store.js";
import { createGroup } from "/static/ui/group.js";
import { RESERVED_GROUP_IDS } from "/static/groups/constants.js";

const defs = new Map(); // id -> group def
const analysisModules = new Map(); // id -> analysis sub-module def
const instances = new Map(); // id -> { el, update, setVisible }
const RESERVED = new Set(RESERVED_GROUP_IDS);

const VALID_MODES = new Set(["realtime", "recorded"]);
const VALID_VIEWS = new Set(["normal", "diagnostics"]);

function validate(def, kind) {
  for (const key of ["id", "title", "order"]) {
    if (def[key] === undefined || def[key] === null) {
      throw new Error(`${kind} is missing "${key}"`);
    }
  }
  for (const fn of ["summary", "render"]) {
    if (typeof def[fn] !== "function") throw new Error(`${kind} "${def.id}" needs a ${fn}() function`);
  }
}

export function registerGroup(def) {
  validate(def, "group");
  if (!Array.isArray(def.modes) || def.modes.length === 0 || !def.modes.every((m) => VALID_MODES.has(m))) {
    throw new Error(`group "${def.id}" has invalid modes: ${JSON.stringify(def.modes)}`);
  }
  if (!VALID_VIEWS.has(def.view)) {
    throw new Error(`group "${def.id}" has invalid view: ${def.view}`);
  }
  if (RESERVED.has(def.id)) {
    throw new Error(`group id "${def.id}" is reserved for a later phase and must not be registered`);
  }
  if (defs.has(def.id)) {
    throw new Error(`duplicate group id "${def.id}"`);
  }
  defs.set(def.id, def);
}

export function registerAnalysisModule(def) {
  validate(def, "analysis module");
  if (analysisModules.has(def.id)) {
    throw new Error(`duplicate analysis module id "${def.id}"`);
  }
  analysisModules.set(def.id, def);
}

export function getRegisteredGroups() {
  return [...defs.values()].sort((a, b) => a.order - b.order);
}

export function getAnalysisModules() {
  return [...analysisModules.values()].sort((a, b) => a.order - b.order);
}

export function mount(container, ctx) {
  container.replaceChildren();
  for (const def of getRegisteredGroups()) {
    const inst = createGroup(def, ctx);
    instances.set(def.id, inst);
    container.append(inst.el);
  }
  applyFilters();
}

/** Show/hide each group for the current mode and Diagnostics toggle. */
export function applyFilters() {
  const { mode, diagnosticsVisible } = store.get();
  for (const def of defs.values()) {
    const inst = instances.get(def.id);
    if (!inst) continue;
    const modeOk = def.modes.includes(mode);
    const viewOk = def.view !== "diagnostics" || diagnosticsVisible;
    inst.setVisible(modeOk && viewOk);
  }
}

/** Cheap per-state refresh: update every mounted group's summary + body. */
export function refresh(state) {
  for (const inst of instances.values()) inst.update(state);
}
