/* Tiny observable holding UI state.
 *
 * Persisted keys (localStorage): mode, diagnosticsVisible, panelCollapsed,
 * panelWidth, openGroups. Runtime-only keys (never persisted): snapshot, status,
 * note, and the feature-published summary blocks (cameraInfo, videoInfo,
 * datasetInfo).
 *
 * Subscribers are held in a Set and each subscribe() returns an unsubscribe
 * function - there is no unbounded listener list.
 */
"use strict";

const KEY = "ps.ui";

const PERSISTED_KEYS = [
  "mode",
  "diagnosticsVisible",
  "panelCollapsed",
  "panelWidth",
  "openGroups",
];

const DEFAULTS = {
  mode: "realtime", // "realtime" | "recorded" - client-side view state only
  diagnosticsVisible: false,
  panelCollapsed: false,
  panelWidth: null, // px; null = use the config default. Clamped by ui/resizer.js.
  openGroups: { input: true, camera: true, video: true, dataset: true, diagnostics: false },
};

function loadPersisted() {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    const out = {};
    for (const k of PERSISTED_KEYS) {
      if (parsed[k] !== undefined) out[k] = parsed[k];
    }
    if (out.openGroups) out.openGroups = { ...DEFAULTS.openGroups, ...out.openGroups };
    return out;
  } catch {
    return {};
  }
}

const state = {
  ...DEFAULTS,
  ...loadPersisted(),
  snapshot: null,
  status: "ready", // "ready" | "running" | "degraded" | "error"
  note: "",
  cameraInfo: null,
  videoInfo: null,
  datasetInfo: null,
};

const listeners = new Set();

function persist() {
  try {
    const slim = {};
    for (const k of PERSISTED_KEYS) slim[k] = state[k];
    localStorage.setItem(KEY, JSON.stringify(slim));
  } catch {
    /* private mode / disabled storage - ignore, state still lives in memory */
  }
}

function notify() {
  for (const fn of listeners) {
    try {
      fn(state);
    } catch (err) {
      console.error("[store] subscriber threw", err);
    }
  }
}

export const store = {
  get: () => state,

  subscribe(fn) {
    listeners.add(fn);
    return () => listeners.delete(fn);
  },

  setMode(mode) {
    if (mode !== "realtime" && mode !== "recorded") return;
    if (state.mode === mode) return;
    state.mode = mode;
    persist();
    notify();
  },

  setDiagnosticsVisible(value) {
    const v = !!value;
    if (state.diagnosticsVisible === v) return;
    state.diagnosticsVisible = v;
    persist();
    notify();
  },

  setPanelCollapsed(value) {
    const v = !!value;
    if (state.panelCollapsed === v) return;
    state.panelCollapsed = v;
    persist();
    notify();
  },

  togglePanel() {
    this.setPanelCollapsed(!state.panelCollapsed);
  },

  setPanelWidth(px) {
    const v = Number.isFinite(px) ? Math.round(px) : null;
    if (state.panelWidth === v) return;
    state.panelWidth = v;
    persist();
    notify();
  },

  isGroupOpen: (id) => state.openGroups[id] !== false,

  setGroupOpen(id, open) {
    const v = !!open;
    if (state.openGroups[id] === v) return;
    state.openGroups = { ...state.openGroups, [id]: v };
    persist();
    notify();
  },

  toggleGroup(id) {
    this.setGroupOpen(id, !this.isGroupOpen(id));
  },

  setSnapshot(snap) {
    state.snapshot = snap;
    notify();
  },

  setStatus(status) {
    if (state.status === status) return;
    state.status = status;
    notify();
  },

  setNote(text) {
    state.note = String(text ?? "");
    notify();
  },

  setCameraInfo(info) {
    state.cameraInfo = info;
    notify();
  },

  setVideoInfo(info) {
    state.videoInfo = info;
    notify();
  },

  setDatasetInfo(info) {
    state.datasetInfo = info;
    notify();
  },
};
