/* The single debug logger for the dashboard.
 *
 * Console output is gated by the Diagnostics toggle: nothing is printed unless
 * the operator has turned Diagnostics on. `note()` also routes the message into
 * the UI store so the Diagnostics group can show the latest line. This is the
 * only file in static/ allowed to call console.log.
 */
"use strict";

import { store } from "/static/ui/store.js";

let debugEnabled = false;

export function setDebugEnabled(value) {
  debugEnabled = !!value;
}

export function debug(...args) {
  if (debugEnabled) console.log("[ps]", ...args);
}

/** Human-facing status line. Shown in the Diagnostics group; logged when debug is on. */
export function note(text) {
  store.setNote(text);
  debug(text);
}
