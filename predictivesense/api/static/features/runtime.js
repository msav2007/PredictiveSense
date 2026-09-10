/* Shared capture-runtime state and a tiny event bus for the feature modules.
 *
 * This is the object that used to be the module-level `state` in app.js - the
 * camera, worker, recorder and metrics code is relocated essentially verbatim
 * and still reads/writes these fields. Cross-feature signals go over `bus`:
 *
 *   "stream"          { stream }   a new preview stream is live
 *   "stream-stopped"  {}           the preview stream + worker were torn down
 *   "worker-metrics"  {}           runtime.workerMetrics was refreshed
 */
"use strict";

export const runtime = {
  config: null,
  owner: "browser",
  stream: null, // full-quality getUserMedia stream (preview + recorder)
  worker: null,
  workerMetrics: {}, // last {type:"metrics"} block from the worker
  lastSnapshot: null, // last StateSnapshot from /ws/state
  ageSamples: [], // frame_age_ms reservoir for p50/p95 (bounded)
  paintAgeSamples: [], // Phase 8: capture(drawImage)->overlay-paint age, ms (bounded)
  lastPaintMs: 0, // Phase 8: overlay draw() duration, ms
  recorder: null,
  recChunks: [],
  recStart: 0,
  recTimer: 0,
  previewFrames: 0,
  previewFps: 0,
  currentDeviceId: "",
  currentSizeKey: "1280x720",
  switchStartMs: 0,
  switchFrom: "",
  switchTimes: [], // [{from, to, ms}] - bounded to 12
};

export const MAX_AGE_SAMPLES = 600;

const bus = new EventTarget();

export function emit(type, detail = {}) {
  bus.dispatchEvent(new CustomEvent(type, { detail }));
}

export function on(type, handler) {
  bus.addEventListener(type, handler);
  return () => bus.removeEventListener(type, handler);
}
