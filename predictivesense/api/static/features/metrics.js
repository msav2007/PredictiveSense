/* Metrics: the /ws/state socket that drives the Diagnostics metric rows, plus
 * the browser-measurement block and its POST /api/metrics/browser action.
 *
 * Relocated from app.js verbatim (Phase 1.6). connectStateSocket now also
 * pushes each snapshot into the UI store (so groups can refresh) and maps
 * stale / socket state onto the top-bar status pill. The metric-row ids and the
 * browser-metrics markup are unchanged - they live in the Diagnostics group.
 */
"use strict";

import { runtime, on, MAX_AGE_SAMPLES } from "/static/features/runtime.js";
import { store } from "/static/ui/store.js";
import { note } from "/static/ui/log.js";

const $ = (id) => document.getElementById(id);
const PREVIEW_SIZES = {
  "640x480": { width: 640, height: 480 },
  "1280x720": { width: 1280, height: 720 },
  "1920x1080": { width: 1920, height: 1080 },
};
const REQUESTED_FPS = 30;

export function initMetrics() {
  connectStateSocket();
  wireMetricsSample();
  on("worker-metrics", refreshBrowserMetricsPanel);
  on("sample-refresh", refreshBrowserMetricsPanel);
  on("stream", refreshBrowserMetricsPanel);
}

function connectStateSocket() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${location.host}/ws/state`);
  ws.onopen = () => {
    const dot = $("ws-dot");
    if (dot) dot.classList.add("ok");
  };
  ws.onclose = () => {
    const dot = $("ws-dot");
    if (dot) {
      dot.classList.remove("ok");
      dot.classList.add("bad");
    }
  };
  ws.onerror = () => {
    const dot = $("ws-dot");
    if (dot) dot.classList.add("bad");
    store.setStatus("error");
  };
  ws.onmessage = (ev) => {
    let snap;
    try {
      snap = JSON.parse(ev.data);
    } catch {
      return;
    }
    runtime.lastSnapshot = snap;
    store.setSnapshot(snap);
    if (runtime.owner !== "backend") {
      store.setStatus(snap.stale ? "degraded" : "running");
    }
    const m = snap.metrics || {};
    const show = (key, val, digits = 1) => {
      const el = $(`m-${key}`);
      if (!el) return;
      el.textContent =
        val === undefined || val === null || val < 0 ? "—" : Number(val).toFixed(digits);
    };
    show("capture_fps", m.capture_fps);
    show("analysis_fps", m.analysis_fps);
    show("dropped_analysis_frames", m.dropped_analysis_frames, 0);
    show("drop_rate", m.drop_rate, 3);
    show("mailbox_depth", m.mailbox_depth, 0);
    show("frame_age_ms", snap.frame_age_ms, 1);
    show("decode_ms", m.decode_ms, 2);
    show("ingest_bytes_per_s", m.ingest_bytes_per_s, 0);
    show("reconnects", m.reconnects, 0);
    show("clock_offset_rtt_ms", m.clock_offset_rtt_ms, 2);
    const staleCell = $("m-stale");
    if (staleCell) staleCell.textContent = snap.stale ? "yes" : "no";
    const pvCell = $("m-preview_fps");
    if (pvCell) pvCell.textContent = runtime.previewFps.toFixed(1);

    if (typeof snap.frame_age_ms === "number" && snap.frame_age_ms >= 0 && !snap.stale) {
      runtime.ageSamples.push(snap.frame_age_ms);
      if (runtime.ageSamples.length > MAX_AGE_SAMPLES) runtime.ageSamples.shift();
    }
    refreshBrowserMetricsPanel();
  };
}

function pct(arr, p) {
  if (!arr.length) return 0;
  const s = [...arr].sort((a, b) => a - b);
  const r = (p / 100) * (s.length - 1);
  const lo = Math.floor(r);
  return lo + 1 >= s.length ? s[s.length - 1] : s[lo] + (r - lo) * (s[lo + 1] - s[lo]);
}

function currentBrowserSample() {
  const track = runtime.stream?.getVideoTracks()[0];
  const s = track ? track.getSettings() : {};
  const size = PREVIEW_SIZES[runtime.currentSizeKey] || {};
  const wm = runtime.workerMetrics || {};
  const snap = runtime.lastSnapshot || {};
  const m = snap.metrics || {};
  const lastSwitch = runtime.switchTimes[runtime.switchTimes.length - 1] || null;
  const cap = runtime.config?.capture || {};
  return {
    requested: {
      width: size.width ?? null,
      height: size.height ?? null,
      frameRate: REQUESTED_FPS,
      analysis_fps: cap.analysis_fps ?? null,
      analysis_width: cap.analysis_width ?? null,
      analysis_height: cap.analysis_height ?? null,
      analysis_jpeg_quality: cap.analysis_jpeg_quality ?? null,
    },
    achieved: {
      width: s.width ?? null,
      height: s.height ?? null,
      frameRate: s.frameRate ?? null,
      deviceId: s.deviceId ?? null,
    },
    preview_fps: Number(runtime.previewFps.toFixed(2)),
    analysis_fps: m.analysis_fps ?? null,
    worker_encode_ms_last: wm.encodeMsLast ?? null,
    worker_encode_ms_avg: wm.encodeMsAvg ?? null,
    ws_buffered_amount: wm.bufferedAmount ?? null,
    ws_rtt_ms: wm.wsRttMs ?? null,
    worker_frames_in: wm.framesIn ?? null,
    worker_frames_closed: wm.framesClosed ?? null,
    worker_frames_leaked: wm.leaked ?? null,
    worker_frames_sent: wm.framesSent ?? null,
    worker_skip_throttle: wm.skipThrottle ?? null,
    worker_skip_busy: wm.skipBusy ?? null,
    worker_skip_backpressure: wm.skipBackpressure ?? null,
    backend_decode_ms: m.decode_ms ?? null,
    backend_frame_age_ms_last: typeof snap.frame_age_ms === "number" ? snap.frame_age_ms : null,
    backend_frame_age_ms_p50: Number(pct(runtime.ageSamples, 50).toFixed(1)),
    backend_frame_age_ms_p95: Number(pct(runtime.ageSamples, 95).toFixed(1)),
    backend_drop_rate: m.drop_rate ?? null,
    backend_ingest_bytes_per_s: m.ingest_bytes_per_s ?? null,
    backend_clock_offset_rtt_ms: m.clock_offset_rtt_ms ?? null,
    stale: !!snap.stale,
    last_switch: lastSwitch,
    switch_times: runtime.switchTimes.slice(),
    ts_client_ms: performance.timeOrigin + performance.now(),
  };
}

function refreshBrowserMetricsPanel() {
  const el = $("browser-metrics");
  if (!el) return;
  const smp = currentBrowserSample();
  const rows = [
    ["Requested resolution", `${smp.requested.width}×${smp.requested.height} @ ${smp.requested.frameRate}`],
    [
      "Achieved resolution",
      `${smp.achieved.width ?? "—"}×${smp.achieved.height ?? "—"} @ ${
        smp.achieved.frameRate ? Math.round(smp.achieved.frameRate) : "—"
      }`,
    ],
    ["Preview FPS", smp.preview_fps.toFixed(1)],
    ["Analysis FPS", smp.analysis_fps == null ? "—" : Number(smp.analysis_fps).toFixed(1)],
    ["Worker encode ms", smp.worker_encode_ms_avg == null ? "—" : smp.worker_encode_ms_avg.toFixed(1)],
    ["WS buffered B", smp.ws_buffered_amount == null ? "—" : String(smp.ws_buffered_amount)],
    ["WS RTT ms", smp.ws_rtt_ms == null ? "—" : smp.ws_rtt_ms.toFixed(1)],
    [
      "Frames in / closed",
      smp.worker_frames_in == null
        ? "—"
        : `${smp.worker_frames_in} / ${smp.worker_frames_closed} (leak ${smp.worker_frames_leaked})`,
    ],
    [
      "Worker skips (throttle / busy / backpressure)",
      smp.worker_skip_throttle == null
        ? "—"
        : `${smp.worker_skip_throttle} / ${smp.worker_skip_busy} / ${smp.worker_skip_backpressure}`,
    ],
    ["Backend decode ms", smp.backend_decode_ms == null ? "—" : Number(smp.backend_decode_ms).toFixed(2)],
    ["Frame age p50 / p95", `${smp.backend_frame_age_ms_p50} / ${smp.backend_frame_age_ms_p95}`],
    ["Drop rate", smp.backend_drop_rate == null ? "—" : Number(smp.backend_drop_rate).toFixed(3)],
    [
      "Last switch ms",
      smp.last_switch ? `${smp.last_switch.from} → ${smp.last_switch.to}: ${smp.last_switch.ms}` : "—",
    ],
  ];
  el.innerHTML = rows.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("");
}

function wireMetricsSample() {
  const btn = $("metrics-sample-btn");
  if (!btn) return;
  btn.addEventListener("click", async () => {
    const label = ($("metrics-label").value || "browser").replace(/[^A-Za-z0-9_-]/g, "-");
    const body = { label, sample: currentBrowserSample() };
    btn.disabled = true;
    try {
      const res = await fetch("/api/metrics/browser", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      });
      const out = await res.json();
      if (!res.ok) note(`metrics sample failed: ${res.status} ${JSON.stringify(out)}`);
      else note(`metrics sample #${out.samples} written to ${out.path}`);
    } catch (err) {
      note(`metrics sample error: ${err}`);
    } finally {
      btn.disabled = false;
    }
  });
}
