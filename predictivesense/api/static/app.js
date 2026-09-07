/* PredictiveSense input-layer dashboard.
 *
 * Preview independence is the rule of this page: the <video> element is fed by
 * getUserMedia via srcObject and is NEVER read from, drawn into a canvas for
 * display, or replaced. The ONLY permitted read of <video> is
 * createImageBitmap() in the analysis fallback (pumpBitmaps), used only when
 * MediaStreamTrackProcessor is unavailable. Analysis frames are produced only in
 * a Web Worker (analysis-worker.js), which owns the WS /ws/ingest socket.
 *
 * Phase 1.5: device switching stops every old track and nulls srcObject before
 * acquiring the next device; switch time is measured both directions;
 * applyConstraints() is preferred when only the preview size changes on the same
 * device; a browser-metrics panel samples the live path and can POST a labelled
 * block to /api/metrics/browser.
 */
"use strict";

const $ = (id) => document.getElementById(id);
const DEVICE_KEY = "ps.deviceId";
const SIZE_KEY = "ps.previewSize";

const PREVIEW_SIZES = {
  "640x480": { width: 640, height: 480 },
  "1280x720": { width: 1280, height: 720 },
  "1920x1080": { width: 1920, height: 1080 },
};
const REQUESTED_FPS = 30;

const state = {
  config: null,
  owner: "browser",
  stream: null, // full-quality getUserMedia stream (preview + recorder)
  worker: null,
  workerMetrics: {}, // last {type:"metrics"} block from the worker
  lastSnapshot: null, // last StateSnapshot from /ws/state
  ageSamples: [], // frame_age_ms reservoir for p50/p95 (bounded)
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
  switchTimes: [], // [{from, to, ms}]
};

const MAX_AGE_SAMPLES = 600;

async function main() {
  state.config = await fetch("/api/config").then((r) => r.json());
  state.owner = state.config.capture?.owner ?? "browser";
  $("owner-badge").textContent = `owner: ${state.owner}`;

  connectStateSocket();
  wireRecordingMeta();
  wirePreviewSize();
  wireMetricsSample();
  assertPreviewUncomposited();
  loadVideos();
  loadClips();

  if (state.owner === "backend") {
    $("preview-unavailable").hidden = false;
    $("device-select").disabled = true;
    $("preview-size").disabled = true;
    $("record-btn").title = "backend owns the camera";
    setNote("preview + recording disabled: capture.owner = backend");
    return;
  }
  await startBrowserCapture();
}

/* ---- state socket: drives the metrics strip ---- */

function connectStateSocket() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${location.host}/ws/state`);
  ws.onopen = () => $("ws-dot").classList.add("ok");
  ws.onclose = () => {
    $("ws-dot").classList.remove("ok");
    $("ws-dot").classList.add("bad");
  };
  ws.onerror = () => $("ws-dot").classList.add("bad");
  ws.onmessage = (ev) => {
    let snap;
    try {
      snap = JSON.parse(ev.data);
    } catch {
      return;
    }
    state.lastSnapshot = snap;
    const m = snap.metrics || {};
    const show = (key, val, digits = 1) => {
      const el = $(`m-${key}`);
      if (!el) return;
      el.textContent =
        val === undefined || val === null || val < 0
          ? "—"
          : Number(val).toFixed(digits);
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
    $("m-stale").textContent = snap.stale ? "yes" : "no";
    $("m-preview_fps").textContent = state.previewFps.toFixed(1);

    if (typeof snap.frame_age_ms === "number" && snap.frame_age_ms >= 0 && !snap.stale) {
      state.ageSamples.push(snap.frame_age_ms);
      if (state.ageSamples.length > MAX_AGE_SAMPLES) state.ageSamples.shift();
    }
    refreshBrowserMetricsPanel();
  };
}

/* ---- browser-owned capture: preview + analysis worker + recorder ---- */

async function startBrowserCapture() {
  try {
    // One permission-granting call so enumerateDevices() returns labels.
    const probe = await navigator.mediaDevices.getUserMedia({ video: true });
    probe.getTracks().forEach((t) => t.stop());
  } catch (err) {
    setNote(`camera permission denied: ${err.name}`);
    return;
  }

  await populateDevices();
  $("device-select").addEventListener("change", () => {
    const id = $("device-select").value;
    localStorage.setItem(DEVICE_KEY, id);
    // switch time is measured from this change event to the first frame shown
    state.switchStartMs = performance.now();
    state.switchFrom = deviceLabel(state.currentDeviceId);
    openStream(id);
  });
  navigator.mediaDevices.addEventListener?.("devicechange", onDeviceChange);

  const savedSize = localStorage.getItem(SIZE_KEY);
  if (savedSize && PREVIEW_SIZES[savedSize]) {
    state.currentSizeKey = savedSize;
    $("preview-size").value = savedSize;
  }
  const saved = localStorage.getItem(DEVICE_KEY) || "";
  if (saved) $("device-select").value = saved;
  await openStream($("device-select").value);
}

function deviceLabel(id) {
  const sel = $("device-select");
  for (const opt of sel.options) if (opt.value === id) return opt.textContent;
  return id ? "selected" : "default";
}

async function populateDevices() {
  const devices = (await navigator.mediaDevices.enumerateDevices()).filter(
    (d) => d.kind === "videoinput",
  );
  const sel = $("device-select");
  const current = sel.value;
  sel.innerHTML = "";
  devices.forEach((d, i) => {
    const opt = document.createElement("option");
    opt.value = d.deviceId;
    opt.textContent = d.label || `Camera ${i}`;
    sel.appendChild(opt);
  });
  if (current && devices.some((d) => d.deviceId === current)) sel.value = current;
}

/* Re-enumerate when a device appears/disappears (OnePlus via Phone Link, USB
 * webcam). If the device currently in use vanished, fall back to the default. */
async function onDeviceChange() {
  await populateDevices();
  const stillThere = Array.from($("device-select").options).some(
    (o) => o.value === state.currentDeviceId,
  );
  if (state.currentDeviceId && !stillThere) {
    setNote("active camera disappeared — falling back to default");
    state.switchStartMs = performance.now();
    state.switchFrom = "(lost)";
    await openStream("");
  }
}

function stopCurrentStream() {
  if (state.worker) {
    state.worker.terminate();
    state.worker = null;
  }
  const video = $("preview");
  // Stop every track and drop srcObject BEFORE acquiring the next device —
  // holding the old device is the usual cause of slow/failed switches and of a
  // virtual camera reporting "busy".
  if (state.stream) {
    state.stream.getTracks().forEach((t) => t.stop());
    state.stream = null;
  }
  video.srcObject = null;
}

async function openStream(deviceId) {
  stopCurrentStream();

  const size = PREVIEW_SIZES[state.currentSizeKey] || PREVIEW_SIZES["1280x720"];
  const ideal = {
    width: { ideal: size.width },
    height: { ideal: size.height },
    frameRate: { ideal: REQUESTED_FPS },
  };
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: deviceId ? { deviceId: { exact: deviceId }, ...ideal } : ideal,
      audio: false,
    });
  } catch (err) {
    setNote(
      `getUserMedia(${deviceId ? "selected" : "default"}) failed: ${err.name}; trying default`,
    );
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: ideal, audio: false });
    } catch (err2) {
      setNote(`no camera available: ${err2.name}`);
      populateDevices();
      return;
    }
  }
  state.stream = stream;
  state.currentDeviceId = deviceId;

  const video = $("preview");
  video.srcObject = stream; // the only thing we ever do with <video>
  measurePreviewFps(video);
  markSwitchComplete(video);
  populateDevices(); // labels are only populated once a stream is live

  const track = stream.getVideoTracks()[0];
  const s = track ? track.getSettings() : {};
  setNote(
    `preview: requested ${size.width}x${size.height}@${REQUESTED_FPS} → got ` +
      `${s.width || "?"}x${s.height || "?"} @ ${Math.round(s.frameRate || 0)} fps`,
  );

  $("record-btn").disabled = false;
  startAnalysisWorker(stream);
  refreshBrowserMetricsPanel();
}

/* Preview-size change on the SAME device: applyConstraints() rather than a full
 * re-acquisition (Requirement 12). Device change still goes through openStream. */
function wirePreviewSize() {
  $("preview-size").addEventListener("change", async () => {
    const key = $("preview-size").value;
    if (!PREVIEW_SIZES[key]) return;
    state.currentSizeKey = key;
    localStorage.setItem(SIZE_KEY, key);
    const size = PREVIEW_SIZES[key];
    const track = state.stream?.getVideoTracks()[0];
    if (!track) {
      await openStream(state.currentDeviceId);
      return;
    }
    state.switchStartMs = performance.now();
    state.switchFrom = `${key} (applyConstraints)`;
    try {
      await track.applyConstraints({
        width: { ideal: size.width },
        height: { ideal: size.height },
        frameRate: { ideal: REQUESTED_FPS },
      });
      markSwitchComplete($("preview"));
      const s = track.getSettings();
      setNote(
        `applyConstraints ${key}: got ${s.width}x${s.height} @ ${Math.round(s.frameRate || 0)} fps`,
      );
    } catch (err) {
      setNote(`applyConstraints failed (${err.name}); re-acquiring`);
      await openStream(state.currentDeviceId);
    }
  });
}

function markSwitchComplete(video) {
  if (!state.switchStartMs) return;
  if (!("requestVideoFrameCallback" in HTMLVideoElement.prototype)) {
    state.switchStartMs = 0;
    return;
  }
  const startedAt = state.switchStartMs;
  const from = state.switchFrom;
  state.switchStartMs = 0;
  video.requestVideoFrameCallback(() => {
    const ms = performance.now() - startedAt;
    const to = deviceLabel(state.currentDeviceId);
    state.switchTimes.push({ from, to, ms: Math.round(ms) });
    if (state.switchTimes.length > 12) state.switchTimes.shift();
    $("m-switch_ms").textContent = Math.round(ms).toString();
    setNote(`switch ${from} → ${to}: ${Math.round(ms)} ms to first frame`);
    refreshBrowserMetricsPanel();
  });
}

function measurePreviewFps(video) {
  if (!("requestVideoFrameCallback" in HTMLVideoElement.prototype)) {
    setNote("requestVideoFrameCallback unavailable: preview fps not measured");
    return;
  }
  state.previewFrames = 0;
  let windowStart = performance.now();
  const tick = () => {
    state.previewFrames += 1;
    const now = performance.now();
    if (now - windowStart >= 1000) {
      state.previewFps = (state.previewFrames * 1000) / (now - windowStart);
      state.previewFrames = 0;
      windowStart = now;
      $("m-preview_fps").textContent = state.previewFps.toFixed(1);
    }
    if (video.srcObject) video.requestVideoFrameCallback(tick);
  };
  video.requestVideoFrameCallback(tick);
}

function startAnalysisWorker(stream) {
  if (state.worker) {
    state.worker.terminate();
    state.worker = null;
  }
  const track = stream.getVideoTracks()[0];
  if (!track) return;

  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const cap = state.config.capture || {};
  const worker = new Worker("/static/analysis-worker.js");
  state.worker = worker;
  state.workerMetrics = {};
  worker.onmessage = (ev) => {
    const d = ev.data || {};
    if (d.type === "status") setNote(`analysis worker: ${d.text}`);
    else if (d.type === "metrics") {
      state.workerMetrics = d.metrics || {};
      refreshBrowserMetricsPanel();
    }
  };

  const init = {
    type: "init",
    wsUrl: `${proto}//${location.host}/ws/ingest`,
    fps: cap.analysis_fps ?? 10,
    width: cap.analysis_width ?? 640,
    height: cap.analysis_height ?? 480,
    quality: cap.analysis_jpeg_quality ?? 0.7,
    maxWsBufferedBytes: cap.max_ws_buffered_bytes ?? 1_000_000,
  };

  if ("MediaStreamTrackProcessor" in self) {
    const processor = new MediaStreamTrackProcessor({ track });
    init.readable = processor.readable;
    worker.postMessage(init, [processor.readable]);
  } else {
    // Fallback: main thread samples via rVFC, ships ImageBitmaps to the worker.
    // This createImageBitmap(video) is the ONLY read of <video> in this file.
    worker.postMessage(init);
    pumpBitmaps(worker, init.fps);
  }
}

function pumpBitmaps(worker, fps) {
  const video = $("preview");
  const minGap = 1000 / fps;
  let last = 0;
  const tick = async (now) => {
    if (video.srcObject && now - last >= minGap) {
      last = now;
      try {
        const bmp = await createImageBitmap(video);
        worker.postMessage(
          { type: "bitmap", bitmap: bmp, ts: performance.timeOrigin + now },
          [bmp],
        );
      } catch {
        /* transient; skip this frame */
      }
    }
    if (video.srcObject && state.worker === worker) video.requestVideoFrameCallback(tick);
  };
  if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
    video.requestVideoFrameCallback(tick);
  } else {
    setInterval(() => tick(performance.now()), minGap);
  }
}

/* Guard Requirement 8: no compositor work on the <video> surface. */
function assertPreviewUncomposited() {
  const video = $("preview");
  if (!video || !window.getComputedStyle) return;
  const cs = getComputedStyle(video);
  const offenders = [];
  if (cs.filter && cs.filter !== "none") offenders.push(`filter:${cs.filter}`);
  if (cs.transform && cs.transform !== "none") offenders.push(`transform:${cs.transform}`);
  if (cs.boxShadow && cs.boxShadow !== "none") offenders.push("box-shadow");
  if (cs.animationName && cs.animationName !== "none")
    offenders.push(`animation:${cs.animationName}`);
  if (offenders.length) {
    console.warn("[preview] compositor work on <video>:", offenders.join(", "));
    setNote(`preview WARNING: ${offenders.join(", ")}`);
  }
}

/* ---- browser-metrics panel + POST /api/metrics/browser ---- */

function pct(arr, p) {
  if (!arr.length) return 0;
  const s = [...arr].sort((a, b) => a - b);
  const r = (p / 100) * (s.length - 1);
  const lo = Math.floor(r);
  return lo + 1 >= s.length ? s[s.length - 1] : s[lo] + (r - lo) * (s[lo + 1] - s[lo]);
}

function currentBrowserSample() {
  const track = state.stream?.getVideoTracks()[0];
  const s = track ? track.getSettings() : {};
  const size = PREVIEW_SIZES[state.currentSizeKey] || {};
  const wm = state.workerMetrics || {};
  const snap = state.lastSnapshot || {};
  const m = snap.metrics || {};
  const lastSwitch = state.switchTimes[state.switchTimes.length - 1] || null;
  return {
    requested: {
      width: size.width ?? null,
      height: size.height ?? null,
      frameRate: REQUESTED_FPS,
      analysis_fps: state.config.capture?.analysis_fps ?? null,
      analysis_width: state.config.capture?.analysis_width ?? null,
      analysis_height: state.config.capture?.analysis_height ?? null,
      analysis_jpeg_quality: state.config.capture?.analysis_jpeg_quality ?? null,
    },
    achieved: {
      width: s.width ?? null,
      height: s.height ?? null,
      frameRate: s.frameRate ?? null,
      deviceId: s.deviceId ?? null,
    },
    preview_fps: Number(state.previewFps.toFixed(2)),
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
    backend_frame_age_ms_p50: Number(pct(state.ageSamples, 50).toFixed(1)),
    backend_frame_age_ms_p95: Number(pct(state.ageSamples, 95).toFixed(1)),
    backend_drop_rate: m.drop_rate ?? null,
    backend_ingest_bytes_per_s: m.ingest_bytes_per_s ?? null,
    backend_clock_offset_rtt_ms: m.clock_offset_rtt_ms ?? null,
    stale: !!snap.stale,
    last_switch: lastSwitch,
    switch_times: state.switchTimes.slice(),
    ts_client_ms: performance.timeOrigin + performance.now(),
  };
}

function refreshBrowserMetricsPanel() {
  const el = $("browser-metrics");
  if (!el) return;
  const smp = currentBrowserSample();
  const rows = [
    ["req resolution", `${smp.requested.width}×${smp.requested.height} @ ${smp.requested.frameRate}`],
    [
      "got resolution",
      `${smp.achieved.width ?? "—"}×${smp.achieved.height ?? "—"} @ ${
        smp.achieved.frameRate ? Math.round(smp.achieved.frameRate) : "—"
      }`,
    ],
    ["preview fps", smp.preview_fps.toFixed(1)],
    ["analysis fps", smp.analysis_fps == null ? "—" : Number(smp.analysis_fps).toFixed(1)],
    ["worker encode ms", smp.worker_encode_ms_avg == null ? "—" : smp.worker_encode_ms_avg.toFixed(1)],
    ["ws buffered B", smp.ws_buffered_amount == null ? "—" : String(smp.ws_buffered_amount)],
    ["ws rtt ms", smp.ws_rtt_ms == null ? "—" : smp.ws_rtt_ms.toFixed(1)],
    [
      "frames in/closed",
      smp.worker_frames_in == null
        ? "—"
        : `${smp.worker_frames_in}/${smp.worker_frames_closed} (leak ${smp.worker_frames_leaked})`,
    ],
    [
      "worker skips t/b/p",
      smp.worker_skip_throttle == null
        ? "—"
        : `${smp.worker_skip_throttle}/${smp.worker_skip_busy}/${smp.worker_skip_backpressure}`,
    ],
    ["backend decode ms", smp.backend_decode_ms == null ? "—" : Number(smp.backend_decode_ms).toFixed(2)],
    [
      "frame age p50/p95",
      `${smp.backend_frame_age_ms_p50} / ${smp.backend_frame_age_ms_p95}`,
    ],
    ["drop rate", smp.backend_drop_rate == null ? "—" : Number(smp.backend_drop_rate).toFixed(3)],
    [
      "last switch ms",
      smp.last_switch ? `${smp.last_switch.from} → ${smp.last_switch.to}: ${smp.last_switch.ms}` : "—",
    ],
  ];
  el.innerHTML = rows
    .map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`)
    .join("");
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
      if (!res.ok) {
        setNote(`metrics sample failed: ${res.status} ${JSON.stringify(out)}`);
      } else {
        setNote(`metrics sample #${out.samples} written to ${out.path}`);
      }
    } catch (err) {
      setNote(`metrics sample error: ${err}`);
    } finally {
      btn.disabled = false;
    }
  });
}

/* ---- raw clip recorder (full-quality preview stream) ---- */

function wireRecordingMeta() {
  $("record-btn").addEventListener("click", () => {
    if (state.recorder && state.recorder.state === "recording") stopRecording();
    else startRecording();
  });
}

function startRecording() {
  if (!state.stream) return;
  const types = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"];
  const mimeType = types.find((t) => MediaRecorder.isTypeSupported(t)) || "";
  let rec;
  try {
    rec = new MediaRecorder(state.stream, mimeType ? { mimeType } : undefined);
  } catch (err) {
    setNote(`MediaRecorder failed: ${err.name}`);
    return;
  }
  state.recorder = rec;
  state.recChunks = [];
  rec.ondataavailable = (e) => {
    if (e.data && e.data.size) state.recChunks.push(e.data);
  };
  rec.onstop = uploadRecording;
  rec.start(1000);

  state.recStart = performance.now();
  $("rec-badge").hidden = false;
  $("record-btn").textContent = "Stop recording";
  $("record-btn").classList.add("recording");
  state.recTimer = setInterval(() => {
    const s = Math.floor((performance.now() - state.recStart) / 1000);
    $("rec-elapsed").textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  }, 500);
}

function stopRecording() {
  clearInterval(state.recTimer);
  $("rec-badge").hidden = true;
  $("record-btn").textContent = "Record clip";
  $("record-btn").classList.remove("recording");
  if (state.recorder && state.recorder.state !== "inactive") state.recorder.stop();
}

async function uploadRecording() {
  const blob = new Blob(state.recChunks, { type: "video/webm" });
  const track = state.stream?.getVideoTracks()[0];
  const s = track ? track.getSettings() : {};
  const form = new FormData();
  form.append("file", blob, "clip.webm");
  form.append("scenario_tag", $("scenario-tag").value || "cabin-unlabelled");
  form.append(
    "device_label",
    track?.label || $("device-select").selectedOptions[0]?.textContent || "unknown",
  );
  form.append("width", String(s.width || 0));
  form.append("height", String(s.height || 0));
  form.append("nominal_fps", String(s.frameRate || 0));
  form.append("notes", $("clip-notes").value || "");
  form.append("consent_ack", $("consent-ack").checked ? "true" : "false");

  setNote(`uploading clip (${(blob.size / 1e6).toFixed(1)} MB)…`);
  const res = await fetch("/api/record/upload", { method: "POST", body: form });
  if (!res.ok) {
    setNote(`clip upload failed: ${res.status} ${await res.text()}`);
    return;
  }
  setNote("clip stored");
  loadClips();
}

/* ---- Mode B: recorded video list + analyze ---- */

async function loadVideos() {
  const list = $("video-list");
  const rows = await fetch("/api/videos")
    .then((r) => r.json())
    .catch(() => []);
  list.innerHTML = "";
  if (!rows.length) {
    list.innerHTML = "<li>no files under data/videos/</li>";
    return;
  }
  for (const row of rows) {
    const li = document.createElement("li");
    const label = document.createElement("span");
    label.textContent = `${row.path} · ${(row.size_bytes / 1e6).toFixed(1)} MB`;
    const btn = document.createElement("button");
    btn.textContent = "Analyze";
    btn.addEventListener("click", () => analyzeVideo(row.path, btn));
    li.append(label, btn);
    list.appendChild(li);
  }
}

async function analyzeVideo(path, btn) {
  btn.disabled = true;
  btn.textContent = "…";
  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ path, replay_mode: $("replay-mode").value }),
    });
    const out = $("analyze-out");
    out.hidden = false;
    out.textContent = JSON.stringify(await res.json(), null, 2);
  } finally {
    btn.disabled = false;
    btn.textContent = "Analyze";
  }
}

async function loadClips() {
  const list = $("clip-list");
  const rows = await fetch("/api/clips")
    .then((r) => r.json())
    .catch(() => []);
  list.innerHTML = "";
  if (!rows.length) {
    list.innerHTML = "<li>no clips recorded yet</li>";
    return;
  }
  for (const c of rows) {
    const li = document.createElement("li");
    const dur = c.duration_s ? `${c.duration_s.toFixed(1)}s` : "?";
    li.textContent = `${c.scenario_tag} · ${dur} · ${(c.size_bytes / 1e6).toFixed(1)} MB · ${
      c.consent_ack ? "consent" : "NO consent"
    }`;
    list.appendChild(li);
  }
}

function setNote(text) {
  $("ingest-note").textContent = text;
}

main().catch((err) => setNote(`init failed: ${err}`));
