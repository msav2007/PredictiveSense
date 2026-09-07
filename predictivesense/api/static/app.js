/* PredictiveSense input-layer dashboard.
 *
 * Preview independence is the rule of this page: the <video> element is fed by
 * getUserMedia via srcObject and is NEVER read from, drawn into a canvas for
 * display, or replaced. Analysis frames are produced only in a Web Worker
 * (analysis-worker.js), which owns the WS /ws/ingest socket. This file just
 * wires devices, metrics, recording and Mode-B controls.
 */
"use strict";

const $ = (id) => document.getElementById(id);
const DEVICE_KEY = "ps.deviceId";

const state = {
  config: null,
  owner: "browser",
  stream: null,        // full-quality getUserMedia stream (preview + recorder)
  worker: null,
  recorder: null,
  recChunks: [],
  recStart: 0,
  recTimer: 0,
  previewFrames: 0,
  previewFps: 0,
};

async function main() {
  state.config = await fetch("/api/config").then((r) => r.json());
  state.owner = state.config.capture?.owner ?? "browser";
  $("owner-badge").textContent = `owner: ${state.owner}`;

  connectStateSocket();
  wireRecordingMeta();
  loadVideos();
  loadClips();

  if (state.owner === "backend") {
    $("preview-unavailable").hidden = false;
    $("device-select").disabled = true;
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
  ws.onclose = () => { $("ws-dot").classList.remove("ok"); $("ws-dot").classList.add("bad"); };
  ws.onerror = () => $("ws-dot").classList.add("bad");
  ws.onmessage = (ev) => {
    let snap;
    try { snap = JSON.parse(ev.data); } catch { return; }
    const m = snap.metrics || {};
    const show = (key, val, digits = 1) => {
      const el = $(`m-${key}`);
      if (!el) return;
      el.textContent = (val === undefined || val === null || val < 0) ? "—" : Number(val).toFixed(digits);
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
    localStorage.setItem(DEVICE_KEY, $("device-select").value);
    openStream($("device-select").value);
  });
  navigator.mediaDevices.addEventListener?.("devicechange", populateDevices);

  const saved = localStorage.getItem(DEVICE_KEY) || "";
  if (saved) $("device-select").value = saved;
  await openStream($("device-select").value);
}

async function populateDevices() {
  const devices = (await navigator.mediaDevices.enumerateDevices())
    .filter((d) => d.kind === "videoinput");
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

async function openStream(deviceId) {
  if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
  const constraints = {
    video: deviceId
      ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
      : { width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  };
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia(constraints);
  } catch (err) {
    setNote(`getUserMedia failed: ${err.name}`);
    return;
  }
  state.stream = stream;

  const video = $("preview");
  video.srcObject = stream;          // the only thing we ever do with <video>
  measurePreviewFps(video);

  $("record-btn").disabled = false;
  startAnalysisWorker(stream);
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
  if (state.worker) { state.worker.terminate(); state.worker = null; }
  const track = stream.getVideoTracks()[0];
  if (!track) return;

  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const cap = state.config.capture || {};
  const worker = new Worker("/static/analysis-worker.js");
  state.worker = worker;
  worker.onmessage = (ev) => {
    const d = ev.data || {};
    if (d.type === "status") setNote(`analysis worker: ${d.text}`);
  };

  const init = {
    type: "init",
    wsUrl: `${proto}//${location.host}/ws/ingest`,
    fps: cap.analysis_fps ?? 10,
    width: cap.analysis_width ?? 640,
    height: cap.analysis_height ?? 480,
    quality: cap.analysis_jpeg_quality ?? 0.7,
  };

  if ("MediaStreamTrackProcessor" in self) {
    const processor = new MediaStreamTrackProcessor({ track });
    init.readable = processor.readable;
    worker.postMessage(init, [processor.readable]);
  } else {
    // Fallback: main thread samples via rVFC, ships ImageBitmaps to the worker.
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
        worker.postMessage({ type: "bitmap", bitmap: bmp, ts: performance.timeOrigin + now }, [bmp]);
      } catch { /* transient; skip this frame */ }
    }
    if (video.srcObject && state.worker === worker) video.requestVideoFrameCallback(tick);
  };
  if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
    video.requestVideoFrameCallback(tick);
  } else {
    setInterval(() => tick(performance.now()), minGap);
  }
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
  rec.ondataavailable = (e) => { if (e.data && e.data.size) state.recChunks.push(e.data); };
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
  form.append("device_label", track?.label || $("device-select").selectedOptions[0]?.textContent || "unknown");
  form.append("width", String(s.width || 0));
  form.append("height", String(s.height || 0));
  form.append("nominal_fps", String(s.frameRate || 0));
  form.append("notes", $("clip-notes").value || "");
  form.append("consent_ack", $("consent-ack").checked ? "true" : "false");

  setNote(`uploading clip (${(blob.size / 1e6).toFixed(1)} MB)…`);
  const res = await fetch("/api/record/upload", { method: "POST", body: form });
  if (!res.ok) { setNote(`clip upload failed: ${res.status} ${await res.text()}`); return; }
  setNote("clip stored");
  loadClips();
}

/* ---- Mode B: recorded video list + analyze ---- */

async function loadVideos() {
  const list = $("video-list");
  const rows = await fetch("/api/videos").then((r) => r.json()).catch(() => []);
  list.innerHTML = "";
  if (!rows.length) { list.innerHTML = "<li>no files under data/videos/</li>"; return; }
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
  const rows = await fetch("/api/clips").then((r) => r.json()).catch(() => []);
  list.innerHTML = "";
  if (!rows.length) { list.innerHTML = "<li>no clips recorded yet</li>"; return; }
  for (const c of rows) {
    const li = document.createElement("li");
    const dur = c.duration_s ? `${c.duration_s.toFixed(1)}s` : "?";
    li.textContent = `${c.scenario_tag} · ${dur} · ${(c.size_bytes / 1e6).toFixed(1)} MB · ${c.consent_ack ? "consent" : "NO consent"}`;
    list.appendChild(li);
  }
}

function setNote(text) { $("ingest-note").textContent = text; }

main().catch((err) => setNote(`init failed: ${err}`));
