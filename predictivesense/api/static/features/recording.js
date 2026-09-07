/* Raw research-clip recorder: MediaRecorder over the full-quality preview
 * stream, POSTed to /api/record/upload with scenario metadata. Relocated from
 * app.js verbatim (Phase 1.6). Wires to elements the Dataset group renders and
 * enables itself when a stream goes live ("stream" bus event).
 */
"use strict";

import { runtime, on } from "/static/features/runtime.js";
import { store } from "/static/ui/store.js";
import { note } from "/static/ui/log.js";
import { fmtMB } from "/static/ui/format.js";

const $ = (id) => document.getElementById(id);

export function initRecording() {
  const btn = $("record-btn");
  if (btn) {
    btn.addEventListener("click", () => {
      if (runtime.recorder && runtime.recorder.state === "recording") stopRecording();
      else startRecording();
    });
    btn.disabled = !runtime.stream;
  }
  on("stream", () => {
    const b = $("record-btn");
    if (b && runtime.owner !== "backend") b.disabled = false;
  });
  on("stream-stopped", () => {
    const b = $("record-btn");
    if (b && (!runtime.recorder || runtime.recorder.state !== "recording")) b.disabled = true;
  });
  loadClips();
}

function startRecording() {
  if (!runtime.stream) return;
  const types = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"];
  const mimeType = types.find((t) => MediaRecorder.isTypeSupported(t)) || "";
  let rec;
  try {
    rec = new MediaRecorder(runtime.stream, mimeType ? { mimeType } : undefined);
  } catch (err) {
    note(`MediaRecorder failed: ${err.name}`);
    return;
  }
  runtime.recorder = rec;
  runtime.recChunks = [];
  rec.ondataavailable = (e) => {
    if (e.data && e.data.size) runtime.recChunks.push(e.data);
  };
  rec.onstop = uploadRecording;
  rec.start(1000);

  runtime.recStart = performance.now();
  $("rec-badge").hidden = false;
  const btn = $("record-btn");
  btn.textContent = "Stop recording";
  btn.classList.remove("btn-primary");
  btn.classList.add("btn-danger", "recording");
  store.setDatasetInfo({ ...(store.get().datasetInfo || {}), recording: true });
  runtime.recTimer = setInterval(() => {
    const s = Math.floor((performance.now() - runtime.recStart) / 1000);
    $("rec-elapsed").textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  }, 500);
}

function stopRecording() {
  clearInterval(runtime.recTimer);
  $("rec-badge").hidden = true;
  const btn = $("record-btn");
  btn.textContent = "Record sample";
  btn.classList.remove("btn-danger", "recording");
  btn.classList.add("btn-primary");
  store.setDatasetInfo({ ...(store.get().datasetInfo || {}), recording: false });
  if (runtime.recorder && runtime.recorder.state !== "inactive") runtime.recorder.stop();
}

async function uploadRecording() {
  const blob = new Blob(runtime.recChunks, { type: "video/webm" });
  const track = runtime.stream?.getVideoTracks()[0];
  const s = track ? track.getSettings() : {};
  const form = new FormData();
  form.append("file", blob, "clip.webm");
  form.append("scenario_tag", $("scenario-tag").value || "cabin-unlabelled");
  form.append(
    "device_label",
    track?.label || $("device-select")?.selectedOptions[0]?.textContent || "unknown",
  );
  form.append("width", String(s.width || 0));
  form.append("height", String(s.height || 0));
  form.append("nominal_fps", String(s.frameRate || 0));
  form.append("notes", $("clip-notes").value || "");
  form.append("consent_ack", $("consent-ack").checked ? "true" : "false");

  note(`uploading clip (${fmtMB(blob.size)})…`);
  const res = await fetch("/api/record/upload", { method: "POST", body: form });
  if (!res.ok) {
    note(`clip upload failed: ${res.status} ${await res.text()}`);
    return;
  }
  note("clip stored");
  loadClips();
}

export async function loadClips() {
  const list = $("clip-list");
  if (!list) return;
  const rows = await fetch("/api/clips")
    .then((r) => r.json())
    .catch(() => []);
  list.innerHTML = "";
  store.setDatasetInfo({ ...(store.get().datasetInfo || {}), clips: rows.length });
  if (!rows.length) {
    list.innerHTML = "<li>No clips recorded yet.</li>";
    return;
  }
  for (const c of rows) {
    const li = document.createElement("li");
    const dur = c.duration_s ? `${c.duration_s.toFixed(1)}s` : "?";
    li.textContent = `${c.scenario_tag} · ${dur} · ${fmtMB(c.size_bytes)} · ${
      c.consent_ack ? "consent" : "NO consent"
    }`;
    li.title = li.textContent;
    list.appendChild(li);
  }
}
