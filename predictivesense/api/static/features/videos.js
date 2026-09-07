/* Recorded-video mode: list files under data/videos/, run the backend
 * RecordedDriver over one via POST /api/analyze, and (client-side only) play a
 * locally-chosen file in the viewport <video> for visual review.
 *
 * Relocated from app.js verbatim (Phase 1.6); loadVideos / analyzeVideo are
 * unchanged. Local-file preview uses URL.createObjectURL and touches no API -
 * mode is client-side view state, analysis still runs through /api/analyze on a
 * server-side path.
 */
"use strict";

import { store } from "/static/ui/store.js";
import { note } from "/static/ui/log.js";
import { fmtMB } from "/static/ui/format.js";

const $ = (id) => document.getElementById(id);
let localObjectUrl = null;

export function initVideos() {
  loadVideos();
  const fileInput = $("local-video");
  if (fileInput) fileInput.addEventListener("change", onLocalFile);
}

function setViewportMessage(text) {
  const el = $("viewport-msg");
  if (!el) return;
  el.textContent = text || "";
  el.hidden = !text;
}

/** Called by app.js when the view mode switches. */
export function showRecordedViewport() {
  const video = $("preview");
  if (video && !video.src) {
    setViewportMessage(
      "Recorded video mode. Load a local file to review it here, or use Analyse below to run the recorded pipeline on a file in data/videos/.",
    );
  }
}

export function clearRecordedViewport() {
  const video = $("preview");
  if (video) {
    video.pause?.();
    video.removeAttribute("src");
    video.removeAttribute("controls");
    video.load?.();
  }
  if (localObjectUrl) {
    URL.revokeObjectURL(localObjectUrl);
    localObjectUrl = null;
  }
  setViewportMessage("");
}

function onLocalFile(ev) {
  const file = ev.target.files && ev.target.files[0];
  if (!file) return;
  const video = $("preview");
  if (!video) return;
  video.srcObject = null;
  if (localObjectUrl) URL.revokeObjectURL(localObjectUrl);
  localObjectUrl = URL.createObjectURL(file);
  video.src = localObjectUrl;
  video.setAttribute("controls", "");
  setViewportMessage("");
  video.onloadedmetadata = () => {
    const dur = Number.isFinite(video.duration) ? `${video.duration.toFixed(1)}s` : "unknown length";
    store.setVideoInfo({ name: file.name, duration: dur, source: "local" });
    const out = $("video-selected");
    if (out) {
      out.textContent = `${file.name} · ${dur}`;
      out.title = file.name;
    }
  };
  video.play().catch(() => {
    /* autoplay may be blocked; controls are visible */
  });
}

async function loadVideos() {
  const list = $("video-list");
  if (!list) return;
  const rows = await fetch("/api/videos")
    .then((r) => r.json())
    .catch(() => []);
  list.innerHTML = "";
  if (!rows.length) {
    list.innerHTML = "<li>No files under data/videos/.</li>";
    return;
  }
  for (const row of rows) {
    const li = document.createElement("li");
    const label = document.createElement("span");
    label.className = "ellipsis";
    label.textContent = `${row.path} · ${fmtMB(row.size_bytes)}`;
    label.title = row.path;
    const btn = document.createElement("button");
    btn.className = "btn btn-primary";
    btn.textContent = "Analyse";
    btn.addEventListener("click", () => analyzeVideo(row.path, btn));
    li.append(label, btn);
    list.appendChild(li);
  }
}

async function analyzeVideo(path, btn) {
  btn.disabled = true;
  btn.textContent = "…";
  store.setVideoInfo({ name: path, duration: store.get().videoInfo?.duration || null, source: "server" });
  const out = $("analyze-out");
  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ path, replay_mode: $("replay-mode").value }),
    });
    const body = await res.json();
    if (out) {
      out.hidden = false;
      out.textContent = JSON.stringify(body, null, 2);
    }
    if (!res.ok) note(`analyse failed: ${res.status}`);
    else note(`analysed ${path}`);
  } catch (err) {
    if (out) {
      out.hidden = false;
      out.textContent = `error: ${err}`;
    }
    note(`analyse error: ${err}`);
  } finally {
    btn.disabled = false;
    btn.textContent = "Analyse";
  }
}
