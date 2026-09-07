/* The analysis path: a Web Worker (analysis-worker.js) that owns the
 * WS /ws/ingest socket, plus the rVFC fallback that ships ImageBitmaps to it
 * when MediaStreamTrackProcessor is unavailable.
 *
 * Relocated from app.js verbatim (Phase 1.6). It now starts/stops in response
 * to runtime bus events instead of being called directly by openStream:
 *   "stream"          -> (re)start the worker on the new track
 *   "stream-stopped"  -> terminate the worker
 * Newest-wins and bufferedAmount backpressure still live entirely in the worker
 * and are unchanged.
 */
"use strict";

import { runtime, emit, on } from "/static/features/runtime.js";
import { note } from "/static/ui/log.js";

export function initAnalysisClient() {
  on("stream", (e) => startAnalysisWorker(e.detail.stream));
  on("stream-stopped", stopAnalysis);
}

export function stopAnalysis() {
  if (runtime.worker) {
    runtime.worker.terminate();
    runtime.worker = null;
  }
}

function startAnalysisWorker(stream) {
  stopAnalysis();
  const track = stream.getVideoTracks()[0];
  if (!track) return;

  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const cap = runtime.config.capture || {};
  const worker = new Worker("/static/analysis-worker.js");
  runtime.worker = worker;
  runtime.workerMetrics = {};
  worker.onmessage = (ev) => {
    const d = ev.data || {};
    if (d.type === "status") note(`analysis worker: ${d.text}`);
    else if (d.type === "metrics") {
      runtime.workerMetrics = d.metrics || {};
      emit("worker-metrics");
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
    // The createImageBitmap(video) below is the ONLY read of <video> anywhere.
    worker.postMessage(init);
    pumpBitmaps(worker, init.fps);
  }
}

function pumpBitmaps(worker, fps) {
  const video = document.getElementById("preview");
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
    if (video.srcObject && runtime.worker === worker) video.requestVideoFrameCallback(tick);
  };
  if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
    video.requestVideoFrameCallback(tick);
  } else {
    setInterval(() => tick(performance.now()), minGap);
  }
}
