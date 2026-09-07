/* PredictiveSense analysis worker.
 *
 * Runs off the page's main thread. Samples the camera stream, downscales each
 * sampled frame to the configured analysis size, encodes JPEG, and pushes it to
 * WS /ws/ingest as one binary message:
 *
 *   [ 4-byte BE header length | UTF-8 JSON {client_ts_ms,seq,w,h} | JPEG bytes ]
 *
 * This is a push path for analysis only. It is not the preview and it never
 * polls. Backpressure is handled by dropping frames (no queue) when the socket
 * buffer is not draining.
 */
"use strict";

const cfg = { wsUrl: "", fps: 10, width: 640, height: 480, quality: 0.7 };
let canvas = null;
let ctx = null;
let ws = null;
let ready = false;       // handshake complete
let seq = 0;
let lastSendMs = 0;
let helloSentMs = 0;

function nowMs() {
  return performance.timeOrigin + performance.now();
}

function status(text) {
  self.postMessage({ type: "status", text });
}

function connect() {
  ws = new WebSocket(cfg.wsUrl);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    helloSentMs = nowMs();
    ws.send(JSON.stringify({ type: "hello", client_ts_ms: helloSentMs }));
  };
  ws.onmessage = (ev) => {
    if (ready) return;
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.type === "hello_ack") {
      ws.send(JSON.stringify({ type: "echo", rtt_probe: msg.rtt_probe }));
      ready = true;
      status("streaming");
    }
  };
  ws.onclose = () => { ready = false; status("socket closed"); };
  ws.onerror = () => status("socket error");
}

function frameMessage(bytes) {
  const header = JSON.stringify({
    client_ts_ms: nowMs(),
    seq: seq++,
    w: cfg.width,
    h: cfg.height,
  });
  const headerBytes = new TextEncoder().encode(header);
  const buf = new ArrayBuffer(4 + headerBytes.length + bytes.length);
  const view = new DataView(buf);
  view.setUint32(0, headerBytes.length, false);
  new Uint8Array(buf, 4, headerBytes.length).set(headerBytes);
  new Uint8Array(buf, 4 + headerBytes.length).set(bytes);
  return buf;
}

async function encodeAndSend(source) {
  if (!ready || !ws || ws.readyState !== WebSocket.OPEN) return;
  if (ws.bufferedAmount > 1_000_000) return; // socket not draining: drop frame
  const t = performance.now();
  if (t - lastSendMs < 1000 / cfg.fps) return;
  lastSendMs = t;

  ctx.drawImage(source, 0, 0, cfg.width, cfg.height);
  const blob = await canvas.convertToBlob({ type: "image/jpeg", quality: cfg.quality });
  const bytes = new Uint8Array(await blob.arrayBuffer());
  if (ws.readyState === WebSocket.OPEN) ws.send(frameMessage(bytes));
}

async function pumpReadable(readable) {
  const reader = readable.getReader();
  status("reading MediaStreamTrackProcessor");
  while (true) {
    let chunk;
    try {
      chunk = await reader.read();
    } catch (err) {
      status(`stream read error: ${err}`);
      break;
    }
    if (chunk.done) break;
    const frame = chunk.value; // VideoFrame
    try {
      await encodeAndSend(frame);
    } finally {
      frame.close();
    }
  }
}

self.onmessage = (ev) => {
  const d = ev.data || {};
  if (d.type === "init") {
    cfg.wsUrl = d.wsUrl;
    cfg.fps = d.fps;
    cfg.width = d.width;
    cfg.height = d.height;
    cfg.quality = d.quality;
    canvas = new OffscreenCanvas(cfg.width, cfg.height);
    ctx = canvas.getContext("2d", { alpha: false });
    connect();
    if (d.readable) pumpReadable(d.readable);
    else status("awaiting bitmaps (rVFC fallback)");
    return;
  }
  if (d.type === "bitmap" && d.bitmap) {
    encodeAndSend(d.bitmap).finally(() => d.bitmap.close());
  }
};
