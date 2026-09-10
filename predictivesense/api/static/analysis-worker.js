/* PredictiveSense analysis worker.
 *
 * Runs off the page's main thread. Samples the camera stream, downscales each
 * sampled frame to the configured analysis size, encodes JPEG, and pushes it to
 * WS /ws/ingest as one binary message:
 *
 *   [ 4-byte BE header length | UTF-8 JSON {client_ts_ms,seq,w,h} | JPEG bytes ]
 *
 * This is a push path for analysis only. It is not the preview and it never
 * polls. Two rules keep it bounded (Phase 1.5):
 *
 *   1. Newest-wins: at most ONE encode+send is in flight. A frame that arrives
 *      while an encode is running is dropped, never queued.
 *   2. Backpressure skip: before sending, if ws.bufferedAmount exceeds
 *      cfg.maxWsBufferedBytes the frame is dropped and counted. This is the
 *      direct fix for frame age creeping upward when the link or consumer lags.
 *
 * Every VideoFrame / ImageBitmap taken from the stream is closed on every path
 * (framesIn === framesClosed is asserted by the metrics report).
 */
"use strict";

const cfg = {
  wsUrl: "",
  fps: 10,
  width: 640,
  height: 480,
  quality: 0.7,
  maxWsBufferedBytes: 1_000_000,
};

let canvas = null;
let ctx = null;
let ws = null;
let ready = false; // handshake complete
let seq = 0;
let lastSendMs = 0;
let helloSentMs = 0;
let wsRttMs = null; // browser-observed hello round trip (approx)

let encodeBusy = false; // newest-wins guard: one encode in flight at a time
let pendingCapMs = 0; // nowMs() at drawImage of the frame currently encoding (Phase 8)

const counters = {
  framesIn: 0, // frames pulled from the stream
  framesClosed: 0, // frames released (must equal framesIn)
  framesSent: 0, // binary messages actually put on the wire
  skipThrottle: 0, // dropped: under the fps interval
  skipBusy: 0, // dropped: an encode was still in flight (newest-wins)
  skipBackpressure: 0, // dropped: ws.bufferedAmount over the ceiling
};
let encMsLast = 0;
let encMsEwma = 0;

function nowMs() {
  return performance.timeOrigin + performance.now();
}

function status(text) {
  self.postMessage({ type: "status", text });
}

function reportMetrics() {
  self.postMessage({
    type: "metrics",
    metrics: {
      ...counters,
      leaked: counters.framesIn - counters.framesClosed,
      seq,
      encodeMsLast: Number(encMsLast.toFixed(2)),
      encodeMsAvg: Number(encMsEwma.toFixed(2)),
      bufferedAmount: ws ? ws.bufferedAmount : 0,
      wsRttMs: wsRttMs === null ? null : Number(wsRttMs.toFixed(2)),
      ready,
    },
  });
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
    try {
      msg = JSON.parse(ev.data);
    } catch {
      return;
    }
    if (msg.type === "hello_ack") {
      wsRttMs = nowMs() - helloSentMs;
      ws.send(JSON.stringify({ type: "echo", rtt_probe: msg.rtt_probe }));
      ready = true;
      status("streaming");
    }
  };
  ws.onclose = () => {
    ready = false;
    status("socket closed");
  };
  ws.onerror = () => status("socket error");
}

function frameMessage(bytes) {
  const header = JSON.stringify({
    client_ts_ms: nowMs(),
    seq: seq++,
    w: cfg.width,
    h: cfg.height,
    // Phase 8 stage attribution: true capture instant (drawImage) and the
    // worker's own JPEG encode duration. Additive; an older server ignores them.
    cap_ts_ms: pendingCapMs || nowMs(),
    enc_ms: Number(encMsLast.toFixed(2)),
  });
  const headerBytes = new TextEncoder().encode(header);
  const buf = new ArrayBuffer(4 + headerBytes.length + bytes.length);
  const view = new DataView(buf);
  view.setUint32(0, headerBytes.length, false);
  new Uint8Array(buf, 4, headerBytes.length).set(headerBytes);
  new Uint8Array(buf, 4 + headerBytes.length).set(bytes);
  return buf;
}

/* Decide whether this sampled frame should be encoded+sent right now.
 * Returns a reason string when the frame is dropped, or null to proceed. */
function dropReason() {
  if (!ready || !ws || ws.readyState !== WebSocket.OPEN) return "notReady";
  const t = performance.now();
  if (t - lastSendMs < 1000 / cfg.fps) return "throttle";
  if (encodeBusy) return "busy"; // newest-wins: never queue behind an encode
  if (ws.bufferedAmount > cfg.maxWsBufferedBytes) return "backpressure";
  return null;
}

/* Draw `source` into the analysis canvas synchronously (so the caller can close
 * the frame immediately), then encode + send asynchronously. */
function handleFrame(source) {
  const reason = dropReason();
  if (reason === "throttle") counters.skipThrottle++;
  else if (reason === "busy") counters.skipBusy++;
  else if (reason === "backpressure") counters.skipBackpressure++;
  if (reason) return;

  lastSendMs = performance.now();
  pendingCapMs = nowMs(); // true capture instant for this frame (Phase 8)
  encodeBusy = true;
  try {
    ctx.drawImage(source, 0, 0, cfg.width, cfg.height);
  } catch (err) {
    encodeBusy = false;
    status(`drawImage failed: ${err}`);
    return;
  }
  void finishEncode();
}

async function finishEncode() {
  const t0 = performance.now();
  try {
    const blob = await canvas.convertToBlob({
      type: "image/jpeg",
      quality: cfg.quality,
    });
    const bytes = new Uint8Array(await blob.arrayBuffer());
    encMsLast = performance.now() - t0;
    encMsEwma = encMsEwma === 0 ? encMsLast : encMsEwma * 0.8 + encMsLast * 0.2;
    if (
      ws &&
      ws.readyState === WebSocket.OPEN &&
      ws.bufferedAmount <= cfg.maxWsBufferedBytes
    ) {
      ws.send(frameMessage(bytes));
      counters.framesSent++;
    } else {
      counters.skipBackpressure++;
    }
  } catch (err) {
    status(`encode error: ${err}`);
  } finally {
    encodeBusy = false;
  }
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
    counters.framesIn++;
    try {
      handleFrame(frame);
    } finally {
      frame.close(); // closed on every path: success, drop, or throw
      counters.framesClosed++;
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
    if (typeof d.maxWsBufferedBytes === "number" && d.maxWsBufferedBytes > 0) {
      cfg.maxWsBufferedBytes = d.maxWsBufferedBytes;
    }
    canvas = new OffscreenCanvas(cfg.width, cfg.height);
    ctx = canvas.getContext("2d", { alpha: false });
    connect();
    setInterval(reportMetrics, 1000);
    if (d.readable) pumpReadable(d.readable);
    else status("awaiting bitmaps (rVFC fallback)");
    return;
  }
  if (d.type === "bitmap" && d.bitmap) {
    counters.framesIn++;
    try {
      handleFrame(d.bitmap);
    } finally {
      d.bitmap.close();
      counters.framesClosed++;
    }
  }
};
