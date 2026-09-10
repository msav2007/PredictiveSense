"""Headless capture->overlay-paint latency, with a fake camera (Phase 8).

Playwright drives the real dashboard in headless Chromium with Chromium's fake
media device (`--use-fake-device-for-media-stream`), so the two stages the
server-side loopback cannot see are measured with a real browser in the loop:

  * **worker_encode** - the Web Worker's real `OffscreenCanvas.convertToBlob`
    JPEG encode (not a `cv2.imencode` proxy);
  * **overlay_paint** - `capture (drawImage) -> overlay canvas paint`, computed
    in the page as a pure client-clock delta against `capture_client_ts_ms`
    which the worker stamps and the snapshot echoes back.

It also captures `ws_transit`, `decode`, both single-slot buffer dwells,
`mailbox_dwell`, `snapshot_build` and `ws_out` from the same `stage_*` snapshot
keys, so this file is a full capture->paint attribution for the **fake-camera
headless** condition. Real-camera absolute values, and the OnePlus virtual
camera's transport latency / capture jitter, remain the developer's physical
verification (they cannot come from a recorded clip or a fake device).

    python scripts/benchmark_capture_paint.py --seconds 20 --label fake_camera_headless

Writes results/latency_stages_<label>.{json,md} in the same shape as
scripts/benchmark_latency.py's loopback table. Skips cleanly (exit 0, a clear
message) when Playwright or its Chromium build is absent.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from predictivesense.config.settings import load_config  # noqa: E402
from predictivesense.core.enums import SourceKind  # noqa: E402
from predictivesense.logging_setup import configure_logging, get_logger  # noqa: E402

_LOG = get_logger("predictivesense.scripts.benchmark_capture_paint")

_FAKE_MEDIA_ARGS = [
    "--use-fake-device-for-media-stream",
    "--use-fake-ui-for-media-stream",
    "--autoplay-policy=no-user-gesture-required",
]

_ATTR_STAGES = (
    ("worker_encode", "browser JPEG encode (real OffscreenCanvas)"),
    ("ws_transit", "send -> server receive"),
    ("decode", "cv2.imdecode"),
    ("src_buffer_dwell", "BrowserSource slot dwell"),
    ("producer_handoff", "producer read -> mailbox put"),
    ("mailbox_dwell", "LatestFrameMailbox dwell"),
    ("detector", "detector inference"),
    ("pose", "pose inference"),
    ("policy", "recognition policy"),
    ("snapshot_build", "snapshot assembly"),
    ("ws_out", "emit -> /ws/state send"),
    ("overlay_paint", "capture -> overlay paint (browser, client clock)"),
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _pct(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    r = (p / 100.0) * (len(s) - 1)
    lo = int(r)
    return s[-1] if lo + 1 >= len(s) else s[lo] + (r - lo) * (s[lo + 1] - s[lo])


class _ThreadedUvicorn:
    def __init__(self, app, host, port):
        import uvicorn

        self._server = uvicorn.Server(
            uvicorn.Config(app, host=host, port=port, log_level="warning")
        )
        self._server.install_signal_handlers = lambda: None
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def start(self, timeout=15.0):
        self._thread.start()
        deadline = time.monotonic() + timeout
        while not self._server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("uvicorn did not start")
            time.sleep(0.02)

    def stop(self):
        self._server.should_exit = True
        self._thread.join(timeout=5.0)


# Runs in the page every /ws/state message: keep rolling reservoirs of the
# capture->paint age and each stage_* key, plus the live worker encode ms.
_PAGE_COLLECTOR = r"""
() => {
  window.__ps = { paint: [], stages: {}, workerEnc: [] };
  const push = (k, v) => {
    (window.__ps.stages[k] = window.__ps.stages[k] || []).push(v);
  };
  const orig = window.WebSocket;
  // The app already opens /ws/state; hook the overlay + metrics via runtime.
  const tick = () => {
    try {
      const mod = window.__psRuntime;
      if (mod) {
        const s = mod.runtime.lastSnapshot;
        const wm = mod.runtime.workerMetrics || {};
        if (typeof wm.encodeMsLast === "number" && wm.encodeMsLast > 0) {
          window.__ps.workerEnc.push(wm.encodeMsLast);
        }
        if (s && !s.stale && s.metrics) {
          for (const k of Object.keys(s.metrics)) {
            if (k.startsWith("stage_") && k.endsWith("_ms")) {
              push(k.slice(6, -3), s.metrics[k]);
            }
          }
        }
        const pa = mod.runtime.paintAgeSamples || [];
        window.__ps.paint = pa.slice();
      }
    } catch (e) {}
    setTimeout(tick, 200);
  };
  tick();
}
"""


def _run(seconds: float, label: str, headed: bool) -> int:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("playwright not installed (%r) - skipping capture->paint bench", exc)
        return 0

    from predictivesense.api.app import create_app

    cfg = load_config("dev")
    cfg = cfg.model_copy(
        update={"source": cfg.source.model_copy(update={"kind": SourceKind.BROWSER})}
    )
    port = _free_port()
    app = create_app(cfg, start_loop=True)
    server = _ThreadedUvicorn(app, "127.0.0.1", port)
    server.start()
    base = f"http://127.0.0.1:{port}"

    paint: list[float] = []
    stages: dict[str, list[float]] = {}
    worker_enc: list[float] = []
    try:
        with sync_playwright() as pw:
            try:
                browser = pw.chromium.launch(headless=not headed, args=_FAKE_MEDIA_ARGS)
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("could not launch Chromium (%r) - run `playwright install chromium`", exc)
                return 0
            ctx = browser.new_context(
                permissions=["camera"], viewport={"width": 1280, "height": 800}
            )
            page = ctx.new_page()
            # Expose the runtime module so the collector can read paint samples.
            page.add_init_script(
                "import('/static/features/runtime.js').then(m => { window.__psRuntime = m; });"
            )
            page.goto(base, wait_until="load")
            page.wait_for_timeout(1500)
            page.evaluate(_PAGE_COLLECTOR)
            # Let the pipeline warm and the reservoirs fill.
            page.wait_for_timeout(int(seconds * 1000))
            data = page.evaluate("() => window.__ps")
            paint = [float(x) for x in (data.get("paint") or [])]
            worker_enc = [float(x) for x in (data.get("workerEnc") or [])]
            for k, v in (data.get("stages") or {}).items():
                stages[k] = [float(x) for x in v]
            # Post a browser-metrics sample too, through the real endpoint.
            page.evaluate(
                """async (label) => {
                  const body = { label, sample: { source: 'benchmark_capture_paint',
                    paint_age_p50: null } };
                  try { await fetch('/api/metrics/browser', { method:'POST',
                    headers:{'content-type':'application/json'}, body: JSON.stringify(body) }); }
                  catch (e) {}
                }""",
                label,
            )
            ctx.close()
            browser.close()
    finally:
        server.stop()

    if not paint and not stages:
        _LOG.error("no samples collected - the page may not have started capture")
        return 1

    # worker_encode: prefer the real per-frame worker number over the stage key
    # (the stage key carries whatever the header reported).
    if worker_enc:
        stages["worker_encode"] = worker_enc
    if paint:
        stages["overlay_paint"] = paint

    cap_to_paint_p50 = _pct(paint, 50) if paint else None
    denom = cap_to_paint_p50 or (_pct(stages.get("capture_to_snapshot", []), 50) or 0.0)

    rows = []
    attributed = 0.0
    for key, human in _ATTR_STAGES:
        v = stages.get(key)
        if not v:
            continue
        p50 = _pct(v, 50)
        p95 = _pct(v, 95)
        share = (p50 / denom * 100.0) if denom else 0.0
        if key != "overlay_paint":
            attributed += p50
        rows.append({
            "stage": key, "what": human, "p50_ms": round(p50, 2),
            "p95_ms": round(p95, 2), "pct_of_capture_to_paint": round(share, 1),
            "n": len(v),
        })

    import platform

    import psutil

    payload = {
        "label": label,
        "kind": "fake-camera headless (real browser: getUserMedia fake device, "
                "real Web Worker encode, real overlay paint; NOT a real camera)",
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "logical_cores": psutil.cpu_count(logical=True),
            "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
            "provider_config": cfg.perception.provider,
            "intra_op_threads": cfg.perception.intra_op_threads,
        },
        "capture_to_paint_ms_p50": round(cap_to_paint_p50, 2) if cap_to_paint_p50 else None,
        "capture_to_paint_ms_p95": round(_pct(paint, 95), 2) if paint else None,
        "paint_samples": len(paint),
        "stages": rows,
        "notes": (
            "Fake-camera headless: worker_encode and overlay_paint are REAL "
            "browser numbers; the fake device has no sensor exposure/AF cost so "
            "absolute capture->paint is a floor, not a real-camera figure. The "
            "OnePlus virtual camera's transport latency and capture jitter can "
            "only come from live physical benchmarking (developer)."
        ),
    }
    results_dir = Path(cfg.eval.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"latency_stages_{label}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    fp = payload["machine"]
    L = [
        f"# Latency stage attribution - `{label}`",
        "",
        f"**Kind:** {payload['kind']}",
        "",
        f"**Machine:** {fp['logical_cores']} logical cores · {fp['ram_gb']} GB · "
        f"{fp['platform']} · Python {fp['python']} · provider `{fp['provider_config']}` "
        f"· intra_op {fp['intra_op_threads']}",
        "",
        f"**capture -> overlay paint p50 / p95:** "
        f"**{payload['capture_to_paint_ms_p50']} / {payload['capture_to_paint_ms_p95']} ms** "
        f"({payload['paint_samples']} paint samples)",
        "",
        "| stage | what | p50 ms | p95 ms | % of capture->paint | n |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        L.append(
            f"| `{r['stage']}` | {r['what']} | {r['p50_ms']} | {r['p95_ms']} "
            f"| {r['pct_of_capture_to_paint']}% | {r['n']} |"
        )
    L += ["", payload["notes"], ""]
    (results_dir / f"latency_stages_{label}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    _LOG.info("wrote %s", results_dir / f"latency_stages_{label}.md")
    print("\n".join(L))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--label", default="fake_camera_headless")
    p.add_argument("--headed", action="store_true", help="show the browser window")
    args = p.parse_args(argv)
    configure_logging("INFO")
    return _run(args.seconds, args.label, args.headed)


if __name__ == "__main__":
    sys.exit(main())
