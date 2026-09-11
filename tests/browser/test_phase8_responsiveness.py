"""Phase 8 browser tests 14 and 15.

14: preview FPS is unaffected during a deliberate 3-second analysis stall, and
    the number is recorded.
15: the overlay marks a stale (reused) pose distinctly, and `/` and `/studio`
    both load with zero console errors.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.browser


def _preview_fps(page: Page, window_ms: int = 1200) -> float:
    """Count requestVideoFrameCallback ticks on #preview over a window."""

    return page.evaluate(
        """async (ms) => {
          const v = document.getElementById('preview');
          if (!v || !('requestVideoFrameCallback' in v)) return -1;
          let n = 0;
          const tick = () => { n++; v.requestVideoFrameCallback(tick); };
          v.requestVideoFrameCallback(tick);
          await new Promise(r => setTimeout(r, ms));
          return (n * 1000) / ms;
        }""",
        window_ms,
    )


def test_preview_fps_survives_a_3s_analysis_stall(live_server: str, page: Page, console) -> None:
    page.goto(f"{live_server}/", wait_until="networkidle")
    # let getUserMedia + the fake camera settle
    page.wait_for_timeout(1500)

    baseline = _preview_fps(page)
    if baseline < 0:
        pytest.skip("requestVideoFrameCallback unavailable in this Chromium")
    assert baseline > 8, f"fake-camera preview FPS unexpectedly low before stall: {baseline}"

    # deliberately stall the analysis consumer for 3 s
    resp = page.request.post(f"{live_server}/api/debug/stall", data={"seconds": "3"})
    assert resp.ok
    during = _preview_fps(page, window_ms=2000)  # sampled inside the stall window

    # preview must not drop meaningfully - it is a separate path from analysis
    assert during > baseline * 0.6, (
        f"preview FPS collapsed during the analysis stall: {baseline:.1f} -> {during:.1f}"
    )
    print(f"[phase8 test 14] preview FPS baseline={baseline:.1f} during-stall={during:.1f}")
    console.assert_clean()


def test_slash_and_studio_load_with_zero_console_errors(live_server: str, browser) -> None:
    from tests.browser.conftest import ConsoleWatch

    for path in ("/", "/studio"):
        ctx = browser.new_context(permissions=["camera"], viewport={"width": 1280, "height": 900})
        pg = ctx.new_page()
        watch = ConsoleWatch().attach(pg)
        pg.goto(f"{live_server}{path}", wait_until="networkidle")
        pg.wait_for_timeout(400)
        watch.assert_clean()
        ctx.close()


def test_overlay_de_emphasises_a_stale_pose(live_server: str, page: Page, console) -> None:
    """Feed the overlay a synthetic snapshot with pose_stale=true and confirm the
    draw path takes the stale branch (dashed line, dimmer alpha) without error."""

    page.goto(f"{live_server}/", wait_until="networkidle")
    page.wait_for_timeout(800)

    result = page.evaluate(
        """async () => {
          const mod = await import('/static/features/runtime.js');
          const store = (await import('/static/ui/store.js')).store;
          const prefs = await import('/static/features/analysis-prefs.js');
          prefs.setLayerEnabled('pose', true);  // live_server has pose_enabled=false
          // spy on canvas 2d context to see how the skeleton was stroked
          const calls = { setLineDash: [], globalAlpha: [] };
          const proto = CanvasRenderingContext2D.prototype;
          const origDash = proto.setLineDash;
          proto.setLineDash = function (p) { calls.setLineDash.push(Array.from(p || [])); return origDash.call(this, p); };
          const gaDesc = Object.getOwnPropertyDescriptor(proto, 'globalAlpha');

          const kp = Array.from({ length: 17 }, () => [50, 50, 0.9]);
          const snap = {
            snapshot_id: 999, mode: 'realtime', frame_id: 5, capture_ts: 1, emitted_ts: 1,
            frame_age_ms: 40, detections: [], poses: [{ keypoints: kp, bbox: [10,10,90,90], score: 0.9, frame_id: 4 }],
            tracks: [], risk: null, metrics: { capture_client_ts_ms: performance.timeOrigin + performance.now() },
            stale: false, pose_stale: true,
          };
          mod.runtime.lastSnapshot = snap;
          store.setSnapshot(snap);
          await new Promise(r => setTimeout(r, 200));
          proto.setLineDash = origDash;
          // a stale skeleton must have been drawn with a dash pattern at least once
          const dashed = calls.setLineDash.some(p => p.length === 2 && p[0] > 0);
          return { dashed, dashCalls: calls.setLineDash.length };
        }"""
    )
    assert result["dashCalls"] > 0, "overlay.draw() never ran for the injected snapshot"
    assert result["dashed"], "stale pose was not drawn with a dashed (de-emphasised) skeleton"
    console.assert_clean()
