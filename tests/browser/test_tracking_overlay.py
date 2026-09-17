"""Phase 9 section 14, browser tests 17-20 - the state/visual mapping and
flicker suppression, verified against the real overlay.js / diagnostics.js
running in real Chromium (item 21, "zero console errors on / and /studio", is
already covered by
``tests/browser/test_phase8_responsiveness.py::test_slash_and_studio_load_with_zero_console_errors``
and needs no Phase-9-specific duplicate).

Synthetic StateSnapshot-shaped objects are injected directly via
``store.setSnapshot()`` (same technique as
``test_phase8_responsiveness.py::test_overlay_de_emphasises_a_stale_pose``),
bypassing the camera/WS entirely, with the real canvas 2D context spied on so
assertions check exactly what was drawn rather than raster pixels.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page

pytestmark = pytest.mark.browser


def _track(track_id, **overrides):
    base = {
        "track_id": track_id,
        "status": "confirmed",
        "class_name": "cup",
        "bbox": [100.0, 100.0, 180.0, 180.0],
        "last_seen_frame_id": 1,
        "observed_class": "cup",
        "track_class": "cup",
        "class_votes": [["cup", 5]],
        "velocity": [0.0, 0.0],
        "fresh": True,
        "age_frames": 5,
        "age_ms": 250.0,
        "hits": 5,
        "consecutive_misses": 0,
        "first_seen_frame_id": 0,
        "last_detection_frame_id": 1,
        "last_detection_capture_ts": 1.0,
        "last_detector_confidence": 0.9,
        "last_detector_confidence_age_ms": 0.0,
        "policy_state": "accepted",
        "tier": "primary",
    }
    base.update(overrides)
    return base


def _snap(tracks, snapshot_id=1, stale=False):
    return {
        "snapshot_id": snapshot_id, "mode": "realtime", "frame_id": snapshot_id,
        "capture_ts": float(snapshot_id), "emitted_ts": float(snapshot_id),
        "frame_age_ms": 40, "detections": [], "poses": [], "tracks": tracks,
        "risk": None, "metrics": {}, "stale": stale, "pose_stale": False,
    }


_SPY_SETUP = """
const mod = await import('/static/features/runtime.js');
const store = (await import('/static/ui/store.js')).store;
const prefs = await import('/static/features/analysis-prefs.js');
prefs.setLayerEnabled('detection', true);  // live_server has detection_enabled=false
const proto = CanvasRenderingContext2D.prototype;
const calls = { setLineDash: [], fillText: [], fillStyle: [] };
const origDash = proto.setLineDash;
proto.setLineDash = function (p) { calls.setLineDash.push(Array.from(p || [])); return origDash.call(this, p); };
const origFillText = proto.fillText;
proto.fillText = function (t, x, y) { calls.fillText.push(String(t)); return origFillText.call(this, t, x, y); };
const fsDesc = Object.getOwnPropertyDescriptor(CanvasRenderingContext2D.prototype, 'fillStyle')
  || Object.getOwnPropertyDescriptor(CanvasGradient.prototype, 'fillStyle');
window.__ps_calls = calls;
window.__ps_inject = (snap) => {
  calls.setLineDash.length = 0;
  calls.fillText.length = 0;
  mod.runtime.lastSnapshot = snap;
  store.setSnapshot(snap);
};
window.__ps_emit = (type, detail) => mod.emit(type, detail);
"""


@pytest.fixture()
def spy_page(live_server: str, page: Page):
    page.goto(f"{live_server}/", wait_until="networkidle")
    page.wait_for_timeout(500)
    page.evaluate(f"async () => {{ {_SPY_SETUP} }}")
    return page


def test_fresh_track_becomes_dashed_on_coasting_and_reverts(spy_page: Page, console) -> None:
    inject = lambda snap: spy_page.evaluate("(s) => window.__ps_inject(s)", snap)  # noqa: E731

    inject(_snap([_track(1, fresh=True)]))
    spy_page.wait_for_timeout(150)
    dashed = spy_page.evaluate("window.__ps_calls.setLineDash.some(p => p.length === 2 && p[0] > 0)")
    assert not dashed, "a fresh track must not render dashed"

    inject(_snap([_track(1, fresh=False, status="coasting")], snapshot_id=2))
    spy_page.wait_for_timeout(150)
    dashed = spy_page.evaluate("window.__ps_calls.setLineDash.some(p => p.length === 2 && p[0] > 0)")
    assert dashed, "a coasting track must render dashed"

    inject(_snap([_track(1, fresh=True)], snapshot_id=3))
    spy_page.wait_for_timeout(150)
    dashed = spy_page.evaluate("window.__ps_calls.setLineDash.some(p => p.length === 2 && p[0] > 0)")
    assert not dashed, "re-detection must revert to a solid stroke"
    console.assert_clean()


def test_unknown_secondary_and_coasting_are_not_rendered_identically(spy_page: Page, console) -> None:
    inject = lambda snap: spy_page.evaluate("(s) => window.__ps_inject(s)", snap)  # noqa: E731

    tracks = [
        _track(1, policy_state="unknown_low_confidence", fresh=True, class_name="unknown"),
        _track(2, policy_state="accepted_secondary", tier="secondary", fresh=True,
               class_name="umbrella", observed_class="umbrella", track_class="umbrella"),
        _track(3, policy_state="accepted", fresh=False, status="coasting",
               last_detector_confidence_age_ms=300.0),
    ]
    inject(_snap(tracks))
    spy_page.wait_for_timeout(200)
    texts = spy_page.evaluate("window.__ps_calls.fillText")

    assert "Unknown" in texts, "unknown track must show the exact label 'Unknown'"
    assert not any(t.startswith("Unknown ") for t in texts), "Unknown must never carry a percentage"
    assert any("tier: secondary" in t for t in texts), "secondary-tier track must show a tier badge"
    assert any("Cup" in t and "%" in t and "old" in t for t in texts), (
        "a coasting accepted track must show its confidence AND its age - "
        f"got: {texts}"
    )
    # None of the three labels collapse into each other.
    assert texts.count("Unknown") == 1
    console.assert_clean()


def test_confirmed_track_does_not_blink_across_a_missed_detection(spy_page: Page, console) -> None:
    inject = lambda snap: spy_page.evaluate("(s) => window.__ps_inject(s)", snap)  # noqa: E731

    inject(_snap([_track(7, fresh=True, status="confirmed")]))
    spy_page.wait_for_timeout(100)
    # A coasting cycle for the SAME track_id: the backend never drops a
    # coasting track from `tracks`, so the box must still be drawn.
    inject(_snap([_track(7, fresh=False, status="coasting", consecutive_misses=1)], snapshot_id=2))
    spy_page.wait_for_timeout(100)
    texts = spy_page.evaluate("window.__ps_calls.fillText")
    assert texts, "a coasting (still-alive) track must still be drawn, not skipped"

    # Rapid policy_state flicker within the hysteresis dwell window must not
    # flip the label to "Unknown" and back - section 9.4.
    inject(_snap([_track(7, fresh=True, policy_state="accepted")], snapshot_id=3))
    spy_page.wait_for_timeout(20)
    inject(_snap([_track(7, fresh=True, policy_state="unknown_low_confidence")], snapshot_id=4))
    spy_page.wait_for_timeout(20)
    texts_mid = spy_page.evaluate("window.__ps_calls.fillText")
    inject(_snap([_track(7, fresh=True, policy_state="accepted")], snapshot_id=5))
    spy_page.wait_for_timeout(20)

    assert "Unknown" not in texts_mid, (
        "a single-cycle policy_state flicker must not blink the label to "
        f"Unknown within the dwell window - got: {texts_mid}"
    )
    console.assert_clean()


def test_diagnostics_shows_selected_track_detail(spy_page: Page, console) -> None:
    inject = lambda snap: spy_page.evaluate("(s) => window.__ps_inject(s)", snap)  # noqa: E731
    t = _track(42, class_name="laptop", observed_class="laptop", track_class="laptop",
               last_detector_confidence=0.81, hits=9, consecutive_misses=2)
    inject(_snap([t]))
    spy_page.wait_for_timeout(150)

    spy_page.evaluate("(t) => window.__ps_emit('detection-selected', { track: t, detection: null })", t)
    spy_page.wait_for_timeout(150)

    text = spy_page.evaluate("document.getElementById('trk-selected')?.textContent || ''")
    assert "42" in text
    assert "laptop" in text
    assert "81.0%" in text or "81%" in text
    assert "9" in text  # hits
    console.assert_clean()
