"""Operations-panel resize contract (Phase 4, P4 Block 4.8 / Block 12.5).

There is no JS runtime in the suite (no new dependency), so this pins the
resize *logic* two ways:

1. an executable Python mirror of ``clampPanelWidth`` from
   ``static/ui/resizer.js`` - boundary behaviour for min/max and the 40% rule,
   asserted directly, and kept byte-checked against the JS constants below;
2. static assertions that the shipped JS actually wires persistence (store),
   the double-click / Home reset, keyboard stepping, and the ARIA separator role.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import predictivesense

pytestmark = pytest.mark.unit

_STATIC = Path(predictivesense.__file__).resolve().parent / "api" / "static"
_RESIZER = (_STATIC / "ui" / "resizer.js").read_text(encoding="utf-8")
_STORE = (_STATIC / "ui" / "store.js").read_text(encoding="utf-8")
_SHELL = (_STATIC / "ui" / "shell.js").read_text(encoding="utf-8")
_INDEX = (_STATIC / "index.html").read_text(encoding="utf-8")
_CSS = (_STATIC / "app.css").read_text(encoding="utf-8")

_DEFAULTS = {
    "default_width_px": 380,
    "min_width_px": 300,
    "max_width_px": 560,
    "max_width_frac": 0.4,
}


def clamp_panel_width(width, win_width, cfg=None) -> int:
    """Mirror of static/ui/resizer.js:clampPanelWidth."""

    c = {**_DEFAULTS, **(cfg or {})}
    w = win_width if (isinstance(win_width, (int, float)) and win_width > 0) else 1280
    desired = width if isinstance(width, (int, float)) else c["default_width_px"]
    frac_cap = int(c["max_width_frac"] * w)  # floor
    viewport_floor = int((1 - c["max_width_frac"] - 0.05) * w)  # floor, 45% at 40% cap
    hard_max = min(c["max_width_px"], frac_cap, max(c["min_width_px"], w - viewport_floor))
    lo = min(c["min_width_px"], hard_max)
    return round(max(lo, min(desired, hard_max)))


def test_clamps_to_min_and_max_on_a_wide_window() -> None:
    win = 1600
    assert clamp_panel_width(380, win) == 380
    assert clamp_panel_width(5000, win) == 560  # max_width_px
    assert clamp_panel_width(50, win) == 300  # min_width_px
    assert clamp_panel_width(None, win) == 380  # default when width missing


def test_forty_percent_rule_and_viewport_floor() -> None:
    # 900px window: 40% cap = 360; viewport keeps >= 45% (405px).
    assert clamp_panel_width(560, 900) == 360
    assert 900 - clamp_panel_width(560, 900) >= round(0.45 * 900)
    # very narrow window: panel never pushes the viewport below 45%
    for win in (600, 700, 800, 1000, 1280, 1920):
        assert win - clamp_panel_width(10_000, win) >= round(0.45 * win) - 1


def test_reset_is_the_configured_default() -> None:
    assert clamp_panel_width(_DEFAULTS["default_width_px"], 1440) == 380
    # a custom config still resets to its own default
    cfg = {"default_width_px": 420, "min_width_px": 320, "max_width_px": 520, "max_width_frac": 0.35}
    assert clamp_panel_width(cfg["default_width_px"], 1440, cfg) == 420


def test_persistence_round_trip_semantics() -> None:
    # store.js persists panelWidth and setPanelWidth rounds + notifies
    assert "panelWidth" in _STORE
    assert re.search(r'PERSISTED_KEYS\s*=\s*\[[^\]]*"panelWidth"', _STORE, re.S)
    assert "setPanelWidth(px)" in _STORE and "Math.round(px)" in _STORE
    # a stored width is read back and re-clamped by the resizer
    assert "store.get().panelWidth" in _RESIZER
    assert "store.setPanelWidth(" in _RESIZER


def test_js_constants_match_the_mirror() -> None:
    for key, val in _DEFAULTS.items():
        assert re.search(rf"{key}:\s*{re.escape(str(val))}\b", _RESIZER), f"resizer.js default {key} drifted"
    assert "clampPanelWidth" in _RESIZER
    assert "Math.min(c.max_width_px, fracCap" in _RESIZER


def test_keyboard_and_reset_wired() -> None:
    assert 'setAttribute("role", "separator")' in _RESIZER
    assert 'aria-orientation' in _RESIZER and "aria-valuenow" in _RESIZER
    assert '"ArrowLeft"' in _RESIZER and '"ArrowRight"' in _RESIZER and '"Home"' in _RESIZER
    assert 'addEventListener("dblclick"' in _RESIZER  # double-click resets
    assert "cfg.default_width_px" in _RESIZER
    assert "requestAnimationFrame" in _RESIZER  # rAF-throttled drag


def test_shell_mounts_the_resizer_and_markup_present() -> None:
    assert "mountResizer" in _SHELL
    assert 'id="panel-resizer"' in _INDEX
    assert re.search(r"\.panel-resizer\s*\{[^}]*col-resize", _CSS, re.S)
    # the handle hides when the panel is collapsed and in the mobile overlay
    assert '.app-shell[data-collapsed="true"] .panel-resizer' in _CSS
