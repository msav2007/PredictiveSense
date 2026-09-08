"""Object Learning Studio state machine (Phase 5, BLOCK 11.4).

The three reported faults were caused by implicit state. ``studio-state.js`` makes
the four states and six transitions explicit. There is no browser here, so - like
``test_panel_resize_contract.py`` / ``test_worker_backpressure_contract.py`` -
this test is (a) an executable Python mirror of the transition table kept in
step with the shipped JS, driving the three fault scenarios, and (b) static
assertions that ``studio.js`` uses the machine and never reloads the page.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import predictivesense

pytestmark = pytest.mark.integration

_STUDIO = Path(predictivesense.__file__).resolve().parent / "api" / "static" / "studio"


# --------------------------------------------------------------------------
# Python mirror of static/studio/studio-state.js
# --------------------------------------------------------------------------

_TABLE = {
    "browsing": {"select": "object_selected"},
    "object_selected": {
        "select": "object_selected",
        "capture": "capturing",
        "review": "reviewing",
        "save_and_return": "browsing",
    },
    "capturing": {
        "select": "object_selected",
        "capture": "capturing",
        "review": "reviewing",
        "back_to_camera": "object_selected",
        "discard": "object_selected",
        "save_and_return": "browsing",
    },
    "reviewing": {
        "select": "object_selected",
        "review": "reviewing",
        "back_to_camera": "object_selected",
        "discard": "object_selected",
        "save_and_return": "browsing",
    },
}


def _fresh():
    return {
        "objectId": None,
        "stage": "camera",
        "editingSampleId": None,
        "pendingUploads": 0,
        "dirtyInspect": False,
    }


class StudioState:
    def __init__(self) -> None:
        self.state = "browsing"
        self.ctx = _fresh()

    def has_pending(self) -> bool:
        return self.ctx["pendingUploads"] > 0 or self.ctx["dirtyInspect"]

    def can(self, t: str) -> bool:
        if t not in _TABLE.get(self.state, {}):
            return False
        if t == "save_and_return":
            return not self.has_pending()
        return True

    def set_pending(self, uploads=None, dirty_inspect=None) -> None:
        if uploads is not None:
            self.ctx["pendingUploads"] = max(0, uploads)
        if dirty_inspect is not None:
            self.ctx["dirtyInspect"] = dirty_inspect

    def dispatch(self, t: str, **payload):
        target = _TABLE.get(self.state, {}).get(t)
        if target is None:
            raise ValueError(f"invalid transition {t} from {self.state}")
        if t == "save_and_return" and self.has_pending():
            return "blocked"
        if t == "select":
            self.ctx = _fresh()
            self.ctx["objectId"] = payload["objectId"]
        elif t == "capture":
            self.ctx["stage"] = "upload"
            self.ctx["editingSampleId"] = None
            self.ctx["pendingUploads"] = max(1, payload.get("count", 1))
            self.ctx["dirtyInspect"] = False
        elif t == "review":
            self.ctx["stage"] = "inspect"
            self.ctx["editingSampleId"] = payload["sampleId"]
            self.ctx["pendingUploads"] = 0
            self.ctx["dirtyInspect"] = False
        elif t in ("back_to_camera", "discard"):
            keep = self.ctx["objectId"]
            self.ctx = _fresh()
            self.ctx["objectId"] = keep
        elif t == "save_and_return":
            self.ctx = _fresh()
        self.state = target
        return self.state


# --------------------------------------------------------------------------
# transitions
# --------------------------------------------------------------------------


def test_happy_path_transitions() -> None:
    sm = StudioState()
    assert sm.state == "browsing"
    sm.dispatch("select", objectId="cup")
    assert sm.state == "object_selected" and sm.ctx["objectId"] == "cup"
    sm.dispatch("capture", count=2)
    assert sm.state == "capturing" and sm.ctx["pendingUploads"] == 2
    sm.dispatch("back_to_camera")
    assert sm.state == "object_selected" and sm.ctx["objectId"] == "cup"
    sm.dispatch("review", sampleId="s1")
    assert sm.state == "reviewing" and sm.ctx["editingSampleId"] == "s1"


def test_invalid_transitions_raise() -> None:
    sm = StudioState()
    with pytest.raises(ValueError):
        sm.dispatch("capture")  # not from browsing
    with pytest.raises(ValueError):
        sm.dispatch("back_to_camera")  # not from browsing


# -- FAULT 1: Back to camera keeps the selected object + samples, no reload -----


def test_back_to_camera_preserves_selection() -> None:
    sm = StudioState()
    sm.dispatch("select", objectId="watch")
    sm.dispatch("review", sampleId="s7")
    sm.dispatch("back_to_camera")
    assert sm.state == "object_selected"
    assert sm.ctx["objectId"] == "watch"  # selection NOT lost
    assert sm.ctx["editingSampleId"] is None  # stale inspect cleared
    assert sm.ctx["stage"] == "camera"
    assert sm.ctx["pendingUploads"] == 0 and sm.ctx["dirtyInspect"] is False


# -- FAULT 2: re-selecting an object restores a clean editor (no stale inspect) -


def test_reselecting_an_object_clears_stale_inspect_state() -> None:
    sm = StudioState()
    sm.dispatch("select", objectId="mug")
    sm.dispatch("review", sampleId="s1")
    sm.set_pending(dirty_inspect=True)
    # user clicks a different object in the sidebar while inspecting
    sm.dispatch("select", objectId="kettle")
    assert sm.state == "object_selected"
    assert sm.ctx["objectId"] == "kettle"
    assert sm.ctx["editingSampleId"] is None
    assert sm.ctx["stage"] == "camera"
    assert sm.has_pending() is False


# -- FAULT 3: Save & Return never silently drops a pending sample --------------


def test_save_and_return_is_blocked_while_a_capture_is_pending() -> None:
    sm = StudioState()
    sm.dispatch("select", objectId="lamp")
    sm.dispatch("capture", count=1)
    assert sm.has_pending() is True
    assert sm.can("save_and_return") is False
    assert sm.dispatch("save_and_return") == "blocked"
    assert sm.state == "capturing"  # not left
    # once the staged upload is saved / discarded, leaving is allowed
    sm.set_pending(uploads=0)
    assert sm.can("save_and_return") is True
    assert sm.dispatch("save_and_return") == "browsing"
    assert sm.ctx["objectId"] is None


def test_dirty_inspect_also_blocks_leaving() -> None:
    sm = StudioState()
    sm.dispatch("select", objectId="pan")
    sm.dispatch("review", sampleId="s2")
    sm.set_pending(dirty_inspect=True)
    assert sm.can("save_and_return") is False
    sm.set_pending(dirty_inspect=False)  # saveInspect() clears it
    assert sm.can("save_and_return") is True


# --------------------------------------------------------------------------
# the shipped JS actually uses the machine and never reloads
# --------------------------------------------------------------------------


def test_studio_js_uses_the_state_machine_and_never_reloads() -> None:
    js = (_STUDIO / "studio.js").read_text(encoding="utf-8")
    assert "createStudioState" in js
    assert "/static/studio/studio-state.js" in js
    for t in ("select", "capture", "review", "back_to_camera", "save_and_return", "discard"):
        assert f'"{t}"' in js, f"studio.js never dispatches {t!r}"
    # BLOCK 3.16 / 3.18: fix the lifecycle, do not paper over it with a reload
    assert "location.reload" not in js
    # exactly one navigation-away, in saveAndReturn
    assert js.count("window.location.href") == 1

    sjs = (_STUDIO / "studio-state.js").read_text(encoding="utf-8")
    for s in ("browsing", "object_selected", "capturing", "reviewing"):
        assert f'"{s}"' in sjs
    assert "hasPending" in sjs


def test_python_mirror_matches_the_shipped_transition_table() -> None:
    """Guard: the _TABLE above must stay in step with studio-state.js."""

    sjs = (_STUDIO / "studio-state.js").read_text(encoding="utf-8")
    m = re.search(r"const TABLE = \{(.+?)\n\};", sjs, re.S)
    assert m, "could not find TABLE in studio-state.js"
    block = m.group(1)
    for from_state, trans in _TABLE.items():
        assert re.search(rf"\b{from_state}:\s*\{{", block), f"{from_state} missing in JS TABLE"
        for t, to in trans.items():
            assert re.search(rf"\b{t}:\s*\"{to}\"", block), f"{from_state}.{t} -> {to} missing in JS"
