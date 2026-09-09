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
    "browsing": {"select": "object_selected", "save_and_return": "browsing"},
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
        self.last_refused = None  # observable; cleared by the next good transition

    def has_pending(self) -> bool:
        return self.ctx["pendingUploads"] > 0 or self.ctx["dirtyInspect"]

    def can(self, t: str) -> bool:
        if t not in _TABLE.get(self.state, {}):
            return False
        if t == "save_and_return":
            return not self.has_pending()
        return True

    def refuse(self, t: str) -> None:
        self.last_refused = t

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
            self.last_refused = t
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
        self.last_refused = None
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


# -- Phase 6 REGRESSION: root cause #2 - "Save and return" dead from `browsing` -
# Phase 5 had no `save_and_return` entry for `browsing`; `dispatch()` threw an
# uncaught error that the async click handler swallowed, so the button no-op'd on
# every clean return (and on the reported bug, where selection never fired so the
# machine was stuck in `browsing`).


def test_save_and_return_is_reachable_from_browsing() -> None:
    sm = StudioState()
    assert sm.state == "browsing"
    assert sm.can("save_and_return") is True  # not pending -> allowed
    result = sm.dispatch("save_and_return")  # must NOT raise
    assert result == "browsing"
    assert sm.ctx["objectId"] is None


def test_save_and_return_reachable_from_every_state() -> None:
    # object_selected
    sm = StudioState()
    sm.dispatch("select", objectId="a")
    assert sm.can("save_and_return") and sm.dispatch("save_and_return") == "browsing"
    # capturing (after the staged upload is resolved)
    sm = StudioState()
    sm.dispatch("select", objectId="b")
    sm.dispatch("capture", count=1)
    sm.set_pending(uploads=0)
    assert sm.dispatch("save_and_return") == "browsing"
    # reviewing (after the inspect edit is saved)
    sm = StudioState()
    sm.dispatch("select", objectId="c")
    sm.dispatch("review", sampleId="s1")
    assert sm.dispatch("save_and_return") == "browsing"


def test_last_refused_is_recorded_then_cleared_by_the_next_good_transition() -> None:
    sm = StudioState()
    sm.dispatch("select", objectId="lamp")
    sm.dispatch("capture", count=1)  # pending
    assert sm.dispatch("save_and_return") == "blocked"
    assert sm.last_refused == "save_and_return"  # observable in Diagnostics
    sm.refuse("back_to_camera")
    assert sm.last_refused == "back_to_camera"
    sm.set_pending(uploads=0)
    sm.dispatch("back_to_camera")
    assert sm.last_refused is None  # cannot latch


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


# -- Phase 6 REGRESSION: root cause #1 - object row clicks were bound with a
# case-wrong `el("li", { onClick })` (-> addEventListener("Click"), never fires)
# on nodes that `loadObjects()` replaces on every refresh. The fix is ONE
# delegated listener on the stable #object-list container.


def test_object_rows_are_selected_by_delegation_from_a_stable_container() -> None:
    js = (_STUDIO / "studio.js").read_text(encoding="utf-8")
    html = (_STUDIO / "index.html").read_text(encoding="utf-8")

    # the stable container exists and is never itself replaced
    assert 'id="object-list"' in html

    # one delegated click listener bound to that container, matching rows by class
    assert re.search(r'"object-list"\)\.addEventListener\("click"', js), (
        "object selection must be delegated from #object-list, not bound per row"
    )
    assert 'closest(".object-item")' in js

    # the per-row `el(...)` must NOT carry an inline handler (the Phase 5 bug)
    assert "onClick:" not in js and "onclick:" not in js, (
        "object rows must not use an inline el() click handler"
    )
    # rows still tag their id for the delegated handler to read
    assert "dataset: { objectId:" in js


def test_studio_surfaces_load_errors_and_exposes_a_state_readout() -> None:
    js = (_STUDIO / "studio.js").read_text(encoding="utf-8")
    html = (_STUDIO / "index.html").read_text(encoding="utf-8")

    # BLOCK 14: a module-load fault / unhandled rejection shows a visible banner
    assert 'id="studio-error"' in html
    assert "showFatal" in js
    assert 'addEventListener("error"' in js
    assert 'addEventListener("unhandledrejection"' in js
    assert "catch" in js and "showFatal(" in js  # main() is wrapped

    # BLOCK 13: Diagnostics-visible state readout, all four fields
    assert 'id="studio-diag"' in html
    for field in (
        "state=",
        "selected_object_id=",
        "has_pending=",
        "last_refused_transition=",
    ):
        assert field in js, f"studio-diag readout missing {field!r}"
    # BLOCK 5.13: a refused transition produces visible feedback
    assert "refuseNote" in js


def test_studio_state_exposes_last_refused_and_clears_it() -> None:
    sjs = (_STUDIO / "studio-state.js").read_text(encoding="utf-8")
    assert "lastRefused" in sjs
    assert "refuse" in sjs
    # snapshot carries it; a successful dispatch resets it
    assert re.search(r"return \{ state, context: \{ \.\.\.ctx \}, lastRefused \}", sjs)
    assert "lastRefused = null" in sjs
