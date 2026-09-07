"""User-facing text rules for the Phase 1.6 dashboard.

Forbidden internal identifiers must not appear in Normal-view markup or in group
modules; the documented replacements must be present somewhere in the shipped
JS; and no ``console.log(`` may survive outside the single gated logger.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import predictivesense

pytestmark = pytest.mark.unit

_STATIC = Path(predictivesense.__file__).resolve().parent / "api" / "static"
_GROUPS = _STATIC / "groups"

FORBIDDEN = ["asfast", "backend owns the camera", "worker_skips_t"]
REQUIRED = ["Fastest", "Real-time speed", "Worker skips", "Live preview unavailable in backend-camera mode."]


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def test_no_forbidden_strings_in_index_or_group_modules() -> None:
    targets = [_STATIC / "index.html", *sorted(_GROUPS.glob("*.js"))]
    for p in targets:
        text = _read(p)
        for bad in FORBIDDEN:
            assert bad not in text, f"{p.name} contains the forbidden user-facing string {bad!r}"


def test_required_replacements_present_somewhere_in_shipped_js() -> None:
    blob = "\n".join(_read(p) for p in _STATIC.rglob("*.js"))
    for want in REQUIRED:
        assert want in blob, f"missing the required replacement string {want!r}"


def test_no_console_log_outside_the_gated_logger() -> None:
    for p in _STATIC.rglob("*.js"):
        if p.name == "log.js":
            continue  # log.js IS the single Diagnostics-gated logger
        assert "console.log(" not in _read(p), f"{p.relative_to(_STATIC)} has a console.log("


def test_log_is_gated_by_diagnostics() -> None:
    src = _read(_STATIC / "ui" / "log.js")
    assert "debugEnabled" in src and "console.log(" in src
    assert "setDebugEnabled" in src


def test_replay_values_still_wire_the_api() -> None:
    # the labels are renamed but the values POSTed to /api/analyze are unchanged
    fmt = _read(_STATIC / "ui" / "format.js")
    assert '"asfast"' in fmt and '"realtime"' in fmt
    assert '"Fastest"' in fmt and '"Real-time speed"' in fmt
