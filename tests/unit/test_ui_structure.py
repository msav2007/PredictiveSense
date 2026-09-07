"""Static analysis of the shipped dashboard assets (Phase 1.6).

No browser automation, no new dependency - just read the files under
``predictivesense/api/static/`` and assert the interface architecture holds:
a shell mount (not a legacy panel stack), five registered groups each exporting
the extension contract, and the three reserved ids declared-but-not-registered.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import predictivesense

pytestmark = pytest.mark.unit

_STATIC = Path(predictivesense.__file__).resolve().parent / "api" / "static"
_GROUPS = _STATIC / "groups"

REGISTERED_GROUP_IDS = ["input", "camera", "video", "dataset", "diagnostics"]
RESERVED_GROUP_IDS = ["analysis", "alerts", "research"]


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def test_index_is_a_shell_mount_not_a_panel_stack() -> None:
    html = _read(_STATIC / "index.html")
    for needle in ('id="app-shell"', 'id="panel-body"', 'id="viewport"', 'id="preview"', 'id="overlay-layer"'):
        assert needle in html, f"index.html missing shell mount point {needle}"
    # the Phase 1 flat panel stack is gone
    for legacy in ('class="panel preview"', 'class="panel metrics"', 'id="metric-grid"', 'id="browser-metrics"'):
        assert legacy not in html, f"index.html still carries legacy markup {legacy}"


def test_video_element_keeps_attrs_and_no_compositing() -> None:
    html = _read(_STATIC / "index.html")
    m = re.search(r"<video[^>]*\bid=\"preview\"[^>]*>", html)
    assert m, "no <video id=\"preview\"> in index.html"
    tag = m.group(0)
    for attr in ("autoplay", "muted", "playsinline"):
        assert attr in tag, f"<video> lost the {attr} attribute"

    css = _read(_STATIC / "app.css")
    block = re.search(r"#preview\s*\{([^}]*)\}", css)
    assert block, "no #preview rule in app.css"
    body = block.group(1)
    for prop in ("filter:", "transform:", "animation:", "opacity:", "transition:"):
        assert prop not in body, f"#preview must not use {prop}"
    # nothing else may animate/transform the preview either
    assert not re.search(r"#preview[^{]*\{[^}]*(filter|transform|animation)\s*:", css)


def test_every_group_module_exports_the_contract() -> None:
    for gid in REGISTERED_GROUP_IDS:
        src = _read(_GROUPS / f"{gid}.js")
        assert re.search(rf'export\s+const\s+id\s*=\s*"{gid}"', src), f"{gid}.js: export const id must be \"{gid}\""
        for name in ("title", "order", "modes"):
            assert re.search(rf"export\s+const\s+{name}\b", src), f"{gid}.js missing `export const {name}`"
        for fn in ("summary", "render"):
            assert re.search(rf"export\s+function\s+{fn}\b", src), f"{gid}.js missing `export function {fn}`"


def test_each_group_registered_exactly_once() -> None:
    app = _read(_STATIC / "app.js")
    assert "registerGroup" in app
    for gid in REGISTERED_GROUP_IDS:
        n = len(re.findall(rf"/static/groups/{gid}\.js", app))
        assert n == 1, f"groups/{gid}.js is imported {n} times by app.js (want 1)"


def test_reserved_ids_declared_but_not_registered() -> None:
    consts = _read(_GROUPS / "constants.js")
    assert "RESERVED_GROUP_IDS" in consts
    for rid in RESERVED_GROUP_IDS:
        assert re.search(rf'"{rid}"', consts), f"constants.js does not declare reserved id {rid}"
        assert not (_GROUPS / f"{rid}.js").exists(), f"groups/{rid}.js must not exist in Phase 1.6"
    app = _read(_STATIC / "app.js")
    for rid in RESERVED_GROUP_IDS:
        assert f"/static/groups/{rid}.js" not in app, f"app.js must not import the reserved group {rid}"


def test_registry_rejects_duplicate_and_reserved_ids() -> None:
    src = _read(_STATIC / "ui" / "registry.js")
    assert "duplicate group id" in src
    assert "reserved" in src
    assert "registerAnalysisModule" in src, "the analysis sub-module contract must exist for later phases"


def test_extension_contract_documented_with_example() -> None:
    arch = (Path(predictivesense.__file__).resolve().parents[1] / "docs" / "architecture.md").read_text(
        encoding="utf-8"
    )
    assert "registerGroup(" in arch and "registerAnalysisModule(" in arch
    assert "Phase 1.6" in arch
