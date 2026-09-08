"""Static analysis of the shipped dashboard assets (Phase 1.6 shell, Phase 2 fills
the Analysis slot).

No browser automation, no new dependency - just read the files under
``predictivesense/api/static/`` and assert the interface architecture holds:
a shell mount (not a legacy panel stack), the registered groups each exporting
the extension contract, and the still-reserved ids declared-but-not-registered.
Phase 2 registered ``analysis`` (Detection + Pose sub-modules); Phase 2.5
registered ``research`` (labelling + evaluation); only ``alerts`` remains
reserved.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import predictivesense

pytestmark = pytest.mark.unit

_STATIC = Path(predictivesense.__file__).resolve().parent / "api" / "static"
_GROUPS = _STATIC / "groups"

REGISTERED_GROUP_IDS = ["input", "camera", "video", "dataset", "analysis", "research", "diagnostics"]
RESERVED_GROUP_IDS = ["alerts"]


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


def test_analysis_group_hosts_detection_and_pose_submodules() -> None:
    analysis = _read(_GROUPS / "analysis.js")
    assert re.search(r'export\s+const\s+id\s*=\s*"analysis"', analysis)
    assert "registerAnalysisModule" in analysis
    assert "/static/features/detection.js" in analysis
    assert "/static/features/pose.js" in analysis
    for feat in ("detection.js", "pose.js", "overlay.js", "analysis-prefs.js"):
        assert (_STATIC / "features" / feat).is_file(), f"features/{feat} missing"
    # the reserved slot was filled, not renegotiated
    consts = _read(_GROUPS / "constants.js")
    assert "analysis: 50" in consts
    assert '"analysis"' not in _read_reserved_ids_line(consts)


def _read_reserved_ids_line(consts: str) -> str:
    m = re.search(r"RESERVED_GROUP_IDS\s*=\s*\[[^\]]*\]", consts)
    return m.group(0) if m else ""


def test_research_group_registered_with_labelling_and_eval() -> None:
    research = _read(_GROUPS / "research.js")
    assert re.search(r'export\s+const\s+id\s*=\s*"research"', research)
    assert "/label" in research, "Research group must link to the labelling tool"
    assert "/api/labels/progress" in research
    assert "/api/labels/eval-summary" in research
    consts = _read(_GROUPS / "constants.js")
    assert "research: 70" in consts
    assert '"research"' not in _read_reserved_ids_line(consts)
    app = _read(_STATIC / "app.js")
    assert app.count("/static/groups/research.js") == 1


def test_policy_controls_and_unknown_rendering_present() -> None:
    detection = _read(_STATIC / "features" / "detection.js")
    assert "/static/features/policy.js" in detection
    assert "Recognition policy" in detection
    overlay = _read(_STATIC / "features" / "overlay.js")
    assert "Unknown" in overlay and "effectiveDetection" in overlay
    diagnostics = _read(_GROUPS / "diagnostics.js")
    assert "policy_rejected_out_of_domain" in diagnostics
    assert "raw" in diagnostics and "decided" in diagnostics


def test_overlay_never_touches_the_video_element() -> None:
    overlay = _read(_STATIC / "features" / "overlay.js")
    # draws on #overlay-layer, not into <video>
    assert "overlay-layer" in overlay
    assert "getContext" in overlay
    for banned in ("video.src =", "video.srcObject =", "drawImage(video", "video.play("):
        assert banned not in overlay, f"overlay.js must not do {banned!r}"


def test_extension_contract_documented_with_example() -> None:
    arch = (Path(predictivesense.__file__).resolve().parents[1] / "docs" / "architecture.md").read_text(
        encoding="utf-8"
    )
    assert "registerGroup(" in arch and "registerAnalysisModule(" in arch
    assert "Phase 1.6" in arch
