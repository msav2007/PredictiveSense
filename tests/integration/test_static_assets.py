"""Every module the dashboard references resolves to a served file (Phase 1.6).

Walks index.html and every ``import`` / ``new Worker`` string under
``static/**/*.js``, checks the file exists on disk, and GETs it through the app
expecting 200 + a JavaScript content type.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import predictivesense
from predictivesense.api.app import create_app

pytestmark = pytest.mark.integration

_STATIC = Path(predictivesense.__file__).resolve().parent / "api" / "static"

_IMPORT_RE = re.compile(r"""(?:import[^"']*?|from\s*)["'](/static/[^"']+\.js)["']""")
_WORKER_RE = re.compile(r"""new\s+Worker\(\s*["'](/static/[^"']+\.js)["']""")
_HTML_SRC_RE = re.compile(r"""<script[^>]*\bsrc=["'](/static/[^"']+\.js)["']""")


@pytest.fixture()
def client(dev_config):
    app = create_app(dev_config, start_loop=False)
    with TestClient(app) as c:
        yield c


def _referenced_modules() -> list[str]:
    refs: set[str] = set()
    html = (_STATIC / "index.html").read_text(encoding="utf-8")
    refs.update(_HTML_SRC_RE.findall(html))
    for js in _STATIC.rglob("*.js"):
        text = js.read_text(encoding="utf-8")
        refs.update(_IMPORT_RE.findall(text))
        refs.update(_WORKER_RE.findall(text))
    return sorted(refs)


def test_all_referenced_modules_exist_on_disk() -> None:
    refs = _referenced_modules()
    assert len(refs) >= 15, f"expected the full module graph, found {refs}"
    for ref in refs:
        rel = ref[len("/static/") :]
        assert (_STATIC / rel).is_file(), f"{ref} is referenced but missing on disk"


def test_all_referenced_modules_serve_as_javascript(client) -> None:
    for ref in _referenced_modules():
        r = client.get(ref)
        assert r.status_code == 200, f"{ref} -> HTTP {r.status_code}"
        ctype = r.headers.get("content-type", "")
        assert "javascript" in ctype, f"{ref} served as {ctype!r}, not JavaScript"


def test_layer_directories_are_all_reachable(client) -> None:
    expected = [
        "ui/shell.js",
        "ui/group.js",
        "ui/registry.js",
        "ui/controls.js",
        "ui/store.js",
        "ui/format.js",
        "ui/log.js",
        "groups/constants.js",
        "groups/input.js",
        "groups/camera.js",
        "groups/video.js",
        "groups/dataset.js",
        "groups/diagnostics.js",
        "features/runtime.js",
        "features/camera-capture.js",
        "features/analysis-client.js",
        "features/recording.js",
        "features/videos.js",
        "features/metrics.js",
    ]
    for rel in expected:
        assert (_STATIC / rel).is_file(), f"static/{rel} missing"
        r = client.get(f"/static/{rel}")
        assert r.status_code == 200, f"/static/{rel} -> {r.status_code}"
