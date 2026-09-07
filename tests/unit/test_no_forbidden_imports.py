"""Static guarantees about what the package is allowed to import."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import predictivesense

pytestmark = pytest.mark.unit

PACKAGE_ROOT = Path(predictivesense.__file__).resolve().parent

FORBIDDEN_MODULES = {
    "cv2",
    "torch",
    "onnxruntime",
    "openvino",
    "mediapipe",
    "ultralytics",
    "requests",
    "httpx",
    "aiohttp",
    "pyttsx3",
    "aiortc",
}
# Submodule imports that must also be caught.
FORBIDDEN_QUALIFIED = {"urllib.request"}

CORE_FORBIDDEN_INTERNAL_PREFIXES = (
    "predictivesense.api",
    "predictivesense.pipeline",
    "predictivesense.camera",
)


def _module_files() -> list[Path]:
    return sorted(PACKAGE_ROOT.rglob("*.py"))


def _imported_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import, not a top-level module
                continue
            mod = node.module or ""
            names.add(mod)
            for alias in node.names:
                names.add(f"{mod}.{alias.name}" if mod else alias.name)
    return names


def test_package_has_modules() -> None:
    assert _module_files(), "no python modules discovered under predictivesense/"


def test_no_module_imports_a_forbidden_library() -> None:
    offenders: list[str] = []
    for path in _module_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for name in _imported_names(tree):
            root = name.split(".")[0]
            if root in FORBIDDEN_MODULES or name in FORBIDDEN_QUALIFIED:
                offenders.append(f"{path.relative_to(PACKAGE_ROOT)} imports {name}")
    assert not offenders, "forbidden imports found:\n" + "\n".join(offenders)


def test_core_does_not_depend_on_api_pipeline_or_camera() -> None:
    core_dir = PACKAGE_ROOT / "core"
    offenders: list[str] = []
    for path in sorted(core_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for name in _imported_names(tree):
            if name.startswith(CORE_FORBIDDEN_INTERNAL_PREFIXES):
                offenders.append(f"{path.relative_to(PACKAGE_ROOT)} imports {name}")
    assert not offenders, "core/ reached outside itself:\n" + "\n".join(offenders)
