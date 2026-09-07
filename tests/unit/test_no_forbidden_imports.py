"""Static guarantees about what the package is allowed to import."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import predictivesense

pytestmark = pytest.mark.unit

PACKAGE_ROOT = Path(predictivesense.__file__).resolve().parent

# Never allowed anywhere in the package.
FORBIDDEN_MODULES = {
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

# Allowed, but only within this subpackage (Phase 1: OpenCV camera / file / codec).
CAMERA_ONLY_MODULES = {"cv2"}
CAMERA_DIR = PACKAGE_ROOT / "camera"

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


def _scan_for_module(source: str, filename: str, target_root: str) -> bool:
    """True if ``source`` imports ``target_root`` (or a submodule of it)."""

    tree = ast.parse(source, filename=filename)
    for name in _imported_names(tree):
        if name == target_root or name.startswith(f"{target_root}."):
            return True
    return False


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


def test_cv2_is_imported_only_under_camera() -> None:
    offenders: list[str] = []
    for path in _module_files():
        source = path.read_text(encoding="utf-8")
        for module in CAMERA_ONLY_MODULES:
            if _scan_for_module(source, str(path), module):
                if CAMERA_DIR not in path.parents:
                    offenders.append(f"{path.relative_to(PACKAGE_ROOT)} imports {module}")
    assert not offenders, (
        "cv2 (and other camera-only modules) may be imported only under "
        "predictivesense/camera/:\n" + "\n".join(offenders)
    )


def test_cv2_is_actually_used_under_camera() -> None:
    # Guards the checker itself: the rule is meaningful only if cv2 really is
    # imported somewhere under camera/.
    used = any(
        _scan_for_module(p.read_text(encoding="utf-8"), str(p), "cv2")
        for p in sorted(CAMERA_DIR.rglob("*.py"))
    )
    assert used, "expected at least one cv2 import under predictivesense/camera/"


def test_synthetic_cv2_violation_outside_camera_is_detected() -> None:
    # A fabricated import of cv2 in a non-camera location must be caught by the
    # same scanning logic the real test uses.
    fake_non_camera_file = PACKAGE_ROOT / "pipeline" / "_fake_violation.py"
    assert CAMERA_DIR not in fake_non_camera_file.parents
    assert _scan_for_module("import cv2\n", str(fake_non_camera_file), "cv2")
    assert _scan_for_module("from cv2 import VideoCapture\n", str(fake_non_camera_file), "cv2")
    assert not _scan_for_module("import numpy as np\n", str(fake_non_camera_file), "cv2")


def test_core_does_not_depend_on_api_pipeline_or_camera() -> None:
    core_dir = PACKAGE_ROOT / "core"
    offenders: list[str] = []
    for path in sorted(core_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for name in _imported_names(tree):
            if name.startswith(CORE_FORBIDDEN_INTERNAL_PREFIXES):
                offenders.append(f"{path.relative_to(PACKAGE_ROOT)} imports {name}")
    assert not offenders, "core/ reached outside itself:\n" + "\n".join(offenders)
