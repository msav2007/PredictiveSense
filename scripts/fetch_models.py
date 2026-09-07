"""Fetch the Phase 2 perception weights - a MANUAL, build-time setup step.

    python scripts\\fetch_models.py [--force] [--models-dir models]

Downloads the detector and pose ONNX files listed in ``models/manifest.json``
(committed), verifies each SHA-256, and writes the licence text beside them.
Nothing here runs at application runtime; this script is not imported by the
``predictivesense`` package and is excluded from the outbound-network guard by
path. A hash mismatch fails loudly (exit 2) and deletes nothing.

The models are pre-exported ONNX published by Ultralytics, so no ``torch`` /
``ultralytics`` install is needed. If a future model needs a ``.pt`` -> ``.onnx``
export step, add it here behind ``--export`` in an isolated environment - never
as a runtime dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LICENSE_URL = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/LICENSE"
_CHUNK = 1 << 20


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    print(f"  downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "predictivesense-fetch"})
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:  # noqa: S310
        while True:
            chunk = resp.read(_CHUNK)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(dest)


def _fetch_one(name: str, spec: dict, models_dir: Path, force: bool) -> bool:
    dest = models_dir / spec["filename"]
    want = spec["sha256"].lower()

    if dest.is_file() and not force:
        have = _sha256(dest)
        if have == want:
            print(f"[{name}] {dest.name} present and verified ({have[:12]}…)")
            return True
        print(
            f"[{name}] {dest.name} present but sha256 MISMATCH\n"
            f"        expected {want}\n        got      {have}\n"
            f"        not overwriting - move it aside and re-run, or pass --force"
        )
        return False

    print(f"[{name}] fetching {dest.name}")
    try:
        _download(spec["url"], dest)
    except Exception as exc:  # noqa: BLE001 - report and fail
        print(f"[{name}] download failed: {exc!r}")
        return False

    have = _sha256(dest)
    if have != want:
        print(
            f"[{name}] sha256 MISMATCH after download\n"
            f"        expected {want}\n        got      {have}\n"
            f"        leaving the file in place for inspection; NOT using it"
        )
        return False
    size = dest.stat().st_size
    if int(spec.get("size_bytes", size)) != size:
        print(f"[{name}] warning: size {size} != manifest {spec.get('size_bytes')}")
    print(f"[{name}] OK {dest.name} ({size} bytes, sha256 {have[:12]}…)")
    return True


def _write_license(models_dir: Path) -> None:
    out = models_dir / "LICENSE-AGPL-3.0.txt"
    if out.is_file():
        return
    try:
        req = urllib.request.Request(
            _LICENSE_URL, headers={"User-Agent": "predictivesense-fetch"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
            text = resp.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        text = (
            "Ultralytics YOLO models are licensed AGPL-3.0-only.\n"
            f"Full text: https://www.gnu.org/licenses/agpl-3.0.txt\n"
            f"(automatic fetch of {_LICENSE_URL} failed: {exc!r})\n"
        )
    out.write_text(text, encoding="utf-8")
    print(f"[license] wrote {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch Phase 2 perception weights.")
    parser.add_argument("--models-dir", default=str(_REPO_ROOT / "models"))
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args(argv)

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = models_dir / "manifest.json"
    if not manifest_path.is_file():
        print(f"manifest not found: {manifest_path}")
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    ok = True
    for name, spec in manifest["models"].items():
        ok = _fetch_one(name, spec, models_dir, args.force) and ok

    if ok:
        _write_license(models_dir)
        print("\nall models present and verified.")
        return 0
    print("\none or more models failed verification - see above.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
