"""Read, validate and resolve ``models/registry.json``.

A model version records ``version_id``, its files (path + SHA-256 + size),
``source``, ``created_utc``, ``license``, ``metrics_ref`` (a path under
``results/``) and an ``active`` flag. Exactly one version is active at a time.

**Activation is a deliberate operation, never an implicit overwrite** - this
module has no activation code; it reads the flag. Phase 4 registers the shipped
Phase 2 detector as baseline ``v1``. No model is trained here.

Failure policy (P4 Block 10): a malformed registry, a bad SHA-256, or zero / more
than one active version raises :class:`ModelRegistryError`. Callers that must not
crash (``GET /api/models/registry``) catch it and report.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "ModelRegistry",
    "ModelRegistryError",
    "ModelVersion",
    "ModelFile",
    "default_registry_path",
]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REPO_ROOT = Path(__file__).resolve().parents[2]


class ModelRegistryError(RuntimeError):
    """The registry is missing, malformed, has a bad hash, or its active set is
    not exactly one."""


def default_registry_path() -> Path:
    return _REPO_ROOT / "models" / "registry.json"


def _manifest_hashes(repo_root: Path) -> dict[str, str]:
    """``filename -> sha256`` from ``models/manifest.json`` (the source of truth
    for the shipped weights), or ``{}`` if it is absent/unreadable."""

    path = repo_root / "models" / "manifest.json"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict[str, str] = {}
    for entry in (doc.get("models") or {}).values():
        name = entry.get("filename")
        sha = entry.get("sha256")
        if name and sha:
            out[str(name)] = str(sha).lower()
    return out


@dataclass(frozen=True)
class ModelFile:
    role: str
    path: str
    sha256: str
    size_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "path": self.path, "sha256": self.sha256, "size_bytes": self.size_bytes}


@dataclass(frozen=True)
class ModelVersion:
    version_id: str
    name: str
    files: list[ModelFile]
    source: str
    created_utc: str
    license: str
    metrics_ref: str
    active: bool = False
    notes: str = ""
    task: str = "detect"

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "name": self.name,
            "task": self.task,
            "files": [f.to_dict() for f in self.files],
            "source": self.source,
            "created_utc": self.created_utc,
            "license": self.license,
            "metrics_ref": self.metrics_ref,
            "active": self.active,
            "notes": self.notes,
        }


class ModelRegistry:
    """In-memory view of ``models/registry.json``."""

    def __init__(self, versions: list[ModelVersion], *, repo_root: Path | None = None) -> None:
        self._versions = list(versions)
        self._repo_root = repo_root or _REPO_ROOT
        self.validate()

    # -- construction ------------------------------------------

    @classmethod
    def load(cls, path: str | Path | None = None, *, repo_root: Path | None = None) -> "ModelRegistry":
        p = Path(path) if path is not None else default_registry_path()
        if not p.is_file():
            raise ModelRegistryError(f"model registry not found: {p}")
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ModelRegistryError(f"{p} is not readable JSON: {exc}") from exc
        if not isinstance(doc, dict) or not isinstance(doc.get("models"), list):
            raise ModelRegistryError(f"{p} must be an object with a 'models' list")

        versions: list[ModelVersion] = []
        for raw in doc["models"]:
            try:
                files = [
                    ModelFile(
                        role=str(f.get("role", "weights")),
                        path=str(f["path"]),
                        sha256=str(f["sha256"]).lower(),
                        size_bytes=int(f.get("size_bytes", 0)),
                    )
                    for f in raw["files"]
                ]
                versions.append(
                    ModelVersion(
                        version_id=str(raw["version_id"]),
                        name=str(raw.get("name", raw["version_id"])),
                        task=str(raw.get("task", "detect")),
                        files=files,
                        source=str(raw.get("source", "")),
                        created_utc=str(raw.get("created_utc", "")),
                        license=str(raw.get("license", "")),
                        metrics_ref=str(raw.get("metrics_ref", "")),
                        active=bool(raw.get("active", False)),
                        notes=str(raw.get("notes", "")),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ModelRegistryError(f"bad model version record {raw!r}: {exc}") from exc
        return cls(versions, repo_root=repo_root or p.resolve().parents[1])

    # -- validation -------------------------------------------

    def validate(self) -> None:
        if not self._versions:
            raise ModelRegistryError("registry has no model versions")

        ids: set[str] = set()
        manifest = _manifest_hashes(self._repo_root)
        for v in self._versions:
            if v.version_id in ids:
                raise ModelRegistryError(f"duplicate version_id {v.version_id!r}")
            ids.add(v.version_id)
            if not v.files:
                raise ModelRegistryError(f"version {v.version_id!r} lists no files")
            for f in v.files:
                if not _SHA256_RE.match(f.sha256):
                    raise ModelRegistryError(
                        f"version {v.version_id!r} file {f.path!r} has a non-SHA-256 hash {f.sha256!r}"
                    )
                name = Path(f.path).name
                if name in manifest and manifest[name] != f.sha256:
                    raise ModelRegistryError(
                        f"version {v.version_id!r} file {name!r} hash {f.sha256!r} "
                        f"does not match models/manifest.json ({manifest[name]!r})"
                    )

        active = [v for v in self._versions if v.active]
        if len(active) != 1:
            raise ModelRegistryError(
                f"exactly one model version must be active, found {len(active)}: "
                f"{[v.version_id for v in active]}"
            )

    # -- queries ----------------------------------------------

    def list(self) -> list[ModelVersion]:
        return list(self._versions)

    def get(self, version_id: str) -> ModelVersion:
        for v in self._versions:
            if v.version_id == version_id:
                return v
        raise ModelRegistryError(f"no model version {version_id!r}")

    def active(self) -> ModelVersion:
        for v in self._versions:
            if v.active:
                return v
        raise ModelRegistryError("no active model version")  # pragma: no cover - validate guards

    def resolve_active_files(self) -> dict[str, str]:
        """``role -> absolute path`` for the active version's files. Does not
        check the files exist (they are git-ignored and may be absent)."""

        out: dict[str, str] = {}
        for f in self.active().files:
            p = Path(f.path)
            out[f.role] = str(p if p.is_absolute() else (self._repo_root / p))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "active": self.active().version_id,
            "models": [v.to_dict() for v in self._versions],
        }
