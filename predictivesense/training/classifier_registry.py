"""Custom crop-classifier registry (Phase 11 Part B section 13).

Deliberately a SEPARATE store from ``predictivesense/models/registry.py``
(the shipped detector's frozen, heavily-tested registry) rather than an
extension of it: the detector registry's "exactly one active version, always"
invariant predates a second task existing at all, and the classifier's
lifecycle (trained -> validated -> active, section 13.1) is materially
different from "ship one baseline detector". Keeping them separate means nothing
about the detector's registry, tests, or activation semantics is touched.

Four distinct states, never collapsed (section 13.1):
    training data   -> `predictivesense/objects/` samples (untouched by this module)
    trained model   -> `register()`  (validated=False, active=False)
    validated model -> `validate()`  (validated=True,  active unchanged)
    active model    -> `activate()`  (validated model only; exactly one, or none)

"Rollback to baseline" (section 13.4) is simply ``activate(None)``: the
running app resolves "no active classifier" to baseline (detector-only)
behaviour, which is also the registry's natural starting state before Part B
ever trains anything - no separate baseline entry to invent or keep in sync.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "ClassifierRegistry",
    "ClassifierRegistryError",
    "ClassifierVersion",
    "default_classifier_registry_path",
]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REGISTRY_VERSION = 1


class ClassifierRegistryError(RuntimeError):
    """The classifier registry is malformed, or an operation violates its
    state machine (e.g. activating an unvalidated version)."""


def default_classifier_registry_path() -> Path:
    return _REPO_ROOT / "models" / "classifier_registry.json"


@dataclass(frozen=True)
class ClassifierVersion:
    version_id: str
    created_utc: str
    class_map: dict[str, int]
    dataset_content_hash: str
    split_content_hash: str
    architecture: str
    image_size: int
    epochs: int
    batch_size: int
    learning_rate: float
    seed: int
    augmentation: str
    metrics: dict[str, Any]  # per-epoch + final train/val metrics
    artifact_path: str
    artifact_sha256: str
    wall_seconds: float
    machine_fingerprint: str
    provider: str
    dataset_human_confirmed_fraction: float
    validated: bool = False
    active: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ClassifierVersion":
        return cls(**{k: raw[k] for k in cls.__dataclass_fields__ if k in raw})


class ClassifierRegistry:
    """Reads/writes ``models/classifier_registry.json``."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path is not None else default_classifier_registry_path()

    @property
    def path(self) -> Path:
        return self._path

    def _load(self) -> list[ClassifierVersion]:
        if not self._path.is_file():
            return []
        try:
            doc = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ClassifierRegistryError(f"{self._path} is not readable JSON: {exc}") from exc
        if not isinstance(doc, dict) or not isinstance(doc.get("versions"), list):
            raise ClassifierRegistryError(f"{self._path} must be an object with a 'versions' list")
        versions = [ClassifierVersion.from_dict(v) for v in doc["versions"]]
        active = [v for v in versions if v.active]
        if len(active) > 1:
            raise ClassifierRegistryError(
                f"more than one active classifier version: {[v.version_id for v in active]}"
            )
        for v in versions:
            if v.active and not v.validated:
                raise ClassifierRegistryError(
                    f"version {v.version_id!r} is active but not validated - "
                    "this should never happen (activate() enforces it)"
                )
        return versions

    def _save(self, versions: list[ClassifierVersion]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        doc = {"version": _REGISTRY_VERSION, "versions": [v.to_dict() for v in versions]}
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self._path)

    # -- queries --------------------------------------------------

    def list(self) -> list[ClassifierVersion]:
        return self._load()

    def get(self, version_id: str) -> ClassifierVersion:
        for v in self._load():
            if v.version_id == version_id:
                return v
        raise ClassifierRegistryError(f"no classifier version {version_id!r}")

    def active(self) -> ClassifierVersion | None:
        for v in self._load():
            if v.active:
                return v
        return None

    # -- state transitions -----------------------------------------

    def register(self, version: ClassifierVersion) -> ClassifierVersion:
        """Record a freshly-trained artifact. Always starts unvalidated and
        inactive - training never has the side effect of validating or
        activating (section 13.2)."""

        versions = self._load()
        if any(v.version_id == version.version_id for v in versions):
            raise ClassifierRegistryError(f"version {version.version_id!r} already registered")
        fresh = ClassifierVersion(**{**version.to_dict(), "validated": False, "active": False})
        versions.append(fresh)
        self._save(versions)
        return fresh

    def validate_version(self, version_id: str) -> ClassifierVersion:
        """Mark a trained version validated - a separate, explicit step from
        training (section 13.2). Does not change ``active``."""

        versions = self._load()
        out: list[ClassifierVersion] = []
        found: ClassifierVersion | None = None
        for v in versions:
            if v.version_id == version_id:
                v = ClassifierVersion(**{**v.to_dict(), "validated": True})
                found = v
            out.append(v)
        if found is None:
            raise ClassifierRegistryError(f"no classifier version {version_id!r}")
        self._save(out)
        return found

    def activate(self, version_id: str | None) -> ClassifierVersion | None:
        """Make ``version_id`` the one active classifier, deactivating every
        other version. ``version_id=None`` deactivates all of them (rollback
        to baseline, section 13.4). Always explicit, never a training side
        effect. Raises if the target is not validated."""

        versions = self._load()
        if version_id is not None and not any(v.version_id == version_id for v in versions):
            raise ClassifierRegistryError(f"no classifier version {version_id!r}")
        if version_id is not None:
            target = next(v for v in versions if v.version_id == version_id)
            if not target.validated:
                raise ClassifierRegistryError(
                    f"version {version_id!r} is not validated - an unvalidated "
                    "model can never become active"
                )
        out = [
            ClassifierVersion(**{**v.to_dict(), "active": (v.version_id == version_id)})
            for v in versions
        ]
        self._save(out)
        return next((v for v in out if v.active), None)
