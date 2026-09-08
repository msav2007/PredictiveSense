"""Object profile registry - ``data/objects/objects.json`` CRUD.

An :class:`ObjectProfile` is a persisted profile with a stable slug id. The
registry supports create / rename / edit-metadata / list / soft-delete. A deleted
profile's folder is *moved* to ``data/objects/_deleted/<id>/`` rather than
destroyed, and the delete result says so.

Failure policy (P4 Block 10): a corrupt ``objects.json`` raises
:class:`ObjectStoreError`; every write is atomic (``os.replace`` of a sibling
temp file) so an interrupted write never leaves a half-file.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from predictivesense.logging_setup import get_logger
from predictivesense.objects.vocab import (
    CONFUSABLE_SEEDS,
    validate_kind,
    validate_status,
)
from predictivesense.telemetry.manifest import utc_now_iso

__all__ = ["ObjectProfile", "ObjectRegistry", "ObjectStoreError"]

_LOG = get_logger(__name__)
_REGISTRY_VERSION = 1
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_MAX_SLUG_LEN = 48


class ObjectStoreError(RuntimeError):
    """``objects.json`` (or an object manifest) is missing a section, has the
    wrong types, or holds ids that do not reconcile."""


def slugify(name: str) -> str:
    """Lower-case ASCII slug; collapses runs of non-alphanumerics to ``-``."""

    slug = _SLUG_RE.sub("-", name.strip().lower()).strip("-")
    slug = slug[:_MAX_SLUG_LEN].strip("-")
    return slug or "object"


@dataclass(frozen=True)
class ObjectProfile:
    """A persisted object profile (P4 Block 9 contract)."""

    object_id: str
    name: str
    kind: str = "class"  # "class" | "instance"
    parent_class: str | None = None
    category: str | None = None
    description: str | None = None
    confusable_with: list[str] = field(default_factory=list)
    created_utc: str = ""
    updated_utc: str = ""
    sample_count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    status: str = "collecting"  # "collecting" | "ready_for_training" | "archived"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ObjectProfile":
        try:
            return cls(
                object_id=str(raw["object_id"]),
                name=str(raw["name"]),
                kind=validate_kind(str(raw.get("kind", "class"))),
                parent_class=raw.get("parent_class"),
                category=raw.get("category"),
                description=raw.get("description"),
                confusable_with=list(raw.get("confusable_with", []) or []),
                created_utc=str(raw.get("created_utc", "")),
                updated_utc=str(raw.get("updated_utc", "")),
                sample_count=int(raw.get("sample_count", 0)),
                positive_count=int(raw.get("positive_count", 0)),
                negative_count=int(raw.get("negative_count", 0)),
                status=validate_status(str(raw.get("status", "collecting"))),
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise ObjectStoreError(f"bad object profile record {raw!r}: {exc}") from exc


class ObjectRegistry:
    """Reads and writes ``<root>/objects.json``."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._path = self._root / "objects.json"
        self._deleted_dir = self._root / "_deleted"

    # -- paths -----------------------------------------------------

    @property
    def root(self) -> Path:
        return self._root

    @property
    def path(self) -> Path:
        return self._path

    def object_dir(self, object_id: str) -> Path:
        return self._root / object_id

    def images_dir(self, object_id: str) -> Path:
        return self._root / object_id / "images"

    # -- load / save ---------------------------------------------

    def _load_doc(self) -> dict[str, Any]:
        if not self._path.is_file():
            return {"version": _REGISTRY_VERSION, "profiles": [], "updated_utc": ""}
        try:
            doc = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ObjectStoreError(f"{self._path} is not readable JSON: {exc}") from exc
        if not isinstance(doc, dict) or not isinstance(doc.get("profiles"), list):
            raise ObjectStoreError(
                f"{self._path} must be an object with a 'profiles' list"
            )
        return doc

    def _profiles(self) -> list[ObjectProfile]:
        doc = self._load_doc()
        seen: set[str] = set()
        out: list[ObjectProfile] = []
        for raw in doc["profiles"]:
            prof = ObjectProfile.from_dict(raw)
            if prof.object_id in seen:
                raise ObjectStoreError(f"duplicate object_id {prof.object_id!r} in {self._path}")
            seen.add(prof.object_id)
            out.append(prof)
        return out

    def _save(self, profiles: list[ObjectProfile]) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _REGISTRY_VERSION,
            "updated_utc": utc_now_iso(),
            "profiles": [p.to_dict() for p in profiles],
        }
        _atomic_write_json(self._path, payload)

    # -- CRUD --------------------------------------------------

    def list(self) -> list[ObjectProfile]:
        return self._profiles()

    def get(self, object_id: str) -> ObjectProfile:
        for prof in self._profiles():
            if prof.object_id == object_id:
                return prof
        raise ObjectStoreError(f"no object with id {object_id!r}")

    def exists(self, object_id: str) -> bool:
        return any(p.object_id == object_id for p in self._profiles())

    def create(
        self,
        name: str,
        *,
        kind: str = "class",
        parent_class: str | None = None,
        category: str | None = None,
        description: str | None = None,
        confusable_with: list[str] | None = None,
    ) -> ObjectProfile:
        """Create a profile. The slug is derived from ``name`` and disambiguated
        (``-2``, ``-3`` ...) on collision. ``confusable_with`` defaults to the
        seed list for the name (Block 4.5.19)."""

        validate_kind(kind)
        if kind == "instance" and not parent_class:
            raise ObjectStoreError("an instance object needs a parent_class")

        profiles = self._profiles()
        taken = {p.object_id for p in profiles}
        base = slugify(name)
        object_id = base
        n = 2
        while object_id in taken:
            object_id = f"{base}-{n}"
            n += 1

        if confusable_with is None:
            confusable_with = list(CONFUSABLE_SEEDS.get(base, []))

        now = utc_now_iso()
        prof = ObjectProfile(
            object_id=object_id,
            name=name.strip() or object_id,
            kind=kind,
            parent_class=parent_class,
            category=category,
            description=description,
            confusable_with=confusable_with,
            created_utc=now,
            updated_utc=now,
        )
        profiles.append(prof)
        self._save(profiles)
        self.images_dir(object_id).mkdir(parents=True, exist_ok=True)
        _LOG.info("object profile created: %s (%s)", object_id, kind)
        return prof

    def update(self, object_id: str, **fields: Any) -> ObjectProfile:
        """Patch metadata. ``object_id``, ``created_utc`` and the count fields are
        not user-editable here (counts are maintained by :meth:`set_counts`)."""

        profiles = self._profiles()
        for i, prof in enumerate(profiles):
            if prof.object_id != object_id:
                continue
            allowed = {
                "name",
                "kind",
                "parent_class",
                "category",
                "description",
                "confusable_with",
                "status",
            }
            patch = {k: v for k, v in fields.items() if k in allowed}
            if "kind" in patch:
                validate_kind(str(patch["kind"]))
            if "status" in patch:
                validate_status(str(patch["status"]))
            if patch.get("kind") == "instance" and not (
                patch.get("parent_class") or prof.parent_class
            ):
                raise ObjectStoreError("an instance object needs a parent_class")
            if "confusable_with" in patch and patch["confusable_with"] is not None:
                patch["confusable_with"] = [str(c) for c in patch["confusable_with"]]
            updated = replace(prof, updated_utc=utc_now_iso(), **patch)
            profiles[i] = updated
            self._save(profiles)
            _LOG.info("object profile updated: %s %s", object_id, sorted(patch))
            return updated
        raise ObjectStoreError(f"no object with id {object_id!r}")

    def set_counts(
        self, object_id: str, *, sample_count: int, positive_count: int, negative_count: int
    ) -> ObjectProfile:
        profiles = self._profiles()
        for i, prof in enumerate(profiles):
            if prof.object_id == object_id:
                updated = replace(
                    prof,
                    sample_count=int(sample_count),
                    positive_count=int(positive_count),
                    negative_count=int(negative_count),
                    updated_utc=utc_now_iso(),
                )
                profiles[i] = updated
                self._save(profiles)
                return updated
        raise ObjectStoreError(f"no object with id {object_id!r}")

    def soft_delete(self, object_id: str) -> dict[str, Any]:
        """Remove the profile from the registry and *move* its folder to
        ``_deleted/``. Returns where the folder went."""

        profiles = self._profiles()
        remaining = [p for p in profiles if p.object_id != object_id]
        if len(remaining) == len(profiles):
            raise ObjectStoreError(f"no object with id {object_id!r}")

        moved_to: str | None = None
        src = self.object_dir(object_id)
        if src.is_dir():
            self._deleted_dir.mkdir(parents=True, exist_ok=True)
            dest = self._deleted_dir / object_id
            if dest.exists():
                dest = self._deleted_dir / f"{object_id}-{utc_now_iso().replace(':', '').replace('.', '')}"
            shutil.move(str(src), str(dest))
            moved_to = str(dest)
            _LOG.info("object %s soft-deleted: folder moved to %s", object_id, dest)

        self._save(remaining)
        return {"object_id": object_id, "soft_deleted": True, "moved_to": moved_to}


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
