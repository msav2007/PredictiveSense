"""Bulk-upload staging store - ``data/objects/<id>/_staging/<batch_id>/`` (Phase 7).

A *batch* is a set of images the developer uploaded in one operation, held apart
from the object's committed samples until **Save all**. Each batch is a
directory with a ``batch.json`` manifest plus the staged image + thumbnail
files. Nothing here reaches ``SampleStore`` - so staged images are excluded from
every sample count, coverage figure and dataset export automatically (they are
never in ``manifest.json``).

Lifecycle:

* :meth:`create` writes the manifest and the item skeletons.
* :meth:`add_staged_image` writes one image + its thumbnail (bytes come from the
  API layer, which owns ``cv2``).
* :meth:`update_item` / :meth:`set_status` mutate the manifest atomically.
* :meth:`discard` removes the whole batch directory.
* :meth:`cleanup_stale` drops batches older than a configured age (called at
  Studio entry so an abandoned batch never lingers).

Stdlib + numpy only (matches the rest of ``predictivesense/objects/``).
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from predictivesense.logging_setup import get_logger
from predictivesense.telemetry.manifest import utc_now_iso

__all__ = ["BatchItem", "BatchStore", "BatchStoreError", "ITEM_STATUSES"]

_LOG = get_logger(__name__)
_MANIFEST_NAME = "batch.json"
_MANIFEST_VERSION = 1

# BLOCK 6 BatchItem.status
ITEM_STATUSES = ("ready", "manual_required", "edited", "flagged", "error")


class BatchStoreError(RuntimeError):
    """A batch directory / manifest is missing, malformed, or names an unknown
    item."""


@dataclass
class BatchItem:
    """One staged image in a batch (BLOCK 6 contract). Mutable: the reviewer
    PATCHes ``box`` / ``role`` / ``conditions`` / ``box_confirmed_by_human``."""

    item_id: str
    filename: str
    staged_path: str | None = None  # relative to the batch dir
    thumbnail_path: str | None = None
    width: int = 0
    height: int = 0
    box: list[float] | None = None
    status: str = "manual_required"
    proposal_source: str | None = None  # "detector" | "manual" | "default_centred"
    proposal_raw_class: str | None = None
    proposal_score: float | None = None
    box_confirmed_by_human: bool = False
    role: str = "positive"
    negative_for: list[str] = field(default_factory=list)
    conditions: dict[str, str] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BatchItem":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in raw.items() if k in known})


class BatchStore:
    """Staging for one object under ``<object_dir>/_staging/``."""

    def __init__(self, object_dir: str | Path, object_id: str) -> None:
        self._dir = Path(object_dir)
        self._object_id = object_id
        self._staging = self._dir / "_staging"

    # -- paths -----------------------------------------------------

    @property
    def staging_root(self) -> Path:
        return self._staging

    def batch_dir(self, batch_id: str) -> Path:
        _guard_id(batch_id)
        return self._staging / batch_id

    def manifest_path(self, batch_id: str) -> Path:
        return self.batch_dir(batch_id) / _MANIFEST_NAME

    def staged_image_path(self, batch_id: str, item_id: str) -> Path:
        item = self._item(self.load(batch_id), item_id)
        rel = item.get("staged_path")
        if not rel:
            raise BatchStoreError(f"item {item_id!r} has no staged image")
        return self.batch_dir(batch_id) / rel

    def thumbnail_image_path(self, batch_id: str, item_id: str) -> Path:
        item = self._item(self.load(batch_id), item_id)
        rel = item.get("thumbnail_path") or item.get("staged_path")
        if not rel:
            raise BatchStoreError(f"item {item_id!r} has no image")
        return self.batch_dir(batch_id) / rel

    # -- load / save ---------------------------------------------

    def exists(self, batch_id: str) -> bool:
        return self.manifest_path(batch_id).is_file()

    def load(self, batch_id: str) -> dict[str, Any]:
        path = self.manifest_path(batch_id)
        if not path.is_file():
            raise BatchStoreError(f"no batch {batch_id!r} for object {self._object_id!r}")
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise BatchStoreError(f"{path} is not readable JSON: {exc}") from exc
        if not isinstance(doc, dict) or not isinstance(doc.get("items"), list):
            raise BatchStoreError(f"{path} must be an object with an 'items' list")
        return doc

    def _save(self, batch_id: str, doc: dict[str, Any]) -> None:
        path = self.manifest_path(batch_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, path)

    # -- create / write -----------------------------------------

    def create(
        self, batch_id: str, *, role: str, items: list[BatchItem]
    ) -> dict[str, Any]:
        _guard_id(batch_id)
        bdir = self.batch_dir(batch_id)
        if bdir.exists():
            raise BatchStoreError(f"batch {batch_id!r} already exists")
        bdir.mkdir(parents=True, exist_ok=True)
        doc = {
            "version": _MANIFEST_VERSION,
            "batch_id": batch_id,
            "object_id": self._object_id,
            "created_utc": utc_now_iso(),
            "role": role,
            "status": "processing",
            "total": len(items),
            "processed": 0,
            "items": [it.to_dict() for it in items],
        }
        self._save(batch_id, doc)
        _LOG.info(
            "batch %s created for object %s: %d item(s), role=%s",
            batch_id, self._object_id, len(items), role,
        )
        return doc

    def add_staged_image(
        self,
        batch_id: str,
        item_id: str,
        *,
        image_bytes: bytes,
        thumb_bytes: bytes,
        width: int,
        height: int,
    ) -> None:
        bdir = self.batch_dir(batch_id)
        rel = f"{item_id}.jpg"
        thumb_rel = f"{item_id}.thumb.jpg"
        (bdir / rel).write_bytes(image_bytes)
        (bdir / thumb_rel).write_bytes(thumb_bytes)
        self.update_item(
            batch_id,
            item_id,
            staged_path=rel,
            thumbnail_path=thumb_rel,
            width=int(width),
            height=int(height),
        )

    def update_item(self, batch_id: str, item_id: str, **fields: Any) -> dict[str, Any]:
        doc = self.load(batch_id)
        for it in doc["items"]:
            if it["item_id"] != item_id:
                continue
            allowed = set(BatchItem.__dataclass_fields__)  # type: ignore[attr-defined]
            for key, value in fields.items():
                if key not in allowed:
                    raise BatchStoreError(f"unknown batch item field {key!r}")
                it[key] = value
            self._save(batch_id, doc)
            return dict(it)
        raise BatchStoreError(f"no item {item_id!r} in batch {batch_id!r}")

    def overwrite(self, batch_id: str, doc: dict[str, Any]) -> None:
        """Replace the whole manifest (used after a partial save trims the
        committed items). ``doc`` must already have the ``items`` list."""

        if not isinstance(doc, dict) or not isinstance(doc.get("items"), list):
            raise BatchStoreError("manifest doc must carry an 'items' list")
        self._save(batch_id, doc)

    def set_status(self, batch_id: str, status: str) -> None:
        doc = self.load(batch_id)
        doc["status"] = status
        self._save(batch_id, doc)

    def set_progress(self, batch_id: str, processed: int) -> None:
        doc = self.load(batch_id)
        doc["processed"] = int(processed)
        self._save(batch_id, doc)

    # -- read helpers ------------------------------------------

    def list_batches(self) -> list[str]:
        if not self._staging.is_dir():
            return []
        return sorted(
            p.name for p in self._staging.iterdir()
            if p.is_dir() and (p / _MANIFEST_NAME).is_file()
        )

    def items(self, batch_id: str) -> list[dict[str, Any]]:
        return list(self.load(batch_id)["items"])

    @staticmethod
    def _item(doc: dict[str, Any], item_id: str) -> dict[str, Any]:
        for it in doc["items"]:
            if it["item_id"] == item_id:
                return it
        raise BatchStoreError(f"no item {item_id!r} in batch {doc.get('batch_id')!r}")

    def existing_batch_hashes(
        self, batch_id: str, *, exclude_item: str | None = None
    ) -> list[tuple[str, str]]:
        """``(item_id, phash)`` for items in this batch that already carry one -
        used for within-batch near-duplicate detection."""

        out: list[tuple[str, str]] = []
        for it in self.load(batch_id)["items"]:
            if exclude_item is not None and it["item_id"] == exclude_item:
                continue
            phash = (it.get("quality") or {}).get("phash")
            if phash:
                out.append((it["item_id"], phash))
        return out

    # -- discard / cleanup ------------------------------------

    def discard(self, batch_id: str) -> dict[str, Any]:
        bdir = self.batch_dir(batch_id)
        removed = bdir.is_dir()
        if removed:
            shutil.rmtree(bdir, ignore_errors=True)
        _LOG.info("batch %s discarded for object %s (existed=%s)", batch_id, self._object_id, removed)
        return {"batch_id": batch_id, "discarded": True, "existed": removed}

    def cleanup_stale(self, ttl_hours: float) -> list[str]:
        """Remove batches older than ``ttl_hours`` (by manifest ``created_utc``,
        falling back to the directory mtime). Returns the ids removed."""

        if not self._staging.is_dir():
            return []
        cutoff = datetime.now(timezone.utc).timestamp() - float(ttl_hours) * 3600.0
        removed: list[str] = []
        for p in self._staging.iterdir():
            if not p.is_dir():
                continue
            age_ts = _batch_created_ts(p)
            if age_ts is not None and age_ts < cutoff:
                shutil.rmtree(p, ignore_errors=True)
                removed.append(p.name)
        if removed:
            _LOG.info("removed %d stale batch(es) for object %s: %s", len(removed), self._object_id, removed)
        return removed


def _guard_id(batch_id: str) -> None:
    if not batch_id or "/" in batch_id or "\\" in batch_id or batch_id in {".", ".."}:
        raise BatchStoreError(f"unsafe batch id {batch_id!r}")


def _batch_created_ts(batch_dir: Path) -> float | None:
    manifest = batch_dir / _MANIFEST_NAME
    if manifest.is_file():
        try:
            doc = json.loads(manifest.read_text(encoding="utf-8"))
            created = doc.get("created_utc")
            if created:
                return datetime.fromisoformat(str(created).replace("Z", "+00:00")).timestamp()
        except (OSError, ValueError):
            pass
    try:
        return batch_dir.stat().st_mtime
    except OSError:
        return None
