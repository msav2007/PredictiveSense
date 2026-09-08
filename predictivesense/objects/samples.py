"""Per-object sample store - ``data/objects/<id>/manifest.json`` + ``images/``.

Every sample carries **exactly one bounding box** (``[x, y, w, h]`` in pixels,
required, inside the image), condition tags from the fixed vocabularies, a
quality block, and full provenance (capture time, camera device label, source,
original filename, app git commit, ``consent_ack``).

The manifest is written atomically (``os.replace`` of a temp file). Sample
deletion is soft: the image files move to ``<id>/_deleted/`` and the record moves
to a ``deleted`` list (P4 Block 10 - never a hard delete).

This module is stdlib + numpy only; the caller decodes / encodes images (via the
camera layer, where ``cv2`` is allowed) and hands bytes in.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from predictivesense.logging_setup import get_logger
from predictivesense.objects.registry import ObjectStoreError
from predictivesense.objects.vocab import default_conditions, validate_conditions, validate_role
from predictivesense.telemetry.manifest import git_state, utc_now_iso

__all__ = ["ObjectSample", "SampleStore"]

_LOG = get_logger(__name__)
_MANIFEST_VERSION = 1
_GIT_COMMIT_CACHE: str | None = None


def _git_commit() -> str:
    """Repo commit for sample provenance. Resolved once, lazily (no subprocess at
    import time)."""

    global _GIT_COMMIT_CACHE
    if _GIT_COMMIT_CACHE is None:
        _GIT_COMMIT_CACHE, _ = git_state()
    return _GIT_COMMIT_CACHE


@dataclass(frozen=True)
class ObjectSample:
    """One stored sample (P4 Block 9 contract)."""

    sample_id: str
    object_id: str
    path: str  # relative to the object dir, e.g. "images/<sid>.jpg"
    role: str = "positive"  # "positive" | "negative" | "hard_negative"
    negative_for: list[str] = field(default_factory=list)
    box: list[float] = field(default_factory=list)  # [x, y, w, h] pixels, required
    width: int = 0
    height: int = 0
    conditions: dict[str, str] = field(default_factory=default_conditions)
    quality: dict[str, Any] = field(default_factory=dict)
    source: str = "camera"  # "camera" | "upload"
    device_label: str | None = None
    original_filename: str | None = None
    captured_utc: str = ""
    git_commit: str = ""
    consent_ack: bool = False
    thumb_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_box(box: Any, width: int, height: int) -> list[float]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ObjectStoreError(f"box must be [x, y, w, h], got {box!r}")
    try:
        x, y, w, h = (float(v) for v in box)
    except (TypeError, ValueError) as exc:
        raise ObjectStoreError(f"box values are not numeric: {box!r}") from exc
    if w <= 0 or h <= 0:
        raise ObjectStoreError(f"box has non-positive width/height: {box!r}")
    # A 1px tolerance absorbs rounding from the browser canvas.
    if x < -1 or y < -1 or (x + w) > width + 1 or (y + h) > height + 1:
        raise ObjectStoreError(
            f"box {box!r} falls outside the {width}x{height} image bounds"
        )
    x = min(max(0.0, x), float(width))
    y = min(max(0.0, y), float(height))
    w = min(w, float(width) - x)
    h = min(h, float(height) - y)
    return [round(x, 2), round(y, 2), round(w, 2), round(h, 2)]


class SampleStore:
    """One object's sample manifest under ``<root>/<object_id>/``."""

    def __init__(self, object_dir: str | Path, object_id: str) -> None:
        self._dir = Path(object_dir)
        self._object_id = object_id
        self._manifest = self._dir / "manifest.json"
        self._images = self._dir / "images"
        self._deleted = self._dir / "_deleted"

    # -- load / save ---------------------------------------------

    def _load(self) -> dict[str, Any]:
        if not self._manifest.is_file():
            return {
                "version": _MANIFEST_VERSION,
                "object_id": self._object_id,
                "samples": [],
                "deleted": [],
            }
        try:
            doc = json.loads(self._manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ObjectStoreError(f"{self._manifest} is not readable JSON: {exc}") from exc
        if not isinstance(doc, dict) or not isinstance(doc.get("samples"), list):
            raise ObjectStoreError(f"{self._manifest} must have a 'samples' list")
        doc.setdefault("deleted", [])
        return doc

    def _save(self, doc: dict[str, Any]) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        tmp = self._manifest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, self._manifest)

    # -- read -------------------------------------------------

    def list(self) -> list[dict[str, Any]]:
        return list(self._load()["samples"])

    def get(self, sample_id: str) -> dict[str, Any]:
        for s in self._load()["samples"]:
            if s["sample_id"] == sample_id:
                return s
        raise ObjectStoreError(f"no sample {sample_id!r} for object {self._object_id!r}")

    def counts(self) -> dict[str, int]:
        samples = self._load()["samples"]
        pos = sum(1 for s in samples if s.get("role") == "positive")
        neg = sum(1 for s in samples if s.get("role") in ("negative", "hard_negative"))
        return {"sample_count": len(samples), "positive_count": pos, "negative_count": neg}

    def existing_hashes(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for s in self._load()["samples"]:
            phash = (s.get("quality") or {}).get("phash")
            if phash:
                out.append((s["sample_id"], phash))
        return out

    def image_path(self, sample_id: str) -> Path:
        return self._dir / self.get(sample_id)["path"]

    # -- write ------------------------------------------------

    def add(
        self,
        *,
        image_bytes: bytes,
        width: int,
        height: int,
        box: Any,
        conditions: dict[str, Any] | None,
        role: str = "positive",
        negative_for: list[str] | None = None,
        quality: dict[str, Any] | None = None,
        source: str = "camera",
        device_label: str | None = None,
        original_filename: str | None = None,
        consent_ack: bool = False,
        thumb_bytes: bytes | None = None,
    ) -> ObjectSample:
        validate_role(role)
        conds = validate_conditions(conditions)
        clean_box = _validate_box(box, int(width), int(height))
        neg_for = [str(o) for o in (negative_for or [])]
        if role in ("negative", "hard_negative") and not neg_for:
            neg_for = [self._object_id]

        sample_id = uuid.uuid4().hex
        self._images.mkdir(parents=True, exist_ok=True)
        rel = f"images/{sample_id}.jpg"
        (self._dir / rel).write_bytes(image_bytes)
        thumb_rel: str | None = None
        if thumb_bytes:
            thumb_rel = f"images/{sample_id}.thumb.jpg"
            (self._dir / thumb_rel).write_bytes(thumb_bytes)

        sample = ObjectSample(
            sample_id=sample_id,
            object_id=self._object_id,
            path=rel,
            role=role,
            negative_for=neg_for,
            box=clean_box,
            width=int(width),
            height=int(height),
            conditions=conds,
            quality=dict(quality or {}),
            source=source,
            device_label=device_label,
            original_filename=original_filename,
            captured_utc=utc_now_iso(),
            git_commit=_git_commit(),
            consent_ack=bool(consent_ack),
            thumb_path=thumb_rel,
        )
        doc = self._load()
        doc["samples"].append(sample.to_dict())
        self._save(doc)
        _LOG.info("object %s: sample %s stored (%s)", self._object_id, sample_id, role)
        return sample

    def update(self, sample_id: str, **fields: Any) -> dict[str, Any]:
        """Adjust box, tags or role in place."""

        doc = self._load()
        for s in doc["samples"]:
            if s["sample_id"] != sample_id:
                continue
            if "box" in fields and fields["box"] is not None:
                s["box"] = _validate_box(fields["box"], int(s["width"]), int(s["height"]))
            if "conditions" in fields and fields["conditions"] is not None:
                s["conditions"] = validate_conditions(fields["conditions"])
            if "role" in fields and fields["role"] is not None:
                s["role"] = validate_role(str(fields["role"]))
                if s["role"] in ("negative", "hard_negative") and not s.get("negative_for"):
                    s["negative_for"] = [self._object_id]
            if "negative_for" in fields and fields["negative_for"] is not None:
                s["negative_for"] = [str(o) for o in fields["negative_for"]]
            if "quality" in fields and fields["quality"] is not None:
                s["quality"] = dict(fields["quality"])
            self._save(doc)
            _LOG.info("object %s: sample %s updated %s", self._object_id, sample_id, sorted(fields))
            return dict(s)
        raise ObjectStoreError(f"no sample {sample_id!r} for object {self._object_id!r}")

    def soft_delete(self, sample_id: str) -> dict[str, Any]:
        doc = self._load()
        kept: list[dict[str, Any]] = []
        removed: dict[str, Any] | None = None
        for s in doc["samples"]:
            if s["sample_id"] == sample_id:
                removed = s
            else:
                kept.append(s)
        if removed is None:
            raise ObjectStoreError(f"no sample {sample_id!r} for object {self._object_id!r}")

        self._deleted.mkdir(parents=True, exist_ok=True)
        for rel_key in ("path", "thumb_path"):
            rel = removed.get(rel_key)
            if not rel:
                continue
            src = self._dir / rel
            if src.is_file():
                shutil.move(str(src), str(self._deleted / Path(rel).name))
        removed["deleted_utc"] = utc_now_iso()
        doc["samples"] = kept
        doc["deleted"].append(removed)
        self._save(doc)
        _LOG.info("object %s: sample %s soft-deleted", self._object_id, sample_id)
        return {"sample_id": sample_id, "soft_deleted": True}
