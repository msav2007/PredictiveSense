"""Read/write the COCO-detection-format annotation store.

COCO detection JSON (``images``, ``annotations``, ``categories``) is chosen so
the same files feed standard fine-tuning tooling in a later phase without
conversion (recorded in ``docs/decisions.md``).

Per-image extras beyond stock COCO:

* ``seeded`` - was this frame pre-seeded with detector output during labelling?
  Seeded labelling biases recall upward, so the harness reports the unseeded
  subset separately (BLOCK 3.1.4).
* ``labelled`` - has the frame been marked done?
* ``ps_provenance`` - source clip, capture timestamp, session id, camera device,
  clip condition tags (BLOCK 3.1.2).

Failure policy (BLOCK 10): a corrupt or schema-invalid file raises
:class:`CocoStoreError`; :meth:`save` never overwrites an existing file without
first writing a ``.bak`` copy.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

from predictivesense.logging_setup import get_logger

__all__ = ["CocoStore", "CocoStoreError", "domain_categories"]

_LOG = get_logger(__name__)

# (x, y, w, h) in pixels, COCO bbox convention (top-left origin).
CocoBBox = tuple[float, float, float, float]


class CocoStoreError(RuntimeError):
    """A COCO file is missing a required section, has the wrong types, or holds
    ids that do not reconcile."""


def domain_categories(names: Iterable[str]) -> list[dict[str, Any]]:
    """A COCO ``categories`` list from an ordered vocabulary. ids start at 1."""

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    next_id = 1
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        out.append({"id": next_id, "name": name, "supercategory": "object"})
        next_id += 1
    if not out:
        raise CocoStoreError("domain vocabulary is empty")
    return out


@dataclass(frozen=True)
class _Box:
    category_id: int
    bbox: CocoBBox  # x, y, w, h


class CocoStore:
    """An in-memory COCO detection store with id allocation and validation."""

    def __init__(self, doc: dict[str, Any]) -> None:
        _validate(doc)
        self._images: dict[int, dict[str, Any]] = {
            int(im["id"]): dict(im) for im in doc["images"]
        }
        self._annotations: dict[int, dict[str, Any]] = {
            int(an["id"]): dict(an) for an in doc["annotations"]
        }
        self._categories: list[dict[str, Any]] = [dict(c) for c in doc["categories"]]
        self._cat_by_name = {c["name"]: int(c["id"]) for c in self._categories}
        self._cat_by_id = {int(c["id"]): c["name"] for c in self._categories}
        self._info: dict[str, Any] = dict(doc.get("info", {}))
        self._next_image_id = (max(self._images) + 1) if self._images else 1
        self._next_ann_id = (max(self._annotations) + 1) if self._annotations else 1

    # -- construction ------------------------------------------------

    @classmethod
    def create(cls, categories: list[dict[str, Any]], *, info: dict[str, Any] | None = None) -> "CocoStore":
        return cls(
            {
                "info": info or {},
                "images": [],
                "annotations": [],
                "categories": categories,
            }
        )

    @classmethod
    def load(cls, path: str | Path) -> "CocoStore":
        p = Path(path)
        if not p.is_file():
            raise CocoStoreError(f"annotation file not found: {p}")
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise CocoStoreError(f"{p} is not readable JSON: {exc}") from exc
        if not isinstance(doc, dict):
            raise CocoStoreError(f"{p} top level must be a JSON object")
        return cls(doc)

    @classmethod
    def load_or_create(
        cls, path: str | Path, categories: list[dict[str, Any]], *, info: dict[str, Any] | None = None
    ) -> "CocoStore":
        p = Path(path)
        if p.is_file():
            store = cls.load(p)
            store._reconcile_categories(categories)
            return store
        return cls.create(categories, info=info)

    # -- categories ------------------------------------------------

    @property
    def category_names(self) -> tuple[str, ...]:
        return tuple(c["name"] for c in self._categories)

    def category_id(self, name: str) -> int:
        try:
            return self._cat_by_name[name]
        except KeyError as exc:
            raise CocoStoreError(
                f"class {name!r} is not in the store vocabulary {self.category_names}"
            ) from exc

    def category_name(self, cid: int) -> str:
        try:
            return self._cat_by_id[int(cid)]
        except KeyError as exc:
            raise CocoStoreError(f"category id {cid} is not in the store") from exc

    def _reconcile_categories(self, wanted: list[dict[str, Any]]) -> None:
        have = {c["name"]: int(c["id"]) for c in self._categories}
        want = {c["name"]: int(c["id"]) for c in wanted}
        if have != want:
            raise CocoStoreError(
                "the store's categories do not match the configured domain "
                f"vocabulary.\n  store: {sorted(have.items())}\n  config: {sorted(want.items())}\n"
                "Rebuild the store or align policy.domain_classes."
            )

    # -- images ------------------------------------------------

    def add_image(
        self,
        *,
        file_name: str,
        width: int,
        height: int,
        session_id: str,
        provenance: dict[str, Any] | None = None,
        image_id: int | None = None,
    ) -> int:
        """Register a frame. Returns its image id. Re-adding the same
        ``file_name`` returns the existing id (idempotent frame index build)."""

        for iid, im in self._images.items():
            if im["file_name"] == file_name:
                return iid
        iid = image_id if image_id is not None else self._alloc_image_id()
        if iid in self._images:
            raise CocoStoreError(f"image id {iid} already exists")
        self._images[iid] = {
            "id": iid,
            "file_name": file_name,
            "width": int(width),
            "height": int(height),
            "ps_session_id": session_id,
            "ps_provenance": dict(provenance or {}),
            "seeded": False,
            "labelled": False,
        }
        return iid

    def _alloc_image_id(self) -> int:
        iid = self._next_image_id
        self._next_image_id += 1
        return iid

    def has_image(self, image_id: int) -> bool:
        return int(image_id) in self._images

    def image(self, image_id: int) -> dict[str, Any]:
        try:
            return dict(self._images[int(image_id)])
        except KeyError as exc:
            raise CocoStoreError(f"no image with id {image_id}") from exc

    def image_ids(self) -> list[int]:
        return sorted(self._images)

    def session_of(self, image_id: int) -> str:
        return str(self._images[int(image_id)]["ps_session_id"])

    def annotations_for(self, image_id: int) -> list[dict[str, Any]]:
        return [
            dict(an)
            for an in self._annotations.values()
            if int(an["image_id"]) == int(image_id)
        ]

    # -- labelling ------------------------------------------------

    def set_frame_boxes(
        self,
        image_id: int,
        boxes: list[dict[str, Any]],
        *,
        seeded: bool,
        labelled: bool = True,
    ) -> list[int]:
        """Replace every annotation for ``image_id``. Each box is
        ``{"category": name | "category_id": int, "bbox": [x, y, w, h]}``.
        Records ``seeded`` and sets ``labelled``. Returns the new annotation ids.
        """

        if int(image_id) not in self._images:
            raise CocoStoreError(f"cannot label unknown image id {image_id}")

        # drop existing annotations for this image
        for aid in [
            aid
            for aid, an in self._annotations.items()
            if int(an["image_id"]) == int(image_id)
        ]:
            del self._annotations[aid]

        new_ids: list[int] = []
        for box in boxes:
            cid = (
                int(box["category_id"])
                if "category_id" in box
                else self.category_id(str(box["category"]))
            )
            if cid not in self._cat_by_id:
                raise CocoStoreError(f"box category id {cid} not in vocabulary")
            x, y, w, h = (float(v) for v in box["bbox"])
            if w <= 0 or h <= 0:
                raise CocoStoreError(f"box for image {image_id} has non-positive size: {box['bbox']}")
            aid = self._alloc_ann_id()
            self._annotations[aid] = {
                "id": aid,
                "image_id": int(image_id),
                "category_id": cid,
                "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                "area": round(w * h, 2),
                "iscrowd": 0,
            }
            new_ids.append(aid)

        im = self._images[int(image_id)]
        im["seeded"] = bool(seeded)
        im["labelled"] = bool(labelled)
        return new_ids

    def _alloc_ann_id(self) -> int:
        aid = self._next_ann_id
        self._next_ann_id += 1
        return aid

    # -- serialisation ------------------------------------------------

    def to_doc(self) -> dict[str, Any]:
        return {
            "info": dict(self._info),
            "images": [self._images[i] for i in sorted(self._images)],
            "annotations": [self._annotations[a] for a in sorted(self._annotations)],
            "categories": [dict(c) for c in self._categories],
        }

    def save(self, path: str | Path) -> Path:
        """Write pretty JSON. If ``path`` exists it is copied to ``path + '.bak'``
        first (BLOCK 10 - never overwrite an annotation file without a backup)."""

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.is_file():
            backup = p.with_suffix(p.suffix + ".bak")
            shutil.copy2(p, backup)
            _LOG.info("backed up %s -> %s", p, backup)
        payload = json.dumps(self.to_doc(), indent=2, ensure_ascii=False, sort_keys=False)
        p.write_text(payload + "\n", encoding="utf-8")
        return p

    def content_hash(self) -> str:
        """Stable hash of images + annotations + categories (not ``info``)."""

        doc = self.to_doc()
        blob = json.dumps(
            {k: doc[k] for k in ("images", "annotations", "categories")},
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256(blob.encode("utf-8")).hexdigest()

    # -- counts ------------------------------------------------

    def counts(self) -> dict[str, Any]:
        images = list(self._images.values())
        labelled = [im for im in images if im.get("labelled")]
        seeded = [im for im in labelled if im.get("seeded")]
        per_class: dict[str, int] = {name: 0 for name in self.category_names}
        for an in self._annotations.values():
            per_class[self.category_name(int(an["category_id"]))] += 1
        per_session: dict[str, dict[str, int]] = {}
        for im in images:
            sess = str(im["ps_session_id"])
            row = per_session.setdefault(sess, {"frames": 0, "labelled": 0, "unseeded": 0})
            row["frames"] += 1
            if im.get("labelled"):
                row["labelled"] += 1
                if not im.get("seeded"):
                    row["unseeded"] += 1
        n_labelled = len(labelled)
        n_unseeded = n_labelled - len(seeded)
        return {
            "images": len(images),
            "labelled": n_labelled,
            "seeded": len(seeded),
            "unseeded": n_unseeded,
            "unseeded_fraction": (n_unseeded / n_labelled) if n_labelled else 0.0,
            "annotations": len(self._annotations),
            "sessions": len(per_session),
            "per_class": per_class,
            "per_session": per_session,
        }


# -- validation ------------------------------------------------


def _validate(doc: dict[str, Any]) -> None:
    for section in ("images", "annotations", "categories"):
        if section not in doc:
            raise CocoStoreError(f"COCO doc missing required section {section!r}")
        if not isinstance(doc[section], list):
            raise CocoStoreError(f"COCO section {section!r} must be a list")

    cat_ids: set[int] = set()
    for c in doc["categories"]:
        if not isinstance(c, dict) or "id" not in c or "name" not in c:
            raise CocoStoreError(f"bad category record: {c!r}")
        cid = _as_int(c["id"], f"category id {c!r}")
        if cid in cat_ids:
            raise CocoStoreError(f"duplicate category id {cid}")
        cat_ids.add(cid)

    img_ids: set[int] = set()
    for im in doc["images"]:
        if not isinstance(im, dict):
            raise CocoStoreError(f"bad image record: {im!r}")
        for key in ("id", "file_name", "width", "height"):
            if key not in im:
                raise CocoStoreError(f"image record missing {key!r}: {im!r}")
        iid = _as_int(im["id"], f"image id {im!r}")
        if iid in img_ids:
            raise CocoStoreError(f"duplicate image id {iid}")
        img_ids.add(iid)
        _as_int(im["width"], "image width")
        _as_int(im["height"], "image height")

    ann_ids: set[int] = set()
    for an in doc["annotations"]:
        if not isinstance(an, dict):
            raise CocoStoreError(f"bad annotation record: {an!r}")
        for key in ("id", "image_id", "category_id", "bbox"):
            if key not in an:
                raise CocoStoreError(f"annotation missing {key!r}: {an!r}")
        aid = _as_int(an["id"], f"annotation id {an!r}")
        if aid in ann_ids:
            raise CocoStoreError(f"duplicate annotation id {aid}")
        ann_ids.add(aid)
        if _as_int(an["image_id"], "annotation image_id") not in img_ids:
            raise CocoStoreError(f"annotation {aid} references unknown image {an['image_id']}")
        if _as_int(an["category_id"], "annotation category_id") not in cat_ids:
            raise CocoStoreError(f"annotation {aid} references unknown category {an['category_id']}")
        bbox = an["bbox"]
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise CocoStoreError(f"annotation {aid} bbox must be [x, y, w, h]: {bbox!r}")
        try:
            _x, _y, w, h = (float(v) for v in bbox)
        except (TypeError, ValueError) as exc:
            raise CocoStoreError(f"annotation {aid} bbox is not numeric: {bbox!r}") from exc
        if w <= 0 or h <= 0:
            raise CocoStoreError(f"annotation {aid} bbox has non-positive w/h: {bbox!r}")


def _as_int(value: Any, what: str) -> int:
    try:
        i = int(value)
    except (TypeError, ValueError) as exc:
        raise CocoStoreError(f"{what} is not an integer: {value!r}") from exc
    if isinstance(value, float) and not float(value).is_integer():
        raise CocoStoreError(f"{what} is not an integer: {value!r}")
    return i
