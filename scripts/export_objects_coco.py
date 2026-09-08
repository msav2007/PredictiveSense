"""Export the Object Learning Studio dataset to COCO detection format (Phase 4).

    python scripts\\export_objects_coco.py --out data\\objects\\coco_train.json

Converts every object's samples into a COCO detection file, with a
sample-disjoint train/val split of its own (each object appears in both splits;
no sample appears in both). Positives contribute one box; negatives and hard
negatives contribute an image with **no** box for that class (a hard-negative
signal for a later fine-tuning phase).

**The object dataset is training data. The Phase 2.5 evaluation set
(``data/eval``) is test data. They must never merge.** This exporter refuses to
run if any object image content-hash, image path, or source identifier also
appears under ``data/eval`` (P4 Block 4.6.22 / Block 10). It never touches
``data/eval``.

No model is trained or fine-tuned here.
"""

from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

from predictivesense.config.settings import ConfigError, ValidationError, load_config
from predictivesense.logging_setup import configure_logging, get_logger
from predictivesense.objects.registry import ObjectRegistry, ObjectStoreError
from predictivesense.objects.samples import SampleStore

_LOG = get_logger("predictivesense.scripts.export_objects_coco")
_REPO_ROOT = Path(__file__).resolve().parents[1]


def image_content_hash(path: Path) -> str:
    """SHA-256 of a file's bytes."""

    return sha256(Path(path).read_bytes()).hexdigest()


def _iter_object_samples(root: Path):
    """Yield ``(object_id, sample_record, image_path)`` for every stored sample."""

    reg = ObjectRegistry(root)
    for prof in reg.list():
        store = SampleStore(reg.object_dir(prof.object_id), prof.object_id)
        for sample in store.list():
            img = reg.object_dir(prof.object_id) / sample["path"]
            yield prof, sample, img


def check_dataset_separation(objects_root: Path, eval_root: Path) -> list[str]:
    """Return a list of separation violations between ``data/objects`` and
    ``data/eval``. Empty list == disjoint.

    Checks three things (Block 4.6.22): image content hash, image path, and
    source identifier (an uploaded original filename / a recorded source clip).
    """

    violations: list[str] = []
    obj_hashes: dict[str, str] = {}
    obj_sources: set[str] = set()
    obj_paths: set[str] = set()
    if objects_root.is_dir():
        for _prof, sample, img in _iter_object_samples(objects_root):
            if img.is_file():
                obj_hashes[image_content_hash(img)] = str(img)
                obj_paths.add(img.resolve().as_posix())
            orig = (sample.get("original_filename") or "").strip()
            if orig:
                obj_sources.add(Path(orig).name.lower())

    eval_frames = eval_root / "frames"
    eval_store = eval_root / "annotations.json"
    if eval_frames.is_dir():
        for frame in sorted(eval_frames.glob("*.jpg")):
            h = image_content_hash(frame)
            if h in obj_hashes:
                violations.append(
                    f"image content hash collision: {frame} == {obj_hashes[h]}"
                )
            if frame.resolve().as_posix() in obj_paths:
                violations.append(f"image path appears in both datasets: {frame}")
    if eval_store.is_file():
        try:
            doc = json.loads(eval_store.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            doc = {}
        for im in doc.get("images", []):
            clip = str((im.get("ps_provenance") or {}).get("source_clip", "")).strip()
            if clip and Path(clip).name.lower() in obj_sources:
                violations.append(
                    f"source clip {clip!r} is also an object sample source filename"
                )
    return violations


def _split_of(sample_id: str, val_fraction: float) -> str:
    """Deterministic per-sample split. Stable across runs; object-balanced
    because every object hashes its own sample ids."""

    bucket = int(sha256(sample_id.encode("utf-8")).hexdigest()[:8], 16) % 1000
    return "val" if bucket < int(val_fraction * 1000) else "train"


def build_coco(root: Path, val_fraction: float) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    reg = ObjectRegistry(root)
    profiles = reg.list()
    if not profiles:
        raise ObjectStoreError("no object profiles to export")

    categories = [
        {"id": i + 1, "name": p.object_id, "ps_name": p.name, "kind": p.kind, "supercategory": "object"}
        for i, p in enumerate(profiles)
    ]
    cat_id = {p.object_id: i + 1 for i, p in enumerate(profiles)}

    docs: dict[str, dict[str, Any]] = {
        "train": {"info": {"description": "PredictiveSense object training set", "split": "train"},
                  "images": [], "annotations": [], "categories": categories},
        "val": {"info": {"description": "PredictiveSense object training set", "split": "val"},
                "images": [], "annotations": [], "categories": categories},
    }
    image_id = 0
    ann_id = 0
    per_object: dict[str, dict[str, int]] = {}
    for prof, sample, img in _iter_object_samples(root):
        if not img.is_file():
            raise ObjectStoreError(f"sample image missing on disk: {img}")
        box = [float(v) for v in sample["box"]]
        x, y, w, h = box
        iw, ih = int(sample["width"]), int(sample["height"])
        if w <= 0 or h <= 0 or x < -1 or y < -1 or x + w > iw + 1 or y + h > ih + 1:
            raise ObjectStoreError(f"sample {sample['sample_id']} box {box} outside {iw}x{ih}")

        split = _split_of(sample["sample_id"], val_fraction)
        row = per_object.setdefault(
            prof.object_id, {"train": 0, "val": 0, "boxes": 0, "negatives": 0}
        )
        # Guarantee at least one training sample per object.
        if split == "val" and row["train"] == 0 and row["val"] >= 1:
            split = "train"
        row[split] += 1

        image_id += 1
        docs[split]["images"].append(
            {
                "id": image_id,
                "file_name": f"{prof.object_id}/{sample['path']}",
                "width": iw,
                "height": ih,
                "ps_object_id": prof.object_id,
                "ps_role": sample.get("role", "positive"),
                "ps_conditions": sample.get("conditions", {}),
                "ps_provenance": {
                    "captured_utc": sample.get("captured_utc"),
                    "device_label": sample.get("device_label"),
                    "source": sample.get("source"),
                    "original_filename": sample.get("original_filename"),
                    "git_commit": sample.get("git_commit"),
                    "consent_ack": sample.get("consent_ack"),
                },
                "ps_split": split,
                "ps_quality_flags": (sample.get("quality") or {}).get("flags", []),
            }
        )
        if sample.get("role") == "positive":
            ann_id += 1
            docs[split]["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cat_id[prof.object_id],
                    "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                    "area": round(w * h, 2),
                    "iscrowd": 0,
                }
            )
            row["boxes"] += 1
        else:
            row["negatives"] += 1

    manifest = {
        "generated_from": str(root),
        "objects": len(profiles),
        "categories": [c["name"] for c in categories],
        "val_fraction": val_fraction,
        "per_object": per_object,
        "counts": {
            "train_images": len(docs["train"]["images"]),
            "val_images": len(docs["val"]["images"]),
            "train_boxes": len(docs["train"]["annotations"]),
            "val_boxes": len(docs["val"]["annotations"]),
        },
        "note": "Training data only. Disjoint from data/eval (asserted at export). No model trained.",
    }
    return docs["train"], docs["val"], manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Export object samples to COCO detection format.")
    p.add_argument("--out", default="data/objects/coco_train.json", help="train COCO json path")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--profile", default="dev")
    args = p.parse_args(argv)
    configure_logging("INFO")

    if not (0.0 <= args.val_fraction < 1.0):
        _LOG.error("--val-fraction must be in [0, 1)")
        return 2
    try:
        config = load_config(args.profile)
    except (ConfigError, ValidationError) as exc:
        _LOG.error("configuration error: %s", exc)
        return 2

    objects_root = Path(config.objects.root)
    eval_root = Path(config.dataset.root)

    violations = check_dataset_separation(objects_root, eval_root)
    if violations:
        _LOG.error(
            "DATASET SEPARATION VIOLATION - refusing to export. The object "
            "training set and the Phase 2.5 evaluation set must never merge:\n  - %s",
            "\n  - ".join(violations),
        )
        return 2

    try:
        train_doc, val_doc, manifest = build_coco(objects_root, args.val_fraction)
    except ObjectStoreError as exc:
        _LOG.error("export failed: %s", exc)
        return 2

    train_path = Path(args.out)
    if "train" in train_path.stem:
        val_path = train_path.with_name(train_path.stem.replace("train", "val") + train_path.suffix)
    else:
        val_path = train_path.with_name(f"{train_path.stem}_val{train_path.suffix}")
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.write_text(json.dumps(train_doc, indent=2) + "\n", encoding="utf-8")
    val_path.write_text(json.dumps(val_doc, indent=2) + "\n", encoding="utf-8")
    (train_path.parent / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    _LOG.info(
        "exported %d train / %d val image(s), %d train / %d val box(es) -> %s , %s",
        manifest["counts"]["train_images"], manifest["counts"]["val_images"],
        manifest["counts"]["train_boxes"], manifest["counts"]["val_boxes"],
        train_path, val_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
