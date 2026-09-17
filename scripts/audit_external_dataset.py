"""Phase 12 section 3: audit a downloaded external dataset BEFORE any import.

Reads ONLY the raw files a FiftyOne Open-Images-V7 export writes to disk
(``<split>/data/*.jpg``, ``<split>/labels/detections.csv``,
``<split>/metadata/classes.csv``) - never through the ``fiftyone`` Python
API, so neither this script nor the later importer require the
``fiftyone`` runtime package (section 9.4: external-data dependencies are
confined to the tool that did the one-time download, not to anything this
repository runs routinely).

The master ``detections.csv`` FiftyOne writes is NOT pre-filtered to the
requested sample subset - it is (an excerpt of) the full Open Images V7
train annotation file, covering every image FiftyOne has ever indexed for
that split. This script filters it down to exactly the image IDs present
in ``<split>/data`` before computing any count. Confirmed by inspection:
the first data row's ``ImageID`` (``000002b66c9c498e``) does not appear
among this download's 5000 images.

Usage::

    python scripts/audit_external_dataset.py \
        --root "C:\\Users\\mummi\\fiftyone\\open-images-v7" --split train

Nothing is imported by this script. It only reads and reports.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = ["audit_external_dataset", "TARGET_CLASSES", "main"]

# Open Images V7 mid -> our candidate PredictiveSense class, derived by
# reading train/metadata/classes.csv (section 4). Every mid here was
# confirmed present in classes.csv by inspection before this list was
# written - nothing is guessed. A PredictiveSense class with no OI mid
# (pencil, charger, comb, can, pen) is simply absent from this dict; the
# audit still reports it as "no matching class found" in that case.
TARGET_CLASSES: dict[str, str] = {
    "/m/0gjkl": "Watch",
    "/m/0jyfg": "Glasses",
    "/m/017ftj": "Sunglasses",
    "/m/02_n6y": "Goggles",
    "/m/01b7fy": "Headphones",
    "/m/02jvh9": "Mug",
    "/m/02p5f1q": "Coffee cup",
    "/m/04dr76w": "Bottle",
    "/m/04kkgm": "Bowl",
    "/m/03hj559": "Mixing bowl",
    "/m/07v9_z": "Measuring cup",
    "/m/01m2v": "Computer keyboard",
    "/m/057cc": "Musical keyboard",
    "/m/050k8": "Mobile phone",
    "/m/07cx4": "Telephone",
    "/m/0h8lkj8": "Corded phone",
    "/m/0440zs": "Cocktail shaker",
    "/m/02x8cch": "Salt and pepper shakers",
    "/m/02mqfb": "Can opener",
    "/m/02ddwp": "Pencil sharpener",
    "/m/05676x": "Pencil case",
    "/m/0k1tl": "Pen",
    "/m/02jnhm": "Tin can",
}

_DETECTION_COLS = ["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax"]
_CHUNK_ROWS = 500_000


@dataclass
class ClassAudit:
    mid: str
    name: str
    boxes: int = 0
    images: set[str] = field(default_factory=set)
    area_fracs: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        fracs = sorted(self.area_fracs)
        n = len(fracs)

        def pct(p: float) -> float:
            if n == 0:
                return 0.0
            idx = min(n - 1, max(0, int(round(p * (n - 1)))))
            return round(fracs[idx], 6)

        return {
            "mid": self.mid,
            "name": self.name,
            "boxes": self.boxes,
            "images": len(self.images),
            "box_area_frac": {
                "min": pct(0.0),
                "median": pct(0.5),
                "p95": pct(0.95),
                "max": pct(1.0),
            },
        }


def _load_class_names(metadata_dir: Path) -> dict[str, str]:
    classes_csv = metadata_dir / "classes.csv"
    if not classes_csv.is_file():
        raise FileNotFoundError(f"classes.csv not found: {classes_csv}")
    df = pd.read_csv(classes_csv, header=None, names=["mid", "name"])
    return dict(zip(df["mid"], df["name"]))


def _load_image_ids(data_dir: Path) -> set[str]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"image data dir not found: {data_dir}")
    return {p.stem for p in data_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")}


def audit_external_dataset(
    root: Path,
    split: str = "train",
    target_classes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Read the raw export under ``root/split`` and return the audit report
    as a JSON-serialisable dict. Reads only; writes nothing."""

    target_classes = TARGET_CLASSES if target_classes is None else target_classes
    split_dir = root / split
    data_dir = split_dir / "data"
    labels_csv = split_dir / "labels" / "detections.csv"
    metadata_dir = split_dir / "metadata"

    for p in (data_dir, labels_csv, metadata_dir):
        if not p.exists():
            raise FileNotFoundError(f"expected export path missing: {p}")

    mid_to_name = _load_class_names(metadata_dir)
    image_ids = _load_image_ids(data_dir)
    total_samples = len(image_ids)

    per_class: dict[str, ClassAudit] = {}
    all_present_mids: dict[str, int] = defaultdict(int)
    annotated_image_ids: set[str] = set()
    per_image_target_mids: dict[str, set[str]] = defaultdict(set)
    total_annotations = 0
    total_rows_scanned = 0

    reader = pd.read_csv(
        labels_csv,
        usecols=_DETECTION_COLS,
        chunksize=_CHUNK_ROWS,
        dtype={"ImageID": str, "LabelName": str},
    )
    for chunk in reader:
        total_rows_scanned += len(chunk)
        chunk = chunk[chunk["ImageID"].isin(image_ids)]
        if chunk.empty:
            continue
        total_annotations += len(chunk)
        annotated_image_ids.update(chunk["ImageID"].unique())
        for label_name, group in chunk.groupby("LabelName"):
            all_present_mids[label_name] += len(group)
            if label_name in target_classes:
                ca = per_class.setdefault(
                    label_name, ClassAudit(mid=label_name, name=mid_to_name.get(label_name, label_name))
                )
                ca.boxes += len(group)
                ca.images.update(group["ImageID"].unique())
                area = (group["XMax"] - group["XMin"]) * (group["YMax"] - group["YMin"])
                ca.area_fracs.extend(area.tolist())
                for img_id in group["ImageID"].unique():
                    per_image_target_mids[img_id].add(label_name)

    images_no_target = total_samples - len(
        {img for img, mids in per_image_target_mids.items() if mids}
    )
    images_multi_target = sum(1 for mids in per_image_target_mids.values() if len(mids) >= 2)

    exact_classes_present = sorted(
        {mid_to_name.get(mid, mid): count for mid, count in all_present_mids.items()}.items(),
        key=lambda kv: (-kv[1], kv[0]),
    )

    report = {
        "source": {
            "root": str(root),
            "split": split,
            "labels_csv": str(labels_csv),
            "labels_csv_note": (
                "detections.csv is the full multi-split Open Images V7 annotation "
                "file, not pre-filtered to this download's samples; every count "
                "below is filtered by this script to the image IDs actually "
                "present under data/."
            ),
        },
        "total_samples": total_samples,
        "total_detection_rows_scanned_in_full_csv": total_rows_scanned,
        "total_annotations_in_subset": total_annotations,
        "images_with_no_target_class_annotation": images_no_target,
        "images_with_multiple_target_classes": images_multi_target,
        "target_class_audit": {
            ca.name: ca.to_dict() for ca in sorted(per_class.values(), key=lambda c: -c.boxes)
        },
        "target_classes_with_zero_matches": sorted(
            name for mid, name in target_classes.items() if mid not in per_class
        ),
        "exact_open_images_classes_present_in_subset": [
            {"name": name, "boxes": count} for name, count in exact_classes_present
        ],
        "classes_present_count": len(exact_classes_present),
    }
    return report


def _to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# External dataset audit",
        "",
        f"- Root: `{report['source']['root']}`",
        f"- Split: `{report['source']['split']}`",
        f"- Total downloaded images: **{report['total_samples']}**",
        f"- Total detection boxes for these images (all OI classes): "
        f"**{report['total_annotations_in_subset']}**",
        f"- Distinct OI classes appearing at all in this subset: "
        f"**{report['classes_present_count']}**",
        f"- Images with no annotation for any candidate target class: "
        f"**{report['images_with_no_target_class_annotation']}** / {report['total_samples']}",
        f"- Images with 2+ candidate target classes: "
        f"**{report['images_with_multiple_target_classes']}**",
        "",
        "## Candidate target classes",
        "",
        "| OI class | boxes | images | area frac (median) | area frac (p95) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, d in report["target_class_audit"].items():
        af = d["box_area_frac"]
        lines.append(f"| {name} | {d['boxes']} | {d['images']} | {af['median']} | {af['p95']} |")
    zero = report["target_classes_with_zero_matches"]
    if zero:
        lines.append("")
        lines.append(f"Zero matches in this download: {', '.join(zero)}")
    lines.append("")
    lines.append("## Top 30 OI classes actually present (any class, not just candidates)")
    lines.append("")
    lines.append("| OI class | boxes |")
    lines.append("|---|---:|")
    for entry in report["exact_open_images_classes_present_in_subset"][:30]:
        lines.append(f"| {entry['name']} | {entry['boxes']} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(r"C:\Users\mummi\fiftyone\open-images-v7"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--out-json", type=Path, default=Path("results/external_dataset_audit.json"))
    parser.add_argument("--out-md", type=Path, default=Path("results/external_dataset_audit.md"))
    args = parser.parse_args(argv)

    report = audit_external_dataset(args.root, args.split)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.out_md.write_text(_to_markdown(report), encoding="utf-8")
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
