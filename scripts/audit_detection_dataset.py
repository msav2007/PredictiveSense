"""Stage 5 section 3.3: duplicate / blur / geometry / licence audit of a
built detection COCO store, run and published BEFORE any split.

Reuses predictivesense/objects/quality.py's dHash/Laplacian-variance
heuristics (already established in earlier phases) rather than inventing a
second set. Needs cv2 to decode images for the pixel-level checks, so this
lives in scripts/, never in predictivesense/ (test_no_forbidden_imports.py).

Nothing here drops a sample - duplicates are grouped and reported, never
deleted (stage5 section 3.3), except exact byte-identical images, which are
reported under a distinct key.
"""

from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from predictivesense.dataset.coco_store import CocoStore
from predictivesense.objects.quality import dhash, hamming, laplacian_variance

__all__ = ["audit_detection_store", "load_eval_phashes", "main"]


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    s = sorted(values)
    n = len(s)

    def pct(p: float) -> float:
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return round(s[idx], 5)

    return {"min": pct(0.0), "median": pct(0.5), "p95": pct(0.95), "max": pct(1.0)}


def load_eval_phashes(eval_frames_dir: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if not eval_frames_dir.is_dir():
        return out
    for frame in sorted(eval_frames_dir.glob("*.jpg")):
        image = cv2.imread(str(frame), cv2.IMREAD_COLOR)
        if image is not None:
            out.append((frame.name, dhash(image)))
    return out


def audit_detection_store(
    store: CocoStore,
    *,
    eval_phashes: list[tuple[str, str]] = (),
    blur_var_min: float = 60.0,
    min_box_area_frac: float = 0.01,
    max_aspect_ratio: float = 6.0,
    duplicate_hamming_max: int = 6,
) -> dict[str, Any]:
    per_class_totals: dict[str, dict[str, int]] = {}
    per_class_blur: dict[str, list[float]] = {}
    per_class_area_frac: dict[str, list[float]] = {}
    per_class_aspect: dict[str, list[float]] = {}
    license_counts: dict[str, int] = {}
    zero_box_images = 0
    boxes_below_min_area = 0

    image_phash: dict[int, str] = {}
    image_exact_sha: dict[int, str] = {}
    unreadable: list[int] = []

    for iid in store.image_ids():
        im = store.image(iid)
        prov = im.get("ps_provenance") or {}
        img_path = Path(str(prov.get("image_path", "")))
        lic = str(prov.get("license", "unknown")) or "unknown"
        license_counts[lic] = license_counts.get(lic, 0) + 1

        anns = store.annotations_for(iid)
        if not anns:
            zero_box_images += 1

        image_arr: np.ndarray | None = None
        if img_path.is_file():
            image_arr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            image_exact_sha[iid] = sha256(img_path.read_bytes()).hexdigest()
        if image_arr is None:
            unreadable.append(iid)
        else:
            image_phash[iid] = dhash(image_arr)

        for ann in anns:
            cname = store.category_name(int(ann["category_id"]))
            totals = per_class_totals.setdefault(cname, {"images": 0, "boxes": 0, "zero_box_images": 0})
            totals["boxes"] += 1
            x, y, w, h = ann["bbox"]
            area_frac = (w * h) / max(1.0, float(im["width"]) * float(im["height"]))
            aspect = max(w, h) / max(1e-9, min(w, h))
            per_class_area_frac.setdefault(cname, []).append(area_frac)
            per_class_aspect.setdefault(cname, []).append(aspect)
            if area_frac < min_box_area_frac or aspect > max_aspect_ratio:
                boxes_below_min_area += 1
            if image_arr is not None:
                x0, y0 = max(0, int(x)), max(0, int(y))
                x1, y1 = min(image_arr.shape[1], int(x + w)), min(image_arr.shape[0], int(y + h))
                crop = image_arr[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else image_arr
                per_class_blur.setdefault(cname, []).append(laplacian_variance(crop))

        classes_here = {store.category_name(int(a["category_id"])) for a in anns}
        for cname in classes_here:
            per_class_totals.setdefault(cname, {"images": 0, "boxes": 0, "zero_box_images": 0})
            per_class_totals[cname]["images"] += 1
        if not classes_here:
            pass  # zero-box images are not attributed to any per-class "images" count

    # -- near-duplicate detection: within class, across class, against eval --
    by_class_images: dict[str, list[int]] = {}
    for iid in store.image_ids():
        classes_here = {store.category_name(int(a["category_id"])) for a in store.annotations_for(iid)}
        for c in classes_here:
            by_class_images.setdefault(c, []).append(iid)

    def _dup_groups(ids: list[int]) -> list[list[int]]:
        seen: list[tuple[int, str]] = []
        groups: dict[int, list[int]] = {}
        parent: dict[int, int] = {}

        def find(a: int) -> int:
            parent.setdefault(a, a)
            while parent[a] != a:
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for iid in ids:
            parent.setdefault(iid, iid)
        for iid in ids:
            ph = image_phash.get(iid)
            if ph is None:
                continue
            for other_id, other_ph in seen:
                if hamming(ph, other_ph) <= duplicate_hamming_max:
                    union(iid, other_id)
            seen.append((iid, ph))
        for iid in ids:
            groups.setdefault(find(iid), []).append(iid)
        return [g for g in groups.values() if len(g) > 1]

    within_class_dup_groups = {c: _dup_groups(ids) for c, ids in by_class_images.items()}
    all_image_ids = store.image_ids()
    across_class_dup_groups = [
        g for g in _dup_groups(all_image_ids)
        if len({c for c, ids in by_class_images.items() for iid in g if iid in ids}) > 1
    ]

    exact_dup_groups: dict[str, list[int]] = {}
    for iid, sha in image_exact_sha.items():
        exact_dup_groups.setdefault(sha, []).append(iid)
    exact_dups = {sha: ids for sha, ids in exact_dup_groups.items() if len(ids) > 1}

    against_eval: list[dict[str, Any]] = []
    if eval_phashes:
        for iid, ph in image_phash.items():
            for eval_name, other_ph in eval_phashes:
                if hamming(ph, other_ph) <= duplicate_hamming_max:
                    against_eval.append({"image_id": iid, "eval_frame": eval_name, "hamming": hamming(ph, other_ph)})

    return {
        "per_class_totals": per_class_totals,
        "per_class_blur_var": {c: _percentiles(v) for c, v in per_class_blur.items()},
        "per_class_box_area_frac": {c: _percentiles(v) for c, v in per_class_area_frac.items()},
        "per_class_box_aspect_ratio": {c: _percentiles(v) for c, v in per_class_aspect.items()},
        "boxes_below_min_area_or_extreme_aspect": boxes_below_min_area,
        "zero_box_images": zero_box_images,
        "license_breakdown": license_counts,
        "unreadable_images": unreadable,
        "duplicate_policy": "grouped and kept together in one split, never deleted, except exact byte-duplicates (reported separately, only one copy is meaningful to keep)",
        "within_class_duplicate_groups": {c: g for c, g in within_class_dup_groups.items() if g},
        "across_class_duplicate_groups": across_class_dup_groups,
        "exact_byte_duplicate_groups": exact_dups,
        "against_eval_duplicates": against_eval,
        "thresholds": {
            "blur_var_min": blur_var_min, "min_box_area_frac": min_box_area_frac,
            "max_aspect_ratio": max_aspect_ratio, "duplicate_hamming_max": duplicate_hamming_max,
        },
    }


def _to_markdown(report: dict[str, Any]) -> str:
    lines = ["# Detection dataset audit", ""]
    lines.append("| class | images | boxes | zero-box images |")
    lines.append("|---|---:|---:|---:|")
    for c, t in report["per_class_totals"].items():
        lines.append(f"| {c} | {t['images']} | {t['boxes']} | {t.get('zero_box_images', 0)} |")
    lines.append("")
    lines.append(f"Total zero-box (hard-negative/background) images: **{report['zero_box_images']}**")
    lines.append(f"Boxes below min area / extreme aspect: **{report['boxes_below_min_area_or_extreme_aspect']}**")
    lines.append("")
    lines.append("## Box area fraction (per class)")
    lines.append("| class | min | median | p95 | max |")
    lines.append("|---|---:|---:|---:|---:|")
    for c, d in report["per_class_box_area_frac"].items():
        lines.append(f"| {c} | {d['min']} | {d['median']} | {d['p95']} | {d['max']} |")
    lines.append("")
    lines.append("## Blur variance (per class, higher = sharper)")
    lines.append("| class | min | median | p95 | max |")
    lines.append("|---|---:|---:|---:|---:|")
    for c, d in report["per_class_blur_var"].items():
        lines.append(f"| {c} | {d['min']} | {d['median']} | {d['p95']} | {d['max']} |")
    lines.append("")
    lines.append(f"Within-class duplicate groups: {sum(len(g) for g in report['within_class_duplicate_groups'].values())} images across {sum(1 for gl in report['within_class_duplicate_groups'].values() for g in gl)} groups")
    lines.append(f"Across-class duplicate groups: {len(report['across_class_duplicate_groups'])}")
    lines.append(f"Exact byte-duplicate groups: {len(report['exact_byte_duplicate_groups'])}")
    lines.append(f"Duplicates against data/eval: {len(report['against_eval_duplicates'])}")
    lines.append("")
    lines.append(f"Licence breakdown: {report['license_breakdown']}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--eval-frames", type=Path, default=_REPO / "data" / "eval" / "frames")
    parser.add_argument("--out-json", type=Path, default=_REPO / "results" / "detection_dataset_audit.json")
    parser.add_argument("--out-md", type=Path, default=_REPO / "results" / "detection_dataset_audit.md")
    args = parser.parse_args(argv)

    store = CocoStore.load(args.store)
    eval_phashes = load_eval_phashes(args.eval_frames)
    report = audit_detection_store(store, eval_phashes=eval_phashes)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    args.out_md.write_text(_to_markdown(report), encoding="utf-8")
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
