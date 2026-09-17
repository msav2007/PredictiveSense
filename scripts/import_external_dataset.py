"""Phase 12 section 5-8: import an external dataset into ``data/external/``,
feeding the EXISTING Phase 11 crop-training pipeline (section 9.1) - never a
parallel one.

Reads ONLY the raw files a FiftyOne export writes to disk (mirrors
``scripts/audit_external_dataset.py`` exactly - see that script's docstring
for why ``detections.csv``/``image_ids.csv`` must be filtered to this
download's own image IDs before use). Never imports ``fiftyone``.

Nothing is copied into the repository (section 5.3): each sample's
``image_path`` references the original file on disk by absolute path, plus a
SHA-256 of its bytes for reproducibility/integrity. Only manifests, hashes and
the class mapping are ever committed (section 5.4) - the images themselves,
under ``data/external/``, are git-ignored.

Usage::

    python scripts\\import_external_dataset.py --class watch
    python scripts\\import_external_dataset.py --class watch \\
        --hard-negative-mid /m/050k8 --hard-negative-limit 40

Requires the ``[external]`` extra (``pip install -e ".[external]"``) for
pandas - nothing under ``predictivesense/`` ever needs it (enforced by
``tests/unit/test_no_forbidden_imports.py``).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from predictivesense.logging_setup import configure_logging, get_logger
from predictivesense.objects.quality import dhash, hamming
from predictivesense.telemetry.manifest import utc_now_iso
from scripts.export_objects_coco import image_content_hash

_LOG = get_logger("predictivesense.scripts.import_external_dataset")

_MIN_BOX_AREA_FRAC = 0.01  # matches ObjectsConfig.min_box_area_frac default
_MAX_ASPECT_RATIO = 6.0  # reject implausibly sliver-thin boxes either way
_DUPLICATE_HAMMING_MAX = 6  # matches ObjectsConfig.duplicate_hamming_max default
_MAPPING_PATH = _REPO / "results" / "external_class_mapping.json"


class ImportAbortedError(RuntimeError):
    """An external<->eval collision, or another hard-fail condition, stopped
    the import before anything was written (section 7.2)."""


@dataclass
class RejectionCounter:
    reasons: dict[str, int] = field(default_factory=dict)

    def add(self, reason: str) -> None:
        self.reasons[reason] = self.reasons.get(reason, 0) + 1


def _load_mapping(ps_class: str, mapping_path: Path = _MAPPING_PATH) -> dict[str, Any]:
    doc = json.loads(mapping_path.read_text(encoding="utf-8"))
    for entry in doc["decisions"]:
        if entry["ps_class"] == ps_class:
            return entry
    raise KeyError(f"no mapping entry for {ps_class!r} in {mapping_path}")


def _existing_hashes_and_phashes(
    objects_root: Path, eval_root: Path
) -> tuple[set[str], list[tuple[str, str]], set[str], list[tuple[str, str]]]:
    """``(objects_hashes, objects_phashes, eval_hashes, eval_phashes)`` for the
    cross-store collision check (section 5.2/7.2). Kept separate BY DESIGN: a
    match against ``data/eval`` fails the whole import (our test data must
    never be contaminated); a match against ``data/objects`` only excludes
    that one image and continues (still reported loudly)."""

    from predictivesense.objects.registry import ObjectRegistry
    from predictivesense.objects.samples import SampleStore

    objects_hashes: set[str] = set()
    objects_phashes: list[tuple[str, str]] = []
    eval_hashes: set[str] = set()
    eval_phashes: list[tuple[str, str]] = []

    if objects_root.is_dir():
        reg = ObjectRegistry(objects_root)
        for prof in reg.list():
            store_dir = reg.object_dir(prof.object_id)
            store = SampleStore(store_dir, prof.object_id)
            for sample in store.list():
                img_path = store_dir / str(sample.get("path", ""))
                if img_path.is_file():
                    objects_hashes.add(image_content_hash(img_path))
                ph = (sample.get("quality") or {}).get("phash")
                if ph:
                    objects_phashes.append((f"objects:{sample.get('sample_id')}", ph))

    eval_frames = eval_root / "frames"
    if eval_frames.is_dir():
        for frame in sorted(eval_frames.glob("*.jpg")):
            eval_hashes.add(image_content_hash(frame))
            image = cv2.imread(str(frame), cv2.IMREAD_COLOR)
            if image is not None:
                eval_phashes.append((f"eval:{frame.name}", dhash(image)))

    return objects_hashes, objects_phashes, eval_hashes, eval_phashes


def _read_target_rows(
    labels_csv: Path, image_ids: set[str], mids: set[str]
) -> pd.DataFrame:
    cols = ["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax"]
    frames = []
    for chunk in pd.read_csv(labels_csv, usecols=cols, chunksize=500_000, dtype={"ImageID": str, "LabelName": str}):
        chunk = chunk[chunk["ImageID"].isin(image_ids) & chunk["LabelName"].isin(mids)]
        if not chunk.empty:
            frames.append(chunk)
    if not frames:
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True)


def _image_ids_with_any_mid(labels_csv: Path, image_ids: set[str], mids: set[str]) -> set[str]:
    """Every image id (within ``image_ids``) that has at least one box of any
    class in ``mids`` - used to exclude "contaminated" images when sourcing
    hard negatives for a different class."""

    hits: set[str] = set()
    for chunk in pd.read_csv(
        labels_csv, usecols=["ImageID", "LabelName"], chunksize=500_000, dtype={"ImageID": str, "LabelName": str}
    ):
        chunk = chunk[chunk["ImageID"].isin(image_ids) & chunk["LabelName"].isin(mids)]
        hits.update(chunk["ImageID"].unique())
    return hits


def _license_lookup(metadata_dir: Path, wanted_ids: set[str]) -> dict[str, dict[str, str]]:
    """Per-image licence/attribution (section 6.1), filtered to ``wanted_ids``
    - ``image_ids.csv`` is also the full, unfiltered OI master file."""

    path = metadata_dir / "image_ids.csv"
    out: dict[str, dict[str, str]] = {}
    cols = ["ImageID", "License", "Author", "OriginalURL", "Title"]
    for chunk in pd.read_csv(path, usecols=cols, chunksize=500_000, dtype=str):
        chunk = chunk[chunk["ImageID"].isin(wanted_ids)]
        for _, row in chunk.iterrows():
            out[row["ImageID"]] = {
                "license": row.get("License") or "",
                "author": row.get("Author") or "",
                "source_url": row.get("OriginalURL") or "",
                "title": row.get("Title") or "",
            }
    return out


def _build_samples_for_mid_group(
    rows: pd.DataFrame,
    *,
    ps_class: str,
    role: str,
    negative_for: list[str],
    data_dir: Path,
    mid_to_name: dict[str, str],
    licenses: dict[str, dict[str, str]],
    rejects: RejectionCounter,
    objects_hashes: set[str],
    objects_phashes: list[tuple[str, str]],
    eval_hashes: set[str],
    eval_phashes: list[tuple[str, str]],
    seen_image_dims: dict[str, tuple[int, int]],
    seen_phashes_this_class: list[tuple[str, str]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Filter + build sample records for one group of detection rows (section
    7.1/7.2). Returns ``(samples, image_ids_used)``. Raises
    :class:`ImportAbortedError` on any external<->eval collision - a match
    against ``data/objects`` instead only excludes that one image."""

    samples: list[dict[str, Any]] = []
    used_image_ids: list[str] = []
    by_image = rows.groupby("ImageID")
    for image_id, group in by_image:
        img_path = data_dir / f"{image_id}.jpg"
        if not img_path.is_file():
            rejects.add("image_file_missing")
            continue
        if image_id not in seen_image_dims:
            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if image is None:
                rejects.add("corrupt_or_undecodable_image")
                continue
            h, w = image.shape[0], image.shape[1]
            seen_image_dims[image_id] = (w, h)
            sha = image_content_hash(img_path)
            ph = dhash(image)
            if sha in eval_hashes:
                raise ImportAbortedError(
                    f"external image {image_id} (sha256 {sha[:12]}...) content-hash-collides with "
                    "data/eval - external<->eval collision (section 5.2/7.2). Import aborted, nothing written."
                )
            near_eval = next(
                ((sid, hamming(ph, other)) for sid, other in eval_phashes if hamming(ph, other) <= _DUPLICATE_HAMMING_MAX),
                None,
            )
            if near_eval is not None:
                raise ImportAbortedError(
                    f"external image {image_id} is a near-duplicate (Hamming {near_eval[1]}) of eval frame "
                    f"{near_eval[0]} - external<->eval collision. Import aborted, nothing written."
                )
            if sha in objects_hashes:
                rejects.add("cross_store_duplicate_of_objects")
                continue
            near_objects = next(
                ((sid, hamming(ph, other)) for sid, other in objects_phashes if hamming(ph, other) <= _DUPLICATE_HAMMING_MAX),
                None,
            )
            if near_objects is not None:
                rejects.add("cross_store_near_duplicate_of_objects")
                continue
            near_within = next(
                ((sid, hamming(ph, other)) for sid, other in seen_phashes_this_class if hamming(ph, other) <= _DUPLICATE_HAMMING_MAX),
                None,
            )
            if near_within is not None:
                rejects.add("near_duplicate_within_external_set")
                continue
            seen_phashes_this_class.append((image_id, ph))
        w, h = seen_image_dims[image_id]

        image_used = False
        for _, r in group.iterrows():
            xmin, xmax, ymin, ymax = float(r.XMin), float(r.XMax), float(r.YMin), float(r.YMax)
            if not (0.0 <= xmin < xmax <= 1.0 and 0.0 <= ymin < ymax <= 1.0):
                rejects.add("box_out_of_bounds")
                continue
            bw_frac, bh_frac = (xmax - xmin), (ymax - ymin)
            area_frac = bw_frac * bh_frac
            if area_frac < _MIN_BOX_AREA_FRAC:
                rejects.add("box_too_small")
                continue
            aspect = max(bw_frac, bh_frac) / max(1e-9, min(bw_frac, bh_frac))
            if aspect > _MAX_ASPECT_RATIO:
                rejects.add("box_extreme_aspect_ratio")
                continue

            px, py = xmin * w, ymin * h
            pw, ph_box = bw_frac * w, bh_frac * h
            mid = str(r.LabelName)
            lic = licenses.get(image_id, {})
            samples.append({
                "sample_id": f"ext_{ps_class}_{image_id}_{len(samples)}",
                "object_id": ps_class,
                "image_id": image_id,
                "image_path": str(img_path.resolve()),
                "image_sha256": image_content_hash(img_path),
                "box": [round(px, 2), round(py, 2), round(pw, 2), round(ph_box, 2)],
                "width": w,
                "height": h,
                "box_area_frac": round(area_frac, 5),
                "role": role,
                "negative_for": negative_for,
                "box_confirmed_by_human": True,  # OI boxes are human-annotated by the dataset's own process
                "oi_label_mid": mid,
                "oi_label_name": mid_to_name.get(mid, mid),
                "license": lic.get("license", ""),
                "author": lic.get("author", ""),
                "source_url": lic.get("source_url", ""),
            })
            image_used = True
        if image_used:
            used_image_ids.append(image_id)
    return samples, used_image_ids


def import_class(
    ps_class: str,
    *,
    root: Path,
    split: str,
    objects_root: Path,
    eval_root: Path,
    external_root: Path,
    hard_negative_mids: list[str] | None = None,
    hard_negative_limit: int = 0,
    mapping_path: Path = _MAPPING_PATH,
) -> dict[str, Any]:
    entry = _load_mapping(ps_class, mapping_path)
    if entry["status"] != "mapped":
        raise ValueError(f"{ps_class!r} is {entry['status']!r} in the class mapping, not mapped - refusing to import")
    mids = {c["mid"] for c in entry["oi_classes"]}

    split_dir = root / split
    data_dir = split_dir / "data"
    labels_csv = split_dir / "labels" / "detections.csv"
    metadata_dir = split_dir / "metadata"
    classes_csv = metadata_dir / "classes.csv"

    classes_df = pd.read_csv(classes_csv, header=None, names=["mid", "name"])
    mid_to_name = dict(zip(classes_df["mid"], classes_df["name"]))
    image_ids = {p.stem for p in data_dir.iterdir() if p.suffix.lower() == ".jpg"}

    rejects = RejectionCounter()
    objects_hashes, objects_phashes, eval_hashes, eval_phashes = _existing_hashes_and_phashes(objects_root, eval_root)
    seen_image_dims: dict[str, tuple[int, int]] = {}
    seen_phashes_this_class: list[tuple[str, str]] = []

    target_rows = _read_target_rows(labels_csv, image_ids, mids)
    positive_image_ids_all_targets = set(target_rows["ImageID"].unique()) if not target_rows.empty else set()
    wanted_ids = set(positive_image_ids_all_targets)

    positive_samples: list[dict[str, Any]] = []
    if not target_rows.empty:
        licenses = _license_lookup(metadata_dir, wanted_ids)
        positive_samples, _used = _build_samples_for_mid_group(
            target_rows, ps_class=ps_class, role="positive", negative_for=[],
            data_dir=data_dir, mid_to_name=mid_to_name, licenses=licenses, rejects=rejects,
            objects_hashes=objects_hashes, objects_phashes=objects_phashes,
            eval_hashes=eval_hashes, eval_phashes=eval_phashes,
            seen_image_dims=seen_image_dims, seen_phashes_this_class=seen_phashes_this_class,
        )

    hard_negative_samples: list[dict[str, Any]] = []
    if hard_negative_mids and hard_negative_limit > 0:
        contaminated = _image_ids_with_any_mid(labels_csv, image_ids, mids)
        hn_rows = _read_target_rows(labels_csv, image_ids - contaminated, set(hard_negative_mids))
        if not hn_rows.empty:
            # One box per image, capped at the limit, deterministic order.
            hn_rows = hn_rows.sort_values("ImageID").groupby("ImageID", as_index=False).first()
            hn_rows = hn_rows.head(hard_negative_limit)
            hn_ids = set(hn_rows["ImageID"].unique())
            licenses = _license_lookup(metadata_dir, hn_ids)
            hard_negative_samples, _used = _build_samples_for_mid_group(
                hn_rows, ps_class=ps_class, role="hard_negative", negative_for=[ps_class],
                data_dir=data_dir, mid_to_name=mid_to_name, licenses=licenses, rejects=rejects,
                objects_hashes=objects_hashes, objects_phashes=objects_phashes,
                eval_hashes=eval_hashes, eval_phashes=eval_phashes,
                seen_image_dims=seen_image_dims, seen_phashes_this_class=seen_phashes_this_class,
            )

    out_dir = external_root / "open-images-v7" / ps_class
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 1,
        "source": "open-images-v7",
        "source_root": str(root),
        "split": split,
        "object_id": ps_class,
        "oi_mids": sorted(mids),
        "generated_utc": utc_now_iso(),
        "samples": positive_samples + hard_negative_samples,
        "rejected_reasons": rejects.reasons,
        "counts": {
            "positive": len(positive_samples),
            "hard_negative": len(hard_negative_samples),
            "positive_images": len({s["image_id"] for s in positive_samples}),
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--class", dest="ps_class", required=True)
    parser.add_argument("--root", type=Path, default=Path(r"C:\Users\mummi\fiftyone\open-images-v7"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--objects-root", type=Path, default=_REPO / "data" / "objects")
    parser.add_argument("--eval-root", type=Path, default=_REPO / "data" / "eval")
    parser.add_argument("--external-root", type=Path, default=_REPO / "data" / "external")
    parser.add_argument("--hard-negative-mid", action="append", default=[], dest="hard_negative_mids")
    parser.add_argument("--hard-negative-limit", type=int, default=0)
    parser.add_argument("--out-run-manifest", type=Path, default=None)
    args = parser.parse_args(argv)
    configure_logging("INFO")

    try:
        manifest = import_class(
            args.ps_class, root=args.root, split=args.split, objects_root=args.objects_root,
            eval_root=args.eval_root, external_root=args.external_root,
            hard_negative_mids=args.hard_negative_mids, hard_negative_limit=args.hard_negative_limit,
        )
    except ImportAbortedError as exc:
        _LOG.error("IMPORT ABORTED: %s", exc)
        return 2

    _LOG.info(
        "imported %s: %d positive samples (%d images), %d hard-negative samples; rejections=%s",
        args.ps_class, manifest["counts"]["positive"], manifest["counts"]["positive_images"],
        manifest["counts"]["hard_negative"], manifest["rejected_reasons"],
    )

    run_manifest_path = args.out_run_manifest or (
        _REPO / "results" / f"external_import_run_{args.ps_class}_{utc_now_iso().replace(':', '').replace('.', '')}.json"
    )
    run_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    run_manifest_path.write_text(json.dumps({
        "ps_class": args.ps_class,
        "source_root": str(args.root),
        "split": args.split,
        "generated_utc": utc_now_iso(),
        "manifest_path": str((args.external_root / "open-images-v7" / args.ps_class / "manifest.json").resolve()),
        "counts": manifest["counts"],
        "rejected_reasons": manifest["rejected_reasons"],
        "tool_versions": {"pandas": pd.__version__, "numpy": np.__version__, "opencv": cv2.__version__},
    }, indent=2), encoding="utf-8")
    _LOG.info("run manifest: %s", run_manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
