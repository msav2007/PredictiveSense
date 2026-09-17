"""Stage 5 section 4: build the detection-format COCO store + splits for
real, end to end - external classes via exhaustive multi-class requery
(section 2.1), own-capture classes via the identical converter (section 3.5).

Order matches the stage prompt's "how to work": class map is read (never
built here - results/detection_class_map.json is the committed, reviewed
source of truth), then query, then convert, then audit (published before any
split), then split (image/duplicate-group/session-disjoint, hard-fails on an
eval collision, refuses a training split below --min-images-per-class).

Usage::

    python scripts/build_detection_dataset.py \\
        --external-classes watch "mug/cup" \\
        --out-dir data/external/detection

    python scripts/build_detection_dataset.py \\
        --own-capture-classes comb --out-dir /tmp/comb_smoke_test
        # expected to FAIL at the split step (section 3.5 smoke test)

Requires the ``[external]`` extra (pandas) when ``--external-classes`` is
used. Own-capture-only runs need no external dependency.
"""

from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from predictivesense.dataset.coco_store import CocoStore, domain_categories
from predictivesense.dataset.detection_convert import (
    DetectionBox,
    DetectionImageRecord,
    add_detection_record,
    own_capture_records,
)
from predictivesense.dataset.detection_splits import (
    DetectionSplitError,
    EvalCollisionError,
    assert_no_eval_collision,
    build_detection_splits,
    merge_duplicate_groups,
)
from predictivesense.logging_setup import configure_logging, get_logger
from predictivesense.objects.registry import ObjectRegistry
from predictivesense.objects.samples import SampleStore
from predictivesense.training.dataset import session_key_for
from scripts.audit_detection_dataset import audit_detection_store, load_eval_phashes
from scripts.import_external_dataset import RejectionCounter

_LOG = get_logger("predictivesense.scripts.build_detection_dataset")

_MIN_BOX_AREA_FRAC = 0.01
_MAX_ASPECT_RATIO = 6.0
_DEFAULT_CLASS_MAP = _REPO / "results" / "detection_class_map.json"


class BuildAbortedError(RuntimeError):
    """A hard-fail condition (eval collision, empty active set) stopped the
    build before splits were written."""


def _load_class_map(path: Path) -> list[dict[str, Any]]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    return doc["entries"]


def _resolve_active_mids(entries: list[dict[str, Any]], ps_classes: list[str]) -> dict[str, str]:
    """``mid -> ps_class`` for every requested ps_class. Refuses a ps_class
    that is not ``status: mapped`` (rejected/unavailable classes are not
    importable, matching ``import_class``'s own refusal)."""

    mid_to_ps_class: dict[str, str] = {}
    wanted = set(ps_classes)
    found: set[str] = set()
    for entry in entries:
        if entry["ps_class"] not in wanted:
            continue
        if entry["status"] != "mapped":
            raise ValueError(f"{entry['ps_class']!r} is {entry['status']!r} in the class map, not mapped - refusing")
        found.add(entry["ps_class"])
        for sc in entry["source_classes"]:
            mid_to_ps_class[sc["mid"]] = entry["ps_class"]
    missing = wanted - found
    if missing:
        raise ValueError(f"no class-map entry for: {sorted(missing)}")
    return mid_to_ps_class


def build_external_records(
    mid_to_ps_class: dict[str, str],
    *,
    root: Path,
    split: str,
    rejects: RejectionCounter,
) -> list[DetectionImageRecord]:
    from scripts.external_detection_source import (
        load_class_names,
        load_image_ids,
        load_licenses,
        query_active_classes,
    )

    split_dir = root / split
    data_dir = split_dir / "data"
    labels_csv = split_dir / "labels" / "detections.csv"
    metadata_dir = split_dir / "metadata"

    mid_to_name = load_class_names(metadata_dir)
    all_image_ids = load_image_ids(data_dir)  # the FULL download population - section 2.1's exhaustive scope
    hits = query_active_classes(labels_csv, all_image_ids, set(mid_to_ps_class))
    licenses = load_licenses(metadata_dir, set(hits))

    import cv2

    records: list[DetectionImageRecord] = []
    for image_id, raw_boxes in sorted(hits.items()):
        img_path = data_dir / f"{image_id}.jpg"
        if not img_path.is_file():
            rejects.add("image_file_missing")
            continue
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            rejects.add("corrupt_or_undecodable_image")
            continue
        h, w = image.shape[0], image.shape[1]
        boxes: list[DetectionBox] = []
        for rb in raw_boxes:
            if not (0.0 <= rb.xmin < rb.xmax <= 1.0 and 0.0 <= rb.ymin < rb.ymax <= 1.0):
                rejects.add("box_out_of_bounds")
                continue
            bw_frac, bh_frac = (rb.xmax - rb.xmin), (rb.ymax - rb.ymin)
            area_frac = bw_frac * bh_frac
            if area_frac < _MIN_BOX_AREA_FRAC:
                rejects.add("box_too_small")
                continue
            aspect = max(bw_frac, bh_frac) / max(1e-9, min(bw_frac, bh_frac))
            if aspect > _MAX_ASPECT_RATIO:
                rejects.add("box_extreme_aspect_ratio")
                continue
            px, py = rb.xmin * w, rb.ymin * h
            pw, phh = bw_frac * w, bh_frac * h
            boxes.append(DetectionBox(
                ps_class=mid_to_ps_class[rb.mid],
                bbox=(round(px, 2), round(py, 2), round(pw, 2), round(phh, 2)),
                source_class=mid_to_name.get(rb.mid, rb.mid),
                box_confirmed_by_human=True,  # OI boxes are human-annotated by the dataset's own process
            ))
        lic = licenses.get(image_id, {})
        records.append(DetectionImageRecord(
            file_name=f"ext:open-images-v7:{image_id}",
            width=w, height=h, session_id=f"ext:{image_id}",  # one image = one split-group (no OI session concept)
            image_path=str(img_path.resolve()),
            image_sha256=sha256(img_path.read_bytes()).hexdigest(),
            source_dataset="open-images-v7", source_version=split,
            original_image_id=image_id,
            license=lic.get("license", ""), author=lic.get("author", ""), source_url=lic.get("source_url", ""),
            boxes=tuple(boxes),
            filtering_notes=() if boxes else ("zero_boxes_survived_filtering",),
        ))
    return records


def build_own_capture_records(
    ps_class: str, object_id: str, *, objects_root: Path,
) -> list[DetectionImageRecord]:
    reg = ObjectRegistry(objects_root)
    profiles = {p.object_id for p in reg.list()}
    if object_id not in profiles:
        return []
    store = SampleStore(reg.object_dir(object_id), object_id)
    samples = [s for s in store.list() if "_staging" not in str(s.get("path", ""))]
    return own_capture_records(
        object_id, ps_class, samples,
        object_dir_str=str(reg.object_dir(object_id)), session_key_fn=session_key_for,
    )


def run_build(
    *,
    external_classes: list[str],
    own_capture: list[tuple[str, str]],  # (ps_class, object_id)
    class_map_path: Path,
    oi_root: Path,
    oi_split: str,
    objects_root: Path,
    eval_root: Path,
    out_dir: Path,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    min_images_per_class: int,
) -> dict[str, Any]:
    entries = _load_class_map(class_map_path)
    all_ps_classes = sorted(set(external_classes) | {c for c, _ in own_capture})
    if not all_ps_classes:
        raise BuildAbortedError("no classes requested (neither --external-classes nor --own-capture-classes)")

    store = CocoStore.create(domain_categories(all_ps_classes))
    rejects = RejectionCounter()

    image_to_group: dict[int, str] = {}
    if external_classes:
        mid_to_ps_class = _resolve_active_mids(entries, external_classes)
        ext_records = build_external_records(mid_to_ps_class, root=oi_root, split=oi_split, rejects=rejects)
        for rec in ext_records:
            iid = add_detection_record(store, rec)
            image_to_group[iid] = rec.session_id

    for ps_class, object_id in own_capture:
        oc_records = build_own_capture_records(ps_class, object_id, objects_root=objects_root)
        for rec in oc_records:
            iid = add_detection_record(store, rec)
            image_to_group[iid] = rec.session_id

    out_dir.mkdir(parents=True, exist_ok=True)
    store_path = out_dir / "coco_detection.json"
    store.save(store_path)

    eval_frames_dir = eval_root / "frames"
    eval_phashes = load_eval_phashes(eval_frames_dir)
    audit = audit_detection_store(store, eval_phashes=eval_phashes)
    (out_dir / "audit.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")

    dup_pairs: list[tuple[str, str]] = []
    for groups in list(audit["within_class_duplicate_groups"].values()) + [audit["across_class_duplicate_groups"]]:
        for g in groups:
            for a, b in zip(g, g[1:]):
                dup_pairs.append((str(a), str(b)))
    image_to_group_str = {str(k): v for k, v in image_to_group.items()}
    merged = merge_duplicate_groups(image_to_group_str, dup_pairs)
    image_to_group_final = {int(k): v for k, v in merged.items()}

    eval_shas: set[str] = set()
    if eval_frames_dir.is_dir():
        for frame in eval_frames_dir.glob("*.jpg"):
            eval_shas.add(sha256(frame.read_bytes()).hexdigest())

    try:
        assert_no_eval_collision(
            train_val_image_ids=store.image_ids(), store=store,
            eval_image_shas=eval_shas, eval_phashes=eval_phashes,
        )
    except EvalCollisionError as exc:
        raise BuildAbortedError(str(exc)) from exc

    splits = build_detection_splits(
        store, image_to_group_final,
        train_fraction=train_fraction, val_fraction=val_fraction, test_fraction=test_fraction,
        seed=seed, min_images_per_class=min_images_per_class,
    )
    splits_path = out_dir / "splits.json"
    splits.save(splits_path)

    summary = {
        "store_path": str(store_path), "splits_path": str(splits_path),
        "audit_path": str(out_dir / "audit.json"),
        "rejected_reasons": rejects.reasons,
        "n_images": len(store.image_ids()),
        "n_annotations": sum(len(store.annotations_for(i)) for i in store.image_ids()),
        "per_class_totals": audit["per_class_totals"],
        "split_fractions_realized": splits.fractions,
        "realized_per_class_box_counts": splits.realized_per_class_box_counts,
        "realized_per_class_image_counts": splits.realized_per_class_image_counts,
        "split_content_hash": splits.content_hash,
    }
    (out_dir / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--class-map", type=Path, default=_DEFAULT_CLASS_MAP)
    parser.add_argument("--external-classes", nargs="*", default=[])
    parser.add_argument("--own-capture-classes", nargs="*", default=[], help="ps_class:object_id pairs, e.g. comb:comb")
    parser.add_argument("--oi-root", type=Path, default=Path(r"C:\Users\mummi\fiftyone\open-images-v7"))
    parser.add_argument("--oi-split", default="train")
    parser.add_argument("--objects-root", type=Path, default=_REPO / "data" / "objects")
    parser.add_argument("--eval-root", type=Path, default=_REPO / "data" / "eval")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-images-per-class", type=int, default=50)
    args = parser.parse_args(argv)
    configure_logging("INFO")

    own_capture = []
    for pair in args.own_capture_classes:
        ps_class, _, object_id = pair.partition(":")
        own_capture.append((ps_class, object_id or ps_class))

    try:
        summary = run_build(
            external_classes=args.external_classes, own_capture=own_capture,
            class_map_path=args.class_map, oi_root=args.oi_root, oi_split=args.oi_split,
            objects_root=args.objects_root, eval_root=args.eval_root, out_dir=args.out_dir,
            train_fraction=args.train_fraction, val_fraction=args.val_fraction, test_fraction=args.test_fraction,
            seed=args.seed, min_images_per_class=args.min_images_per_class,
        )
    except BuildAbortedError as exc:
        _LOG.error("BUILD ABORTED: %s", exc)
        return 2
    except DetectionSplitError as exc:
        _LOG.error("SPLIT REFUSED: %s", exc)
        return 3

    _LOG.info("build complete: %s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
