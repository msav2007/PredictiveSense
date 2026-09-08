"""Fit per-class confidence thresholds and the top-2 margin on the ``val`` split.

    python scripts\\fit_thresholds.py --split val

For each domain class it runs the detector over every ``val`` frame, matches
predictions to ground truth within the class at IoU 0.5, builds the
precision/recall-vs-threshold curve, and picks the threshold that maximises F1
(ties broken toward higher precision - a confidently wrong label is worse than a
miss here). The ``margin_min`` recommendation is the 10th-percentile of
``score - runner_up_score`` over *correct* accepted detections, i.e. keep ~90% of
correct detections while sending genuinely ambiguous ones to ``unknown``. Every
number is written with its sample count; thin classes are flagged.

Only ``val`` is accepted - fitting on ``test`` is refused (BLOCK 17.4). Writes
``results/fit_thresholds_val.{json,md}`` and prints a YAML block to paste into
``policy.per_class_thresholds`` / ``policy.margin_min`` (the profile comment must
name this result file - BLOCK 3.16).
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[1]
if str(_REPO) not in _sys.path:
    _sys.path.insert(0, str(_REPO))

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from predictivesense.config.settings import ConfigError, ValidationError, load_config
from predictivesense.dataset.coco_store import CocoStoreError
from predictivesense.dataset.splits import SplitLeakageError
from predictivesense.eval.matching import greedy_match
from predictivesense.logging_setup import configure_logging, get_logger
from predictivesense.perception.detector import ObjectDetector

from scripts._eval_common import (
    EvalInputError,
    load_store_and_splits,
    model_spec,
    select_split,
)

_LOG = get_logger("predictivesense.scripts.fit_thresholds")
_REPO_ROOT = Path(__file__).resolve().parents[1]
_GRID = np.round(np.arange(0.05, 0.95 + 1e-9, 0.05), 2)


def _fit_class(scores: list[float], is_tp: list[int], n_pos: int) -> dict:
    """Best-F1 threshold over a fixed grid for one class."""

    s = np.array(scores, dtype=np.float64)
    t = np.array(is_tp, dtype=np.int64)
    best = {"threshold": 0.35, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    for thr in _GRID:
        sel = s >= thr
        tp = int(t[sel].sum())
        fp = int(sel.sum() - tp)
        fn = int(n_pos - tp)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if (f1, prec) > (best["f1"], best["precision"]):
            best = {"threshold": float(thr), "f1": f1, "precision": prec, "recall": rec}
    return best


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fit per-class thresholds + margin on val.")
    p.add_argument("--split", default="val", choices=("val",),
                   help="only 'val' is allowed - fitting on test is refused (BLOCK 17.4)")
    p.add_argument("--model", default="yolo11n")
    p.add_argument("--profile", default="dev")
    args = p.parse_args(argv)
    configure_logging("INFO")

    try:
        config = load_config(args.profile)
    except (ConfigError, ValidationError) as exc:
        _LOG.error("configuration error: %s", exc)
        return 2

    try:
        store, splits = load_store_and_splits(config)
        spec = model_spec(args.model, config, models_dir=_REPO_ROOT / "models")
        image_ids = select_split(store, splits, "val")
    except (EvalInputError, SplitLeakageError, CocoStoreError) as exc:
        _LOG.error("%s", exc)
        return 2

    try:
        detector = ObjectDetector(
            spec.detector_config, provider=config.perception.provider, warmup=True,
            intra_op_threads=config.perception.intra_op_threads,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        _LOG.error("cannot start detector: %s", exc)
        return 2

    import cv2

    domain = list(config.policy.domain_classes)
    frames_dir = Path(config.dataset.root) / config.dataset.frames_dirname
    per_class_scores: dict[str, list[float]] = {c: [] for c in domain}
    per_class_is_tp: dict[str, list[int]] = {c: [] for c in domain}
    per_class_pos: dict[str, int] = {c: 0 for c in domain}
    correct_margins: list[float] = []

    for i, iid in enumerate(image_ids):
        im = store.image(iid)
        img = cv2.imread(str(frames_dir / im["file_name"]))
        if img is None:
            _LOG.warning("skipping unreadable frame %s", im["file_name"])
            continue
        from predictivesense.core.types import Frame

        frame = Frame(
            frame_id=i, capture_ts=float(i), image=np.ascontiguousarray(img),
            width=int(img.shape[1]), height=int(img.shape[0]), source_id="fit", seq=i,
        )
        dets = detector.infer(frame)
        gt = store.annotations_for(iid)
        for cls in domain:
            g = [
                [a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                for a in gt if store.category_name(int(a["category_id"])) == cls
            ]
            per_class_pos[cls] += len(g)
            d = [dd for dd in dets if dd.raw_class_name == cls or dd.class_name == cls]
            if not d:
                continue
            gb = np.array(g, dtype=np.float64).reshape(-1, 4)
            pb = np.array([list(dd.bbox) for dd in d], dtype=np.float64)
            ps = np.array([dd.score for dd in d], dtype=np.float64)
            m = greedy_match(gb, pb, ps, iou_threshold=config.eval.iou_threshold)
            tp_idx = {pi for _g, pi, _i in m.matches}
            for j, dd in enumerate(d):
                per_class_scores[cls].append(float(dd.score))
                is_tp = 1 if j in tp_idx else 0
                per_class_is_tp[cls].append(is_tp)
                if is_tp and dd.runner_up is not None:
                    correct_margins.append(float(dd.score) - float(dd.runner_up[1]))

    fitted: dict[str, float] = {}
    rows = []
    for cls in domain:
        n_pos = per_class_pos[cls]
        n_pred = len(per_class_scores[cls])
        if n_pos == 0 or n_pred == 0:
            rows.append({"class": cls, "support": n_pos, "predictions": n_pred,
                         "threshold": None, "f1": None, "precision": None, "recall": None,
                         "note": "no support / no predictions - keeps default_threshold"})
            continue
        best = _fit_class(per_class_scores[cls], per_class_is_tp[cls], n_pos)
        fitted[cls] = round(best["threshold"], 2)
        rows.append({"class": cls, "support": n_pos, "predictions": n_pred,
                     "threshold": round(best["threshold"], 2),
                     "f1": round(best["f1"], 3), "precision": round(best["precision"], 3),
                     "recall": round(best["recall"], 3),
                     "note": "thin (<10 GT) - treat as indicative" if n_pos < 10 else ""})

    margin_min = (
        round(float(np.percentile(correct_margins, 10)), 3) if correct_margins else config.policy.margin_min
    )

    payload = {
        "split": "val",
        "model": spec.name,
        "n_frames": len(image_ids),
        "iou_threshold": config.eval.iou_threshold,
        "default_threshold": config.policy.default_threshold,
        "fitted_per_class_thresholds": fitted,
        "margin_min_recommendation": margin_min,
        "margin_sample_count": len(correct_margins),
        "per_class": rows,
    }
    results_dir = Path(config.eval.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "fit_thresholds_val.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (results_dir / "fit_thresholds_val.md").write_text(_markdown(payload), encoding="utf-8")

    print("\n# paste into policy.per_class_thresholds (justified by results/fit_thresholds_val.md)")
    print("  per_class_thresholds:")
    for cls, thr in fitted.items():
        print(f"    {cls}: {thr}")
    print(f"  margin_min: {margin_min}   # 10th pct of (score - runner_up) over {len(correct_margins)} correct dets")
    _LOG.info("wrote %s", results_dir / "fit_thresholds_val.md")
    return 0


def _markdown(p: dict) -> str:
    L = [
        f"# Fitted thresholds — {p['model']} on `val` ({p['n_frames']} frames, IoU {p['iou_threshold']})",
        "",
        f"`default_threshold` fallback: {p['default_threshold']}  ·  "
        f"`margin_min` recommendation: **{p['margin_min_recommendation']}** "
        f"(10th percentile of score − runner-up over {p['margin_sample_count']} correct detections).",
        "",
        "| class | support | predictions | threshold | precision | recall | F1 | note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in p["per_class"]:
        thr = "—" if r["threshold"] is None else r["threshold"]
        pr = "—" if r["precision"] is None else r["precision"]
        rc = "—" if r["recall"] is None else r["recall"]
        f1 = "—" if r["f1"] is None else r["f1"]
        L.append(f"| {r['class']} | {r['support']} | {r['predictions']} | {thr} | {pr} | {rc} | {f1} | {r['note']} |")
    L += ["", "Classes with no support keep `default_threshold`. Thin classes "
          "(<10 GT boxes) are indicative only - widen the eval set before trusting them.", ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
