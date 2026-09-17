"""PHASE 11 PART B section 15 - baseline (pretrained detector) vs custom
crop classifier, on the identical held-out split.

    python scripts\\evaluate_classifier_baseline.py --fixture

Requires the ``[train]`` extra (trains a fresh classifier to compare) and the
detector weights (``-m models`` - baseline inference needs the real ONNX
detector). Writes ``results/baseline_vs_custom_<label>.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from predictivesense.logging_setup import configure_logging, get_logger

_LOG = get_logger("predictivesense.scripts.evaluate_classifier_baseline")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--objects-root", default="data/objects")
    p.add_argument("--fixture", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", default=None)
    p.add_argument("--results-dir", default="results")
    args = p.parse_args(argv)
    configure_logging("INFO")

    try:
        import torch  # noqa: F401
    except ImportError:
        _LOG.error("the [train] extra is not installed - `pip install -e \".[train]\"`")
        return 2

    from predictivesense.config.settings import load_config
    from predictivesense.perception.detector import ObjectDetector
    from predictivesense.training.classifier_registry import ClassifierRegistry
    from predictivesense.training.dataset import (
        class_map_for,
        positive_records,
        rejection_records,
        validate_dataset,
    )
    from predictivesense.training.evaluate import evaluate_predictor, predictor_from_detector
    from predictivesense.training.splits import build_crop_splits
    from predictivesense.training.train import TrainConfig, run_training

    fixture_dir = None
    objects_root = args.objects_root
    label = args.label
    if args.fixture:
        from predictivesense.training.synthetic_fixture import build_synthetic_object_store

        fixture_dir = tempfile.TemporaryDirectory(prefix="ps_eval_fixture_")
        objects_root = str(Path(fixture_dir.name) / "objects")
        build_synthetic_object_store(objects_root, n_per_class=18, n_sessions_per_class=3, seed=args.seed)
        label = label or "fixture"
    label = label or "objects"

    dcfg = load_config("dev")
    try:
        detector = ObjectDetector(dcfg.perception.detector, provider="cpu", warmup=True, intra_op_threads=2)
    except FileNotFoundError as exc:
        _LOG.error("detector weights required for the baseline comparison: %s", exc)
        if fixture_dir is not None:
            fixture_dir.cleanup()
        return 2

    try:
        with tempfile.TemporaryDirectory(prefix="ps_eval_run_") as run_dir:
            reg = ClassifierRegistry(path=Path(run_dir) / "classifier_registry.json")
            cfg = TrainConfig(
                objects_root=objects_root, results_dir=str(Path(run_dir) / "results"),
                models_dir=str(Path(run_dir) / "models"), epochs=12, seed=args.seed,
            )
            result = run_training(cfg, registry=reg)

            records, _summary = validate_dataset(objects_root)
            class_map = class_map_for(records)
            splits = build_crop_splits(
                records, train_fraction=0.6, val_fraction=0.2, test_fraction=0.2, seed=args.seed
            )
            by_id = {r.crop_id: r for r in records}
            test = [by_id[c] for c in splits.crop_ids["test"]]
            test_pos = positive_records(test)
            test_rej = rejection_records(test)

            custom_metrics = result.test_metrics  # already computed by run_training, same split

            baseline_predict = predictor_from_detector(detector)
            baseline_metrics = evaluate_predictor(baseline_predict, test_pos, test_rej, sorted(class_map))
    finally:
        if fixture_dir is not None:
            fixture_dir.cleanup()

    payload = {
        "label": label, "classes": sorted(class_map),
        "custom": custom_metrics, "baseline": baseline_metrics.to_dict(),
    }
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"baseline_vs_custom_{label}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md = _markdown(payload)
    (results_dir / f"baseline_vs_custom_{label}.md").write_text(md, encoding="utf-8")
    _LOG.info(
        "custom accuracy=%.3f (latency %.2fms) vs baseline accuracy=%.3f (latency %.2fms) -> %s",
        custom_metrics["accuracy"], custom_metrics["latency_ms_mean"],
        baseline_metrics.to_dict()["accuracy"], baseline_metrics.to_dict()["latency_ms_mean"],
        results_dir / f"baseline_vs_custom_{label}.md",
    )
    return 0


def _markdown(p: dict) -> str:
    c, b = p["custom"], p["baseline"]
    return (
        f"# Baseline (pretrained detector) vs custom crop classifier (Phase 11 Part B section 15) - `{p['label']}`\n\n"
        f"Classes: {p['classes']}. Identical held-out test split for both.\n\n"
        "| metric | custom classifier | baseline (pretrained detector) |\n"
        "|---|---|---|\n"
        f"| accuracy | {c['accuracy']} | {b['accuracy']} |\n"
        f"| false_class_rate (on negatives/hard-negatives) | {c['false_class_rate']} | {b['false_class_rate']} |\n"
        f"| latency mean (ms) | {c['latency_ms_mean']} | {b['latency_ms_mean']} |\n"
        f"| latency p95 (ms) | {c['latency_ms_p95']} | {b['latency_ms_p95']} |\n\n"
        f"Custom confusion matrix: `{json.dumps(c['confusion_matrix'])}`\n\n"
        f"Baseline confusion matrix: `{json.dumps(b['confusion_matrix'])}`\n\n"
        "The baseline is the EXISTING pretrained detector's own top class name for each crop - its "
        "COCO-80 vocabulary has no concept of these custom classes, so a near-zero baseline accuracy "
        "is the expected, honest result (section 15), not a bug in this evaluator. Latency is reported "
        "separately, never blended into one score: the custom classifier is also far cheaper per crop "
        "(a tiny 64x64 CNN) than running the full detector again.\n"
    )


if __name__ == "__main__":
    sys.exit(main())
