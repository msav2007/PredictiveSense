"""PHASE 11 PART B - Studio samples -> a trained, registered crop classifier.

    # Real collected data:
    python scripts\\train_object_classifier.py --objects-root data\\objects

    # Prove the pipeline end-to-end without any real collection (section 12.4):
    python scripts\\train_object_classifier.py --fixture

Every run is versioned and recorded (never overwritten): a JSON manifest
under ``results/train_run_<version_id>.json`` and an entry in
``models/classifier_registry.json`` (unvalidated, inactive - training never
has the side effect of validating or activating, section 13.2). Use
``scripts/validate_object_classifier.py`` and
``scripts/activate_object_classifier.py`` next.

Requires the ``[train]`` extra: ``pip install -e ".[train]"``.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from predictivesense.logging_setup import configure_logging, get_logger

_LOG = get_logger("predictivesense.scripts.train_object_classifier")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--objects-root", default="data/objects")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--models-dir", default="models/custom")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-fraction", type=float, default=0.6)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument(
        "--fixture", action="store_true",
        help="train on a freshly-generated synthetic fixture dataset instead of --objects-root "
             "(section 12.4 - proves the pipeline end to end with no real collection)",
    )
    p.add_argument(
        "--external-manifest", action="append", default=[], metavar="PATH",
        help="Phase 12: path to an external-dataset class manifest (from "
             "scripts/import_external_dataset.py), combined with --objects-root. "
             "May be given multiple times.",
    )
    p.add_argument(
        "--classifier-registry", default=None, metavar="PATH",
        help="override the classifier registry path (default: config.training.classifier_registry_path, "
             "i.e. models/classifier_registry.json)",
    )
    args = p.parse_args(argv)
    configure_logging("INFO")

    try:
        import torch  # noqa: F401
    except ImportError:
        _LOG.error(
            "the [train] extra is not installed - run `pip install -e \".[train]\"` "
            "(torch/torchvision; the serving app never needs them)."
        )
        return 2

    from predictivesense.config.settings import load_config
    from predictivesense.training.classifier_registry import ClassifierRegistry
    from predictivesense.training.train import TrainConfig, run_training

    registry_path = (
        Path(args.classifier_registry)
        if args.classifier_registry
        else _REPO / load_config("dev").training.classifier_registry_path
    )
    registry = ClassifierRegistry(path=registry_path)

    fixture_dir: tempfile.TemporaryDirectory | None = None
    objects_root = args.objects_root
    if args.fixture:
        from predictivesense.training.synthetic_fixture import build_synthetic_object_store

        fixture_dir = tempfile.TemporaryDirectory(prefix="ps_train_fixture_")
        objects_root = str(Path(fixture_dir.name) / "objects")
        ids = build_synthetic_object_store(objects_root, n_per_class=18, seed=args.seed)
        _LOG.info("built synthetic fixture with classes: %s", ids)

    try:
        config = TrainConfig(
            objects_root=objects_root, results_dir=args.results_dir, models_dir=args.models_dir,
            epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, seed=args.seed,
            train_fraction=args.train_fraction, val_fraction=args.val_fraction,
            test_fraction=args.test_fraction, external_manifests=tuple(args.external_manifest),
        )
        result = run_training(config, registry=registry)
    finally:
        if fixture_dir is not None:
            fixture_dir.cleanup()

    _LOG.info(
        "trained %s: test accuracy=%.3f false_class_rate=%.3f artifact=%s (sha256 %s)",
        result.version_id, result.test_metrics["accuracy"], result.test_metrics["false_class_rate"],
        result.onnx_path, result.registry_entry.artifact_sha256[:12],
    )
    _LOG.info("manifest: %s", result.manifest_path)
    _LOG.info(
        "NOT active (unvalidated) - run scripts/validate_object_classifier.py %s next", result.version_id
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
