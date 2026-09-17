"""PHASE 11 PART B section 13.2 - the explicit validation step.

Training never validates its own result (a trained model recording good
metrics is not the same as a human deciding those metrics are good enough).
This script prints the recorded test metrics for one trained version and, if
they clear ``--min-accuracy``, marks it validated - a separate, explicit
operation from training. An unvalidated model can never be activated.

    python scripts\\validate_object_classifier.py <version_id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from predictivesense.logging_setup import configure_logging, get_logger
from predictivesense.training.classifier_registry import ClassifierRegistry, ClassifierRegistryError

_LOG = get_logger("predictivesense.scripts.validate_object_classifier")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("version_id")
    p.add_argument("--min-accuracy", type=float, default=0.5)
    p.add_argument("--max-false-class-rate", type=float, default=0.5)
    p.add_argument("--classifier-registry", default=None, metavar="PATH")
    args = p.parse_args(argv)
    configure_logging("INFO")

    from predictivesense.config.settings import load_config

    registry_path = (
        Path(args.classifier_registry)
        if args.classifier_registry
        else _REPO / load_config("dev").training.classifier_registry_path
    )
    reg = ClassifierRegistry(path=registry_path)
    try:
        version = reg.get(args.version_id)
    except ClassifierRegistryError as exc:
        _LOG.error("%s", exc)
        return 2

    metrics = version.metrics
    _LOG.info(
        "%s: accuracy=%.3f false_class_rate=%.3f classes=%s",
        version.version_id, metrics.get("accuracy", -1), metrics.get("false_class_rate", -1),
        sorted(version.class_map),
    )
    if metrics.get("accuracy", 0.0) < args.min_accuracy:
        _LOG.error(
            "refusing to validate: accuracy %.3f < --min-accuracy %.3f",
            metrics.get("accuracy", 0.0), args.min_accuracy,
        )
        return 1
    if metrics.get("false_class_rate", 1.0) > args.max_false_class_rate:
        _LOG.error(
            "refusing to validate: false_class_rate %.3f > --max-false-class-rate %.3f",
            metrics.get("false_class_rate", 1.0), args.max_false_class_rate,
        )
        return 1

    reg.validate_version(args.version_id)
    _LOG.info("%s marked validated. Activate with scripts/activate_object_classifier.py", args.version_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
