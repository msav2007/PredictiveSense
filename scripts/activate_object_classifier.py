"""PHASE 11 PART B section 13 - explicit activation / rollback.

Exactly one classifier version is active at a time (or none - the baseline
state, detector-only recognition). Activation always requires a VALIDATED
version; it is never a side effect of training or validation.

    python scripts\\activate_object_classifier.py <version_id>
    python scripts\\activate_object_classifier.py --rollback
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

_LOG = get_logger("predictivesense.scripts.activate_object_classifier")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("version_id", nargs="?", default=None)
    group.add_argument("--rollback", action="store_true", help="deactivate the custom classifier (baseline)")
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
    target = None if args.rollback else args.version_id
    try:
        active = reg.activate(target)
    except ClassifierRegistryError as exc:
        _LOG.error("%s", exc)
        return 2

    if active is None:
        _LOG.info("rolled back: no custom classifier active (baseline / detector-only recognition)")
    else:
        _LOG.info("active classifier: %s (classes=%s)", active.version_id, sorted(active.class_map))
    return 0


if __name__ == "__main__":
    sys.exit(main())
