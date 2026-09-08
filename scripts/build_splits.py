"""Build session-disjoint ``val`` / ``test`` splits for the evaluation set.

    python scripts\\build_splits.py [--profile dev]

Whole recording sessions are assigned to ``val`` or ``test`` - never individual
frames (adjacent frames are near-duplicates; a frame split leaks). Writes
``data/eval/splits.json`` with a content hash so a later run detects that the
annotation store has changed underneath it. Exits non-zero when there is nothing
labelled to split, or the resulting split would leak.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from predictivesense.config.settings import ConfigError, ValidationError, load_config
from predictivesense.dataset.coco_store import CocoStore, CocoStoreError, domain_categories
from predictivesense.dataset.splits import SplitLeakageError, build_splits
from predictivesense.logging_setup import configure_logging, get_logger

_LOG = get_logger("predictivesense.scripts.build_splits")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Session-disjoint val/test split builder.")
    p.add_argument("--profile", default="dev")
    args = p.parse_args(argv)
    configure_logging("INFO")

    try:
        config = load_config(args.profile)
    except (ConfigError, ValidationError) as exc:
        _LOG.error("configuration error: %s", exc)
        return 2

    coco_path = Path(config.dataset.coco_path)
    if not coco_path.is_file():
        _LOG.error(
            "no annotation store at %s - run scripts/build_eval_frames.py and "
            "label frames at /label first", coco_path,
        )
        return 2
    try:
        store = CocoStore.load_or_create(
            coco_path, domain_categories(config.policy.domain_classes)
        )
    except CocoStoreError as exc:
        _LOG.error("annotation store invalid: %s", exc)
        return 2

    counts = store.counts()
    if counts["labelled"] == 0:
        _LOG.error(
            "0 of %d frames are labelled - label at /label before splitting "
            "(BLOCK 3.11: the harness refuses to run on an unlabelled split)",
            counts["images"],
        )
        return 2

    try:
        splits = build_splits(
            store,
            val_fraction=config.eval.val_fraction,
            test_fraction=config.eval.test_fraction,
            seed=config.eval.split_seed,
        )
    except SplitLeakageError as exc:
        _LOG.error("split build failed: %s", exc)
        return 2

    out = splits.save(config.dataset.splits_path)
    _LOG.info("=== splits ===")
    for name in ("val", "test"):
        _LOG.info(
            "%-4s : %d session(s) %s -> %d image(s)",
            name, len(splits.sessions[name]),
            splits.sessions[name], len(splits.image_ids[name]),
        )
    _LOG.info("content_hash : %s", splits.content_hash)
    _LOG.info("written      : %s", out)
    if not splits.sessions["test"]:
        _LOG.warning(
            "test split is EMPTY (only %d labelled session[s]). Record and label "
            "more sessions before the final test run.", counts["sessions"],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
