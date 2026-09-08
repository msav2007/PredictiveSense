"""Evaluate a detector (+ optional recognition policy) on a labelled split.

    python scripts\\eval_detection.py --split val  --model yolo11n   --policy off
    python scripts\\eval_detection.py --split val  --model yolo11n   --policy on
    python scripts\\eval_detection.py --split val  --model yolox_tiny --policy on
    python scripts\\eval_detection.py --split test --model yolo11n   --policy on   # once, at the end

Computes, at IoU 0.5 with greedy score-ordered matching: per-class precision /
recall / F1 / support, a confusion matrix including ``background`` and
``unknown``, the **false-class rate** (the headline), mAP@0.5, and the most
frequent confusions with example image ids. Writes
``results/eval_<model>_<policy>_<split>.json`` + a markdown table. Refuses to run
on an empty or partially-labelled split and says what is missing (BLOCK 3.11).
Every ``--split test`` run is appended to ``results/test_set_openings.md`` and a
second opening is refused unless ``--allow-reopen`` (BLOCK 17.4).
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[1]
if str(_REPO) not in _sys.path:
    _sys.path.insert(0, str(_REPO))

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from predictivesense.config.settings import ConfigError, ValidationError, load_config
from predictivesense.dataset.coco_store import CocoStoreError
from predictivesense.dataset.splits import SplitLeakageError
from predictivesense.eval import evaluate, write_reports
from predictivesense.eval.report import run_metadata
from predictivesense.logging_setup import configure_logging, get_logger
from predictivesense.perception.detector import ObjectDetector
from predictivesense.perception.policy import RecognitionPolicy
from predictivesense.telemetry.manifest import git_state

from scripts._eval_common import (
    EvalInputError,
    load_store_and_splits,
    model_spec,
    run_predictions,
    select_split,
)

_LOG = get_logger("predictivesense.scripts.eval_detection")
_REPO_ROOT = Path(__file__).resolve().parents[1]
_TEST_LOG = "test_set_openings.md"


def _log_test_opening(results_dir: Path, *, model: str, policy: str, reason: str, allow_reopen: bool) -> bool:
    path = results_dir / _TEST_LOG
    prior = path.read_text(encoding="utf-8") if path.is_file() else ""
    has_prior = "| 20" in prior  # a dated table row
    if has_prior and not allow_reopen:
        _LOG.error(
            "%s already records a test-set opening. BLOCK 17.4: `test` is opened "
            "exactly once. Re-run with --allow-reopen ONLY with an explicit, "
            "recorded reason.", path,
        )
        return False
    header = (
        "# `test` split openings\n\n"
        "BLOCK 3.6 / 17.4: `test` is opened once, at the end. Every opening is "
        "logged here with the date and reason.\n\n"
        "| date (UTC) | model | policy | reason |\n|---|---|---|---|\n"
    )
    row = f"| {datetime.now(timezone.utc).isoformat(timespec='seconds')} | {model} | {policy} | {reason} |\n"
    path.write_text((prior or header) + row, encoding="utf-8")
    _LOG.warning("recorded a test-set opening in %s", path)
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Detection evaluation harness.")
    p.add_argument("--split", required=True, choices=("val", "test"))
    p.add_argument("--model", default="yolo11n", help="yolo11n | yolox_tiny")
    p.add_argument("--policy", required=True, choices=("on", "off"))
    p.add_argument("--profile", default="dev")
    p.add_argument("--model-path", default=None)
    p.add_argument("--decode", default=None, choices=("yolo", "yolox"))
    p.add_argument("--input-size", type=int, default=None)
    p.add_argument("--reason", default="", help="required for --split test")
    p.add_argument("--allow-reopen", action="store_true")
    args = p.parse_args(argv)
    configure_logging("INFO")

    try:
        config = load_config(args.profile)
    except (ConfigError, ValidationError) as exc:
        _LOG.error("configuration error: %s", exc)
        return 2

    results_dir = Path(config.eval.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.split == "test":
        if not args.reason:
            _LOG.error("--split test requires --reason (it is logged).")
            return 2
        if not _log_test_opening(
            results_dir, model=args.model, policy=args.policy,
            reason=args.reason, allow_reopen=args.allow_reopen,
        ):
            return 2

    try:
        store, splits = load_store_and_splits(config)
        spec = model_spec(args.model, config, models_dir=_REPO_ROOT / "models")
        det_cfg = spec.detector_config
        if args.model_path:
            det_cfg = det_cfg.model_copy(update={"model_path": Path(args.model_path)})
        if args.decode:
            det_cfg = det_cfg.model_copy(update={"decode": args.decode})
        if args.input_size:
            det_cfg = det_cfg.model_copy(update={"input_size": args.input_size})
        image_ids = select_split(store, splits, args.split)
    except (EvalInputError, SplitLeakageError, CocoStoreError) as exc:
        _LOG.error("%s", exc)
        return 2

    _LOG.info(
        "evaluating %s (decode=%s, input=%d) on %d frames of split %r, policy %s",
        spec.name, det_cfg.decode, det_cfg.input_size, len(image_ids), args.split, args.policy,
    )
    try:
        detector = ObjectDetector(
            det_cfg, provider=config.perception.provider, warmup=True,
            intra_op_threads=config.perception.intra_op_threads,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        _LOG.error("cannot start detector: %s", exc)
        return 2

    policy = RecognitionPolicy(config.policy) if args.policy == "on" else None
    frames_dir = Path(config.dataset.root) / config.dataset.frames_dirname
    try:
        samples, seen_labels, state_counts = run_predictions(
            store, image_ids, frames_dir, detector, policy=policy,
        )
    except EvalInputError as exc:
        _LOG.error("%s", exc)
        return 2

    # class axis: domain classes + any raw label the detector emitted (policy
    # off) so `donut`/`wine glass` get their own confusion column.
    domain = list(config.policy.domain_classes)
    extra = sorted(l for l in seen_labels if l not in set(domain) and l != "unknown")
    class_names = domain + extra

    metrics = evaluate(samples, class_names, iou_threshold=config.eval.iou_threshold)
    commit, _dirty = git_state()
    meta = run_metadata(
        model_name=spec.name,
        model_hash=spec.model_hash(),
        policy=(config.policy.model_dump(mode="json") if policy is not None else {"enabled": False}),
        split=args.split,
        split_hash=splits.content_hash,
        git_commit=commit,
        seed=config.eval.split_seed,
        extra={
            "profile": args.profile,
            "decode": det_cfg.decode,
            "input_size": det_cfg.input_size,
            "provider": config.perception.provider,
            "n_frames": len(image_ids),
            "policy_state_counts": state_counts,
        },
    )
    stem = f"eval_{spec.name}_{args.policy}_{args.split}"
    jp, mp = write_reports(
        metrics, meta,
        json_path=results_dir / f"{stem}.json",
        md_path=results_dir / f"{stem}.md",
    )
    _LOG.info(
        "false-class rate %.4f (%d/%d matched) · macro F1 %.4f · mAP@0.5 %s · %d frames",
        metrics.false_class_rate, metrics.false_class_count, metrics.n_matched,
        metrics.macro_f1(),
        "n/a" if metrics.map50 is None else f"{metrics.map50:.4f}",
        len(image_ids),
    )
    _LOG.info("wrote %s and %s", jp, mp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
