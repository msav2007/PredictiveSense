"""Measure the recognition policy's *mechanical* effect on unlabelled frames.

    python scripts\\policy_effect.py --frames data\\eval\\frames

This is **not accuracy** - there are no labels. It runs the detector over the
sampled evaluation frames twice (policy off, then on) and reports, with
reconciling counts, how many raw detections each policy rule rejected or
relabelled, plus the most frequent ``raw -> decided`` changes. It is the
before/labelling artifact for BLOCK 3.13 ("each rule ... each counted") and the
"Measured" part of the phase report; the labelled before/after on ``val`` is the
developer-gated follow-up.

Writes ``results/policy_effect_unlabelled.{json,md}``.
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
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from predictivesense.config.settings import ConfigError, ValidationError, load_config
from predictivesense.core.types import Frame
from predictivesense.logging_setup import configure_logging, get_logger
from predictivesense.perception.detector import ObjectDetector
from predictivesense.perception.policy import RecognitionPolicy

from scripts._eval_common import model_spec

_LOG = get_logger("predictivesense.scripts.policy_effect")
_IMG_SUFFIXES = {".jpg", ".jpeg", ".png"}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Unlabelled policy-effect measurement.")
    p.add_argument("--frames", default="data/eval/frames")
    p.add_argument("--model", default="yolo11n")
    p.add_argument("--profile", default="dev")
    args = p.parse_args(argv)
    configure_logging("INFO")

    try:
        config = load_config(args.profile)
    except (ConfigError, ValidationError) as exc:
        _LOG.error("configuration error: %s", exc)
        return 2

    frames_dir = Path(args.frames)
    images = sorted(q for q in frames_dir.rglob("*") if q.suffix.lower() in _IMG_SUFFIXES)
    if not images:
        _LOG.error("no frame images under %s - run scripts/build_eval_frames.py", frames_dir)
        return 2

    spec = model_spec(args.model, config, models_dir=_REPO / "models")
    try:
        detector = ObjectDetector(
            spec.detector_config, provider=config.perception.provider, warmup=True,
            intra_op_threads=config.perception.intra_op_threads,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        _LOG.error("cannot start detector: %s", exc)
        return 2
    policy = RecognitionPolicy(config.policy)

    raw_total = 0
    raw_label_counts: Counter = Counter()
    decided_label_counts: Counter = Counter()
    relabels: Counter = Counter()
    state_counts: Counter = Counter()
    frames_done = 0
    from predictivesense.perception.policy import PolicyCounts

    acc = PolicyCounts()
    for path in images:
        img = cv2.imread(str(path))
        if img is None:
            _LOG.warning("unreadable frame %s - skipped", path)
            continue
        frames_done += 1
        frame = Frame(
            frame_id=frames_done, capture_ts=float(frames_done),
            image=np.ascontiguousarray(img), width=int(img.shape[1]),
            height=int(img.shape[0]), source_id=str(path), seq=frames_done,
        )
        raw = detector.infer(frame)
        raw_total += len(raw)
        for d in raw:
            raw_label_counts[d.raw_class_name or d.class_name] += 1
        outcome = policy.apply(raw, frame_width=frame.width, frame_height=frame.height)
        acc = acc.merged(outcome.counts)
        for d in outcome.detections:
            decided_label_counts[d.class_name] += 1
            state_counts[d.policy_state] += 1
        for rawn, decn in outcome.raw_to_decided:
            if rawn != decn:
                relabels[f"{rawn} -> {decn}"] += 1

    payload = {
        "note": "MECHANICAL policy effect on UNLABELLED frames - NOT accuracy, "
                "precision, recall or mAP. Counts only.",
        "model": spec.name,
        "frames": frames_done,
        "raw_detections": raw_total,
        "reconciles": acc.reconciles,
        "counts": acc.as_metrics(""),
        "policy_state_counts": dict(state_counts),
        "top_raw_labels": raw_label_counts.most_common(15),
        "top_decided_labels": decided_label_counts.most_common(15),
        "top_relabels": relabels.most_common(20),
        "active_thresholds": policy.active_thresholds(),
        "policy": policy.info(),
    }
    results_dir = Path(config.eval.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"policy_effect_unlabelled_{spec.name}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (results_dir / f"policy_effect_unlabelled_{spec.name}.md").write_text(_markdown(payload), encoding="utf-8")
    _LOG.info(
        "policy over %d frames: %d raw dets -> accepted %d / unknown %d / rejected %d "
        "(reconciles=%s)",
        frames_done, raw_total, acc.accepted, acc.unknown, acc.rejected, acc.reconciles,
    )
    _LOG.info("wrote %s", results_dir / f"policy_effect_unlabelled_{spec.name}.md")
    return 0


def _markdown(p: dict) -> str:
    c = p["counts"]
    L = [
        "# Recognition policy — mechanical effect (UNLABELLED)",
        "",
        f"> **{p['note']}**",
        "",
        f"Model `{p['model']}` · {p['frames']} sampled frames · {p['raw_detections']} raw detections.",
        "",
        "| outcome | count | share of raw |",
        "|---|---|---|",
    ]
    tot = max(1, int(p["raw_detections"]))
    for k in ("accepted", "unknown_low_confidence", "unknown_margin",
              "rejected_out_of_domain", "rejected_size"):
        L.append(f"| {k} | {int(c[k])} | {int(c[k]) / tot:.3f} |")
    L.append(f"| **reconciles (accepted+unknown+rejected == raw)** | {p['reconciles']} | — |")
    L += ["", "## Most frequent raw → decided relabels", "",
          "| change | count |", "|---|---|"]
    for change, n in p["top_relabels"]:
        L.append(f"| {change} | {n} |")
    if not p["top_relabels"]:
        L.append("| _(none)_ | 0 |")
    L += ["", "## Top raw detector labels (before policy)", "",
          "| class | count |", "|---|---|"]
    for name, n in p["top_raw_labels"]:
        L.append(f"| {name} | {n} |")
    L += ["", "Active per-class thresholds (fallback `default_threshold` until "
          "`scripts/fit_thresholds.py` runs on labelled `val`):", "",
          "```", json.dumps(p["active_thresholds"], indent=2), "```", ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
