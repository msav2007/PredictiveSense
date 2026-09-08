"""Phase 5 policy audit - evidence for every policy change, WITHOUT labels.

    python scripts\\policy_audit.py --source data\\raw --frames 400

Runs the detector over real footage (a clip or a directory of clips/stills) and
reports, with reconciling counts:

* total raw detections and the per-state breakdown after the policy;
* per **raw class**: accepted / accepted_secondary / unknown / suppressed
  (implausible) / rejected_size;
* the confidence distribution (p10 / p50 / p90 / max) of the detections each
  non-accepted state captured - so an over-aggressive threshold is visible;
* the top raw classes suppressed by the implausible tier.

It also runs a **synthetic-frame control** (deterministic noise) to confirm the
pipeline does not hallucinate detections on content with no objects.

**These are detection FREQUENCIES on UNLABELLED footage - not accuracy,
precision, recall or mAP.** No policy threshold may be changed without a number
from this audit or a stated, documented reason (BLOCK 3.2).

Writes ``results/policy_audit_<label>.{json,md}``.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[1]
if str(_REPO) not in _sys.path:
    _sys.path.insert(0, str(_REPO))

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from predictivesense.camera.synthetic import SyntheticSource
from predictivesense.config.settings import ConfigError, ValidationError, load_config
from predictivesense.core.types import Frame
from predictivesense.logging_setup import configure_logging, get_logger
from predictivesense.perception.detector import ObjectDetector
from predictivesense.perception.policy import POLICY_STATES, RecognitionPolicy

from scripts._eval_common import model_spec

_LOG = get_logger("predictivesense.scripts.policy_audit")
_VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}
_IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
_HEADER = (
    "DETECTION FREQUENCIES on UNLABELLED footage - NOT accuracy, precision, "
    "recall or mAP. A state count here is how often the pretrained detector + "
    "policy landed in that state in this room, not whether it was correct."
)


def _pct(xs: list[float], p: float) -> float:
    return round(float(np.percentile(xs, p)), 3) if xs else -1.0


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-").lower() or "audit"


def _iter_real_frames(source: Path, limit: int):
    files = (
        [source]
        if source.is_file()
        else sorted(
            q for q in source.rglob("*")
            if q.suffix.lower() in (_VIDEO_SUFFIXES | _IMG_SUFFIXES)
        )
    )
    n = 0
    for f in files:
        if f.suffix.lower() in _IMG_SUFFIXES:
            img = cv2.imread(str(f))
            if img is not None:
                n += 1
                yield np.ascontiguousarray(img)
            if n >= limit:
                return
            continue
        cap = cv2.VideoCapture(str(f))
        try:
            while n < limit:
                ok, img = cap.read()
                if not ok:
                    break
                n += 1
                yield np.ascontiguousarray(img)
        finally:
            cap.release()
        if n >= limit:
            return


def _audit(detector: ObjectDetector, policy: RecognitionPolicy, frames_iter, *, source_id: str):
    from predictivesense.perception.policy import PolicyCounts

    acc = PolicyCounts()
    raw_total = 0
    state_scores: dict[str, list[float]] = {s: [] for s in POLICY_STATES}
    per_raw: dict[str, Counter] = defaultdict(Counter)
    suppressed_raw: Counter = Counter()
    relabels: Counter = Counter()
    frames_done = 0

    for i, img in enumerate(frames_iter):
        frames_done += 1
        fr = Frame(
            frame_id=i, capture_ts=float(i), image=np.ascontiguousarray(img),
            width=int(img.shape[1]), height=int(img.shape[0]),
            source_id=source_id, seq=i,
        )
        raw = detector.infer(fr)
        raw_total += len(raw)
        outcome = policy.apply(raw, frame_width=fr.width, frame_height=fr.height)
        acc = acc.merged(outcome.counts)
        for d in outcome.detections:
            st = d.policy_state
            rawn = d.raw_class_name or d.class_name
            state_scores.setdefault(st, []).append(float(d.score))
            per_raw[rawn][st] += 1
            if st == "suppressed_implausible":
                suppressed_raw[rawn] += 1
        for rawn, decn in outcome.raw_to_decided:
            if rawn != decn:
                relabels[f"{rawn} -> {decn}"] += 1

    return {
        "frames": frames_done,
        "raw_detections": raw_total,
        "reconciles": acc.reconciles,
        "counts": acc.as_metrics(""),
        "state_confidence": {
            s: {
                "n": len(v), "p10": _pct(v, 10), "p50": _pct(v, 50),
                "p90": _pct(v, 90), "max": round(max(v), 3) if v else -1.0,
            }
            for s, v in state_scores.items()
        },
        "per_raw_class": {
            k: {s: int(c.get(s, 0)) for s in POLICY_STATES} for k, c in sorted(per_raw.items())
        },
        "top_suppressed_implausible_raw": suppressed_raw.most_common(20),
        "top_relabels": relabels.most_common(25),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 5 policy audit (unlabelled).")
    p.add_argument("--source", default="data/raw", help="clip, or directory of clips/stills")
    p.add_argument("--frames", type=int, default=400, help="max real frames to read")
    p.add_argument("--model", default="yolo11n")
    p.add_argument("--profile", default="dev")
    p.add_argument("--synthetic-control", type=int, default=30,
                   help="synthetic (noise) frames to run as a control; 0 to skip")
    p.add_argument("--label", default=None, help="output label (default: from --source)")
    args = p.parse_args(argv)
    configure_logging("INFO")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    try:
        config = load_config(args.profile)
    except (ConfigError, ValidationError) as exc:
        _LOG.error("configuration error: %s", exc)
        return 2

    source = Path(args.source)
    if not source.exists():
        _LOG.error("source not found: %s", source)
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

    real = _audit(
        detector, policy, _iter_real_frames(source, args.frames), source_id=str(source)
    )
    if real["frames"] == 0:
        _LOG.error("no frames decoded from %s", source)
        return 2

    control = None
    if args.synthetic_control > 0:
        syn = SyntheticSource(
            width=config.source.width, height=config.source.height,
            target_fps=max(config.source.target_fps, 120.0), seed=config.source.seed,
        )
        syn.start()
        try:
            def _syn_frames():
                for _ in range(args.synthetic_control):
                    f = syn.read()
                    if f is None:
                        break
                    yield f.image
            control = _audit(detector, policy, _syn_frames(), source_id="synthetic")
        finally:
            syn.stop()

    label = _slug(args.label or source.name or "audit")
    payload = {
        "header": _HEADER,
        "model": spec.name,
        "profile": args.profile,
        "source": str(source),
        "policy": policy.info(),
        "active_thresholds": policy.active_thresholds(),
        "real_footage": real,
        "synthetic_control": control,
    }
    results_dir = Path(config.eval.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"policy_audit_{label}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (results_dir / f"policy_audit_{label}.md").write_text(_markdown(payload), encoding="utf-8")
    r = real
    _LOG.info(
        "audit over %d real frames: %d raw dets -> accepted %d / accepted_secondary %d / "
        "unknown %d / suppressed %d / rejected_size %d (reconciles=%s)",
        r["frames"], r["raw_detections"],
        int(r["counts"]["accepted"]), int(r["counts"]["accepted_secondary"]),
        int(r["counts"]["unknown_low_confidence"] + r["counts"]["unknown_margin"]),
        int(r["counts"]["suppressed_implausible"]), int(r["counts"]["rejected_size"]),
        r["reconciles"],
    )
    if control is not None:
        _LOG.info(
            "synthetic control: %d frames -> %d raw detections (expect ~0)",
            control["frames"], control["raw_detections"],
        )
    _LOG.info("wrote %s", results_dir / f"policy_audit_{label}.md")
    return 0


def _state_table(block: dict) -> list[str]:
    c = block["counts"]
    tot = max(1, int(block["raw_detections"]))
    L = ["| state | count | share | conf p10 | conf p50 | conf p90 | conf max | n |",
         "|---|---|---|---|---|---|---|---|"]
    for s in POLICY_STATES:
        sc = block["state_confidence"].get(s, {})
        L.append(
            f"| {s} | {int(c[s])} | {int(c[s]) / tot:.3f} | {sc.get('p10', -1)} "
            f"| {sc.get('p50', -1)} | {sc.get('p90', -1)} | {sc.get('max', -1)} | {sc.get('n', 0)} |"
        )
    L.append(
        "| **reconciles (all six == raw)** | "
        f"{block['reconciles']} | — | — | — | — | — | {int(block['raw_detections'])} |"
    )
    return L


def _markdown(p: dict) -> str:
    r = p["real_footage"]
    L = [
        "# Recognition policy audit (Phase 5, UNLABELLED)",
        "",
        f"> **{p['header']}**",
        "",
        f"Model `{p['model']}` · profile `{p['profile']}` · source `{p['source']}` · "
        f"{r['frames']} real frames · {r['raw_detections']} raw detections.",
        "",
        "## Per-state breakdown (real footage)",
        "",
        *_state_table(r),
        "",
        "## Per raw class (real footage)",
        "",
        "| raw class | accepted | accepted_secondary | unknown_low_conf | unknown_margin "
        "| suppressed_implausible | rejected_size |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, d in r["per_raw_class"].items():
        L.append(
            f"| {name} | {d['accepted']} | {d['accepted_secondary']} "
            f"| {d['unknown_low_confidence']} | {d['unknown_margin']} "
            f"| {d['suppressed_implausible']} | {d['rejected_size']} |"
        )
    L += ["", "## Top raw classes suppressed by the implausible tier", "",
          "| raw class | frames suppressed |", "|---|---|"]
    for name, n in r["top_suppressed_implausible_raw"]:
        L.append(f"| {name} | {n} |")
    if not r["top_suppressed_implausible_raw"]:
        L.append("| _(none)_ | 0 |")
    L += ["", "## Most frequent raw -> decided relabels", "", "| change | count |", "|---|---|"]
    for change, n in r["top_relabels"]:
        L.append(f"| {change} | {n} |")
    if not r["top_relabels"]:
        L.append("| _(none)_ | 0 |")

    ctrl = p["synthetic_control"]
    L += ["", "## Synthetic-frame control", ""]
    if ctrl is None:
        L.append("_(skipped)_")
    else:
        L.append(
            f"{ctrl['frames']} synthetic (noise) frames -> **{ctrl['raw_detections']}** raw "
            f"detections (expected ~0 - the pipeline should not hallucinate objects on noise). "
            f"Reconciles: {ctrl['reconciles']}."
        )
    L += ["", "## Active per-class thresholds", "",
          "UNFITTED - `scripts/fit_thresholds.py` has not run on a labelled `val` split; "
          "these are the `default_threshold` fallback.", "",
          "```", json.dumps(p["active_thresholds"], indent=2), "```", ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
