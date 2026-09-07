"""Class-coverage audit - the most important Phase 2 output.

    python scripts\\class_coverage_audit.py --source data\\raw

Runs the detector over the developer's own footage (recorded clips under
``data/raw/`` and/or a directory of stills) and reports, per required class:
frames with >=1 detection, detection rate, confidence distribution (p10/p50/p90),
median box area as a fraction of the frame, and the count of frames where the
class was detected more than once. It also lists the most frequent UNEXPECTED
classes (likely false positives).

THIS PRODUCES DETECTION FREQUENCY AND CONFIDENCE, NOT ACCURACY. There are no
labels, so no precision / recall / mAP figure exists or is implied. The verdict
column (`reliable` / `marginal` / `unusable`) is the developer's to fill in.

Writes ``results/class_coverage.json`` and a readable ``results/class_coverage.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from predictivesense.config.settings import load_config
from predictivesense.core.types import Frame
from predictivesense.logging_setup import configure_logging, get_logger
from predictivesense.perception.classes import REQUIRED_CLASSES
from predictivesense.perception.detector import ObjectDetector

_LOG = get_logger("predictivesense.scripts.class_coverage_audit")
_REPO_ROOT = Path(__file__).resolve().parents[1]
_VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

_HEADER = (
    "These are DETECTION FREQUENCIES and CONFIDENCE DISTRIBUTIONS on UNLABELLED "
    "footage. They are NOT accuracy, precision, recall or mAP - no labelled data "
    "exists yet. A class appearing often here means the pretrained detector fires "
    "on it in this room, not that it fires correctly. The verdict column is the "
    "developer's call."
)


def _pct(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    frac = rank - low
    if low + 1 >= len(ordered):
        return ordered[-1]
    return ordered[low] + frac * (ordered[low + 1] - ordered[low])


def _iter_sources(source: Path):
    if source.is_file():
        yield source
        return
    for path in sorted(source.rglob("*")):
        if path.is_file() and path.suffix.lower() in (_VIDEO_SUFFIXES | _IMAGE_SUFFIXES):
            yield path


def _iter_frames(path: Path, stride: int, max_frames: int):
    if path.suffix.lower() in _IMAGE_SUFFIXES:
        img = cv2.imread(str(path))
        if img is not None:
            yield np.ascontiguousarray(img)
        return
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        _LOG.warning("could not open %s", path)
        return
    idx = 0
    emitted = 0
    try:
        while emitted < max_frames:
            ok, img = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                emitted += 1
                yield np.ascontiguousarray(img)
            idx += 1
    finally:
        cap.release()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Detector class-coverage audit over real footage.")
    p.add_argument("--source", required=True, help="clip, or directory of clips/stills")
    p.add_argument("--profile", default="dev")
    p.add_argument("--stride", type=int, default=1, help="use every Nth video frame")
    p.add_argument("--max-frames-per-source", type=int, default=5000)
    p.add_argument("--results-dir", default=str(_REPO_ROOT / "results"))
    args = p.parse_args(argv)
    configure_logging("INFO")
    try:  # the Windows console is cp1252; the report has non-latin-1 glyphs
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    source = Path(args.source)
    if not source.exists():
        _LOG.error("source not found: %s", source)
        return 2
    sources = list(_iter_sources(source))
    if not sources:
        _LOG.error("no clips or stills under %s", source)
        return 2

    pcfg = load_config(args.profile).perception
    try:
        detector = ObjectDetector(
            pcfg.detector, provider=pcfg.provider, warmup=True,
            intra_op_threads=pcfg.intra_op_threads,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        _LOG.error("cannot start detector: %s", exc)
        return 2

    required = set(REQUIRED_CLASSES)
    total_frames = 0
    # per class: frames_with>=1, frames_with>1, [confidences], [box_area_fracs]
    frames_with_1: dict[str, int] = {c: 0 for c in REQUIRED_CLASSES}
    frames_with_multi: dict[str, int] = {c: 0 for c in REQUIRED_CLASSES}
    confs: dict[str, list[float]] = {c: [] for c in REQUIRED_CLASSES}
    areas: dict[str, list[float]] = {c: [] for c in REQUIRED_CLASSES}
    unexpected_frames: dict[str, int] = {}

    for src in sources:
        n_before = total_frames
        for img in _iter_frames(src, args.stride, args.max_frames_per_source):
            total_frames += 1
            h, w = img.shape[:2]
            frame_area = float(w * h)
            fr = Frame(
                frame_id=total_frames, capture_ts=float(total_frames), image=img,
                width=w, height=h, source_id=str(src), seq=total_frames,
            )
            per_class_count: dict[str, int] = {}
            per_class_best: dict[str, float] = {}
            per_class_area: dict[str, float] = {}
            for det in detector.infer(fr):
                name = det.class_name
                per_class_count[name] = per_class_count.get(name, 0) + 1
                x1, y1, x2, y2 = det.bbox
                area_frac = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)) / frame_area
                if det.score > per_class_best.get(name, -1.0):
                    per_class_best[name] = det.score
                    per_class_area[name] = area_frac
            for name, count in per_class_count.items():
                if name in required:
                    frames_with_1[name] += 1
                    if count > 1:
                        frames_with_multi[name] += 1
                    confs[name].append(per_class_best[name])
                    areas[name].append(per_class_area[name])
                else:
                    unexpected_frames[name] = unexpected_frames.get(name, 0) + 1
        _LOG.info("%s: +%d frames (total %d)", src.name, total_frames - n_before, total_frames)

    per_class = []
    for c in REQUIRED_CLASSES:
        n = frames_with_1[c]
        per_class.append(
            {
                "class": c,
                "frames_with_detection": n,
                "detection_rate": round(n / total_frames, 4) if total_frames else 0.0,
                "conf_p10": round(_pct(confs[c], 10), 3),
                "conf_p50": round(_pct(confs[c], 50), 3),
                "conf_p90": round(_pct(confs[c], 90), 3),
                "median_box_area_frac": round(_pct(areas[c], 50), 5),
                "frames_with_multiple": frames_with_multi[c],
                "verdict": "?",  # developer fills: reliable / marginal / unusable
            }
        )

    top_unexpected = sorted(
        unexpected_frames.items(), key=lambda kv: kv[1], reverse=True
    )[:15]

    payload = {
        "header": _HEADER,
        "source": str(source),
        "clips": [str(s) for s in sources],
        "total_frames": total_frames,
        "stride": args.stride,
        "detector_model": detector.model_name,
        "detector_input_size": detector.input_size,
        "default_conf": pcfg.detector.default_conf,
        "class_thresholds": pcfg.detector.class_thresholds,
        "required_classes": per_class,
        "top_unexpected_classes": [
            {"class": k, "frames_seen": v, "frame_rate": round(v / total_frames, 4)}
            for k, v in top_unexpected
        ],
    }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "class_coverage.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (results_dir / "class_coverage.md").write_text(_markdown(payload), encoding="utf-8")
    _LOG.info("wrote %s", results_dir / "class_coverage.md")
    print("\n" + _markdown(payload))
    return 0


def _markdown(p: dict) -> str:
    lines = [
        "# Class-coverage audit",
        "",
        "> **" + p["header"] + "**",
        "",
        f"Source: `{p['source']}` · {len(p['clips'])} file(s) · "
        f"{p['total_frames']} frames (stride {p['stride']}) · "
        f"detector `{p['detector_model']}` @ {p['detector_input_size']} · "
        f"default_conf {p['default_conf']}",
        "",
        "## Required classes",
        "",
        "| class | frames w/ ≥1 | rate | conf p10 | conf p50 | conf p90 "
        "| median box area frac | frames w/ >1 | verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in p["required_classes"]:
        lines.append(
            f"| {r['class']} | {r['frames_with_detection']} | {r['detection_rate']} "
            f"| {r['conf_p10']} | {r['conf_p50']} | {r['conf_p90']} "
            f"| {r['median_box_area_frac']} | {r['frames_with_multiple']} "
            f"| {r['verdict']} |"
        )
    lines += [
        "",
        "Verdict column: the developer confirms `reliable` / `marginal` / "
        "`unusable` per class after watching the overlay (BLOCK 11). The MVP "
        "scenario list depends on this table.",
        "",
        "## Top unexpected classes (likely false positives)",
        "",
        "| class | frames seen | frame rate |",
        "|---|---|---|",
    ]
    for r in p["top_unexpected_classes"]:
        lines.append(f"| {r['class']} | {r['frames_seen']} | {r['frame_rate']} |")
    if not p["top_unexpected_classes"]:
        lines.append("| _(none)_ | 0 | 0 |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
