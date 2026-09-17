"""Phase 10 section 3.1 - re-compare detector `input_size` 640 vs 480 on
footage, by PER-CLASS RECALL, not totals.

    python scripts\\input_size_recall_compare.py --source data\\raw

The Phase 8 sweep that justified 480 found identical primary-tier totals
(197/197/197) at 320/480/640 - which proves the one clip it used contained no
object small enough to discriminate, not that recall holds in general. This
script re-runs the real ``ObjectDetector`` at both input sizes over the same
clip(s) and reports per-class raw-detection counts side by side, so a
recall difference (if any exists on the available footage) is visible per
class rather than hidden in a single combined total.

Writes ``results/input_size_recall_<label>.{json,md}``.
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
from collections import Counter
from pathlib import Path

from predictivesense.camera.file_source import FileSource
from predictivesense.config.settings import ConfigError, ValidationError, load_config
from predictivesense.logging_setup import configure_logging, get_logger
from predictivesense.perception.detector import ObjectDetector

_LOG = get_logger("predictivesense.scripts.input_size_recall_compare")

REQUIRED_CLASSES: tuple[str, ...] = (
    "person", "cell phone", "bottle", "cup", "laptop", "keyboard", "mouse",
    "chair", "book", "backpack",
)
_VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}


def _find_clips(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(q for q in source.rglob("*") if q.suffix.lower() in _VIDEO_SUFFIXES)


def _run_size(detector: ObjectDetector, clip: Path, *, limit: int | None) -> dict:
    per_class: Counter = Counter()
    per_class_score_sum: Counter = Counter()
    other: Counter = Counter()
    frames = 0
    src = FileSource(clip, replay_mode="asfast")
    src.start()
    try:
        while limit is None or frames < limit:
            frame = src.read()
            if frame is None:
                break
            frames += 1
            for d in detector.infer(frame):
                if d.class_name in REQUIRED_CLASSES:
                    per_class[d.class_name] += 1
                    per_class_score_sum[d.class_name] += d.score
                else:
                    other[d.class_name] += 1
    finally:
        src.stop()
    return {
        "frames": frames,
        "per_class_raw_count": {c: per_class.get(c, 0) for c in REQUIRED_CLASSES},
        "per_class_mean_score": {
            c: round(per_class_score_sum[c] / per_class[c], 4) if per_class[c] else None
            for c in REQUIRED_CLASSES
        },
        "other_classes_top10": dict(other.most_common(10)),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 10 input_size 640 vs 480 per-class recall.")
    p.add_argument("--source", default="data/raw")
    p.add_argument("--frames", type=int, default=0, help="max frames per clip (0 = all)")
    p.add_argument("--profile", default="dev")
    p.add_argument("--label", default=None)
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
    clips = _find_clips(source)
    if not clips:
        _LOG.error("no video clips found under %s", source)
        return 2

    limit = args.frames if args.frames > 0 else None
    label = args.label or re.sub(r"[^A-Za-z0-9_-]", "-", source.name or "clips")

    results_by_size: dict[int, list[dict]] = {}
    for size in (480, 640):
        det_cfg = config.perception.detector.model_copy(update={"input_size": size})
        detector = ObjectDetector(
            det_cfg, provider=config.perception.provider, warmup=True,
            intra_op_threads=config.perception.intra_op_threads,
        )
        clip_results = []
        for clip in clips:
            _LOG.info("input_size=%d, clip %s ...", size, clip.name)
            res = _run_size(detector, clip, limit=limit)
            res["clip"] = str(clip)
            clip_results.append(res)
            _LOG.info(
                "input_size=%d clip=%s frames=%d per_class=%s",
                size, clip.name, res["frames"], res["per_class_raw_count"],
            )
        results_by_size[size] = clip_results

    results_dir = Path(config.eval.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": label, "profile": args.profile, "required_classes": list(REQUIRED_CLASSES),
        "results_by_input_size": {str(k): v for k, v in results_by_size.items()},
    }
    (results_dir / f"input_size_recall_{label}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (results_dir / f"input_size_recall_{label}.md").write_text(
        _markdown(label, results_by_size), encoding="utf-8"
    )
    _LOG.info("wrote %s", results_dir / f"input_size_recall_{label}.md")
    return 0


def _markdown(label: str, results_by_size: dict[int, list[dict]]) -> str:
    clips = results_by_size[480]
    L = [
        f"# Detector input_size 640 vs 480 - per-class recall (Phase 10 section 3.1) - `{label}`",
        "",
        "Re-comparison on the available footage, by per-class raw-detection "
        "RECALL (not totals) - the Phase 8 sweep's identical 197/197/197 "
        "primary-tier totals proved only that its one clip had nothing small "
        "enough to discriminate, not that recall holds in general.",
        "",
    ]
    for i, clip480 in enumerate(clips):
        clip640 = results_by_size[640][i]
        L += [
            f"## Clip `{clip480['clip']}`",
            "",
            f"{clip480['frames']} frames.",
            "",
            "| class | raw count @480 | raw count @640 | delta (640-480) | "
            "mean score @480 | mean score @640 |",
            "|---|---|---|---|---|---|",
        ]
        for cls in REQUIRED_CLASSES:
            c480 = clip480["per_class_raw_count"][cls]
            c640 = clip640["per_class_raw_count"][cls]
            s480 = clip480["per_class_mean_score"][cls]
            s640 = clip640["per_class_mean_score"][cls]
            L.append(
                f"| {cls} | {c480} | {c640} | {c640 - c480:+d} | "
                f"{'-' if s480 is None else f'{s480:.3f}'} | "
                f"{'-' if s640 is None else f'{s640:.3f}'} |"
            )
        L += [
            "",
            f"Other classes @480, top 10: `{clip480['other_classes_top10']}`. "
            f"@640: `{clip640['other_classes_top10']}`.",
            "",
        ]
    L += [
        "## Verdict",
        "",
        "Every required class with ANY raw detections at either size on the "
        "available footage should be compared row by row above. A class at "
        "zero for both sizes is a footage-content or model-capability "
        "question this clip cannot answer either way (it never appears in "
        "the frame, or the model does not recognise it regardless of input "
        "resolution) - not evidence for or against changing the default. "
        "The default stays `input_size: 480` unless a row above shows a "
        "material recall loss at 480 relative to 640 for a required class "
        "that DOES appear in this footage; see the counts above for that "
        "determination on this specific clip.",
        "",
    ]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
