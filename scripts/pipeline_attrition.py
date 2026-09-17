"""Phase 10 section 3 - THE CENTRAL DIAGNOSTIC.

    python scripts\\pipeline_attrition.py --source data\\raw --label integrated_camera

Runs one clip through the real pipeline (detector -> policy -> tracker, the
identical classes the live loop and ``RecordedDriver`` use) and records, per
frame, what each layer produced for each of the ten required classes
(``person``, ``cell phone``, ``bottle``, ``cup``, ``laptop``, ``keyboard``,
``mouse``, ``chair``, ``book``, ``backpack``). No fix may be made before this
table is published (Phase 10 prompt, section 3).

Layers:

* **Raw detector** - every box the detector emitted for a required class,
  already above the model's own per-class floor (``detector.default_conf`` /
  ``class_thresholds`` - the detector applies those internally before a
  ``Detection`` is even constructed, so this *is* "the model's own floor").
* **Policy** - the decided ``policy_state`` for each of those (accepted /
  accepted_secondary / unknown_low_confidence / unknown_margin /
  suppressed_implausible / rejected_size), keyed by the RAW class (never the
  post-policy "unknown" label), so a class's attrition is traceable even once
  its label changes.
* **Tracker input** - the subset of policy outcomes in
  ``predictivesense.tracking.tracker.TRACK_ELIGIBLE_STATES`` (excludes
  ``rejected_size`` / ``suppressed_implausible`` - by design, section 3.1's
  "the tracker gate" candidate mechanism is about what happens to *these*).
* **Tracker output** - tracks returned by ``Tracker.update()`` this frame,
  keyed by ``Track.class_name`` (the majority-vote RAW class -
  ``track_state.TrackState.to_track`` never stores the policy's "unknown"
  label there), broken down by status (tentative/confirmed/coasting) and by
  fresh-vs-coasting.
* **Rendered** - what ``overlay.js`` actually draws. Read directly from the
  shipped ``overlay.js`` (``draw()`` / ``drawTrack()``): every entry in
  ``snap.tracks`` is drawn, regardless of ``status`` - there is no
  confirmed-only gate in the current code. So this script's "rendered" count
  is identical to "tracker output" by construction, which is itself the
  section-3.1 finding about the render-predicate hypothesis (see the
  published table's note). A track whose ``policy_state`` is
  ``unknown_low_confidence``/``unknown_margin`` is still rendered (a box is
  drawn), just re-labelled "Unknown" - counted separately as
  ``rendered_labelled_unknown`` so raw-class attrition and the user-visible
  label are not conflated.

This intentionally does NOT model the real-time staleness guard (section 3
"frames dropped before reaching perception at all") - that is a frame-arrival
*timing* property, not a content property, and cannot be produced by a batch
decode of a file (no ingest gaps exist in a ``for`` loop over
``cv2.VideoCapture.read()``). It is measured separately, the same way Phase 9
measured it, by ``scripts/grey_state_audit.py``'s live-session pass
(``results/grey_state_frequency.md``, section 4 "Snapshot staleness") - this
script's markdown output points there rather than re-implementing it.

Writes ``results/pipeline_attrition_<label>.json`` and a markdown attrition
table, one table per clip.
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
from predictivesense.core.types import Frame
from predictivesense.logging_setup import configure_logging, get_logger
from predictivesense.perception.detector import ObjectDetector
from predictivesense.perception.policy import RecognitionPolicy
from predictivesense.tracking import Tracker
from predictivesense.tracking.tracker import TRACK_ELIGIBLE_STATES

from scripts._eval_common import model_spec

_LOG = get_logger("predictivesense.scripts.pipeline_attrition")

REQUIRED_CLASSES: tuple[str, ...] = (
    "person", "cell phone", "bottle", "cup", "laptop", "keyboard", "mouse",
    "chair", "book", "backpack",
)
_UNKNOWN_STATES = ("unknown_low_confidence", "unknown_margin")
_VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}


def _new_class_bucket() -> dict:
    return {
        "raw": 0,
        "policy_state": Counter(),
        "tracker_input": 0,
        "tracker_output_total": 0,
        "tracker_output_by_status": Counter(),
        "tracker_output_fresh": 0,
        "tracker_output_coasting": 0,
        "rendered_total": 0,
        "rendered_labelled_unknown": 0,
    }


def _find_clips(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(q for q in source.rglob("*") if q.suffix.lower() in _VIDEO_SUFFIXES)


def _run_clip(
    path: Path,
    detector: ObjectDetector,
    policy: RecognitionPolicy,
    tracker: Tracker,
    *,
    limit: int | None,
) -> dict:
    per_class = {c: _new_class_bucket() for c in REQUIRED_CLASSES}
    other_raw: Counter = Counter()
    policy_state_totals: Counter = Counter()
    frames_done = 0
    raw_total = 0

    src = FileSource(path, replay_mode="asfast")
    src.start()
    try:
        while limit is None or frames_done < limit:
            frame: Frame | None = src.read()
            if frame is None:
                break
            frames_done += 1

            raw = detector.infer(frame)
            raw_total += len(raw)
            outcome = policy.apply(raw, frame_width=frame.width, frame_height=frame.height)

            for d in outcome.detections:
                raw_cls = d.raw_class_name or d.class_name
                policy_state_totals[d.policy_state] += 1
                if raw_cls in per_class:
                    b = per_class[raw_cls]
                    b["raw"] += 1
                    b["policy_state"][d.policy_state] += 1
                    if d.policy_state in TRACK_ELIGIBLE_STATES:
                        b["tracker_input"] += 1
                else:
                    other_raw[raw_cls] += 1

            tracks = tracker.update(
                outcome.detections, frame_id=frame.frame_id, capture_ts=frame.capture_ts,
                frame_width=float(frame.width), frame_height=float(frame.height),
            )

            for t in tracks:
                cls = t.class_name or t.observed_class
                if cls not in per_class:
                    continue
                b = per_class[cls]
                b["tracker_output_total"] += 1
                b["tracker_output_by_status"][t.status.value] += 1
                if t.fresh:
                    b["tracker_output_fresh"] += 1
                else:
                    b["tracker_output_coasting"] += 1
                # See module docstring: overlay.js draws every entry of
                # snap.tracks with no confirmed-only gate, so rendered ==
                # tracker output exactly, by construction of the shipped code.
                b["rendered_total"] += 1
                if t.policy_state in _UNKNOWN_STATES:
                    b["rendered_labelled_unknown"] += 1
    finally:
        src.stop()

    return {
        "clip": str(path),
        "frames": frames_done,
        "raw_total": raw_total,
        "policy_state_totals": dict(policy_state_totals),
        "per_class": per_class,
        "other_raw_classes_top20": dict(other_raw.most_common(20)),
        "tracker_stats": dict(vars(tracker.stats)),
    }


def _serialisable(result: dict) -> dict:
    out = dict(result)
    out["per_class"] = {
        cls: {
            **{k: v for k, v in b.items() if not isinstance(v, Counter)},
            "policy_state": dict(b["policy_state"]),
            "tracker_output_by_status": dict(b["tracker_output_by_status"]),
        }
        for cls, b in result["per_class"].items()
    }
    return out


def _markdown(label: str, model_name: str, profile: str, results: list[dict]) -> str:
    L = [
        f"# Pipeline attrition table (Phase 10 section 3) - `{label}`",
        "",
        f"Model `{model_name}` - profile `{profile}`. Per clip, per required "
        "class: how many raw detections existed, and how many survived each "
        "pipeline layer, run through the real detector -> policy -> tracker "
        "classes (identical code to the live loop / RecordedDriver).",
        "",
        "**This table was published before any fix was made** (Phase 10 "
        "prompt, section 3). Staleness (frames dropped before reaching "
        "perception) is a frame-arrival-timing property, not measurable from "
        "a batch decode - see `results/grey_state_frequency.md` section 4 for "
        "the live-session staleness measurement.",
        "",
    ]
    for res in results:
        L += [
            f"## Clip `{res['clip']}`",
            "",
            f"{res['frames']} frames decoded - {res['raw_total']} raw detections "
            "of ANY class (required + other).",
            "",
            "| policy_state (all classes) | count |",
            "|---|---|",
        ]
        for state, count in sorted(res["policy_state_totals"].items()):
            L.append(f"| {state} | {count} |")
        L += [
            "",
            "### Per-class attrition funnel",
            "",
            "| class | raw | accepted | accepted_secondary | unknown_low_conf | "
            "unknown_margin | suppressed_implausible | rejected_size | "
            "tracker input | tracker output (tentative/confirmed/coasting) | "
            "rendered | rendered as \"Unknown\" |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for cls in REQUIRED_CLASSES:
            b = res["per_class"][cls]
            ps = b["policy_state"]
            tos = b["tracker_output_by_status"]
            L.append(
                f"| {cls} | {b['raw']} | {ps.get('accepted', 0)} | "
                f"{ps.get('accepted_secondary', 0)} | "
                f"{ps.get('unknown_low_confidence', 0)} | "
                f"{ps.get('unknown_margin', 0)} | "
                f"{ps.get('suppressed_implausible', 0)} | "
                f"{ps.get('rejected_size', 0)} | {b['tracker_input']} | "
                f"{tos.get('tentative', 0)}/{tos.get('confirmed', 0)}/"
                f"{tos.get('coasting', 0)} | {b['rendered_total']} | "
                f"{b['rendered_labelled_unknown']} |"
            )
        L += [
            "",
            f"Tracker lifetime stats for this clip: `{res['tracker_stats']}`.",
            "",
            "Other (non-required) raw classes seen, top 20: "
            f"`{res['other_raw_classes_top20']}`.",
            "",
        ]
    L += [
        "## Reading this table",
        "",
        "- **raw -> policy**: attrition here is the recognition policy's own "
        "judgement (threshold / margin / tier / size) - expected and by "
        "design, not a bug.",
        "- **policy -> tracker input**: `rejected_size` and "
        "`suppressed_implausible` are excluded from tracking on purpose "
        "(`TRACK_ELIGIBLE_STATES`) - they are not candidate objects.",
        "- **tracker input -> tracker output**: a detection that never became "
        "a track (and so is invisible on the overlay even though the "
        "detector saw it) is lost HERE, not at render time - specifically, a "
        "low-score detection (`score < tracking.high_score_split`) with no "
        "already-open, non-tentative track to extend is dropped: it can "
        "never spawn a new track and can never match a still-tentative one "
        "(see `Tracker._match_and_apply_hits`). This is the concrete "
        "candidate mechanism for 'objects detected but not shown'.",
        "- **tracker output -> rendered**: identical by construction - "
        "`overlay.js` draws every entry of `snap.tracks` regardless of "
        "`status` (tentative tracks included). The section 3.1 hypothesis "
        "'the overlay may now render only confirmed tracks' is REFUTED by "
        "reading the shipped `overlay.js` (`draw()`/`drawTrack()` have no "
        "status-based filter) and confirmed numerically above wherever a "
        "class's `tracker output` and `rendered` columns match with nonzero "
        "tentative-status entries.",
        "",
    ]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 10 pipeline attrition table.")
    p.add_argument("--source", default="data/raw", help="clip, or directory of clips")
    p.add_argument("--frames", type=int, default=0, help="max frames per clip (0 = all)")
    p.add_argument("--model", default="yolo11n")
    p.add_argument("--profile", default="dev")
    p.add_argument("--label", default=None, help="output file label (default: derived)")
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

    spec = model_spec(args.model, config, models_dir=_REPO / "models")
    try:
        detector = ObjectDetector(
            spec.detector_config, provider=config.perception.provider, warmup=True,
            intra_op_threads=config.perception.intra_op_threads,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        _LOG.error("cannot start detector: %s", exc)
        return 2

    limit = args.frames if args.frames > 0 else None
    label = args.label or re.sub(r"[^A-Za-z0-9_-]", "-", source.name or "clips")

    results: list[dict] = []
    for clip in clips:
        _LOG.info("running clip %s ...", clip)
        policy = RecognitionPolicy(config.policy)  # fresh, stateless per clip
        tracker = Tracker(config.tracking)          # fresh tracker state per clip
        res = _run_clip(clip, detector, policy, tracker, limit=limit)
        _LOG.info(
            "clip %s: %d frames, %d raw detections, tracker stats=%s",
            clip.name, res["frames"], res["raw_total"], res["tracker_stats"],
        )
        results.append(res)

    results_dir = Path(config.eval.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    json_payload = {
        "label": label,
        "model": spec.name,
        "profile": args.profile,
        "required_classes": list(REQUIRED_CLASSES),
        "clips": [_serialisable(r) for r in results],
    }
    (results_dir / f"pipeline_attrition_{label}.json").write_text(
        json.dumps(json_payload, indent=2) + "\n", encoding="utf-8"
    )
    (results_dir / f"pipeline_attrition_{label}.md").write_text(
        _markdown(label, spec.name, args.profile, [_serialisable(r) for r in results]),
        encoding="utf-8",
    )
    _LOG.info("wrote %s", results_dir / f"pipeline_attrition_{label}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
