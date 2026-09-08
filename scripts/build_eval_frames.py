"""Sample frames from the developer's recorded clips for labelling (Phase 2.5).

    python scripts\\build_eval_frames.py --source data\\raw --out data\\eval\\frames \\
        --every-n 15 --max-per-clip 40

For every ``*.webm`` / ``*.mp4`` under ``--source`` that has a sibling
``ClipManifest`` JSON, this decodes every ``--every-n``-th frame (up to
``--max-per-clip``), writes it as a JPEG under ``--out``, and records provenance
per frame: source clip, capture timestamp, session id, camera device, and the
clip's condition tags (BLOCK 3.1.2). It then creates / updates the COCO
annotation store (images only, ``labelled=false``) so the developer can label at
``/label``.

Idempotent: re-running does not duplicate frames or images (files are content-
addressed by session+clip+frame index; :meth:`CocoStore.add_image` de-dupes by
file name). No model is loaded here - detector-assisted seeding happens in the
labelling tool, per-frame, and is recorded on the image (BLOCK 3.1.4).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from predictivesense.config.settings import ConfigError, ValidationError, load_config
from predictivesense.dataset.coco_store import CocoStore, domain_categories
from predictivesense.dataset.quality import FrameProvenance
from predictivesense.logging_setup import configure_logging, get_logger

_LOG = get_logger("predictivesense.scripts.build_eval_frames")
_REPO_ROOT = Path(__file__).resolve().parents[1]
_VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}


def _iter_clips(source: Path):
    if source.is_file():
        yield source
        return
    for path in sorted(source.rglob("*")):
        if path.is_file() and path.suffix.lower() in _VIDEO_SUFFIXES:
            yield path


def _manifest_for(clip: Path) -> dict:
    sidecar = clip.with_suffix(".json")
    if sidecar.is_file():
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            _LOG.warning("clip manifest %s unreadable: %s", sidecar, exc)
    return {}


def _condition_tags(manifest: dict) -> list[str]:
    tags: list[str] = []
    raw = str(manifest.get("scenario_tag", "")).strip()
    if raw:
        tags.extend(t for t in raw.replace(",", " ").split() if t)
    notes = str(manifest.get("notes", "")).strip()
    if notes:
        tags.append(f"notes:{notes}")
    return tags


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Sample evaluation frames for labelling.")
    p.add_argument("--source", default="data/raw", help="clip or directory of clips")
    p.add_argument("--out", default="data/eval/frames", help="output frames dir")
    p.add_argument("--every-n", type=int, default=15, help="keep every Nth decoded frame")
    p.add_argument("--max-per-clip", type=int, default=40, help="cap frames sampled per clip")
    p.add_argument("--profile", default="dev")
    args = p.parse_args(argv)
    configure_logging("INFO")

    if args.every_n < 1:
        _LOG.error("--every-n must be >= 1")
        return 2

    try:
        config = load_config(args.profile)
    except (ConfigError, ValidationError) as exc:
        _LOG.error("configuration error: %s", exc)
        return 2

    source = Path(args.source)
    if not source.exists():
        _LOG.error("source not found: %s", source)
        return 2
    clips = list(_iter_clips(source))
    if not clips:
        _LOG.error("no video clips under %s", source)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    coco_path = Path(config.dataset.coco_path)
    cats = domain_categories(config.policy.domain_classes)
    store = CocoStore.load_or_create(coco_path, cats, info={"description": "PredictiveSense Phase 2.5 eval set"})

    index: list[dict] = []
    total_written = 0
    sessions: set[str] = set()
    for clip in clips:
        manifest = _manifest_for(clip)
        session_id = str(manifest.get("session_id") or clip.parent.name)
        clip_id = str(manifest.get("clip_id") or clip.stem)
        device = str(manifest.get("device_label") or "unknown")
        cond = _condition_tags(manifest)
        sessions.add(session_id)

        cap = cv2.VideoCapture(str(clip))
        if not cap.isOpened():
            _LOG.warning("could not open %s - skipping", clip)
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or float(manifest.get("nominal_fps") or 30.0)
        idx = 0
        kept = 0
        try:
            while kept < args.max_per_clip:
                ok, img = cap.read()
                if not ok:
                    break
                if idx % args.every_n == 0:
                    h, w = img.shape[:2]
                    fname = f"{session_id}_{clip_id}_{idx:06d}.jpg"
                    dest = out_dir / fname
                    cv2.imwrite(str(dest), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                    prov = FrameProvenance(
                        source_clip=str(clip),
                        timestamp_s=round(idx / fps, 4) if fps else float(idx),
                        session_id=session_id,
                        camera_device=device,
                        condition_tags=cond,
                        frame_index=idx,
                        clip_id=clip_id,
                    )
                    store.add_image(
                        file_name=fname, width=int(w), height=int(h),
                        session_id=session_id, provenance=prov.to_dict(),
                    )
                    index.append({"file_name": fname, **prov.to_dict(), "width": int(w), "height": int(h)})
                    kept += 1
                    total_written += 1
                idx += 1
        finally:
            cap.release()
        _LOG.info("%s: sampled %d frame(s) (session %s)", clip.name, kept, session_id)

    store.save(coco_path)
    (out_dir / "index.json").write_text(
        json.dumps({"every_n": args.every_n, "max_per_clip": args.max_per_clip,
                    "frames": index}, indent=2) + "\n",
        encoding="utf-8",
    )
    c = store.counts()
    _LOG.info(
        "wrote %d frames across %d session(s) -> %s ; store now %d images (%d labelled) -> %s",
        total_written, len(sessions), out_dir, c["images"], c["labelled"], coco_path,
    )
    if len(sessions) < 4:
        _LOG.warning(
            "only %d recording session(s) available - BLOCK 3.1.7 targets >=4. "
            "Record more clips (both cameras) for a stronger eval set; the "
            "harness prints the sample count beside every metric so this stays honest.",
            len(sessions),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
