"""Labelling API for the Phase 2.5 evaluation set.

| Method | Path | Behaviour |
|---|---|---|
| GET  | ``/api/labels/frames``        | paged frame list (``labelled``, ``seeded``, session id) |
| GET  | ``/api/labels/frame/{id}``    | frame metadata + existing boxes (+ optional detector seed) |
| POST | ``/api/labels/frame/{id}``    | replace that frame's boxes; set ``labelled``, record ``seeded`` |
| GET  | ``/api/labels/progress``      | counts by class, session, seeded/unseeded |
| GET  | ``/api/labels/image/{id}``    | the frame JPEG |
| GET  | ``/api/labels/eval-summary`` | latest ``results/eval_*`` summary, or ``{available: false}`` |

The annotation store (COCO detection JSON) is created by
``scripts/build_eval_frames.py``. Until it exists every endpoint here returns
503 with the command to run. A single module lock serialises writes (labelling
is a one-person task); :meth:`CocoStore.save` writes a ``.bak`` before every
overwrite.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from predictivesense.dataset.coco_store import CocoStore, CocoStoreError, domain_categories
from predictivesense.dataset.quality import ProgressSummary
from predictivesense.logging_setup import get_logger

__all__ = ["router"]

_LOG = get_logger(__name__)
_LOCK = threading.Lock()

router = APIRouter()


# -- request models ------------------------------------------------


class LabelBox(BaseModel):
    category: str
    bbox: list[float] = Field(min_length=4, max_length=4)  # x, y, w, h


class SaveFrameBody(BaseModel):
    boxes: list[LabelBox] = Field(default_factory=list)
    seeded: bool = False
    done: bool = True


# -- store access ------------------------------------------------


def _paths(request: Request) -> tuple[Path, Path]:
    cfg = request.app.state.config
    coco = Path(cfg.dataset.coco_path)
    frames = Path(cfg.dataset.root) / cfg.dataset.frames_dirname
    return coco, frames


def _load_store(request: Request) -> CocoStore:
    coco_path, _frames = _paths(request)
    if not coco_path.is_file():
        raise HTTPException(
            status_code=503,
            detail=(
                f"no annotation store at {coco_path}. Run "
                f"`python scripts/build_eval_frames.py --source data/raw "
                f"--out {Path(request.app.state.config.dataset.root) / 'frames'} "
                f"--every-n 15 --max-per-clip 40` first."
            ),
        )
    try:
        cats = domain_categories(request.app.state.config.policy.domain_classes)
        return CocoStore.load_or_create(coco_path, cats)
    except CocoStoreError as exc:
        raise HTTPException(status_code=500, detail=f"annotation store invalid: {exc}") from exc


def _frame_payload(store: CocoStore, image_id: int) -> dict[str, Any]:
    im = store.image(image_id)
    boxes = [
        {
            "category": store.category_name(int(an["category_id"])),
            "category_id": int(an["category_id"]),
            "bbox": [float(v) for v in an["bbox"]],
        }
        for an in store.annotations_for(image_id)
    ]
    return {
        "image_id": image_id,
        "file_name": im["file_name"],
        "width": int(im["width"]),
        "height": int(im["height"]),
        "session_id": im["ps_session_id"],
        "seeded": bool(im.get("seeded")),
        "labelled": bool(im.get("labelled")),
        "provenance": im.get("ps_provenance", {}),
        "boxes": boxes,
        "categories": [
            {"id": i + 1, "name": n} for i, n in enumerate(store.category_names)
        ],
    }


# -- routes ------------------------------------------------


@router.get("/api/labels/frames")
async def list_frames(request: Request, offset: int = 0, limit: int = 200) -> dict[str, Any]:
    store = _load_store(request)
    ids = store.image_ids()
    offset = max(0, offset)
    limit = max(1, min(1000, limit))
    page = ids[offset : offset + limit]
    rows = []
    for iid in page:
        im = store.image(iid)
        rows.append(
            {
                "image_id": iid,
                "file_name": im["file_name"],
                "session_id": im["ps_session_id"],
                "labelled": bool(im.get("labelled")),
                "seeded": bool(im.get("seeded")),
                "boxes": len(store.annotations_for(iid)),
            }
        )
    return {
        "total": len(ids),
        "offset": offset,
        "limit": limit,
        "labelled": sum(1 for i in ids if store.image(i).get("labelled")),
        "frames": rows,
    }


@router.get("/api/labels/frame/{image_id}")
async def get_frame(request: Request, image_id: int, seed: int = 0) -> dict[str, Any]:
    store = _load_store(request)
    if not store.has_image(image_id):
        raise HTTPException(status_code=404, detail=f"no frame with id {image_id}")
    payload = _frame_payload(store, image_id)
    payload["seed_available"] = _seed_detector(request) is not None
    if seed:
        payload["seed_boxes"] = _seed_boxes(request, store, image_id)
    return payload


def _seed_detector(request: Request):
    """The loop's detector, if perception is running (used for optional
    detector-assisted seeding - always recorded per image, BLOCK 3.1.4)."""

    engine = getattr(request.app.state, "perception", None)
    return getattr(engine, "_detector", None) if engine is not None else None


def _seed_boxes(request: Request, store: CocoStore, image_id: int) -> list[dict[str, Any]]:
    det = _seed_detector(request)
    if det is None:
        return []
    from predictivesense.camera._opencv import read_image_bgr
    from predictivesense.core.types import Frame

    _coco, frames_dir = _paths(request)
    path = (frames_dir / store.image(image_id)["file_name"]).resolve()
    img = read_image_bgr(path)
    if img is None:
        return []
    frame = Frame(
        frame_id=0, capture_ts=0.0, image=img,
        width=int(img.shape[1]), height=int(img.shape[0]), source_id="seed", seq=0,
    )
    domain = set(store.category_names)
    out: list[dict[str, Any]] = []
    for d in det.infer(frame):
        if d.raw_class_name in domain or d.class_name in domain:
            name = d.raw_class_name if d.raw_class_name in domain else d.class_name
            x1, y1, x2, y2 = d.bbox
            out.append(
                {"category": name, "bbox": [round(x1, 2), round(y1, 2),
                                            round(x2 - x1, 2), round(y2 - y1, 2)],
                 "score": round(float(d.score), 3)}
            )
    return out


@router.post("/api/labels/frame/{image_id}")
async def save_frame(request: Request, image_id: int, body: SaveFrameBody) -> dict[str, Any]:
    coco_path, _frames = _paths(request)
    with _LOCK:
        store = _load_store(request)
        if not store.has_image(image_id):
            raise HTTPException(status_code=404, detail=f"no frame with id {image_id}")
        try:
            store.set_frame_boxes(
                image_id,
                [{"category": b.category, "bbox": b.bbox} for b in body.boxes],
                seeded=body.seeded,
                labelled=body.done,
            )
        except CocoStoreError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        store.save(coco_path)
    _LOG.info(
        "labelled frame %s: %d boxes, seeded=%s, done=%s",
        image_id, len(body.boxes), body.seeded, body.done,
    )
    return _frame_payload(_load_store(request), image_id)


@router.get("/api/labels/progress")
async def progress(request: Request) -> dict[str, Any]:
    store = _load_store(request)
    cfg = request.app.state.config
    summary = ProgressSummary.from_store(
        store, min_unseeded_fraction=cfg.dataset.min_unseeded_fraction
    )
    return {
        **summary.to_dict(),
        "target_frames": 200,
        "coco_path": str(cfg.dataset.coco_path),
    }


@router.get("/api/labels/image/{image_id}")
async def frame_image(request: Request, image_id: int) -> FileResponse:
    store = _load_store(request)
    if not store.has_image(image_id):
        raise HTTPException(status_code=404, detail=f"no frame with id {image_id}")
    _coco, frames_dir = _paths(request)
    path = (frames_dir / store.image(image_id)["file_name"]).resolve()
    if frames_dir.resolve() not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="frame image not found on disk")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/api/labels/eval-summary")
async def eval_summary(request: Request) -> dict[str, Any]:
    """Latest ``results/eval_*_*.json`` summary for the Research group. Shows
    'not yet run' rather than erroring when nothing is there (BLOCK 10)."""

    results_dir = Path(request.app.state.config.eval.results_dir)
    files = sorted(results_dir.glob("eval_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return {"available": False, "reason": "no results/eval_*.json - run scripts/eval_detection.py"}
    try:
        doc = json.loads(files[0].read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"available": False, "reason": f"latest eval file unreadable: {exc}"}
    metrics = doc.get("metrics", {})
    meta = doc.get("metadata", {})
    return {
        "available": True,
        "file": files[0].name,
        "model": meta.get("model_name"),
        "split": meta.get("split"),
        "policy_enabled": (meta.get("policy") or {}).get("enabled"),
        "false_class_rate": metrics.get("headline", {}).get("false_class_rate"),
        "macro_f1": metrics.get("macro_f1"),
        "map50": metrics.get("map50"),
        "sample_counts": metrics.get("sample_counts", {}),
        "all_files": [p.name for p in files[:12]],
    }
