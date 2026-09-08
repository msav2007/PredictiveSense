"""Object Learning Studio API (Phase 4).

| Method | Path | Behaviour |
|---|---|---|
| GET | ``/api/objects`` | list profiles with counts + a coverage summary |
| POST | ``/api/objects`` | create a profile |
| GET/PATCH/DELETE | ``/api/objects/{id}`` | read / patch metadata / soft-delete |
| GET | ``/api/objects/{id}/samples`` | list samples |
| POST | ``/api/objects/{id}/samples`` | multipart: image + box + conditions + role -> sample |
| PATCH | ``/api/objects/{id}/samples/{sid}`` | adjust box, tags, role |
| DELETE | ``/api/objects/{id}/samples/{sid}`` | soft-delete a sample |
| GET | ``/api/objects/{id}/coverage`` | coverage summary + guidance strings |
| GET | ``/api/objects/{id}/samples/{sid}/image`` | the stored JPEG (``?thumb=1`` for the thumbnail) |
| GET | ``/api/objects/vocab`` | condition vocabularies + role/kind/status sets |

Storing images is not training. Every sample carries exactly one bounding box,
condition tags, quality fields and full provenance. The object dataset lives
under ``config.objects.root`` and is strictly separate from ``data/eval``.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from predictivesense.camera._opencv import decode_image_bgr, encode_jpeg, thumbnail_jpeg
from predictivesense.logging_setup import get_logger
from predictivesense.objects.quality import (
    CoverageTargets,
    compute_sample_quality,
    coverage_summary,
)
from predictivesense.objects.registry import ObjectRegistry, ObjectStoreError
from predictivesense.objects.samples import SampleStore
from predictivesense.objects.vocab import (
    CONDITION_VOCAB,
    KINDS,
    ROLES,
    STATUSES,
    default_conditions,
)

__all__ = ["router"]

_LOG = get_logger(__name__)
router = APIRouter()


def _parse_float(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# -- request models ------------------------------------------------


class CreateObjectBody(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    kind: str = "class"
    parent_class: str | None = None
    category: str | None = None
    description: str | None = None
    confusable_with: list[str] | None = None


class PatchObjectBody(BaseModel):
    name: str | None = None
    kind: str | None = None
    parent_class: str | None = None
    category: str | None = None
    description: str | None = None
    confusable_with: list[str] | None = None
    status: str | None = None


class PatchSampleBody(BaseModel):
    box: list[float] | None = Field(default=None, min_length=4, max_length=4)
    conditions: dict[str, str] | None = None
    role: str | None = None
    negative_for: list[str] | None = None


# -- store access ------------------------------------------------


def _registry(request: Request) -> ObjectRegistry:
    return ObjectRegistry(request.app.state.config.objects.root)


def _sample_store(request: Request, object_id: str) -> SampleStore:
    reg = _registry(request)
    if not reg.exists(object_id):
        raise HTTPException(status_code=404, detail=f"no object with id {object_id!r}")
    return SampleStore(reg.object_dir(object_id), object_id)


def _targets(request: Request) -> CoverageTargets:
    return CoverageTargets.from_config(request.app.state.config.objects.coverage_targets)


def _sync_counts(request: Request, object_id: str) -> None:
    store = SampleStore(_registry(request).object_dir(object_id), object_id)
    c = store.counts()
    _registry(request).set_counts(
        object_id,
        sample_count=c["sample_count"],
        positive_count=c["positive_count"],
        negative_count=c["negative_count"],
    )


def _coverage_for(request: Request, object_id: str) -> dict[str, Any]:
    store = SampleStore(_registry(request).object_dir(object_id), object_id)
    return coverage_summary(store.list(), _targets(request))


# -- object routes ------------------------------------------------


@router.get("/api/objects/vocab")
async def get_vocab() -> dict[str, Any]:
    return {
        "conditions": {k: list(v) for k, v in CONDITION_VOCAB.items()},
        "default_conditions": default_conditions(),
        "roles": list(ROLES),
        "kinds": list(KINDS),
        "statuses": list(STATUSES),
    }


@router.get("/api/objects")
async def list_objects(request: Request) -> dict[str, Any]:
    try:
        reg = _registry(request)
        profiles = reg.list()
    except ObjectStoreError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    rows = []
    for p in profiles:
        cov = _coverage_for(request, p.object_id)
        rows.append(
            {
                **p.to_dict(),
                "coverage": {
                    "total_positives": cov["total_positives"],
                    "total_hard_negatives": cov["total_hard_negatives"],
                    "distinct_views": cov["distinct_views"],
                    "meets_all": cov["meets_all"],
                    "guidance": cov["guidance"][:3],
                },
            }
        )
    return {"objects": rows, "count": len(rows)}


@router.post("/api/objects", status_code=201)
async def create_object(request: Request, body: CreateObjectBody) -> dict[str, Any]:
    if body.kind not in KINDS:
        raise HTTPException(status_code=422, detail=f"kind must be one of {list(KINDS)}")
    try:
        prof = _registry(request).create(
            body.name,
            kind=body.kind,
            parent_class=body.parent_class,
            category=body.category,
            description=body.description,
            confusable_with=body.confusable_with,
        )
    except ObjectStoreError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return prof.to_dict()


@router.get("/api/objects/{object_id}")
async def get_object(request: Request, object_id: str) -> dict[str, Any]:
    try:
        prof = _registry(request).get(object_id)
    except ObjectStoreError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    store = SampleStore(_registry(request).object_dir(object_id), object_id)
    return {
        **prof.to_dict(),
        "samples": store.list(),
        "coverage": _coverage_for(request, object_id),
    }


@router.patch("/api/objects/{object_id}")
async def patch_object(request: Request, object_id: str, body: PatchObjectBody) -> dict[str, Any]:
    fields = {k: v for k, v in body.model_dump(exclude_unset=True).items()}
    try:
        prof = _registry(request).update(object_id, **fields)
    except ObjectStoreError as exc:
        code = 404 if "no object" in str(exc) else 400
        raise HTTPException(status_code=code, detail=str(exc)) from exc
    return prof.to_dict()


@router.delete("/api/objects/{object_id}")
async def delete_object(request: Request, object_id: str) -> dict[str, Any]:
    try:
        return _registry(request).soft_delete(object_id)
    except ObjectStoreError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# -- sample routes ------------------------------------------------


@router.get("/api/objects/{object_id}/samples")
async def list_samples(request: Request, object_id: str) -> dict[str, Any]:
    store = _sample_store(request, object_id)
    return {"object_id": object_id, "samples": store.list()}


@router.post("/api/objects/{object_id}/samples", status_code=201)
async def add_sample(
    request: Request,
    object_id: str,
    image: UploadFile = File(...),
    box: str = Form(...),
    conditions: str = Form("{}"),
    role: str = Form("positive"),
    negative_for: str = Form("[]"),
    source: str = Form("camera"),
    device_label: str = Form(""),
    original_filename: str = Form(""),
    consent_ack: str = Form("false"),
    capture_path: str = Form(""),
    requested_resolution: str = Form(""),
    encoded_quality: str = Form(""),
) -> dict[str, Any]:
    cfg = request.app.state.config.objects
    store = _sample_store(request, object_id)

    max_bytes = int(cfg.max_image_mb * 1024 * 1024)
    data = await image.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="empty image upload")
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"image exceeds {cfg.max_image_mb} MB limit (objects.max_image_mb)",
        )

    img = decode_image_bgr(data)
    if img is None:
        raise HTTPException(status_code=400, detail="upload is not a readable image")
    height, width = int(img.shape[0]), int(img.shape[1])

    try:
        box_val = json.loads(box)
        conditions_val = json.loads(conditions) if conditions else {}
        negative_for_val = json.loads(negative_for) if negative_for else []
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"bad JSON form field: {exc}") from exc
    if not isinstance(box_val, (list, tuple)) or len(box_val) != 4:
        raise HTTPException(status_code=400, detail="box must be a JSON array [x, y, w, h]")

    quality = compute_sample_quality(
        img,
        tuple(float(v) for v in box_val),
        existing_hashes=store.existing_hashes(),
        blur_var_min=cfg.blur_var_min,
        min_box_area_frac=cfg.min_box_area_frac,
        duplicate_hamming_max=cfg.duplicate_hamming_max,
    )

    # Store the FULL-resolution original. A JPEG upload is kept verbatim so a
    # Studio capture is not re-compressed (generation loss); any other format is
    # normalised to JPEG q92 once. The thumbnail is always a separate, smaller
    # file and never stands in for the original (BLOCK 3.23).
    is_jpeg = data[:3] == b"\xff\xd8\xff"
    try:
        if is_jpeg:
            norm_jpeg = data
            stored_quality = _parse_float(encoded_quality, 0.92)
        else:
            norm_jpeg = encode_jpeg(img, quality=92)
            stored_quality = 0.92
        thumb = thumbnail_jpeg(img, cfg.thumbnail_px)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"could not encode image: {exc}") from exc
    achieved_resolution = f"{width}x{height}"

    src = "upload" if source not in ("camera", "upload") else source
    try:
        sample = store.add(
            image_bytes=norm_jpeg,
            width=width,
            height=height,
            box=box_val,
            conditions=conditions_val,
            role=role,
            negative_for=[str(o) for o in negative_for_val],
            quality=quality.to_dict(),
            source=src,
            device_label=device_label or None,
            original_filename=(original_filename or image.filename) if src == "upload" else None,
            consent_ack=str(consent_ack).strip().lower() in {"true", "1", "yes", "on"},
            thumb_bytes=thumb,
            capture_path=(capture_path or ("upload" if src == "upload" else None)) or None,
            requested_resolution=requested_resolution or None,
            achieved_resolution=achieved_resolution,
            encoded_quality=stored_quality,
        )
    except (ObjectStoreError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _sync_counts(request, object_id)
    return {"sample": sample.to_dict(), "coverage": _coverage_for(request, object_id)}


@router.patch("/api/objects/{object_id}/samples/{sample_id}")
async def patch_sample(
    request: Request, object_id: str, sample_id: str, body: PatchSampleBody
) -> dict[str, Any]:
    store = _sample_store(request, object_id)
    fields = body.model_dump(exclude_unset=True)
    try:
        updated = store.update(sample_id, **fields)
    except ObjectStoreError as exc:
        code = 404 if "no sample" in str(exc) else 400
        raise HTTPException(status_code=code, detail=str(exc)) from exc
    _sync_counts(request, object_id)
    return {"sample": updated, "coverage": _coverage_for(request, object_id)}


@router.delete("/api/objects/{object_id}/samples/{sample_id}")
async def delete_sample(request: Request, object_id: str, sample_id: str) -> dict[str, Any]:
    store = _sample_store(request, object_id)
    try:
        result = store.soft_delete(sample_id)
    except ObjectStoreError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    _sync_counts(request, object_id)
    return result


@router.get("/api/objects/{object_id}/coverage")
async def get_coverage(request: Request, object_id: str) -> dict[str, Any]:
    _sample_store(request, object_id)  # 404s for an unknown object
    return _coverage_for(request, object_id)


@router.get("/api/objects/{object_id}/samples/{sample_id}/image")
async def get_sample_image(
    request: Request, object_id: str, sample_id: str, thumb: int = 0
) -> FileResponse:
    store = _sample_store(request, object_id)
    try:
        record = store.get(sample_id)
    except ObjectStoreError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    obj_dir = _registry(request).object_dir(object_id)
    rel = record.get("thumb_path") if (thumb and record.get("thumb_path")) else record["path"]
    path = (obj_dir / rel).resolve()
    if obj_dir.resolve() not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="sample image not on disk")
    return FileResponse(path, media_type="image/jpeg")
