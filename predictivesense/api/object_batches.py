"""Bulk image upload for the Object Learning Studio (Phase 7).

A data-collection accelerator: upload ~20 images for one object in a single
operation instead of capture -> box -> save, twenty times. The system proposes a
bounding box per image from the **existing** detector's *raw* output (before the
recognition policy - the policy would suppress exactly the box we want; the
bypass is deliberate, see ``docs/decisions.md``), the developer reviews and
corrects them, and **Save all** commits the reviewed samples through the *same*
creation path, quality checks and provenance writing the camera path uses.

| Method | Path | Behaviour |
|---|---|---|
| POST   | ``/api/objects/{id}/batches``                  | multipart: many files + role -> stage + start proposals |
| GET    | ``/api/objects/{id}/batches/{bid}``            | poll: ``{status, processed, total, items}`` |
| PATCH  | ``/api/objects/{id}/batches/{bid}/items/{iid}``| update box / role / conditions / box_confirmed_by_human |
| POST   | ``/api/objects/{id}/batches/{bid}/save``       | commit valid items -> ``{saved, remaining, skipped}`` |
| DELETE | ``/api/objects/{id}/batches/{bid}``            | discard the batch + its staging |
| GET    | ``/api/objects/{id}/batches/{bid}/items/{iid}/image`` | staged image (``?thumb=1``) |

No model is trained, fine-tuned or activated. Storing images is not training.
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from predictivesense.api.objects import persist_sample
from predictivesense.camera._opencv import decode_image_bgr, encode_jpeg, thumbnail_jpeg
from predictivesense.core.types import Frame
from predictivesense.logging_setup import get_logger
from predictivesense.objects.batches import BatchItem, BatchStore, BatchStoreError
from predictivesense.objects.proposals import centred_default_box, propose_box
from predictivesense.objects.quality import compute_sample_quality
from predictivesense.objects.registry import ObjectRegistry, ObjectStoreError
from predictivesense.objects.samples import SampleStore
from predictivesense.objects.vocab import validate_conditions, validate_role

__all__ = ["router", "cleanup_stale_batches"]

_LOG = get_logger(__name__)
router = APIRouter()

# One worker: the proposal step reuses the single shared ONNX session, so
# detector calls are serialised (never one session per image, BLOCK 7).
_PROPOSALS = ThreadPoolExecutor(max_workers=1, thread_name_prefix="batch-proposals")
# Coarse guard around every manifest read-modify-write so a background proposal
# update and a concurrent PATCH cannot lose each other.
_MANIFEST_LOCK = threading.Lock()


# -- request models ---------------------------------------------------


class PatchItemBody(BaseModel):
    box: list[float] | None = Field(default=None, min_length=4, max_length=4)
    role: str | None = None
    conditions: dict[str, str] | None = None
    negative_for: list[str] | None = None
    box_confirmed_by_human: bool | None = None


class SaveBody(BaseModel):
    consent_ack: bool = False


# -- store access ---------------------------------------------------


def _registry(request: Request) -> ObjectRegistry:
    return ObjectRegistry(request.app.state.config.objects.root)


def _require_object(request: Request, object_id: str) -> ObjectRegistry:
    reg = _registry(request)
    if not reg.exists(object_id):
        raise HTTPException(status_code=404, detail=f"no object with id {object_id!r}")
    return reg


def _batch_store(request: Request, object_id: str) -> BatchStore:
    reg = _require_object(request, object_id)
    return BatchStore(reg.object_dir(object_id), object_id)


def _sample_store(request: Request, object_id: str) -> SampleStore:
    reg = _require_object(request, object_id)
    return SampleStore(reg.object_dir(object_id), object_id)


# -- box helpers --------------------------------------------------


def _box_is_valid(box: Any, width: int, height: int) -> bool:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    try:
        x, y, w, h = (float(v) for v in box)
    except (TypeError, ValueError):
        return False
    if w <= 0 or h <= 0:
        return False
    return x >= -1 and y >= -1 and (x + w) <= width + 1 and (y + h) <= height + 1


# -- POST: upload + stage ---------------------------------------


@router.post("/api/objects/{object_id}/batches", status_code=201)
async def create_batch(
    request: Request,
    object_id: str,
    files: list[UploadFile] = File(...),
    role: str = Form("positive"),
) -> dict[str, Any]:
    reg = _require_object(request, object_id)
    cfg = request.app.state.config.objects
    bcfg = cfg.batch
    try:
        role = validate_role(role)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not files:
        raise HTTPException(status_code=422, detail="no files uploaded")
    if len(files) > bcfg.max_images:
        raise HTTPException(
            status_code=400,
            detail=f"{len(files)} images exceeds the batch limit of {bcfg.max_images} "
            f"(objects.batch.max_images)",
        )

    per_file_max = int(cfg.max_image_mb * 1024 * 1024)
    total_max = int(bcfg.max_total_mb * 1024 * 1024)

    raw: list[tuple[str, bytes]] = []
    total = 0
    for f in files:
        data = await f.read()
        total += len(data)
        raw.append((f.filename or "upload.jpg", data))
    if total > total_max:
        raise HTTPException(
            status_code=400,
            detail=f"batch total {total / 1e6:.1f} MB exceeds objects.batch.max_total_mb "
            f"({bcfg.max_total_mb} MB)",
        )

    accepted: list[BatchItem] = []
    staged: list[tuple[str, bytes, int, int]] = []  # item_id, jpeg_bytes, w, h
    thumbs: dict[str, bytes] = {}
    rejected: list[dict[str, str]] = []
    for filename, data in raw:
        if len(data) == 0:
            rejected.append({"filename": filename, "reason": "empty file"})
            continue
        if len(data) > per_file_max:
            rejected.append(
                {"filename": filename, "reason": f"exceeds {cfg.max_image_mb} MB per-image limit"}
            )
            continue
        img = decode_image_bgr(data)
        if img is None:
            rejected.append({"filename": filename, "reason": "unsupported or corrupt image"})
            continue
        h, w = int(img.shape[0]), int(img.shape[1])
        if min(w, h) < bcfg.min_image_px:
            rejected.append(
                {"filename": filename, "reason": f"smaller than {bcfg.min_image_px}px on its short side"}
            )
            continue
        item_id = uuid.uuid4().hex
        is_jpeg = data[:3] == b"\xff\xd8\xff"
        try:
            norm = data if is_jpeg else encode_jpeg(img, quality=92)
            thumb = thumbnail_jpeg(img, bcfg.thumbnail_px)
        except ValueError:
            rejected.append({"filename": filename, "reason": "could not re-encode image"})
            continue
        accepted.append(
            BatchItem(
                item_id=item_id,
                filename=filename,
                width=w,
                height=h,
                role=role,
                status="manual_required",
            )
        )
        staged.append((item_id, norm, w, h))
        thumbs[item_id] = thumb

    if not accepted:
        raise HTTPException(
            status_code=400,
            detail={"error": "no usable images in the batch", "rejected": rejected},
        )

    batch_id = uuid.uuid4().hex
    store = BatchStore(reg.object_dir(object_id), object_id)
    with _MANIFEST_LOCK:
        store.create(batch_id, role=role, items=accepted)
        for item_id, norm, w, h in staged:
            store.add_staged_image(
                batch_id,
                item_id,
                image_bytes=norm,
                thumb_bytes=thumbs[item_id],
                width=w,
                height=h,
            )

    _PROPOSALS.submit(_run_proposals, request.app, object_id, batch_id)

    return {
        "batch_id": batch_id,
        "accepted": [it.filename for it in accepted],
        "rejected": rejected,
        "status": "processing",
        "total": len(accepted),
    }


# -- background: propose one box per staged image --------------


def _run_proposals(app: Any, object_id: str, batch_id: str) -> None:
    """Compute a box proposal + quality flags for every staged image. Runs off
    the request thread; one image failing never blocks the batch; never resumes
    the monitoring loop (it only calls the detector)."""

    cfg = app.state.config.objects
    bcfg = cfg.batch
    reg = ObjectRegistry(cfg.root)
    store = BatchStore(reg.object_dir(object_id), object_id)
    sample_store = SampleStore(reg.object_dir(object_id), object_id)
    perception = getattr(app.state, "perception", None)

    committed_hashes = list(sample_store.existing_hashes())
    processed = 0
    try:
        items = store.items(batch_id)
    except BatchStoreError:
        return

    for item in items:
        item_id = item["item_id"]
        w, h = int(item["width"]), int(item["height"])
        fields: dict[str, Any] = {}
        try:
            data = store.staged_image_path(batch_id, item_id).read_bytes()
            img = decode_image_bgr(data)
            if img is None:
                raise ValueError("staged image no longer decodable")

            raw_dets: list = []
            if perception is not None:
                frame = Frame(
                    frame_id=0, capture_ts=0.0, image=img,
                    width=w, height=h, source_id=f"batch:{batch_id}", seq=0,
                )
                raw_dets = perception.detect(frame)

            proposal = propose_box(raw_dets, w, h, min_score=bcfg.proposal_min_score)
            with _MANIFEST_LOCK:
                batch_hashes = store.existing_batch_hashes(batch_id, exclude_item=item_id)
            quality = compute_sample_quality(
                img,
                tuple(float(v) for v in proposal.box),
                existing_hashes=committed_hashes + batch_hashes,
                blur_var_min=cfg.blur_var_min,
                min_box_area_frac=cfg.min_box_area_frac,
                duplicate_hamming_max=cfg.duplicate_hamming_max,
            ).to_dict()

            status = proposal.status
            if status == "ready" and quality.get("flags"):
                status = "flagged"
            fields = {
                "box": proposal.box,
                "proposal_source": proposal.source,
                "proposal_raw_class": proposal.raw_class,
                "proposal_score": proposal.score,
                "quality": quality,
                "status": status,
                "box_confirmed_by_human": False,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 - one image must never block the batch
            _LOG.warning("batch %s item %s proposal failed: %r", batch_id, item_id, exc)
            fields = {
                "box": centred_default_box(w, h),
                "proposal_source": "default_centred",
                "status": "manual_required",
                "box_confirmed_by_human": False,
                "error": f"proposal failed: {exc}",
            }
        processed += 1
        with _MANIFEST_LOCK:
            try:
                store.update_item(batch_id, item_id, **fields)
                store.set_progress(batch_id, processed)
            except BatchStoreError:
                return

    with _MANIFEST_LOCK:
        try:
            store.set_status(batch_id, "ready")
        except BatchStoreError:
            pass
    _LOG.info("batch %s proposals done: %d image(s)", batch_id, processed)


# -- GET: poll --------------------------------------------------


@router.get("/api/objects/{object_id}/batches")
async def list_batches(request: Request, object_id: str) -> dict[str, Any]:
    store = _batch_store(request, object_id)
    out = []
    for bid in store.list_batches():
        try:
            doc = store.load(bid)
        except BatchStoreError:
            continue
        out.append(
            {
                "batch_id": bid,
                "status": doc.get("status"),
                "total": doc.get("total"),
                "processed": doc.get("processed"),
                "role": doc.get("role"),
                "created_utc": doc.get("created_utc"),
            }
        )
    return {"object_id": object_id, "batches": out}


@router.get("/api/objects/{object_id}/batches/{batch_id}")
async def get_batch(request: Request, object_id: str, batch_id: str) -> dict[str, Any]:
    store = _batch_store(request, object_id)
    try:
        doc = store.load(batch_id)
    except BatchStoreError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "batch_id": batch_id,
        "status": doc["status"],
        "processed": doc["processed"],
        "total": doc["total"],
        "role": doc.get("role"),
        "items": doc["items"],
    }


# -- PATCH: review one item -----------------------------------


@router.patch("/api/objects/{object_id}/batches/{batch_id}/items/{item_id}")
async def patch_item(
    request: Request, object_id: str, batch_id: str, item_id: str, body: PatchItemBody
) -> dict[str, Any]:
    store = _batch_store(request, object_id)
    with _MANIFEST_LOCK:
        try:
            doc = store.load(batch_id)
        except BatchStoreError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        item = next((it for it in doc["items"] if it["item_id"] == item_id), None)
        if item is None:
            raise HTTPException(status_code=404, detail=f"no item {item_id!r} in batch {batch_id!r}")

        fields = body.model_dump(exclude_unset=True)
        w, h = int(item["width"]), int(item["height"])
        human_touched = False
        box_cleared = False

        if "box" in fields and fields["box"] is not None:
            if not _box_is_valid(fields["box"], w, h):
                raise HTTPException(
                    status_code=400,
                    detail=f"box {fields['box']} is zero-area or outside the {w}x{h} image",
                )
            fields["box"] = [round(float(v), 2) for v in fields["box"]]
            human_touched = True
        elif "box" in fields and fields["box"] is None:
            # explicit delete-box from the reviewer
            fields["status"] = "manual_required"
            fields["box_confirmed_by_human"] = False
            box_cleared = True
        if "role" in fields and fields["role"] is not None:
            try:
                fields["role"] = validate_role(str(fields["role"]))
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            if fields["role"] in ("negative", "hard_negative") and not (
                fields.get("negative_for") or item.get("negative_for")
            ):
                fields["negative_for"] = [object_id]
        if "conditions" in fields and fields["conditions"] is not None:
            try:
                fields["conditions"] = validate_conditions(fields["conditions"])
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        if "negative_for" in fields and fields["negative_for"] is not None:
            fields["negative_for"] = [str(o) for o in fields["negative_for"]]
        if fields.get("box_confirmed_by_human"):
            human_touched = True

        if human_touched and not box_cleared:
            fields["box_confirmed_by_human"] = True
            if item.get("status") not in ("error",):
                fields["status"] = "edited"

        updated = store.update_item(batch_id, item_id, **fields)
    return {"item": updated}


# -- POST: commit valid items -------------------------------


@router.post("/api/objects/{object_id}/batches/{batch_id}/save")
async def save_batch(
    request: Request, object_id: str, batch_id: str, body: SaveBody | None = None
) -> dict[str, Any]:
    reg = _require_object(request, object_id)
    cfg = request.app.state.config.objects
    store = BatchStore(reg.object_dir(object_id), object_id)
    sample_store = SampleStore(reg.object_dir(object_id), object_id)
    consent = bool(body.consent_ack) if body else False

    with _MANIFEST_LOCK:
        try:
            doc = store.load(batch_id)
        except BatchStoreError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        saved_ids: list[str] = []
        skipped: list[dict[str, str]] = []
        for item in doc["items"]:
            item_id = item["item_id"]
            w, h = int(item["width"]), int(item["height"])
            if not _box_is_valid(item.get("box"), w, h):
                skipped.append({"item_id": item_id, "reason": "no valid bounding box"})
                continue
            try:
                data = store.staged_image_path(batch_id, item_id).read_bytes()
            except (BatchStoreError, OSError) as exc:
                skipped.append({"item_id": item_id, "reason": f"staged image unreadable: {exc}"})
                continue
            role = item.get("role", doc.get("role", "positive"))
            neg_for = item.get("negative_for") or (
                [object_id] if role in ("negative", "hard_negative") else []
            )
            try:
                persist_sample(
                    sample_store,
                    cfg,
                    data=data,
                    box=item["box"],
                    conditions=item.get("conditions") or {},
                    role=role,
                    negative_for=neg_for,
                    source="upload_batch",
                    original_filename=item.get("filename"),
                    consent_ack=consent,
                    capture_path="upload_batch",
                    extra_provenance={
                        "batch_id": batch_id,
                        "proposal_source": item.get("proposal_source") or "manual",
                        "proposal_raw_class": item.get("proposal_raw_class"),
                        "proposal_score": item.get("proposal_score"),
                        "box_confirmed_by_human": bool(item.get("box_confirmed_by_human")),
                    },
                )
            except HTTPException as exc:
                skipped.append({"item_id": item_id, "reason": str(exc.detail)})
                continue
            saved_ids.append(item_id)

        remaining_items = [it for it in doc["items"] if it["item_id"] not in saved_ids]
        if remaining_items:
            doc["items"] = remaining_items
            doc["total"] = len(remaining_items)
            doc["processed"] = min(doc.get("processed", 0), len(remaining_items))
            doc["status"] = "ready"
            store.overwrite(batch_id, doc)
        else:
            store.discard(batch_id)

    if saved_ids:
        c = sample_store.counts()
        reg.set_counts(
            object_id,
            sample_count=c["sample_count"],
            positive_count=c["positive_count"],
            negative_count=c["negative_count"],
        )

    return {
        "saved": len(saved_ids),
        "remaining": len(skipped),
        "skipped": skipped,
        "batch_cleared": not remaining_items,
    }


# -- DELETE: discard ----------------------------------------


@router.delete("/api/objects/{object_id}/batches/{batch_id}")
async def discard_batch(request: Request, object_id: str, batch_id: str) -> dict[str, Any]:
    store = _batch_store(request, object_id)
    if not store.exists(batch_id):
        raise HTTPException(status_code=404, detail=f"no batch {batch_id!r}")
    with _MANIFEST_LOCK:
        return store.discard(batch_id)


# -- GET: staged image ------------------------------------


@router.get("/api/objects/{object_id}/batches/{batch_id}/items/{item_id}/image")
async def get_item_image(
    request: Request, object_id: str, batch_id: str, item_id: str, thumb: int = 0
) -> FileResponse:
    store = _batch_store(request, object_id)
    try:
        path = (
            store.thumbnail_image_path(batch_id, item_id)
            if thumb
            else store.staged_image_path(batch_id, item_id)
        ).resolve()
    except BatchStoreError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    bdir = store.batch_dir(batch_id).resolve()
    if bdir not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="staged image not on disk")
    return FileResponse(path, media_type="image/jpeg")


# -- Studio-entry cleanup ---------------------------------


def cleanup_stale_batches(app: Any) -> int:
    """Drop staging directories older than ``objects.batch.staging_ttl_hours``
    across every object. Called from ``POST /api/studio/enter`` so an abandoned
    batch never lingers. Returns the number of batches removed."""

    try:
        cfg = app.state.config.objects
        reg = ObjectRegistry(cfg.root)
        removed = 0
        for prof in reg.list():
            store = BatchStore(reg.object_dir(prof.object_id), prof.object_id)
            removed += len(store.cleanup_stale(cfg.batch.staging_ttl_hours))
        return removed
    except (ObjectStoreError, BatchStoreError, OSError) as exc:
        _LOG.warning("stale-batch cleanup skipped: %r", exc)
        return 0
