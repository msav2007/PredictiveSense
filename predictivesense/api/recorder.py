"""Raw clip recorder: ``POST /api/record/upload`` + ``GET /api/clips``.

The browser records the *full-quality* preview stream with ``MediaRecorder``
(not the downscaled analysis frames) and POSTs the blob here on stop. The clip
lands at ``data/raw/<session_id>/<clip_id>.webm`` with a sibling
``<clip_id>.json`` :class:`ClipManifest`. ``data/`` is git-ignored; clips are
never committed.

Uploads above ``recorder.max_clip_mb`` are rejected with 413. An unwritable
output directory fails the request loudly with 500.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from predictivesense.camera.file_source import video_duration_s
from predictivesense.core.types import ClipManifest
from predictivesense.logging_setup import get_logger
from predictivesense.telemetry.manifest import git_state, utc_now_iso

__all__ = ["router"]

_LOG = get_logger(__name__)
_CHUNK = 1 << 20  # 1 MiB
_TRUE = {"true", "1", "yes", "on"}

router = APIRouter()


def _output_dir(request: Request) -> Path:
    return Path(request.app.state.config.recorder.output_dir)


def _probe_duration_s(path: Path) -> float | None:
    """Best-effort clip duration; delegates to the camera layer (cv2 lives there)."""

    try:
        return video_duration_s(path)
    except Exception as exc:  # noqa: BLE001 - duration is optional metadata
        _LOG.debug("clip duration probe failed for %s: %r", path, exc)
        return None


@router.post("/api/record/upload")
async def upload_clip(
    request: Request,
    file: UploadFile = File(...),
    scenario_tag: str = Form(...),
    device_label: str = Form(...),
    width: int = Form(...),
    height: int = Form(...),
    nominal_fps: float = Form(...),
    notes: str = Form(""),
    consent_ack: str = Form("false"),
) -> dict[str, object]:
    config = request.app.state.config
    if not config.recorder.enabled:
        raise HTTPException(status_code=403, detail="clip recorder is disabled for this profile")

    session_id = request.app.state.session_id
    max_bytes = int(config.recorder.max_clip_mb * 1024 * 1024)
    clip_id = uuid.uuid4().hex

    out_dir = _output_dir(request) / session_id
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _LOG.error("clip output dir %s is not writable: %s", out_dir, exc)
        raise HTTPException(status_code=500, detail=f"output dir not writable: {exc}") from exc

    clip_path = out_dir / f"{clip_id}.webm"
    size = 0
    try:
        with clip_path.open("wb") as fh:
            while True:
                chunk = await file.read(_CHUNK)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    fh.close()
                    clip_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"clip exceeds {config.recorder.max_clip_mb} MB limit "
                            f"(recorder.max_clip_mb)"
                        ),
                    )
                fh.write(chunk)
    except HTTPException:
        raise
    except OSError as exc:
        clip_path.unlink(missing_ok=True)
        _LOG.error("writing clip %s failed: %s", clip_path, exc)
        raise HTTPException(status_code=500, detail=f"could not write clip: {exc}") from exc

    commit, _dirty = git_state()
    manifest = ClipManifest(
        clip_id=clip_id,
        session_id=session_id,
        path=str(clip_path.resolve()),
        scenario_tag=scenario_tag,
        device_label=device_label,
        width=int(width),
        height=int(height),
        nominal_fps=float(nominal_fps),
        duration_s=_probe_duration_s(clip_path),
        size_bytes=size,
        recorded_utc=utc_now_iso(),
        git_commit=commit,
        config_profile=config.profile,
        consent_ack=str(consent_ack).strip().lower() in _TRUE,
        notes=notes,
    )
    (out_dir / f"{clip_id}.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    _LOG.info(
        "clip stored: %s (%d bytes, tag=%r, consent=%s)",
        clip_path,
        size,
        scenario_tag,
        manifest.consent_ack,
    )
    return manifest.model_dump(mode="json")


@router.get("/api/clips")
async def list_clips(request: Request) -> list[dict[str, object]]:
    base = _output_dir(request)
    if not base.is_dir():
        return []
    clips: list[dict[str, object]] = []
    for meta in sorted(base.rglob("*.json")):
        try:
            clips.append(json.loads(meta.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            _LOG.warning("skipping unreadable clip manifest %s: %s", meta, exc)
    clips.sort(key=lambda c: c.get("recorded_utc", ""), reverse=True)
    return clips
