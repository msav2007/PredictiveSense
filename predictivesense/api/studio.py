"""Object Learning Studio page + lifecycle + model-registry read (Phase 4).

| Method | Path | Behaviour |
|---|---|---|
| GET  | ``/studio``               | the Studio page (standalone, not the shell) |
| POST | ``/api/studio/enter``     | stop monitoring, return a prior-state token |
| POST | ``/api/studio/leave``     | restore monitoring from the token |
| GET  | ``/api/studio/status``    | whether the Studio is active |
| GET  | ``/api/models/registry``  | model versions + the active one |

Entering the Studio pauses the analysis loop (no perception, no snapshot) and,
combined with the ingest-socket guard in ``api/ingest.py``, guarantees that
nothing keeps detecting and no frames reach ``/ws/ingest`` while the Studio is
open. Leaving restores the previous monitoring state exactly.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from predictivesense.logging_setup import get_logger
from predictivesense.models.registry import ModelRegistry, ModelRegistryError

__all__ = ["router", "studio_is_active"]

_LOG = get_logger(__name__)
_STATIC_DIR = Path(__file__).resolve().parent / "static"

router = APIRouter()


class LeaveBody(BaseModel):
    token: str | None = None


def _studio_state(request: Request) -> dict[str, Any]:
    state = getattr(request.app.state, "studio", None)
    if not isinstance(state, dict):
        state = {"active": False, "token": None, "prior": None}
        request.app.state.studio = state
    return state


def studio_is_active(app: Any) -> bool:
    """True while a Studio session holds the pause. Used by ``api/ingest.py``."""

    state = getattr(app.state, "studio", None)
    return bool(isinstance(state, dict) and state.get("active"))


@router.get("/studio")
async def studio_page() -> FileResponse:
    return FileResponse(_STATIC_DIR / "studio" / "index.html", media_type="text/html")


@router.get("/api/studio/status")
async def studio_status(request: Request) -> dict[str, Any]:
    state = _studio_state(request)
    loop = getattr(request.app.state, "loop", None)
    return {
        "active": bool(state.get("active")),
        "token": state.get("token") if state.get("active") else None,
        "loop_paused": bool(getattr(loop, "paused", False)),
        "stop_monitoring_on_enter": request.app.state.config.studio.stop_monitoring_on_enter,
    }


@router.post("/api/studio/enter")
async def studio_enter(request: Request) -> dict[str, Any]:
    state = _studio_state(request)
    config = request.app.state.config
    loop = getattr(request.app.state, "loop", None)

    if state.get("active"):
        # A page reload re-enters; hand back the same token rather than erroring.
        return {
            "token": state["token"],
            "already_active": True,
            "monitoring_stopped": state.get("prior", {}).get("stopped", {}),
        }

    # Phase 7: clear staging directories older than the configured TTL so an
    # abandoned bulk-upload batch never lingers (BLOCK 3.16).
    try:
        from predictivesense.api.object_batches import cleanup_stale_batches

        removed = cleanup_stale_batches(request.app)
        if removed:
            _LOG.info("studio entry: removed %d stale upload batch(es)", removed)
    except Exception as exc:  # noqa: BLE001 - cleanup must never block entry
        _LOG.warning("studio entry: stale-batch cleanup failed: %r", exc)

    paused_by_us = False
    if config.studio.stop_monitoring_on_enter and loop is not None and not loop.paused:
        loop.pause()
        paused_by_us = True

    token = uuid.uuid4().hex
    prior = {
        "paused_by_us": paused_by_us,
        "loop_was_paused": bool(getattr(loop, "paused", False)) and not paused_by_us,
        "perception_present": getattr(loop, "perception", None) is not None,
        "mode": config.mode.value,
        "stopped": {
            "analysis_loop": "paused" if paused_by_us or getattr(loop, "paused", False) else "n/a",
            "perception": "disabled (loop paused)",
            "ingest_socket": "closed to new frames while Studio active",
            "analysis_worker": "terminated by the client on entering the Studio",
        },
    }
    state.update(active=True, token=token, prior=prior)
    _LOG.info("studio entered: token=%s monitoring paused=%s", token[:8], paused_by_us)
    return {"token": token, "already_active": False, "monitoring_stopped": prior["stopped"]}


@router.post("/api/studio/leave")
async def studio_leave(request: Request, body: LeaveBody | None = None) -> dict[str, Any]:
    state = _studio_state(request)
    loop = getattr(request.app.state, "loop", None)

    if not state.get("active"):
        return {"restored": True, "was_active": False}

    token = (body.token if body else None)
    if token is not None and token != state.get("token"):
        raise HTTPException(status_code=409, detail="studio token mismatch")

    prior = state.get("prior") or {}
    if prior.get("paused_by_us") and loop is not None and loop.paused:
        loop.resume()

    state.update(active=False, token=None, prior=None)
    _LOG.info("studio left: monitoring restored (paused_by_us=%s)", prior.get("paused_by_us"))
    return {
        "restored": True,
        "was_active": True,
        "prior": prior,
        "loop_paused": bool(getattr(loop, "paused", False)),
    }


@router.get("/api/models/registry")
async def models_registry(request: Request) -> dict[str, Any]:
    try:
        reg = ModelRegistry.load()
    except ModelRegistryError as exc:
        _LOG.warning("model registry unavailable: %s", exc)
        return {"available": False, "reason": str(exc)}
    return {"available": True, **reg.to_dict()}
