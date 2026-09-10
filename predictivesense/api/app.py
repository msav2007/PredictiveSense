"""FastAPI application factory and routes.

Phase 0 endpoints (unchanged shapes):
    GET /health       liveness + uptime + mode + version
    GET /api/config   the fully resolved configuration
    WS  /ws/state     last-value-wins stream of StateSnapshot JSON
    GET /             the dashboard page

Phase 1 additions:
    GET  /api/cameras        backend camera enumeration
    WS   /ws/ingest          binary analysis frames from the browser Web Worker
    GET  /api/videos         recorded video files under data/videos/
    POST /api/analyze        run the RecordedDriver over one file
    POST /api/record/upload  store a raw clip + ClipManifest under data/raw/
    GET  /api/clips          list clip manifests
    POST /api/debug/stall    stall the analysis consumer (preview-independence)
    /static/*                dashboard JS/CSS

The app binds a local listening socket and makes no outbound connections.
"""

from __future__ import annotations

import json
import mimetypes
import re
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from predictivesense import __version__
from predictivesense.api import ingest as ingest_router
from predictivesense.api import labels as labels_router
from predictivesense.api import object_batches as object_batches_router
from predictivesense.api import objects as objects_router
from predictivesense.api import recorder as recorder_router
from predictivesense.api import studio as studio_router
from predictivesense.api import videos as videos_router
from predictivesense.api.broadcast import Broadcaster, serve_state_client
from predictivesense.camera.browser import BrowserSource
from predictivesense.camera.enumerate import as_api_rows, enumerate_devices
from predictivesense.config.settings import AppConfig
from predictivesense.logging_setup import get_logger
from predictivesense.pipeline.loop import AnalysisLoop, build_loop

__all__ = ["create_app"]

_LOG = get_logger(__name__)
_STATIC_DIR = Path(__file__).resolve().parent / "static"
_LABEL_RE = re.compile(r"[^A-Za-z0-9_-]")

# The dashboard loads ES modules from static/ui/ and static/groups/. Some Windows
# registries map .js to text/plain, which the browser refuses for `type=module`.
# Pin the correct type before StaticFiles resolves it. Phase 1.6.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")


class BrowserMetricsIn(BaseModel):
    """Body of ``POST /api/metrics/browser``: a labelled measurement block."""

    label: str = Field(default="browser", max_length=64)
    sample: dict[str, Any]


def _available_ort_providers() -> list[str]:
    """ONNX Runtime EPs present in this build, or ``[]`` when ORT is absent.
    Delegates to ``perception.runtime`` so ``onnxruntime`` stays scoped there."""

    try:
        from predictivesense.perception.runtime import available_execution_providers

        return available_execution_providers()
    except Exception:  # noqa: BLE001 - ORT not installed (bare `pip install -e .`)
        return []


def _ort_version() -> str:
    try:
        from predictivesense.perception.runtime import onnxruntime_version

        return onnxruntime_version()
    except Exception:  # noqa: BLE001
        return "absent"


def _machine_fingerprint() -> dict[str, object]:
    """CPU / cores / RAM / OS / Python / ORT - so results from two laptops are
    never conflated (Phase 8 section 14). No secrets, no user data."""

    import platform

    import psutil

    return {
        "cpu": platform.processor() or platform.machine(),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "os": platform.platform(),
        "python": platform.python_version(),
        "onnxruntime": _ort_version(),
    }


def create_app(
    config: AppConfig,
    *,
    loop: AnalysisLoop | None = None,
    start_loop: bool = True,
    write_manifest: bool = False,
) -> FastAPI:
    """Build the application for ``config``.

    If ``loop`` is not supplied one is built from the config. When ``start_loop``
    is true the loop runs for the lifetime of the app. When ``write_manifest`` is
    true a session manifest (with the ingest clock-offset RTT) is written to
    ``results/`` on shutdown.
    """

    analysis_loop = loop or build_loop(config)
    broadcaster = Broadcaster()
    analysis_loop.add_snapshot_listener(broadcaster.publish)
    started_mono = time.monotonic()

    browser_source = (
        analysis_loop.source
        if isinstance(analysis_loop.source, BrowserSource)
        else None
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if start_loop:
            analysis_loop.start()
        try:
            yield
        finally:
            if start_loop:
                analysis_loop.stop()
                if analysis_loop.error is not None:
                    _LOG.error("analysis loop ended with error: %r", analysis_loop.error)
            if write_manifest:
                _write_session_manifest(config, _app)

    app = FastAPI(
        title="PredictiveSense",
        version=__version__,
        summary="Phase 1 input layer - camera ownership, browser-worker ingest, recorded video, clip recorder.",
        lifespan=lifespan,
    )
    app.state.config = config
    app.state.loop = analysis_loop
    app.state.broadcaster = broadcaster
    app.state.browser_source = browser_source
    # Shared by POST /api/analyze so recorded runs use the identical perception
    # code and do not re-create ONNX sessions per request. None when perception is
    # disabled or its weights are absent (then recorded output is the Phase 1.6 shape).
    app.state.perception = analysis_loop.perception
    # The recognition policy (Phase 2.5) - shared with POST /api/analyze so
    # recorded runs apply the identical policy code.
    app.state.policy = analysis_loop.policy
    app.state.session_id = uuid.uuid4().hex
    # Phase 4: Object Learning Studio lifecycle. While active, the analysis loop
    # is paused and api/ingest.py refuses new frames.
    app.state.studio = {"active": False, "token": None, "prior": None}
    app.state.ingest_stats = {
        "frames": 0.0,
        "malformed": 0.0,
        "clients": 0.0,
        "clock_offset_s": 0.0,
        "clock_offset_rtt_ms": 0.0,
    }

    app.include_router(ingest_router.router)
    app.include_router(recorder_router.router)
    app.include_router(videos_router.router)
    app.include_router(labels_router.router)
    app.include_router(objects_router.router)
    app.include_router(object_batches_router.router)
    app.include_router(studio_router.router)
    if _STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "uptime_s": time.monotonic() - started_mono,
            "mode": config.mode.value,
            "version": __version__,
        }

    @app.get("/api/config")
    async def api_config() -> dict[str, object]:
        return config.as_json_dict()

    @app.get("/api/runtime")
    async def api_runtime() -> dict[str, object]:
        """Phase 8: what is *actually* running now - the ONNX Runtime execution
        provider in use (from the live session, not from config), the active
        model-registry version, the scheduling / staleness / pose-cadence policy,
        and a machine fingerprint. Powers the Diagnostics 'active EP' row and is
        the honest answer to 'which provider is live'."""

        loop = analysis_loop
        perc = loop.perception.info() if loop.perception is not None else {}
        return {
            "requested_provider": config.perception.provider,
            "active_provider": perc.get("detector_ep") or loop.active_provider,
            "provider_resolved": perc.get("provider_resolved"),
            "provider_reason": perc.get("provider_reason"),
            "available_providers": _available_ort_providers(),
            "model_version": loop.model_version,
            "scheduler": config.consumer.scheduler,
            "max_frame_age_ms": config.analysis.max_frame_age_ms,
            "pose_cadence": config.perception.pose_cadence,
            "pose_cadence_resolved": perc.get("pose_cadence_resolved"),
            "machine": _machine_fingerprint(),
        }

    @app.get("/api/cameras")
    async def api_cameras() -> list[dict[str, object]]:
        return as_api_rows(enumerate_devices(backend_hint=config.capture.device_backend))

    @app.post("/api/debug/stall")
    async def api_debug_stall(seconds: float = 3.0) -> dict[str, object]:
        applied = analysis_loop.request_consumer_stall(seconds)
        return {"stalled_seconds": applied}

    @app.post("/api/metrics/browser")
    async def api_metrics_browser(body: BrowserMetricsIn) -> dict[str, object]:
        """Append one labelled browser-measured sample block to results/.

        The page pushes ``track.getSettings()`` (requested vs achieved),
        preview / analysis FPS, worker encode ms, WS bufferedAmount, backend
        decode ms, frame-age p50/p95, drop rate and switch times. Written to
        ``results/browser_metrics_<label>.json`` as a JSON array.
        """

        slug = _LABEL_RE.sub("-", body.label)[:64] or "browser"
        out = Path(config.results_dir) / f"browser_metrics_{slug}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        existing: list[Any] = []
        if out.is_file():
            try:
                loaded = json.loads(out.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    existing = loaded
            except ValueError:
                existing = []
        existing.append(
            {
                "received_utc": datetime.now(timezone.utc).isoformat(),
                "sample": body.sample,
            }
        )
        out.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        _LOG.info("browser metrics sample #%d -> %s", len(existing), out)
        return {"ok": True, "path": str(out), "samples": len(existing)}

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html", media_type="text/html")

    @app.get("/label")
    async def label_page() -> FileResponse:
        """The Phase 2.5 labelling tool (linked from the Research group)."""

        return FileResponse(_STATIC_DIR / "label" / "index.html", media_type="text/html")

    @app.websocket("/ws/state")
    async def ws_state(websocket: WebSocket) -> None:
        await websocket.accept()
        await serve_state_client(
            websocket,
            broadcaster,
            rate_hz=config.broadcast.rate_hz,
            send_timeout_s=config.broadcast.client_send_timeout_s,
            metrics=analysis_loop.metrics,
        )

    return app


def _write_session_manifest(config: AppConfig, app: FastAPI) -> None:
    """Write a run manifest carrying the ingest clock-offset RTT. Non-fatal."""

    try:
        from predictivesense.telemetry.manifest import build_manifest

        stats = dict(app.state.ingest_stats)
        source = app.state.browser_source
        if source is not None:
            stats.update(source.info())
        loop = app.state.loop
        perc = loop.perception.info() if loop.perception is not None else {}
        manifest = build_manifest(
            config,
            session_id=app.state.session_id,
            extra={
                "kind": "app",
                "ingest": stats,
                # Phase 8: what actually ran, not what config asked for.
                "perception": {
                    "requested_provider": config.perception.provider,
                    "active_provider": perc.get("detector_ep") or loop.active_provider,
                    "provider_resolved": perc.get("provider_resolved"),
                    "model_version": loop.model_version,
                    "intra_op_threads": config.perception.intra_op_threads,
                    "detector_input_size": config.perception.detector.input_size,
                    "pose_cadence": config.perception.pose_cadence,
                },
                "scheduler": config.consumer.scheduler,
                "max_frame_age_ms": config.analysis.max_frame_age_ms,
                "machine": _machine_fingerprint(),
                "available_providers": _available_ort_providers(),
            },
        )
        manifest.finalize()
        out = Path(config.results_dir) / f"session_{app.state.session_id}.json"
        manifest.write(out)
    except Exception as exc:  # noqa: BLE001 - a manifest failure must not crash shutdown
        _LOG.warning("session manifest not written: %r", exc)
