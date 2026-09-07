"""FastAPI application factory and routes.

Endpoints:
    GET /health       liveness + uptime + mode + version
    GET /api/config   the fully resolved configuration
    WS  /ws/state     last-value-wins stream of StateSnapshot JSON
    GET /             a minimal static snapshot viewer

The app binds a local listening socket. It makes no outbound connections.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse

from predictivesense import __version__
from predictivesense.api.broadcast import Broadcaster, serve_state_client
from predictivesense.config.settings import AppConfig
from predictivesense.logging_setup import get_logger
from predictivesense.pipeline.loop import AnalysisLoop, build_loop

__all__ = ["create_app"]

_LOG = get_logger(__name__)
_STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(
    config: AppConfig,
    *,
    loop: AnalysisLoop | None = None,
    start_loop: bool = True,
) -> FastAPI:
    """Build the application for ``config``.

    If ``loop`` is not supplied one is built from the config. When ``start_loop``
    is true the loop runs for the lifetime of the app.
    """

    analysis_loop = loop or build_loop(config)
    broadcaster = Broadcaster()
    analysis_loop.add_snapshot_listener(broadcaster.publish)
    started_mono = time.monotonic()

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

    app = FastAPI(
        title="PredictiveSense",
        version=__version__,
        summary="Phase 0 foundation - no perception, tracking, risk, or audio.",
        lifespan=lifespan,
    )
    app.state.config = config
    app.state.loop = analysis_loop
    app.state.broadcaster = broadcaster

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

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html", media_type="text/html")

    @app.websocket("/ws/state")
    async def ws_state(websocket: WebSocket) -> None:
        await websocket.accept()
        await serve_state_client(
            websocket,
            broadcaster,
            rate_hz=config.broadcast.rate_hz,
            send_timeout_s=config.broadcast.client_send_timeout_s,
        )

    return app
