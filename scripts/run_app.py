"""Start the FastAPI application for a named profile.

    python scripts\\run_app.py --profile dev [--host 127.0.0.1] [--port 8000]
    python scripts\\run_app.py --profile dev --source-kind browser   # Phase 1 input layer
    python scripts\\run_app.py --profile eval --source-kind device   # backend-owned camera

Exits non-zero on any configuration error (including an unopenable device when
``capture.owner == "backend"`` - the loop start raises and uvicorn exits).
"""

from __future__ import annotations

import argparse
import sys

from predictivesense.config.settings import (
    AppConfig,
    ConfigError,
    ValidationError,
    load_config,
)
from predictivesense.core.enums import SourceKind
from predictivesense.logging_setup import configure_logging, get_logger

_LOG = get_logger("predictivesense.scripts.run_app")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PredictiveSense API.")
    parser.add_argument("--profile", required=True, help="config profile name (dev, eval)")
    parser.add_argument("--host", default=None, help="override api.host")
    parser.add_argument("--port", type=int, default=None, help="override api.port")
    parser.add_argument(
        "--source-kind",
        choices=("synthetic", "browser", "device"),
        default=None,
        help="override source.kind (browser/device select the Phase 1 input layer)",
    )
    return parser.parse_args(argv)


def _apply_overrides(
    config: AppConfig,
    host: str | None,
    port: int | None,
    source_kind: str | None,
) -> AppConfig:
    updates: dict[str, object] = {}
    if host is not None:
        updates["host"] = host
    if port is not None:
        updates["port"] = port
    new_config = config
    if updates:
        new_config = new_config.model_copy(
            update={"api": new_config.api.model_copy(update=updates)}
        )
    if source_kind is not None:
        new_config = new_config.model_copy(
            update={
                "source": new_config.source.model_copy(
                    update={"kind": SourceKind(source_kind)}
                )
            }
        )
    return new_config


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config = _apply_overrides(
            load_config(args.profile), args.host, args.port, args.source_kind
        )
    except (ConfigError, ValidationError) as exc:
        configure_logging("ERROR")
        _LOG.error("configuration error: %s", exc)
        return 2

    configure_logging(config.logging.level)
    _LOG.info(
        "starting api profile=%s mode=%s source=%s owner=%s http=http://%s:%d",
        config.profile,
        config.mode.value,
        config.source.kind.value,
        config.capture.owner,
        config.api.host,
        config.api.port,
    )

    import uvicorn

    from predictivesense.api.app import create_app

    uvicorn.run(
        create_app(config, write_manifest=True),
        host=config.api.host,
        port=config.api.port,
        log_level=config.logging.level.lower(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
