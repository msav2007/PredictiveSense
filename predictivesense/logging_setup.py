"""Structured logging setup.

Library modules call :func:`get_logger`; entrypoints call :func:`configure_logging`
once at startup. No module in this package uses ``print()``.
"""

from __future__ import annotations

import logging
import sys

__all__ = ["configure_logging", "get_logger"]

_CONFIGURED = False
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S%z"


def configure_logging(level: int | str = logging.INFO) -> None:
    """Attach a single stdout handler to the root logger. Idempotent."""

    global _CONFIGURED
    root = logging.getLogger()

    if isinstance(level, str):
        resolved = logging.getLevelName(level.upper())
        level = resolved if isinstance(resolved, int) else logging.INFO

    root.setLevel(level)

    if _CONFIGURED:
        for handler in root.handlers:
            handler.setLevel(level)
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=_FORMAT, datefmt=_DATEFMT))
    handler.setLevel(level)
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Does not configure handlers."""

    return logging.getLogger(name)
