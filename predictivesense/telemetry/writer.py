"""Append-only CSV metrics writer.

A write failure is logged exactly once and does not stop the run.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from predictivesense.logging_setup import get_logger

__all__ = ["MetricsWriter"]

_LOG = get_logger(__name__)


class MetricsWriter:
    """Writes one header row then one row per :meth:`write_row` call."""

    def __init__(self, path: str | Path, fieldnames: Iterable[str]) -> None:
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self._failed = False
        self._warned = False
        self._rows_written = 0

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._fh, fieldnames=self.fieldnames, extrasaction="ignore"
        )
        self._writer.writeheader()
        self._fh.flush()

    @property
    def rows_written(self) -> int:
        return self._rows_written

    @property
    def failed(self) -> bool:
        return self._failed

    def write_row(self, row: Mapping[str, Any]) -> None:
        if self._failed:
            return
        try:
            self._writer.writerow(dict(row))
            self._fh.flush()
            self._rows_written += 1
        except (OSError, ValueError) as exc:
            self._failed = True
            if not self._warned:
                self._warned = True
                _LOG.error("metrics write failed for %s: %s (run continues)", self.path, exc)

    def close(self) -> None:
        try:
            self._fh.close()
        except OSError as exc:  # pragma: no cover - close rarely fails
            _LOG.error("closing metrics file %s failed: %s", self.path, exc)
        _LOG.info("metrics csv written: %s (%d rows)", self.path, self._rows_written)

    def __enter__(self) -> "MetricsWriter":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
