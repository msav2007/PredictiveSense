"""Backend camera enumeration for headless use.

The browser enumerates devices itself with ``navigator.mediaDevices``. This
module is the independent backend view used by ``GET /api/cameras`` and the
transport benchmark: probe OpenCV indices ``0..max_index`` and, when
``pygrabber`` imports and works, attach human-readable DirectShow names.

``pygrabber`` is optional and Windows-only. A missing or throwing ``pygrabber``
is logged exactly once at INFO and degrades to ``"Camera <index>"`` - it is
never fatal.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from predictivesense.core.enums import SourceKind
from predictivesense.core.types import SourceInfo
from predictivesense.logging_setup import get_logger

__all__ = ["enumerate_devices", "as_api_rows", "DeviceProbe", "NameResolver"]

_LOG = get_logger(__name__)
_DEFAULT_MAX_INDEX = 9
_warned_no_pygrabber = False

# (index, backend_hint) -> (available, backend_that_opened_or_None)
DeviceProbe = Callable[[int, str], tuple[bool, str | None]]
NameResolver = Callable[[], Sequence[str]]


def _default_name_resolver() -> Sequence[str]:
    """DirectShow input-device names by index, or ``[]`` if pygrabber is absent."""

    global _warned_no_pygrabber
    try:
        from pygrabber.dshow_graph import FilterGraph  # type: ignore import-not-found

        return list(FilterGraph().get_input_devices())
    except Exception as exc:  # noqa: BLE001 - optional dep, never fatal
        if not _warned_no_pygrabber:
            _warned_no_pygrabber = True
            _LOG.info(
                "pygrabber unavailable (%s); camera names fall back to 'Camera <index>'",
                exc,
            )
        return []


def _default_probe(index: int, backend_hint: str) -> tuple[bool, str | None]:
    """Open ``index`` briefly with OpenCV; report whether it opened and how.

    No frame is grabbed, so a camera indicator light is not turned on.
    """

    import cv2

    from predictivesense.camera._opencv import quiet_opencv_logging

    quiet_opencv_logging()

    order: list[tuple[str, int]]
    if backend_hint == "msmf":
        order = [("msmf", cv2.CAP_MSMF)]
    elif backend_hint == "dshow":
        order = [("dshow", cv2.CAP_DSHOW)]
    else:
        order = [("msmf", cv2.CAP_MSMF), ("dshow", cv2.CAP_DSHOW)]

    for name, flag in order:
        cap = cv2.VideoCapture(index, flag)
        try:
            if cap.isOpened():
                return True, name
        finally:
            cap.release()
    return False, None


def enumerate_devices(
    *,
    max_index: int = _DEFAULT_MAX_INDEX,
    backend_hint: str = "auto",
    probe: DeviceProbe | None = None,
    name_resolver: NameResolver | None = None,
) -> list[SourceInfo]:
    """Probe ``0..max_index`` and return one :class:`SourceInfo` per index.

    ``probe`` and ``name_resolver`` are injectable for tests; the defaults touch
    OpenCV and (optionally) ``pygrabber``.
    """

    probe = probe or _default_probe
    name_resolver = name_resolver or _default_name_resolver

    try:
        names = list(name_resolver())
    except Exception as exc:  # noqa: BLE001 - resolver must never break enumeration
        _LOG.info("camera name resolver failed (%s); using fallback names", exc)
        names = []

    found: list[SourceInfo] = []
    for index in range(max_index + 1):
        try:
            available, backend = probe(index, backend_hint)
        except Exception as exc:  # noqa: BLE001 - one bad index must not abort the sweep
            _LOG.info("probe of camera index %d failed (%s)", index, exc)
            available, backend = False, None

        label = names[index] if index < len(names) and names[index] else f"Camera {index}"
        found.append(
            SourceInfo(
                kind=SourceKind.DEVICE,
                source_id=f"device:{index}",
                label=label,
                width=0,
                height=0,
                achieved_fps=None,
                backend=backend,
                extra={"index": str(index), "available": str(available).lower()},
            )
        )
    return found


def as_api_rows(infos: list[SourceInfo]) -> list[dict[str, Any]]:
    """Shape :func:`enumerate_devices` output as ``GET /api/cameras`` rows."""

    rows: list[dict[str, Any]] = []
    for info in infos:
        rows.append(
            {
                "index": int(info.extra.get("index", "-1")),
                "name": info.label,
                "available": info.extra.get("available", "false") == "true",
                "backend": info.backend,
            }
        )
    return rows
