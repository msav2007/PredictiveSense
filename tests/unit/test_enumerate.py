"""Backend camera enumeration: pygrabber names when available, safe fallback."""

from __future__ import annotations

import pytest

from predictivesense.camera import enumerate as enum_mod
from predictivesense.camera.enumerate import as_api_rows, enumerate_devices

pytestmark = pytest.mark.unit


def _probe_two_available(index: int, hint: str) -> tuple[bool, str | None]:
    return (index < 2, "msmf" if index < 2 else None)


def test_names_resolve_when_resolver_succeeds() -> None:
    infos = enumerate_devices(
        max_index=2,
        probe=_probe_two_available,
        name_resolver=lambda: ["Integrated Camera", "OnePlus Nord 4"],
    )
    rows = as_api_rows(infos)
    assert [r["name"] for r in rows] == ["Integrated Camera", "OnePlus Nord 4", "Camera 2"]
    assert [r["available"] for r in rows] == [True, True, False]
    assert [r["index"] for r in rows] == [0, 1, 2]
    assert rows[0]["backend"] == "msmf"


def test_resolver_exception_falls_back_without_propagating() -> None:
    def _boom() -> list[str]:
        raise RuntimeError("pygrabber exploded")

    infos = enumerate_devices(max_index=1, probe=_probe_two_available, name_resolver=_boom)
    assert [i.label for i in infos] == ["Camera 0", "Camera 1"]


def test_probe_exception_on_one_index_does_not_abort_sweep() -> None:
    def _probe(index: int, hint: str) -> tuple[bool, str | None]:
        if index == 1:
            raise OSError("device busy")
        return (True, "dshow")

    infos = enumerate_devices(max_index=2, probe=_probe, name_resolver=lambda: [])
    rows = as_api_rows(infos)
    assert [r["available"] for r in rows] == [True, False, True]


def test_default_name_resolver_is_never_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the import path inside the resolver to raise; it must return [].
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name.startswith("pygrabber"):
            raise ImportError("no pygrabber here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    monkeypatch.setattr(enum_mod, "_warned_no_pygrabber", False)
    assert enum_mod._default_name_resolver() == []


def test_monkeypatched_pygrabber_names_are_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        enum_mod, "_default_name_resolver", lambda: ["Fake Cam A", "Fake Cam B"]
    )
    infos = enumerate_devices(
        max_index=1, probe=lambda i, h: (True, "msmf"), name_resolver=None
    )
    assert [i.label for i in infos] == ["Fake Cam A", "Fake Cam B"]
