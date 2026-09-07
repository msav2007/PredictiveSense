"""Backend camera enumeration: pygrabber names when available, safe fallback."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from predictivesense.camera import enumerate as enum_mod
from predictivesense.camera.enumerate import (
    as_api_rows,
    enumerate_devices,
    load_backend_hint,
    save_backend_hint,
)

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


# -- Phase 1.5: per-device winning-backend cache ----------------------


def test_backend_hint_absent_cache_returns_none(tmp_path: Path) -> None:
    assert load_backend_hint(0, tmp_path) is None


def test_backend_hint_roundtrip_and_merge(tmp_path: Path) -> None:
    save_backend_hint(0, "dshow", tmp_path)
    save_backend_hint(1, "msmf", tmp_path)
    assert load_backend_hint(0, tmp_path) == "dshow"
    assert load_backend_hint(1, tmp_path) == "msmf"
    # merge, not overwrite
    save_backend_hint(0, "msmf", tmp_path)
    assert load_backend_hint(0, tmp_path) == "msmf"
    assert load_backend_hint(1, tmp_path) == "msmf"


def test_backend_hint_rejects_junk(tmp_path: Path) -> None:
    save_backend_hint(0, "not-a-backend", tmp_path)
    assert load_backend_hint(0, tmp_path) is None
    (tmp_path / "camera_backends.json").write_text("{ not json", encoding="utf-8")
    assert load_backend_hint(0, tmp_path) is None


def test_backend_hint_ignores_unknown_cached_value(tmp_path: Path) -> None:
    (tmp_path / "camera_backends.json").write_text(
        json.dumps({"0": "v4l2"}), encoding="utf-8"
    )
    assert load_backend_hint(0, tmp_path) is None
