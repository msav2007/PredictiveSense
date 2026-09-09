"""Fixtures for the Playwright browser smoke suite (Phase 6, BLOCK 4).

Two consecutive phases shipped Studio frontend defects that the Python-only
suite reported as fixed, because no test loaded the page and clicked anything.
These fixtures load the *real* app in headless Chromium.

Degrade-gracefully contract (BLOCK 4.5 / 14):
  * Playwright not installed          -> the whole ``tests/browser`` dir skips.
  * Chromium binary not installed     -> every ``browser`` test skips with the
    ``playwright install chromium`` hint, never fails.
  * No camera in the environment      -> Chromium's fake media device is used
    (``--use-fake-device-for-media-stream``), so capture flows run headlessly.

Hand-rolled on the base ``playwright`` package only - no ``pytest-playwright``
(Playwright is the single new dependency this phase).
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip(
    "playwright.sync_api",
    reason="playwright not installed - `pip install -e \".[dev]\"` to run browser tests",
)

from playwright.sync_api import Browser, Page, Playwright, sync_playwright  # noqa: E402

from predictivesense.api.app import create_app  # noqa: E402
from predictivesense.config.settings import load_config  # noqa: E402
from predictivesense.core.enums import SourceKind  # noqa: E402
from predictivesense.objects.registry import ObjectRegistry  # noqa: E402

pytestmark = pytest.mark.browser

# Chromium flags that make getUserMedia work headlessly with a synthetic camera.
_FAKE_MEDIA_ARGS = [
    "--use-fake-device-for-media-stream",
    "--use-fake-ui-for-media-stream",
    "--autoplay-policy=no-user-gesture-required",
]

# Console noise that is not a real defect (a missing favicon is a 404 that
# Chromium logs at error level on every page).
_BENIGN_CONSOLE = ("favicon.ico",)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _ThreadedUvicorn:
    """Run a uvicorn server in a background thread for the test session."""

    def __init__(self, app, host: str, port: int) -> None:
        import uvicorn

        self._config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        self._server = uvicorn.Server(self._config)
        self._server.install_signal_handlers = lambda: None  # type: ignore[assignment]
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def start(self, timeout: float = 15.0) -> None:
        self._thread.start()
        deadline = time.monotonic() + timeout
        while not self._server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("uvicorn did not start in time")
            time.sleep(0.02)

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5.0)


# NOTE: every fixture here is **package**-scoped, not session-scoped. While
# `sync_playwright()` is started it keeps a *running* asyncio event loop on the
# main thread, which would break any later test that calls `asyncio.run()`
# (e.g. `test_ws_snapshots.py`). Package scope tears the Playwright manager down
# when `tests/browser/` finishes - before `tests/integration/` runs.


@pytest.fixture(scope="package")
def seeded_objects_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A fresh objects store with two pre-created profiles (no samples).

    Never touches the repo's ``data/objects``.
    """

    root = tmp_path_factory.mktemp("browser_objects")
    reg = ObjectRegistry(root)
    reg.create("cup", confusable_with=["bowl", "mug"])
    reg.create("bottle")
    return root


@pytest.fixture(scope="package")
def live_server(seeded_objects_root: Path) -> Iterator[str]:
    """The real FastAPI app on a real port; browser-ingest source, perception
    off (fast), objects store pointed at the seeded temp dir."""

    cfg = load_config("dev")
    cfg = cfg.model_copy(
        update={
            "source": cfg.source.model_copy(update={"kind": SourceKind.BROWSER}),
            "perception": cfg.perception.model_copy(
                update={"detection_enabled": False, "pose_enabled": False}
            ),
            "objects": cfg.objects.model_copy(update={"root": seeded_objects_root}),
        }
    )
    port = _free_port()
    app = create_app(cfg, start_loop=False)
    server = _ThreadedUvicorn(app, "127.0.0.1", port)
    server.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.stop()


@pytest.fixture(scope="package")
def _playwright() -> Iterator[Playwright]:
    pw = sync_playwright().start()
    try:
        yield pw
    finally:
        pw.stop()  # clears the running asyncio loop it holds on the main thread


@pytest.fixture(scope="package")
def browser(_playwright: Playwright) -> Iterator[Browser]:
    try:
        b = _playwright.chromium.launch(headless=True, args=_FAKE_MEDIA_ARGS)
    except Exception as exc:  # noqa: BLE001 - any launch failure -> skip, never fail
        pytest.skip(
            f"could not launch Chromium ({exc!r}); run `playwright install chromium`"
        )
    try:
        yield b
    finally:
        b.close()


class ConsoleWatch:
    """Collects console errors + uncaught page errors for a page."""

    def __init__(self) -> None:
        self.errors: list[str] = []

    def _on_console(self, msg) -> None:
        if msg.type == "error" and not any(b in msg.text for b in _BENIGN_CONSOLE):
            self.errors.append(f"console.error: {msg.text}")

    def _on_pageerror(self, exc) -> None:
        self.errors.append(f"pageerror: {exc}")

    def attach(self, page: Page) -> "ConsoleWatch":
        page.on("console", self._on_console)
        page.on("pageerror", self._on_pageerror)
        return self

    def assert_clean(self) -> None:
        assert not self.errors, "unexpected console output:\n  " + "\n  ".join(self.errors)


@pytest.fixture()
def page(browser: Browser) -> Iterator[Page]:
    context = browser.new_context(permissions=["camera"], viewport={"width": 1440, "height": 900})
    pg = context.new_page()
    try:
        yield pg
    finally:
        context.close()


@pytest.fixture()
def console(page: Page) -> ConsoleWatch:
    return ConsoleWatch().attach(page)
