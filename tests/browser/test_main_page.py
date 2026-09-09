"""The main dashboard `/` loads clean and its operations panel works
(Phase 6, BLOCK 4.7 / 15.2). The Studio bugs must not have a mirror on `/`.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.browser


def test_main_page_loads_with_zero_console_errors(live_server: str, page: Page, console) -> None:
    page.goto(f"{live_server}/", wait_until="networkidle")
    expect(page.locator("h1, .tb-brand")).to_have_count(1)
    expect(page.locator("#app-shell")).to_be_visible()
    expect(page.locator("#panel-body")).to_be_visible()
    # groups mounted
    expect(page.locator("#panel-body .group, #panel-body [data-group]")).not_to_have_count(0)
    console.assert_clean()


def test_panel_collapse_and_reopen(live_server: str, page: Page) -> None:
    page.goto(f"{live_server}/", wait_until="networkidle")
    shell = page.locator("#app-shell")
    expect(shell).to_have_attribute("data-collapsed", "false")

    page.locator("#btn-collapse").click()
    expect(shell).to_have_attribute("data-collapsed", "true")

    page.locator("#panel-reopen").click()
    expect(shell).to_have_attribute("data-collapsed", "false")


def test_panel_resizes_by_keyboard(live_server: str, page: Page) -> None:
    page.goto(f"{live_server}/", wait_until="networkidle")

    def panel_width() -> float:
        return page.evaluate(
            "() => document.getElementById('control-panel').getBoundingClientRect().width"
        )

    handle = page.locator("#panel-resizer")
    expect(handle).to_have_attribute("role", "separator")

    before = panel_width()
    handle.focus()
    for _ in range(6):
        page.keyboard.press("ArrowLeft")  # widen the panel (handle on its left edge)
    page.wait_for_timeout(150)
    widened = panel_width()

    for _ in range(12):
        page.keyboard.press("ArrowRight")  # narrow it again
    page.wait_for_timeout(150)
    narrowed = panel_width()

    assert widened != before, f"panel width did not change on ArrowLeft ({before} -> {widened})"
    assert narrowed < widened, f"panel width did not shrink on ArrowRight ({widened} -> {narrowed})"


def test_main_page_has_no_studio_state_machine_error(live_server: str, page: Page, console) -> None:
    """Regression guard: the `invalid transition save_and_return from browsing`
    error was observed leaking into `/`'s console during Phase 6 triage (from a
    prior Studio visit). `/` must never carry a Studio transition error."""

    page.goto(f"{live_server}/", wait_until="networkidle")
    page.wait_for_timeout(300)
    assert not any("invalid transition" in e for e in console.errors), console.errors
