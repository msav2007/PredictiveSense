"""Object Learning Studio - bulk image upload (Phase 7, BLOCK 9.7-9.12).

Loads the real ``/studio`` page in headless Chromium, uploads several generated
fixture images to one object, drives the grid + Review All reviewer, saves with
one item unresolved, and asserts the page never claims the model learned or was
trained. Zero console errors throughout.

Uses a runtime-created object so the shared (package-scoped) ``cup`` / ``bottle``
counts other browser tests rely on are untouched.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.browser

OBJ_NAME = "bulkwidget"
OBJ_ID = "bulkwidget"


def _fixture_images(tmp_path, n: int) -> list[str]:
    import cv2

    rng = np.random.default_rng(7)
    paths = []
    for i in range(n):
        img = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
        cv2.rectangle(img, (40 + i * 5, 40), (200, 180), (255, 255, 255), 3)
        p = tmp_path / f"fix_{i}.jpg"
        cv2.imwrite(str(p), img)
        paths.append(str(p))
    return paths


@pytest.fixture()
def bulk_object(live_server: str, page: Page):
    """Create a throwaway object via the API for this test only."""

    r = page.request.post(f"{live_server}/api/objects", data={"name": OBJ_NAME})
    assert r.ok, r.text()
    oid = r.json()["object_id"]
    yield oid
    page.request.delete(f"{live_server}/api/objects/{oid}")


def open_studio(page: Page, base: str) -> None:
    page.goto(f"{base}/studio", wait_until="networkidle")
    page.wait_for_selector("#object-list .object-item")


def select(page: Page, oid: str) -> None:
    page.locator(f'#object-list .object-item[data-object-id="{oid}"]').click()
    expect(page.locator("#studio-diag")).to_contain_text(f"selected_object_id={oid}")


def wait_batch_ready(page: Page) -> None:
    expect(page.locator("#batch-note")).to_contain_text("ready", timeout=15000)


# --------------------------------------------------------------------------


def test_upload_images_shows_a_grid_with_per_item_status(
    live_server: str, page: Page, bulk_object, tmp_path, console
) -> None:
    open_studio(page, live_server)
    select(page, bulk_object)

    files = _fixture_images(tmp_path, 5)
    page.set_input_files("#bulk-upload-input", files)

    expect(page.locator("#batch-panel")).to_be_visible()
    expect(page.locator("#batch-grid .batch-cell")).to_have_count(5, timeout=15000)
    wait_batch_ready(page)

    # every cell carries a status badge (perception is off in the browser
    # fixture -> every item needs a manual box)
    for badge in page.locator("#batch-grid .batch-cell .bc-badge").all():
        assert badge.inner_text().strip() != ""
    expect(page.locator("#batch-counts")).to_contain_text("5 uploaded")
    expect(page.locator("#batch-counts")).to_contain_text("need a box")
    console.assert_clean()


def test_progress_indicator_and_page_stays_interactive(
    live_server: str, page: Page, bulk_object, tmp_path, console
) -> None:
    open_studio(page, live_server)
    select(page, bulk_object)
    page.set_input_files("#bulk-upload-input", _fixture_images(tmp_path, 4))

    # the reviewer opens even if a batch is still processing (UI not blocked)
    page.locator("#batch-grid .batch-cell").first.wait_for(timeout=15000)
    page.locator("#btn-batch-review").click()
    expect(page.locator("#batch-reviewer")).to_be_visible()
    expect(page.locator("#br-pos")).to_contain_text("/ 4")
    console.assert_clean()


def test_review_all_navigation_and_box_editing(
    live_server: str, page: Page, bulk_object, tmp_path, console
) -> None:
    open_studio(page, live_server)
    select(page, bulk_object)
    page.set_input_files("#bulk-upload-input", _fixture_images(tmp_path, 3))
    expect(page.locator("#batch-grid .batch-cell")).to_have_count(3, timeout=15000)
    wait_batch_ready(page)

    page.locator("#btn-batch-review").click()
    expect(page.locator("#batch-reviewer")).to_be_visible()
    expect(page.locator("#br-pos")).to_have_text("1 / 3")

    def box_rect():
        return page.evaluate(
            "() => { const b = document.getElementById('batch-box').getBoundingClientRect();"
            " return {x: b.x, y: b.y, w: b.width, h: b.height}; }"
        )

    # next / previous
    page.locator("#btn-br-next").click()
    expect(page.locator("#br-pos")).to_have_text("2 / 3")
    page.locator("#btn-br-prev").click()
    expect(page.locator("#br-pos")).to_have_text("1 / 3")

    # move + resize the box with the keyboard (reuses the shared editor)
    before = box_rect()
    page.locator("#batch-box").focus()
    for _ in range(12):
        page.keyboard.press("Shift+ArrowRight")
        page.keyboard.press("Shift+ArrowDown")
    moved = box_rect()
    assert (moved["x"], moved["y"]) != (before["x"], before["y"])

    # delete the box on the current item -> it now needs a box
    page.locator("#btn-br-delete").click()
    expect(page.locator("#br-status")).to_contain_text("box removed")

    # add a box back manually
    page.locator("#btn-br-add").click()
    added = box_rect()
    assert added["w"] > 0 and added["h"] > 0
    console.assert_clean()


def test_save_all_with_one_unresolved_reports_both_counts(
    live_server: str, page: Page, bulk_object, tmp_path, console
) -> None:
    open_studio(page, live_server)
    select(page, bulk_object)
    page.set_input_files("#bulk-upload-input", _fixture_images(tmp_path, 3))
    expect(page.locator("#batch-grid .batch-cell")).to_have_count(3, timeout=15000)
    wait_batch_ready(page)

    page.locator("#btn-batch-review").click()
    expect(page.locator("#batch-reviewer")).to_be_visible()

    # accept the box on items 1 and 2, delete the box on item 3
    page.locator("#btn-br-accept").click()  # -> advances to 2
    page.locator("#btn-br-accept").click()  # -> advances to 3
    expect(page.locator("#br-pos")).to_have_text("3 / 3")
    page.locator("#btn-br-delete").click()

    before = int(page.locator("#obj-count").inner_text())
    page.locator("#btn-batch-save").click()

    note = page.locator("#batch-note")
    expect(note).to_contain_text("2 samples saved")
    expect(note).to_contain_text("1 still need")

    # the object's sample count rose by exactly the saved number
    expect(page.locator("#obj-count")).to_have_text(str(before + 2))
    api = page.request.get(f"{live_server}/api/objects/{bulk_object}").json()
    assert api["sample_count"] == before + 2
    assert all(s["source"] == "upload_batch" for s in api["samples"])
    console.assert_clean()


def test_no_text_claims_the_model_learned_or_was_trained(
    live_server: str, page: Page, bulk_object, tmp_path, console
) -> None:
    open_studio(page, live_server)
    select(page, bulk_object)
    page.set_input_files("#bulk-upload-input", _fixture_images(tmp_path, 2))
    expect(page.locator("#batch-grid .batch-cell")).to_have_count(2, timeout=15000)
    wait_batch_ready(page)
    page.locator("#btn-batch-review").click()
    page.locator("#btn-br-accept").click()
    page.locator("#btn-br-accept").click()
    page.locator("#btn-batch-save").click()
    expect(page.locator("#batch-note")).to_contain_text("saved")

    body_text = page.locator("body").inner_text().lower()
    for claim in ("learned", "trained", "recognises now", "recognizes now", "now recognises", "now recognizes"):
        assert claim not in body_text, f"page implies the model changed: {claim!r}"
    console.assert_clean()


def test_discard_batch_removes_the_panel_and_staging(
    live_server: str, page: Page, bulk_object, tmp_path, console
) -> None:
    open_studio(page, live_server)
    select(page, bulk_object)
    page.set_input_files("#bulk-upload-input", _fixture_images(tmp_path, 2))
    expect(page.locator("#batch-panel")).to_be_visible()
    expect(page.locator("#batch-grid .batch-cell")).to_have_count(2, timeout=15000)

    page.once("dialog", lambda d: d.accept())
    page.locator("#btn-batch-discard").click()
    expect(page.locator("#batch-panel")).to_be_hidden()

    batches = page.request.get(f"{live_server}/api/objects/{bulk_object}/batches").json()
    assert batches["batches"] == []
    console.assert_clean()
