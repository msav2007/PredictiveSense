"""Object Learning Studio - full browser flow (Phase 6, BLOCK 4.7 / 7.18 / 15).

Loads the real `/studio` page in headless Chromium and drives the six reported
symptoms + the capture path. This is the structural fix for two phases of
frontend defects the Python-only suite reported as fixed.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.browser

CUP = "cup"
BOTTLE = "bottle"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def open_studio(page: Page, base: str) -> None:
    page.goto(f"{base}/studio", wait_until="networkidle")
    page.wait_for_selector("#object-list .object-item")


def row(page: Page, object_id: str):
    return page.locator(f'#object-list .object-item[data-object-id="{object_id}"]')


def diag_text(page: Page) -> str:
    return page.locator("#studio-diag").inner_text()


def select_object(page: Page, object_id: str) -> None:
    row(page, object_id).click()
    expect(page.locator("#studio-diag")).to_contain_text(f"selected_object_id={object_id}")
    expect(page.locator("#capture-pane")).to_be_visible()


def wait_for_camera(page: Page) -> None:
    page.wait_for_function(
        "() => { const v = document.getElementById('preview');"
        " return v && v.videoWidth > 0 && v.readyState >= 2; }",
        timeout=8000,
    )


def box_rect(page: Page) -> dict:
    return page.evaluate(
        "() => { const b = document.getElementById('sample-box').getBoundingClientRect();"
        " return {x: b.x, y: b.y, w: b.width, h: b.height}; }"
    )


# --------------------------------------------------------------------------
# BLOCK 4.7 - the page loads clean
# --------------------------------------------------------------------------


def test_studio_loads_with_zero_console_errors(live_server: str, page: Page, console) -> None:
    open_studio(page, live_server)
    expect(page.locator("#empty-state")).to_be_visible()
    expect(page.locator("#studio-diag")).to_contain_text("state=browsing")
    expect(page.locator("#studio-error")).to_be_hidden()
    console.assert_clean()


# --------------------------------------------------------------------------
# BLOCK 4.7 - symptoms 1-4: clicking an object row opens the editor
# --------------------------------------------------------------------------


def test_clicking_object_row_opens_the_editor(live_server: str, page: Page, console) -> None:
    open_studio(page, live_server)
    row(page, CUP).click()

    expect(page.locator("#empty-state")).to_be_hidden()
    expect(page.locator("#capture-pane")).to_be_visible()
    expect(page.locator("#obj-title")).to_have_text(CUP)
    expect(page.locator("#obj-count")).to_have_text("0")
    expect(page.locator("#stage")).to_be_visible()  # camera area
    expect(page.locator("#btn-capture")).to_be_visible()
    expect(page.locator("#conditions select")).not_to_have_count(0)  # condition tags
    expect(page.locator("#guidance-list li")).not_to_have_count(0)  # collection guidance
    expect(page.locator(f'#object-list .object-item[data-object-id="{CUP}"].active')).to_have_count(1)
    expect(page.locator("#studio-diag")).to_contain_text("state=object_selected")
    console.assert_clean()


def test_repeated_selection_of_the_same_row_is_stable(live_server: str, page: Page, console) -> None:
    open_studio(page, live_server)
    for _ in range(4):
        row(page, CUP).click()
        page.wait_for_timeout(80)
    expect(page.locator("#obj-title")).to_have_text(CUP)
    expect(page.locator("#studio-diag")).to_contain_text("state=object_selected")
    expect(page.locator("#studio-diag")).to_contain_text("has_pending=no")
    console.assert_clean()


def test_selecting_a_different_object_replaces_editor_state(
    live_server: str, page: Page, console
) -> None:
    open_studio(page, live_server)
    select_object(page, CUP)
    select_object(page, BOTTLE)

    expect(page.locator("#obj-title")).to_have_text(BOTTLE)
    expect(page.locator("#studio-diag")).to_contain_text(f"selected_object_id={BOTTLE}")
    expect(page.locator(f'#object-list .object-item[data-object-id="{BOTTLE}"].active')).to_have_count(1)
    expect(page.locator(f'#object-list .object-item[data-object-id="{CUP}"].active')).to_have_count(0)
    console.assert_clean()


# --------------------------------------------------------------------------
# BLOCK 4.7 - Back to camera: object kept, box reset
# --------------------------------------------------------------------------


def test_back_to_camera_preserves_object_and_resets_the_box(
    live_server: str, page: Page, console
) -> None:
    open_studio(page, live_server)
    select_object(page, CUP)
    wait_for_camera(page)

    # capture a frame from the fake camera (saves immediately - not "pending")
    page.locator("#btn-capture").click()
    expect(page.locator("#review-strip .review-item")).to_have_count(1, timeout=8000)

    # inspect that sample -> reviewing; nudge the box well off-centre
    page.locator("#review-strip .review-item").first.click()
    expect(page.locator("#studio-diag")).to_contain_text("state=reviewing")
    expect(page.locator("#inspect-actions")).to_be_visible()
    page.locator("#sample-box").focus()
    for _ in range(15):
        page.keyboard.press("Shift+ArrowLeft")
        page.keyboard.press("Shift+ArrowUp")
    moved = box_rect(page)

    # Back to camera -> object_selected, object kept, inspect controls gone,
    # box back to a fresh centred default (BLOCK 5.11).
    page.locator("#btn-inspect-back").click()
    expect(page.locator("#studio-diag")).to_contain_text("state=object_selected")
    expect(page.locator("#studio-diag")).to_contain_text(f"selected_object_id={CUP}")
    expect(page.locator("#inspect-actions")).to_be_hidden()
    expect(page.locator("#preview")).to_be_visible()

    reset = box_rect(page)
    stage = page.evaluate(
        "() => { const s = document.getElementById('stage').getBoundingClientRect();"
        " return {x: s.x, y: s.y, w: s.width, h: s.height}; }"
    )
    reset_cx = reset["x"] + reset["w"] / 2
    stage_cx = stage["x"] + stage["w"] / 2
    assert reset != moved, "box was not reset when returning to camera"
    # a centred default sits within ~15% of the stage centre
    assert abs(reset_cx - stage_cx) < stage["w"] * 0.15, (
        f"box not re-centred: box cx {reset_cx:.0f} vs stage cx {stage_cx:.0f}"
    )
    console.assert_clean()


# --------------------------------------------------------------------------
# BLOCK 4.7 - Save and return
# --------------------------------------------------------------------------


def test_save_and_return_from_browsing_navigates_home(live_server: str, page: Page, console) -> None:
    open_studio(page, live_server)
    page.locator("#btn-return").click()
    page.wait_for_url(f"{live_server}/")
    expect(page.locator("#app-shell")).to_be_visible()
    status = page.request.get(f"{live_server}/api/studio/status").json()
    assert status["active"] is False
    console.assert_clean()


def test_save_and_return_from_the_editor_navigates_home(
    live_server: str, page: Page, console
) -> None:
    open_studio(page, live_server)
    select_object(page, CUP)
    page.locator("#btn-return").click()
    page.wait_for_url(f"{live_server}/")
    expect(page.locator("#app-shell")).to_be_visible()
    console.assert_clean()


# --------------------------------------------------------------------------
# BLOCK 5.13 / 14 - a refused / blocked transition gives visible feedback
# --------------------------------------------------------------------------


def test_leaving_with_a_staged_upload_warns_and_can_be_cancelled(
    live_server: str, page: Page, tmp_path, console
) -> None:
    import cv2

    img = np.random.default_rng(7).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    p = tmp_path / "u.jpg"
    cv2.imwrite(str(p), img)

    open_studio(page, live_server)
    select_object(page, CUP)
    page.set_input_files("#upload-input", str(p))
    expect(page.locator("#studio-diag")).to_contain_text("state=capturing")
    expect(page.locator("#studio-diag")).to_contain_text("has_pending=yes")

    # cancel the confirm -> stay in the Studio with a visible note
    page.once("dialog", lambda d: d.dismiss())
    page.locator("#btn-return").click()
    page.wait_for_timeout(200)
    assert page.url.endswith("/studio")
    expect(page.locator("#capture-note")).to_contain_text("Still in the Studio")

    # accept the confirm -> leaves, pending discarded after the warning
    page.once("dialog", lambda d: d.accept())
    page.locator("#btn-return").click()
    page.wait_for_url(f"{live_server}/")
    console.assert_clean()


# --------------------------------------------------------------------------
# BLOCK 7.18 / 15.5 - Studio capture quality, verified against a fake camera
# --------------------------------------------------------------------------


def test_fake_camera_capture_stores_full_res_original_and_separate_thumbnail(
    live_server: str, page: Page, console
) -> None:
    import cv2

    open_studio(page, live_server)
    select_object(page, CUP)
    wait_for_camera(page)
    page.locator("#btn-capture").click()

    # poll the API until the sample lands
    sample = None
    for _ in range(40):
        r = page.request.get(f"{live_server}/api/objects/{CUP}")
        samples = r.json().get("samples", [])
        if samples:
            sample = samples[0]
            break
        time.sleep(0.2)
    assert sample is not None, "capture produced no sample"

    aw, ah = (int(v) for v in sample["achieved_resolution"].split("x"))
    assert sample["width"] == aw and sample["height"] == ah
    assert sample["capture_path"] in ("imagebitmap", "element")
    assert sample["source"] == "camera"
    assert sample["thumb_path"] and sample["thumb_path"] != sample["path"]
    assert sample["original_bytes"] > 0

    orig = page.request.get(
        f"{live_server}/api/objects/{CUP}/samples/{sample['sample_id']}/image"
    ).body()
    thumb = page.request.get(
        f"{live_server}/api/objects/{CUP}/samples/{sample['sample_id']}/image?thumb=1"
    ).body()
    assert orig != thumb and len(thumb) < len(orig)

    dec = cv2.imdecode(np.frombuffer(orig, np.uint8), cv2.IMREAD_COLOR)
    assert dec.shape[1] == aw and dec.shape[0] == ah, "stored original is not the achieved resolution"
    tdec = cv2.imdecode(np.frombuffer(thumb, np.uint8), cv2.IMREAD_COLOR)
    assert max(tdec.shape[:2]) <= 240 and max(tdec.shape[:2]) < max(dec.shape[:2])

    # surface the observed resolution for the phase report
    print(f"[phase6] fake-camera achieved_resolution = {sample['achieved_resolution']}, "
          f"capture_path = {sample['capture_path']}")
    console.assert_clean()
