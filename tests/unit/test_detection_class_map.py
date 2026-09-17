"""Stage 5 section 3.1/5: the committed detection_class_map.json carries the
right decisions - watch mapped, mug/cup merged, shaker rejected (not merely
unmapped), pencil/charger/comb unavailable from the external source.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_PATH = Path(__file__).resolve().parents[2] / "results" / "detection_class_map.json"


def _entries() -> list[dict]:
    doc = json.loads(_PATH.read_text(encoding="utf-8"))
    return doc["entries"]


def _entry(ps_class: str) -> dict:
    for e in _entries():
        if e["ps_class"] == ps_class:
            return e
    raise AssertionError(f"no entry for {ps_class!r}")


def test_class_map_file_exists_and_is_versioned():
    assert _PATH.is_file()
    doc = json.loads(_PATH.read_text(encoding="utf-8"))
    assert "version" in doc
    assert isinstance(doc["entries"], list) and len(doc["entries"]) > 0


def test_watch_is_mapped_with_a_real_mid():
    e = _entry("watch")
    assert e["status"] == "mapped"
    mids = {sc["mid"] for sc in e["source_classes"]}
    assert "/m/0gjkl" in mids


def test_mug_cup_is_an_explicit_merge_of_two_source_classes():
    e = _entry("mug/cup")
    assert e["status"] == "mapped"
    assert e["merge_decision"] == "merged"
    names = {sc["name"] for sc in e["source_classes"]}
    assert names == {"Mug", "Coffee cup"}
    assert e["reason"]  # a merge without a recorded reason is not reviewable


def test_shaker_is_rejected_not_merely_unmapped():
    e = _entry("shaker")
    assert e["status"] == "rejected"
    assert "confusable" in e["reason"].lower() or "different" in e["reason"].lower() or "generic cylindrical" in e["reason"].lower()


@pytest.mark.parametrize("ps_class", ["pencil", "charger", "comb"])
def test_classes_with_no_open_images_match_are_marked_unavailable(ps_class):
    e = _entry(ps_class)
    assert e["status"] == "unavailable"
    assert e["source_classes"] == []


def test_every_mapped_entry_has_at_least_one_source_class_with_a_mid():
    for e in _entries():
        if e["status"] == "mapped":
            assert e["source_classes"], f"{e['ps_class']!r} is mapped but has no source_classes"
            for sc in e["source_classes"]:
                assert sc.get("mid", "").startswith("/m/"), f"{e['ps_class']!r} source class missing a real mid"


def test_every_entry_has_a_status_in_the_allowed_set():
    allowed = {"mapped", "unavailable", "rejected"}
    for e in _entries():
        assert e["status"] in allowed, f"{e['ps_class']!r} has unrecognised status {e['status']!r}"
