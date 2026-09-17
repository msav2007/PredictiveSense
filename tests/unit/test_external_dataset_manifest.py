"""Phase 12: sanity-checks the REAL imported external manifest, when present
on this developer machine. Skips cleanly (require_external) everywhere else -
this is the one place the real download is touched, never the fast unit
suite in test_import_external_dataset.py (synthetic fixtures only)."""

from __future__ import annotations

import json

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.external]


def test_real_watch_manifest_is_well_formed(require_external):
    doc = json.loads(require_external.read_text(encoding="utf-8"))
    assert doc["object_id"] == "watch"
    assert doc["source"] == "open-images-v7"
    positives = [s for s in doc["samples"] if s["role"] == "positive"]
    assert len(positives) >= 100
    for s in positives[:5]:
        assert s["box_confirmed_by_human"] is True
        assert len(s["box"]) == 4
        import os

        assert os.path.isfile(s["image_path"])
