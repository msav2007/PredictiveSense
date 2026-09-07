"""A short in-process no-op run: clean lifecycle, CSV, and manifest."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from predictivesense.telemetry.manifest import MANIFEST_REQUIRED_KEYS
from scripts import run_noop

pytestmark = [pytest.mark.integration, pytest.mark.slow]

RUN_SECONDS = 10
LOOP_THREAD_NAMES = {"ps-producer", "ps-consumer"}


@pytest.fixture()
def results_dir(repo_root: Path) -> Path:
    return repo_root / "results"


def _new_files(directory: Path, since: float, pattern: str) -> list[Path]:
    return sorted(
        p for p in directory.glob(pattern) if p.stat().st_mtime >= since
    )


def test_ten_second_run_is_clean_and_produces_artifacts(results_dir: Path) -> None:
    before = time.time() - 1.0

    exit_code = run_noop.main(["--profile", "dev", "--seconds", str(RUN_SECONDS)])
    assert exit_code == 0

    live = {t.name for t in threading.enumerate()} & LOOP_THREAD_NAMES
    assert not live, f"loop threads still alive: {live}"

    csvs = _new_files(results_dir, before, "noop_*.csv")
    manifests = _new_files(results_dir, before, "manifest_*.json")
    assert csvs, "no metrics CSV written"
    assert manifests, "no manifest written"

    header = csvs[-1].read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == run_noop.CSV_FIELDS
    assert len(csvs[-1].read_text(encoding="utf-8").splitlines()) >= 3  # header + rows

    payload = json.loads(manifests[-1].read_text(encoding="utf-8"))
    for key in MANIFEST_REQUIRED_KEYS:
        assert key in payload
    assert isinstance(payload["config"], dict)
    assert payload["git_commit"]
    assert payload["ended_utc"] is not None
    assert payload["extra"]["run"]["kind"] == "noop"

    # tidy up the artifacts this test produced
    for path in csvs + manifests:
        path.unlink(missing_ok=True)
