"""Shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the repo-root ``scripts/`` package importable from tests without shipping
# an __init__.py in it.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from predictivesense.config.settings import AppConfig, default_profiles_dir, load_config
from predictivesense.core.types import Frame

_VALID_PROFILE = """\
mode: realtime
results_dir: results
logging:
  level: INFO
source:
  kind: synthetic
  width: 64
  height: 48
  target_fps: 1000.0
  seed: 7
consumer:
  sample_rate_hz: 200.0
  stale_after_ms: 250.0
broadcast:
  rate_hz: 50.0
  client_send_timeout_s: 1.0
api:
  host: 127.0.0.1
  port: 8123
noop:
  default_seconds: 5
  csv_sample_period_s: 0.5
  max_rss_growth_mb: 25.0
"""


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def profiles_dir() -> Path:
    return default_profiles_dir()


@pytest.fixture()
def dev_config() -> AppConfig:
    return load_config("dev")


@pytest.fixture()
def eval_config() -> AppConfig:
    return load_config("eval")


@pytest.fixture()
def valid_profile_text() -> str:
    return _VALID_PROFILE


@pytest.fixture()
def write_profile(tmp_path: Path):
    """Return a helper that writes ``<name>.yaml`` into a temp dir and returns the dir."""

    def _write(name: str, text: str) -> Path:
        (tmp_path / f"{name}.yaml").write_text(text, encoding="utf-8")
        return tmp_path

    return _write


@pytest.fixture()
def make_frame():
    """Return a factory for small deterministic :class:`Frame` objects."""

    def _make(frame_id: int, *, w: int = 8, h: int = 6) -> Frame:
        image = np.full((h, w, 3), frame_id % 256, dtype=np.uint8)
        return Frame(
            frame_id=frame_id,
            capture_ts=float(frame_id) * 0.01,
            image=image,
            width=w,
            height=h,
            source_id="test-source",
            seq=frame_id,
        )

    return _make
