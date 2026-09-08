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
from predictivesense.core.enums import SourceKind
from predictivesense.core.types import Frame
from tests.fixtures.make_fixture_video import FIXTURE_NAME, SPEC, make_fixture_video

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


# -- hardware marker gate -------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-hardware",
        action="store_true",
        default=False,
        help="run tests marked `hardware` (needs a physical camera attached)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--run-hardware"):
        return
    skip = pytest.mark.skip(reason="needs a camera; pass --run-hardware to run")
    for item in items:
        if "hardware" in item.keywords:
            item.add_marker(skip)

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


_PERCEPTION_MODELS = [
    _REPO_ROOT / "models" / "yolo11n.onnx",
    _REPO_ROOT / "models" / "yolo11n-pose.onnx",
]


def _without_perception(cfg: AppConfig) -> AppConfig:
    """Disable perception on a config.

    The broad regression suite must stay fast and must reproduce Phase 1.6
    behaviour exactly; perception has dedicated ``models``-marked coverage. The
    shipped ``dev`` / ``eval`` profiles keep perception ON - asserted by
    ``tests/unit/test_config.py`` - this only affects the shared fixtures.
    """

    return cfg.model_copy(
        update={
            "perception": cfg.perception.model_copy(
                update={"detection_enabled": False, "pose_enabled": False}
            )
        }
    )


@pytest.fixture()
def dev_config() -> AppConfig:
    return _without_perception(load_config("dev"))


@pytest.fixture()
def eval_config() -> AppConfig:
    return _without_perception(load_config("eval"))


@pytest.fixture()
def browser_config() -> AppConfig:
    """``dev`` profile with the browser ingest source (capture.owner stays browser)."""

    cfg = _without_perception(load_config("dev"))
    return cfg.model_copy(
        update={"source": cfg.source.model_copy(update={"kind": SourceKind.BROWSER})}
    )


@pytest.fixture()
def perception_dev_config() -> AppConfig:
    """``dev`` profile with perception left ON (models-marked tests)."""

    return load_config("dev")


@pytest.fixture()
def require_models() -> None:
    """Skip cleanly (never fail, never silently pass) when weights are absent."""

    missing = [p.name for p in _PERCEPTION_MODELS if not p.is_file()]
    if missing:
        pytest.skip(
            f"perception weights absent ({', '.join(missing)}); "
            f"run `python scripts/fetch_models.py`"
        )


_EVAL_STORE = _REPO_ROOT / "data" / "eval" / "annotations.json"


@pytest.fixture()
def require_eval_set():
    """Skip cleanly when the labelled evaluation set is absent or unlabelled.

    Returns the loaded :class:`CocoStore`. The evaluation set is produced by
    ``scripts/build_eval_frames.py`` + labelling at ``/label``; ``dataset``-marked
    tests depend on it and must never fail merely because it does not exist yet.
    """

    if not _EVAL_STORE.is_file():
        pytest.skip(
            f"no evaluation set at {_EVAL_STORE}; run "
            f"`python scripts/build_eval_frames.py` and label at /label"
        )
    from predictivesense.dataset.coco_store import CocoStore

    store = CocoStore.load(_EVAL_STORE)
    if store.counts()["labelled"] == 0:
        pytest.skip(
            f"{_EVAL_STORE} has 0 labelled frames; label at /label before "
            f"running dataset-marked tests"
        )
    return store


@pytest.fixture(scope="session")
def fixture_video() -> Path:
    """Path to the deterministic test clip, regenerated if missing."""

    dest = _FIXTURE_DIR / FIXTURE_NAME
    if not dest.is_file() or dest.stat().st_size == 0:
        make_fixture_video(dest)
    return dest


@pytest.fixture(scope="session")
def fixture_video_frames(fixture_video: Path) -> int:
    """The number of frames OpenCV actually decodes from the fixture clip."""

    import cv2

    cap = cv2.VideoCapture(str(fixture_video))
    try:
        count = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            count += 1
    finally:
        cap.release()
    assert count > 0, "fixture clip decoded zero frames"
    return count


@pytest.fixture(scope="session")
def fixture_video_spec():
    return SPEC


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
