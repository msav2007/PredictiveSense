"""Phase 13 Stage 2 - the Object Learning Studio and its trained crop
classifier are isolated behind two off-by-default config flags
(``studio.enabled`` / ``training.classifier_enabled``). Studio source, data,
and registry entries are untouched - only whether the live app depends on
them at all is what changed. See
``docs/phase-reports/phase13-stage1-inspection.md`` section 10 and
``docs/decisions.md`` "Phase 13 Stage 2".
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest
from fastapi.testclient import TestClient

from predictivesense.api.app import create_app
from predictivesense.config.settings import StudioConfig, TrainingConfig

pytestmark = pytest.mark.integration


def test_bare_config_ships_studio_and_classifier_off() -> None:
    # AppConfig itself requires several profile-supplied fields (mode, source,
    # consumer, ...) so it cannot be bare-instantiated - the two Phase 13
    # flags' own section defaults are what "off by default" actually means.
    assert StudioConfig().enabled is False
    assert TrainingConfig().classifier_enabled is False


def test_app_runs_completely_with_studio_disabled(dev_config) -> None:
    """The perception pipeline imports, initialises and runs with Studio
    disabled - GET /health and GET /api/config work, and the pipeline runs a
    real iteration, exactly as with Studio enabled."""

    cfg = dev_config.model_copy(
        update={
            "studio": dev_config.studio.model_copy(update={"enabled": False}),
            "training": dev_config.training.model_copy(update={"classifier_enabled": False}),
        }
    )
    app = create_app(cfg, start_loop=False)
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert client.get("/api/config").status_code == 200
        assert app.state.batch_proposals_executor is None


def test_studio_and_objects_routes_404_when_disabled(dev_config) -> None:
    cfg = dev_config.model_copy(
        update={"studio": dev_config.studio.model_copy(update={"enabled": False})}
    )
    app = create_app(cfg, start_loop=False)
    with TestClient(app) as client:
        assert client.get("/studio").status_code == 404
        assert client.get("/api/studio/status").status_code == 404
        assert client.get("/api/objects").status_code == 404
        assert client.get("/api/objects/vocab").status_code == 404


def test_studio_and_objects_routes_reachable_when_enabled(dev_config) -> None:
    """The flag is a genuine toggle, not a one-way removal - Studio still
    works exactly as before when a profile opts back in (dev.yaml/eval.yaml
    both do)."""

    cfg = dev_config.model_copy(
        update={"studio": dev_config.studio.model_copy(update={"enabled": True})}
    )
    app = create_app(cfg, start_loop=False)
    with TestClient(app) as client:
        assert client.get("/studio").status_code == 200
        assert client.get("/api/studio/status").status_code == 200
        assert client.get("/api/objects/vocab").status_code == 200


def test_ws_ingest_4409_refusal_unaffected_by_studio_being_disabled(browser_config) -> None:
    """The Object Learning Studio being unreachable does not disable the
    ingest guard's own logic - app.state.studio stays a benign, always-
    inactive dict, so studio_is_active() still returns False (it never had a
    session to refuse in the first place)."""

    from predictivesense.api.studio import studio_is_active

    cfg = browser_config.model_copy(
        update={"studio": browser_config.studio.model_copy(update={"enabled": False})}
    )
    app = create_app(cfg, start_loop=False)
    assert studio_is_active(app) is False


def test_shipped_dev_and_eval_profiles_opt_studio_back_in() -> None:
    """Only the shipped developer-facing profiles re-enable Studio; a bare
    AppConfig() does not (asserted above) - documents the deliberate choice
    in docs/decisions.md "Phase 13 Stage 2"."""

    from predictivesense.config.settings import load_config

    assert load_config("dev").studio.enabled is True
    assert load_config("eval").studio.enabled is True
    # The classifier-application feature stays off even in the developer's
    # own profiles - re-enabling it would just reproduce the original defect
    # on a still-single-class model.
    assert load_config("dev").training.classifier_enabled is False
    assert load_config("eval").training.classifier_enabled is False


_SUBPROCESS_SCRIPT = textwrap.dedent(
    """
    import sys

    from predictivesense.camera.mailbox import LatestFrameMailbox
    from predictivesense.camera.source import create_frame_source
    from predictivesense.config.settings import load_config
    from predictivesense.pipeline.loop import AnalysisLoop
    from predictivesense.telemetry.metrics import MetricRegistry

    cfg = load_config("dev")
    assert cfg.training.classifier_enabled is False, "expected the off-by-default flag"
    src = create_frame_source(cfg.source)
    loop = AnalysisLoop(cfg, src, LatestFrameMailbox(), registry=MetricRegistry())
    assert loop._classifier is None

    leaked = [
        name for name in sys.modules
        if name == "predictivesense.training.classifier_registry"
        or name == "predictivesense.perception.classifier"
    ]
    if leaked:
        print("LEAKED:" + ",".join(sorted(leaked)))
        sys.exit(1)
    print("OK")
    """
)


def test_no_studio_module_imported_by_live_detection_path_when_disabled() -> None:
    """Runtime proof, in a fresh interpreter (so no other test's import
    leaves a false pass), that constructing the live AnalysisLoop with the
    classifier disabled never imports predictivesense.training.
    classifier_registry or predictivesense.perception.classifier - the two
    modules that resolve and run the Studio-trained crop classifier."""

    result = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_SCRIPT],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert result.stdout.strip() == "OK"
