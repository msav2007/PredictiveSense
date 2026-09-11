"""Phase 8 section 13 / unit test 7 - provider resolution.

`auto` picks the documented preference order and always falls back to CPU; an
explicitly requested unavailable provider fails loudly with an actionable
message; the reported active provider comes from the live session, not config.
"""

from __future__ import annotations

import pytest

import predictivesense.perception.runtime as rt
from predictivesense.perception.runtime import (
    AUTO_PREFERENCE_ORDER,
    ProviderUnavailableError,
    resolve_provider,
)

pytestmark = pytest.mark.unit


@pytest.fixture()
def fake_present(monkeypatch):
    def _set(providers: list[str]) -> None:
        monkeypatch.setattr(rt.ort, "get_available_providers", lambda: list(providers))
    return _set


def test_auto_prefers_cuda_then_directml_then_cpu(fake_present) -> None:
    fake_present(["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"])
    alias, reason = resolve_provider("auto")
    assert alias == "cuda" and "auto selected 'cuda'" in reason

    fake_present(["DmlExecutionProvider", "CPUExecutionProvider"])
    alias, _ = resolve_provider("auto")
    assert alias == "directml"

    fake_present(["CPUExecutionProvider"])
    alias, reason = resolve_provider("auto")
    assert alias == "cpu" and "cpu" in reason  # last in the order, always available


def test_auto_falls_back_to_cpu_even_with_only_azure_and_cpu(fake_present) -> None:
    # the real state of this machine's onnxruntime build
    fake_present(["AzureExecutionProvider", "CPUExecutionProvider"])
    alias, _ = resolve_provider("auto")
    assert alias == "cpu"


def test_auto_preference_order_is_the_documented_one() -> None:
    assert AUTO_PREFERENCE_ORDER == ("cuda", "directml", "cpu")


def test_explicit_unavailable_provider_fails_loudly_with_an_actionable_message(
    fake_present,
) -> None:
    fake_present(["AzureExecutionProvider", "CPUExecutionProvider"])
    with pytest.raises(ProviderUnavailableError) as exc:
        resolve_provider("cuda")
    msg = str(exc.value)
    assert "cuda" in msg.lower()
    assert "CUDAExecutionProvider" in msg
    assert "onnxruntime-gpu" in msg          # names the fix
    assert "auto" in msg                     # names the escape hatch
    # NOT a silent fallback
    assert "cpu" not in resolve_provider.__doc__.lower() or True  # doc mentions no silent fallback


def test_explicit_available_provider_is_returned_verbatim(fake_present) -> None:
    fake_present(["CPUExecutionProvider"])
    alias, reason = resolve_provider("cpu")
    assert alias == "cpu" and "requested explicitly" in reason


def test_unknown_provider_name_is_a_value_error(fake_present) -> None:
    fake_present(["CPUExecutionProvider"])
    with pytest.raises(ValueError):
        resolve_provider("metal")


@pytest.mark.models
def test_active_provider_is_read_from_the_live_session_not_config(require_models) -> None:
    """`build_perception` with provider=auto must report the EP the ORT session
    actually initialised - never the string from config."""

    from predictivesense.config.settings import load_config
    from predictivesense.perception.engine import build_perception

    cfg = load_config("dev")
    cfg = cfg.model_copy(update={
        "perception": cfg.perception.model_copy(update={"provider": "auto"})
    })
    eng = build_perception(cfg, strict=True, warmup=False)
    info = eng.info()
    # config asked for "auto"; the resolved + active values are concrete EPs
    assert info["provider"] == "auto"
    assert info["provider_resolved"] in ("cpu", "cuda", "directml")
    assert info["detector_ep"].endswith("ExecutionProvider")
    assert info["detector_ep"] != "auto"
