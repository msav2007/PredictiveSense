"""ONNX Runtime session creation, explicit provider selection, and warm-up.

Session objects are created once here and reused for the life of the process -
never per frame (Block 4 / Block 13.8). Provider selection is explicit and never
assumes CUDA. If the requested execution provider is not active on the created
session the call fails loudly rather than silently running on a different one
(Block 7): a requested provider is never reported as the one in use.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from predictivesense.logging_setup import get_logger

__all__ = [
    "PROVIDER_ALIASES",
    "SessionHandle",
    "create_session",
    "available_provider_aliases",
]

_LOG = get_logger(__name__)

# Friendly name (config ``perception.provider``) -> ONNX Runtime EP name.
PROVIDER_ALIASES: dict[str, str] = {
    "cpu": "CPUExecutionProvider",
    "dml": "DmlExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
}
_CPU_EP = "CPUExecutionProvider"


@dataclass(frozen=True)
class SessionHandle:
    """A ready ONNX Runtime session plus what actually backs it."""

    session: ort.InferenceSession
    provider: str          # friendly alias of the EP actually in use
    ep_name: str           # the ONNX Runtime EP name actually in use
    input_name: str
    fixed_input_size: int | None  # spatial size the model locks to, else None
    warmup_ms: float
    model_path: str


def available_provider_aliases() -> list[str]:
    """Friendly aliases whose EP is present in this onnxruntime build."""

    present = set(ort.get_available_providers())
    return [alias for alias, ep in PROVIDER_ALIASES.items() if ep in present]


def _fixed_size_from_shape(shape: list[object]) -> int | None:
    """Return the locked square spatial size, or ``None`` when it is dynamic."""

    if len(shape) != 4:
        return None
    h, w = shape[2], shape[3]
    if isinstance(h, int) and isinstance(w, int) and h == w and h > 0:
        return h
    return None


def create_session(
    model_path: str | Path,
    *,
    provider: str,
    input_size: int,
    warmup: bool = True,
    intra_op_threads: int = 0,
) -> SessionHandle:
    """Create one reusable inference session.

    Fails loudly (non-zero exit for a script) when: the model file is missing;
    ``provider`` is not a known alias or its EP is absent from this build; the
    requested EP does not end up active on the session; or ``input_size`` is one
    the model does not accept.
    """

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"perception model not found: {path} - run `python scripts/fetch_models.py`"
        )

    alias = provider.lower()
    if alias not in PROVIDER_ALIASES:
        raise ValueError(
            f"unknown perception provider {provider!r}; known: {sorted(PROVIDER_ALIASES)}"
        )
    ep = PROVIDER_ALIASES[alias]
    present = ort.get_available_providers()
    if ep not in present:
        raise ValueError(
            f"provider {alias!r} ({ep}) is not available in this onnxruntime build; "
            f"available: {present}. Install the matching onnxruntime package in a "
            f"separate environment (see BLOCK 9 / docs/decisions.md)."
        )

    # Request the chosen EP first, with CPU as a per-op fallback for unsupported
    # ops only. Whether the chosen EP is really active is verified below.
    requested = [ep] if ep == _CPU_EP else [ep, _CPU_EP]
    so = ort.SessionOptions()
    so.log_severity_level = 3  # warnings+; keep provider-init noise down
    if intra_op_threads and intra_op_threads > 0:
        so.intra_op_num_threads = int(intra_op_threads)
    session = ort.InferenceSession(str(path), sess_options=so, providers=requested)

    active = session.get_providers()
    if ep not in active:
        raise RuntimeError(
            f"requested provider {alias!r} ({ep}) did not initialise; session is "
            f"running on {active}. Refusing to silently use a different provider."
        )

    spec = session.get_inputs()[0]
    fixed = _fixed_size_from_shape(list(spec.shape))
    if fixed is not None and fixed != int(input_size):
        raise ValueError(
            f"model {path.name} locks its input to {fixed}x{fixed}; "
            f"config asks for {input_size}. Change perception input_size or the model."
        )
    run_size = fixed or int(input_size)

    warmup_ms = 0.0
    if warmup:
        blob = np.zeros((1, 3, run_size, run_size), dtype=np.float32)
        t0 = time.perf_counter()
        session.run(None, {spec.name: blob})
        warmup_ms = (time.perf_counter() - t0) * 1000.0

    _LOG.info(
        "perception session %s: provider requested=%s active=%s input=%s%s "
        "intra_op=%s warmup=%.1fms",
        path.name,
        alias,
        active[0],
        run_size,
        " (model-locked)" if fixed is not None else "",
        intra_op_threads or "auto",
        warmup_ms,
    )
    return SessionHandle(
        session=session,
        provider=alias,
        ep_name=active[0],
        input_name=spec.name,
        fixed_input_size=fixed,
        warmup_ms=warmup_ms,
        model_path=str(path),
    )
