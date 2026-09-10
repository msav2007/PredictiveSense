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
    "AUTO_PREFERENCE_ORDER",
    "SessionHandle",
    "create_session",
    "available_provider_aliases",
    "resolve_provider",
    "ProviderUnavailableError",
]

_LOG = get_logger(__name__)

# Friendly name (config ``perception.provider``) -> ONNX Runtime EP name.
# ``directml`` and ``dml`` are the same EP (``dml`` kept for the historical
# ``.venv-dml`` benchmark path and ``results/providers_dml.json``).
PROVIDER_ALIASES: dict[str, str] = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "directml": "DmlExecutionProvider",
    "dml": "DmlExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
}
_CPU_EP = "CPUExecutionProvider"

# ``perception.provider = "auto"`` tries these in order and takes the first whose
# EP is present in this onnxruntime build; CPU is always the final fallback.
# Rationale: on an NVIDIA laptop CUDA should win; DirectML is a distant second
# (measured ~2.2x slower than CPU for these nano models on this machine's Arc
# iGPU - so ``auto`` only picks it when nothing better exists, and a developer
# who wants to benchmark it sets ``provider: directml`` explicitly).
AUTO_PREFERENCE_ORDER: tuple[str, ...] = ("cuda", "directml", "cpu")

_UNAVAILABLE_HINT: dict[str, str] = {
    "cuda": (
        "install the CUDA build in a separate environment: "
        "`pip install -e \".[cuda]\"` (onnxruntime-gpu; needs a matching CUDA + "
        "cuDNN runtime - see docs/setup.md). onnxruntime, onnxruntime-gpu and "
        "onnxruntime-directml share the module name and cannot coexist."
    ),
    "directml": (
        "install onnxruntime-directml in a separate environment (never the main "
        ".venv); see docs/setup.md."
    ),
    "dml": "install onnxruntime-directml in a separate environment; see docs/setup.md.",
    "openvino": "onnxruntime-openvino is optional/deferred; not supported here.",
}


class ProviderUnavailableError(RuntimeError):
    """An explicitly requested execution provider is absent from this ORT build.

    Never raised for ``auto`` (which falls back to CPU) - only when the config
    named a specific GPU provider that is not installed. The message is
    actionable. Silent fallback is how a "GPU result" turns out to have been CPU.
    """


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


def available_execution_providers() -> list[str]:
    """Raw ONNX Runtime EP names present in this build (keeps ``onnxruntime``
    imports scoped to ``perception/`` - callers elsewhere use this)."""

    return list(ort.get_available_providers())


def onnxruntime_version() -> str:
    """The installed ``onnxruntime`` version string."""

    return str(getattr(ort, "__version__", "unknown"))


def available_provider_aliases() -> list[str]:
    """Friendly aliases whose EP is present in this onnxruntime build (one entry
    per distinct EP - ``dml``/``directml`` collapse to one)."""

    present = set(ort.get_available_providers())
    seen: set[str] = set()
    out: list[str] = []
    for alias, ep in PROVIDER_ALIASES.items():
        if ep in present and ep not in seen:
            out.append(alias)
            seen.add(ep)
    return out


def resolve_provider(requested: str) -> tuple[str, str]:
    """Map a config ``perception.provider`` value to a concrete alias.

    Returns ``(alias, reason)``. ``"auto"`` picks the first available provider in
    :data:`AUTO_PREFERENCE_ORDER`, always ending at CPU, and logs the choice and
    why. An explicitly named provider whose EP is absent raises
    :class:`ProviderUnavailableError` with an actionable message - **never** a
    silent fallback.
    """

    req = (requested or "auto").lower()
    present = set(ort.get_available_providers())

    if req == "auto":
        for alias in AUTO_PREFERENCE_ORDER:
            if PROVIDER_ALIASES[alias] in present:
                reason = (
                    f"provider=auto selected {alias!r} "
                    f"({PROVIDER_ALIASES[alias]} available); order tried: "
                    f"{', '.join(AUTO_PREFERENCE_ORDER)}"
                )
                _LOG.info("%s", reason)
                return alias, reason
        reason = (
            f"provider=auto fell back to 'cpu' (no GPU EP present; "
            f"available: {sorted(present)})"
        )
        _LOG.info("%s", reason)
        return "cpu", reason

    if req not in PROVIDER_ALIASES:
        raise ValueError(
            f"unknown perception provider {requested!r}; "
            f"expected one of: auto, cpu, cuda, directml"
        )
    ep = PROVIDER_ALIASES[req]
    if ep not in present:
        raise ProviderUnavailableError(
            f"perception.provider={requested!r} was requested explicitly but its "
            f"execution provider {ep} is not in this onnxruntime build "
            f"(available: {sorted(present)}). {_UNAVAILABLE_HINT.get(req, '')} "
            f"Set perception.provider: auto to fall back to CPU automatically."
        )
    reason = f"provider={req!r} requested explicitly; {ep} is available"
    _LOG.info("%s", reason)
    return req, reason


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

    # ``auto`` -> concrete alias (documented order, CPU fallback, logged);
    # an explicitly requested but absent GPU provider -> ProviderUnavailableError.
    alias, _reason = resolve_provider(provider)
    ep = PROVIDER_ALIASES[alias]

    # Request the chosen EP first, with CPU as a per-op fallback for unsupported
    # ops only. Whether the chosen EP is really active is verified below. For
    # CUDA a bad CUDA/cuDNN pairing typically makes the EP silently absent from
    # get_providers() below rather than raising - that path is turned into a
    # clear diagnostic, not a stack trace.
    requested = [ep] if ep == _CPU_EP else [ep, _CPU_EP]
    so = ort.SessionOptions()
    so.log_severity_level = 3  # warnings+; keep provider-init noise down
    # Thread options are set EXPLICITLY (BLOCK 3.5.21), not left to ORT's
    # defaults: two sessions + the loop's own threads over-subscribe an 18-thread
    # machine badly (measured in Phase 2). intra_op is swept; inter_op is pinned
    # to 1 and the graph runs sequentially - there is only one output to produce
    # per call, so parallel op scheduling only adds contention here.
    if intra_op_threads and intra_op_threads > 0:
        so.intra_op_num_threads = int(intra_op_threads)
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(path), sess_options=so, providers=requested)

    active = session.get_providers()
    if ep not in active:
        extra = ""
        if alias == "cuda":
            extra = (
                " This is usually a CUDA/cuDNN runtime-version mismatch for this "
                "onnxruntime-gpu build - check the versions in docs/setup.md "
                "against `nvcc --version` / the installed cuDNN."
            )
        raise RuntimeError(
            f"requested provider {alias!r} ({ep}) did not initialise; session is "
            f"running on {active}. Refusing to silently use a different provider.{extra}"
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
