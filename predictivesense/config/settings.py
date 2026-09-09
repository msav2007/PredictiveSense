"""Typed settings, YAML profile loading, and strict validation.

An unknown profile, an unknown field, or a value that fails validation raises a
clear error. Nothing here falls back silently to a default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator
from pydantic import ValidationError as _PydanticValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from predictivesense.core.enums import Mode, SourceKind

__all__ = [
    "AppConfig",
    "ConfigError",
    "LoggingConfig",
    "SourceConfig",
    "ConsumerConfig",
    "BroadcastConfig",
    "ApiConfig",
    "NoopConfig",
    "CaptureConfig",
    "RecorderConfig",
    "VideoConfig",
    "DetectorConfig",
    "PoseConfig",
    "PerceptionConfig",
    "PolicyVocabularyConfig",
    "PolicyConfig",
    "DatasetConfig",
    "EvalConfig",
    "ObjectsConfig",
    "BatchConfig",
    "CoverageTargetsConfig",
    "StudioConfig",
    "PanelConfig",
    "UiConfig",
    "available_profiles",
    "load_config",
    "default_profiles_dir",
]


class ConfigError(RuntimeError):
    """Raised when a profile is missing, unreadable, or not a mapping."""


def default_profiles_dir() -> Path:
    """Repo-root ``config/profiles`` directory (this file is two packages deep)."""

    return Path(__file__).resolve().parents[2] / "config" / "profiles"


class _Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _tier_default(tier: str) -> tuple[str, ...]:
    """Default class list for one policy vocabulary tier.

    Deferred import: ``perception.vocabulary`` is stdlib-only but importing it
    at this module's load time would pull in ``perception/__init__`` ->
    ``perception.engine`` -> ``config.settings`` (a cycle). Called only from a
    ``default_factory``, i.e. at config instantiation, when both modules exist.
    """

    from predictivesense.perception import vocabulary as _vocab

    return {
        "primary": _vocab.PRIMARY_TIER,
        "secondary": _vocab.SECONDARY_TIER,
        "implausible": _vocab.IMPLAUSIBLE_TIER,
    }[tier]


class LoggingConfig(_Section):
    level: str = Field(default="INFO")

    def resolved_level(self) -> int:
        """Map the configured name to a :mod:`logging` numeric level, or fail."""

        import logging

        value = logging.getLevelName(self.level.upper())
        if not isinstance(value, int):
            raise ValueError(
                f"logging.level {self.level!r} is not a valid level name"
            )
        return value


class SourceConfig(_Section):
    kind: SourceKind = Field(default=SourceKind.SYNTHETIC)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    target_fps: float = Field(gt=0)
    seed: int = Field(ge=0)


class ConsumerConfig(_Section):
    sample_rate_hz: float = Field(gt=0)
    stale_after_ms: float = Field(gt=0)


class BroadcastConfig(_Section):
    rate_hz: float = Field(gt=0)
    client_send_timeout_s: float = Field(gt=0, default=5.0)


class ApiConfig(_Section):
    host: str = Field(default="127.0.0.1")
    port: int = Field(gt=0, lt=65536, default=8000)


class NoopConfig(_Section):
    default_seconds: int = Field(gt=0, default=60)
    csv_sample_period_s: float = Field(gt=0, default=1.0)
    max_rss_growth_mb: float = Field(gt=0, default=25.0)


class CaptureConfig(_Section):
    """Camera ownership and the browser-worker / backend-device capture path."""

    owner: Literal["browser", "backend"] = "browser"
    device_index: int = Field(ge=0, default=0)
    device_backend: Literal["auto", "msmf", "dshow"] = "auto"
    request_width: int = Field(gt=0, default=1280)
    request_height: int = Field(gt=0, default=720)
    request_fps: float = Field(gt=0, default=30.0)
    # "auto" leaves the backend to negotiate the pixel format (measured best on
    # the integrated camera - forcing MJPG gave no FPS gain and adds JPEG-decode
    # CPU). Set a 4-char code (e.g. "MJPG") only for a webcam that needs it.
    fourcc: str = Field(default="auto")
    # CAP_PROP_BUFFERSIZE. 1 = keep only the newest frame (lowest latency).
    # No-op on MSMF; honoured by some DSHOW / virtual-camera paths.
    buffer_size: int = Field(ge=1, default=1)
    # Frames to read and discard immediately after (re)open, for cameras that
    # emit a few dark/garbage frames on start. Measured unnecessary for the
    # integrated camera; left at 0.
    warmup_frames: int = Field(ge=0, default=0)
    open_timeout_s: float = Field(gt=0, default=5.0)
    reconnect_initial_s: float = Field(gt=0, default=0.5)
    reconnect_max_s: float = Field(gt=0, default=8.0)
    # Analysis path defaults chosen from the Phase 1.5 sweep
    # (results/analysis_sweep_loopback.md): 10 fps / 640x480 / q0.70 minimises
    # frame age and drops at acceptable quality - not the highest numbers.
    analysis_fps: float = Field(gt=0, default=10.0)
    analysis_width: int = Field(gt=0, default=640)
    analysis_height: int = Field(gt=0, default=480)
    analysis_jpeg_quality: float = Field(gt=0.0, le=1.0, default=0.7)
    max_ingest_message_bytes: int = Field(gt=0, default=2_000_000)
    # Worker backpressure ceiling: before sending an analysis frame the worker
    # checks socket.bufferedAmount and skips (never queues) the frame when it
    # exceeds this. The direct fix for frame age creeping upward under load.
    max_ws_buffered_bytes: int = Field(gt=0, default=1_000_000)

    @field_validator("fourcc")
    @classmethod
    def _check_fourcc(cls, value: str) -> str:
        if value != "auto" and len(value) != 4:
            raise ValueError("fourcc must be 'auto' or a 4-character code like 'MJPG'")
        return value


class RecorderConfig(_Section):
    """Raw full-quality clip recorder (browser MediaRecorder -> disk + manifest)."""

    enabled: bool = True
    max_clip_mb: float = Field(gt=0, default=500.0)
    output_dir: Path = Field(default=Path("data/raw"))


class VideoConfig(_Section):
    """Mode B: recorded video files analysed retrospectively by RecordedDriver."""

    input_dir: Path = Field(default=Path("data/videos"))
    replay_mode: Literal["realtime", "asfast"] = "asfast"


class DetectorConfig(_Section):
    """ONNX object detector. Model-agnostic: path, input size, thresholds and the
    NMS variant all come from here, so swapping the ONNX model is a config change,
    not a code change (see ``docs/decisions.md``)."""

    model_path: Path = Field(default=Path("models/yolo11n.onnx"))
    input_size: int = Field(gt=0, default=640)
    # Head-decode variant. "yolo" = Ultralytics YOLO11 detect head (RGB, /255,
    # centred letterbox). "yolox" = Megvii YOLOX raw head (BGR, no /255,
    # top-left letterbox, grid decode) - the Phase 2.5 permissive-licence
    # comparison model. Adding this variant does not change the "yolo" path.
    decode: Literal["yolo", "yolox"] = "yolo"
    nms_iou: float = Field(gt=0.0, le=1.0, default=0.5)
    # Documented default; per-class overrides live in ``class_thresholds`` and are
    # set from the class-coverage audit. A single global threshold is not enough.
    default_conf: float = Field(gt=0.0, le=1.0, default=0.35)
    class_thresholds: dict[str, float] = Field(default_factory=dict)
    # Detections whose score falls in this half-open band render dashed / dimmed.
    low_confidence_band: tuple[float, float] = Field(default=(0.35, 0.50))
    max_detections: int = Field(gt=0, default=100)

    @field_validator("input_size")
    @classmethod
    def _multiple_of_stride(cls, value: int) -> int:
        if value % 32 != 0:
            raise ValueError("detector input_size must be a multiple of 32")
        return value

    @field_validator("low_confidence_band")
    @classmethod
    def _band_ordered(cls, value: tuple[float, float]) -> tuple[float, float]:
        lo, hi = value
        if not (0.0 <= lo <= hi <= 1.0):
            raise ValueError("low_confidence_band must be [lo, hi] with 0 <= lo <= hi <= 1")
        return value

    @field_validator("class_thresholds")
    @classmethod
    def _thresholds_in_range(cls, value: dict[str, float]) -> dict[str, float]:
        for name, thr in value.items():
            if not (0.0 < thr <= 1.0):
                raise ValueError(f"class_thresholds[{name!r}] must be in (0, 1]")
        return value


class PoseConfig(_Section):
    """ONNX pose estimator. Same model-agnostic rule as :class:`DetectorConfig`."""

    model_path: Path = Field(default=Path("models/yolo11n-pose.onnx"))
    input_size: int = Field(gt=0, default=640)
    conf: float = Field(gt=0.0, le=1.0, default=0.4)
    max_persons: int = Field(gt=0, default=4)
    keypoint_visibility_threshold: float = Field(ge=0.0, le=1.0, default=0.3)

    @field_validator("input_size")
    @classmethod
    def _multiple_of_stride(cls, value: int) -> int:
        if value % 32 != 0:
            raise ValueError("pose input_size must be a multiple of 32")
        return value


class PerceptionConfig(_Section):
    """Phase 2 per-frame perception. With both ``*_enabled`` off the pipeline
    behaves exactly as Phase 1.6 (no models loaded, empty detections/poses)."""

    detection_enabled: bool = True
    pose_enabled: bool = True
    # Run pose only every Nth sampled frame if the latency budget demands it.
    # Measure before raising this above 1.
    pose_every_n: int = Field(ge=1, default=1)
    # ONNX Runtime intra-op threads per session. 0 = ORT default (all cores),
    # which OVER-subscribes badly with two sessions + the loop's other threads
    # (measured: combined detector+pose p50 collapses from ~90 ms to ~440 ms
    # under contention). 6 is the measured knee on this 14C/18T machine.
    intra_op_threads: int = Field(ge=0, default=6)
    # Chosen from results/providers_*.json. "cpu" is plain onnxruntime; "dml"
    # requires onnxruntime-directml in a separate environment (never the main
    # .venv); "openvino" is optional/deferred.
    provider: Literal["cpu", "dml", "openvino"] = "cpu"
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    pose: PoseConfig = Field(default_factory=PoseConfig)

    @property
    def any_enabled(self) -> bool:
        return self.detection_enabled or self.pose_enabled

    # -- Phase 2.5: person-gated pose --------------------------------
    # Measured before/after in docs/phase-reports/phase2_5.md. When true, pose
    # runs on a sampled frame only if the detector found at least one `person`
    # on that frame (still also subject to pose_every_n). Cuts combined
    # per-frame cost on frames with no person without losing any real skeleton.
    pose_requires_person: bool = False


class PolicyVocabularyConfig(_Section):
    """Phase 5 three-tier class vocabulary - replaces the Phase 2.5
    ``domain_classes`` whitelist.

    ``primary`` + ``secondary`` + ``implausible`` must be a **total, disjoint
    partition** of the shipped detector's class list (COCO-80). A typo, a
    duplicate, an overlap, or an unassigned class fails loudly at config load
    (BLOCK 10). The defaults are an environment-specific judgement, not a
    measured result (``docs/decisions.md``); revising them is cheap.
    """

    # The canonical tier lists live in perception.vocabulary (stdlib-only). They
    # are pulled in via default_factory (at instantiation, not class-body) to
    # avoid a settings <- perception import cycle at module load.
    primary: tuple[str, ...] = Field(default_factory=lambda: _tier_default("primary"))
    secondary: tuple[str, ...] = Field(default_factory=lambda: _tier_default("secondary"))
    implausible: tuple[str, ...] = Field(default_factory=lambda: _tier_default("implausible"))

    @model_validator(mode="after")
    def _partition_is_total_and_disjoint(self) -> "PolicyVocabularyConfig":
        from predictivesense.perception.vocabulary import validate_partition

        validate_partition(self.primary, self.secondary, self.implausible)
        return self


class PolicyConfig(_Section):
    """Recognition policy layer (Phase 2.5, redesigned in Phase 5) - sits AFTER
    the detector and BEFORE the snapshot. Every rule is independently switchable
    and every outcome is counted; the six ``policy_state`` values reconcile
    exactly (every input detection maps to one). ``enabled=False`` is a
    pass-through and the pipeline behaves exactly as Phase 2 did.

    Phase 5: the binary ``domain_classes`` whitelist is replaced by three tiers
    (``vocabulary``). ``suppressed_implausible`` (a confident prediction of a
    class that cannot be here) is now distinct from ``unknown`` (the model is
    unsure *what* the object is). ``thresholds_fitted`` is ``False`` until
    ``scripts/fit_thresholds.py`` runs on a labelled ``val`` split; while it is
    ``False`` the UI states plainly that the thresholds are unfitted defaults.
    """

    enabled: bool = True
    # Phase 5: suppress the `implausible` tier (was: reject out-of-domain).
    domain_restriction: bool = True
    margin_rule: bool = True
    size_rule: bool = True
    per_class_threshold_rule: bool = True
    vocabulary: PolicyVocabularyConfig = Field(default_factory=PolicyVocabularyConfig)
    # Set true ONLY by scripts/fit_thresholds.py after a labelled `val` split
    # exists. While false, per_class_thresholds / default_threshold / margin_min
    # are unfitted defaults and the UI says so (BLOCK 3.5).
    thresholds_fitted: bool = False
    per_class_thresholds: dict[str, float] = Field(default_factory=dict)
    default_threshold: float = Field(gt=0.0, le=1.0, default=0.35)
    # top1 - top2 below this -> emit `unknown` instead of the top-1 guess.
    margin_min: float = Field(ge=0.0, le=1.0, default=0.10)
    min_box_area_frac: float = Field(ge=0.0, le=1.0, default=0.0005)
    # A rejected-but-real detection keeps its box and renders as `unknown`
    # (BLOCK 3.14). Setting this false discards the box - not recommended.
    emit_unknown: bool = True
    # Reveal `suppressed_implausible` detections in the Diagnostics-only overlay
    # toggle. Never on the main overlay (BLOCK 3.8).
    show_suppressed_in_diagnostics: bool = True
    # Per-class aspect-ratio sanity bound (w/h), applied only to classes listed.
    # Empty by default - a judgement call, documented in decisions.md.
    aspect_ratio_bounds: dict[str, tuple[float, float]] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def domain_classes(self) -> tuple[str, ...]:
        """Back-compat alias: the classes the MVP scenarios depend on and that
        the evaluation set / threshold fitting are scoped to = the ``primary``
        tier. Kept so the dataset store, eval harness and ``/api/config``
        consumers do not need reshaping (Phase 0 additive rule)."""

        return tuple(self.vocabulary.primary)

    @field_validator("per_class_thresholds")
    @classmethod
    def _pct_in_range(cls, value: dict[str, float]) -> dict[str, float]:
        for name, thr in value.items():
            if not (0.0 < thr <= 1.0):
                raise ValueError(f"per_class_thresholds[{name!r}] must be in (0, 1]")
        return value

    @field_validator("aspect_ratio_bounds")
    @classmethod
    def _aspect_bounds_ordered(
        cls, value: dict[str, tuple[float, float]]
    ) -> dict[str, tuple[float, float]]:
        for name, (lo, hi) in value.items():
            if not (0.0 < lo <= hi):
                raise ValueError(
                    f"aspect_ratio_bounds[{name!r}] must be (lo, hi) with 0 < lo <= hi"
                )
        return value


class DatasetConfig(_Section):
    """Phase 2.5 labelled evaluation set (COCO detection JSON)."""

    root: Path = Field(default=Path("data/eval"))
    coco_path: Path = Field(default=Path("data/eval/annotations.json"))
    splits_path: Path = Field(default=Path("data/eval/splits.json"))
    frames_dirname: str = Field(default="frames")
    # At least this fraction of labelled frames must be labelled with detector
    # seeding OFF, as a recall-bias check (BLOCK 3.1.4).
    min_unseeded_fraction: float = Field(ge=0.0, le=1.0, default=0.30)


class EvalConfig(_Section):
    """Phase 2.5 detection evaluation harness."""

    iou_threshold: float = Field(gt=0.0, le=1.0, default=0.5)
    results_dir: Path = Field(default=Path("results"))
    # Session fractions for scripts/build_splits.py (test is opened once).
    val_fraction: float = Field(gt=0.0, lt=1.0, default=0.5)
    test_fraction: float = Field(gt=0.0, lt=1.0, default=0.5)
    split_seed: int = Field(ge=0, default=20260908)


class CoverageTargetsConfig(_Section):
    """Phase 4 data-collection guidance thresholds.

    These are **heuristics chosen for data-collection guidance, not scientific
    quality metrics** (``docs/decisions.md``). They drive the Studio's
    "3 views, no far-distance samples" prompts; they are not a pass/fail gate and
    there is no composite score.
    """

    min_positives: int = Field(gt=0, default=20)
    min_views: int = Field(gt=0, default=4)
    min_distances: int = Field(gt=0, default=2)
    min_lighting: int = Field(gt=0, default=2)
    min_occluded: int = Field(ge=0, default=2)
    min_hard_negatives: int = Field(ge=0, default=5)


class BatchConfig(_Section):
    """Phase 7 bulk image upload - a data-collection accelerator for the Object
    Learning Studio. Boxes are *proposed* from the existing detector's raw
    output; nothing here trains, fine-tunes or activates a model."""

    max_images: int = Field(gt=0, default=60)
    max_total_mb: float = Field(gt=0, default=400.0)
    staging_ttl_hours: float = Field(gt=0, default=24.0)
    # Deliberately low - a proposal only needs a rectangle, not a confident
    # class. The detector's predicted label is stored as a hint, never used as
    # the sample's class (see docs/decisions.md).
    proposal_min_score: float = Field(ge=0.0, le=1.0, default=0.10)
    thumbnail_px: int = Field(gt=0, default=240)
    # An image whose shorter side is below this is rejected with a reason.
    min_image_px: int = Field(gt=0, default=32)


class ObjectsConfig(_Section):
    """Phase 4 Object Learning Studio - environment-specific object data
    collection. Storing images is not training; nothing here trains a model."""

    root: Path = Field(default=Path("data/objects"))
    max_image_mb: float = Field(gt=0, default=12.0)
    thumbnail_px: int = Field(gt=0, default=240)
    # Heuristic guidance threshold - variance-of-Laplacian below this marks a
    # sample "blurry" for the developer to review (not auto-deleted). See
    # docs/decisions.md.
    blur_var_min: float = Field(gt=0.0, default=60.0)
    min_box_area_frac: float = Field(gt=0.0, le=1.0, default=0.01)
    # dHash Hamming distance at/below which two samples of the same object are
    # flagged near-duplicates.
    duplicate_hamming_max: int = Field(ge=0, default=6)
    coverage_targets: CoverageTargetsConfig = Field(default_factory=CoverageTargetsConfig)
    batch: BatchConfig = Field(default_factory=BatchConfig)


class StudioConfig(_Section):
    """Phase 4 Studio lifecycle."""

    # Entering /studio stops the monitoring pipeline (worker terminated, ingest
    # socket closed, perception disabled, analysis loop paused). Leaving restores
    # it. Off is unsupported by the UI and only exists for tests.
    stop_monitoring_on_enter: bool = True


class PanelConfig(_Section):
    """Phase 4 operations-panel resize constraints (Block 4.8)."""

    default_width_px: int = Field(gt=0, default=380)
    min_width_px: int = Field(gt=0, default=300)
    max_width_px: int = Field(gt=0, default=560)
    # The panel also never exceeds this fraction of the window, and the viewport
    # never drops below (1 - this) - 0.05 of the window.
    max_width_frac: float = Field(gt=0.0, lt=1.0, default=0.40)

    @field_validator("max_width_px")
    @classmethod
    def _max_ge_min(cls, value: int, info: Any) -> int:
        min_px = info.data.get("min_width_px")
        if isinstance(min_px, int) and value < min_px:
            raise ValueError("ui.panel.max_width_px must be >= min_width_px")
        return value


class UiConfig(_Section):
    """Phase 4 UI state served to the dashboard shell."""

    panel: PanelConfig = Field(default_factory=PanelConfig)


class AppConfig(BaseSettings):
    """The fully resolved configuration for one run.

    Frozen once built. Environment overrides use the ``PS_`` prefix and ``__`` as
    the nested delimiter (e.g. ``PS_API__PORT=9001``).
    """

    model_config = SettingsConfigDict(
        frozen=True,
        extra="forbid",
        env_prefix="PS_",
        env_nested_delimiter="__",
    )

    profile: str
    mode: Mode
    results_dir: Path = Field(default=Path("results"))
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    source: SourceConfig
    consumer: ConsumerConfig
    broadcast: BroadcastConfig
    api: ApiConfig = Field(default_factory=ApiConfig)
    noop: NoopConfig = Field(default_factory=NoopConfig)
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    recorder: RecorderConfig = Field(default_factory=RecorderConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    objects: ObjectsConfig = Field(default_factory=ObjectsConfig)
    studio: StudioConfig = Field(default_factory=StudioConfig)
    ui: UiConfig = Field(default_factory=UiConfig)

    def as_json_dict(self) -> dict[str, Any]:
        """Resolved config as a JSON-serialisable dict (for the manifest and API)."""

        return self.model_dump(mode="json")


def available_profiles(profiles_dir: Path | None = None) -> list[str]:
    """Names (without extension) of every ``*.yaml`` profile on disk, sorted."""

    directory = profiles_dir or default_profiles_dir()
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.yaml"))


def load_config(
    profile: str,
    *,
    overrides: dict[str, Any] | None = None,
    profiles_dir: Path | None = None,
) -> AppConfig:
    """Load and validate a named profile.

    Raises :class:`ConfigError` for an unknown or malformed profile file, and
    :class:`pydantic.ValidationError` for an unknown field or a value that fails
    validation (including an invalid ``mode``).
    """

    directory = profiles_dir or default_profiles_dir()
    path = directory / f"{profile}.yaml"
    if not path.is_file():
        known = available_profiles(directory)
        raise ConfigError(
            f"unknown config profile {profile!r}; expected one of {known or '(none found)'} "
            f"under {directory}"
        )

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # malformed YAML
        raise ConfigError(f"profile {path} is not valid YAML: {exc}") from exc

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"profile {path} must contain a mapping at the top level")

    data: dict[str, Any] = dict(raw)
    if overrides:
        data.update(overrides)
    data["profile"] = profile

    # ValidationError intentionally propagates to the caller.
    return AppConfig(**data)


# Re-export so callers can catch a single symbol.
ValidationError = _PydanticValidationError
