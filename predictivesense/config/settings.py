"""Typed settings, YAML profile loading, and strict validation.

An unknown profile, an unknown field, or a value that fails validation raises a
clear error. Nothing here falls back silently to a default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
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
    open_timeout_s: float = Field(gt=0, default=5.0)
    reconnect_initial_s: float = Field(gt=0, default=0.5)
    reconnect_max_s: float = Field(gt=0, default=8.0)
    analysis_fps: float = Field(gt=0, default=10.0)
    analysis_width: int = Field(gt=0, default=640)
    analysis_height: int = Field(gt=0, default=480)
    analysis_jpeg_quality: float = Field(gt=0.0, le=1.0, default=0.7)
    max_ingest_message_bytes: int = Field(gt=0, default=2_000_000)


class RecorderConfig(_Section):
    """Raw full-quality clip recorder (browser MediaRecorder -> disk + manifest)."""

    enabled: bool = True
    max_clip_mb: float = Field(gt=0, default=500.0)
    output_dir: Path = Field(default=Path("data/raw"))


class VideoConfig(_Section):
    """Mode B: recorded video files analysed retrospectively by RecordedDriver."""

    input_dir: Path = Field(default=Path("data/videos"))
    replay_mode: Literal["realtime", "asfast"] = "asfast"


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
