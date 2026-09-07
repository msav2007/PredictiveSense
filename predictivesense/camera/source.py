"""The :class:`FrameSource` interface and a factory that honours phase discipline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from predictivesense.core.enums import SourceKind
from predictivesense.core.types import Frame

__all__ = ["FrameSource", "create_frame_source"]


class FrameSource(ABC):
    """A pull source of :class:`Frame` objects.

    ``stop()`` must be safe to call twice. Instances double as context managers:
    ``__enter__`` calls ``start()``, ``__exit__`` calls ``stop()``.
    """

    @abstractmethod
    def start(self) -> None:
        """Begin producing frames. Idempotent."""

    @abstractmethod
    def stop(self) -> None:
        """Stop producing frames and release resources. Safe to call twice."""

    @abstractmethod
    def read(self) -> Frame | None:
        """Return the next frame, or ``None`` when the source is not running."""

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """Whether the source is currently producing frames."""

    @abstractmethod
    def info(self) -> dict[str, Any]:
        """Static and counter metadata about this source."""

    def __enter__(self) -> "FrameSource":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()


def create_frame_source(source_config: Any) -> FrameSource:
    """Build a :class:`SyntheticSource` from a bare :class:`SourceConfig`.

    This minimal factory only handles ``SYNTHETIC``. ``DEVICE`` / ``FILE`` /
    ``BROWSER`` need parameters that ``SourceConfig`` does not carry (a device
    index and backend, a file path, ingest dimensions); those are constructed by
    :func:`predictivesense.pipeline.loop.build_loop` and
    :class:`predictivesense.pipeline.recorded.RecordedDriver` from the full
    ``AppConfig``. ``WEBRTC`` is never constructible.
    """

    kind = SourceKind(source_config.kind)
    if kind is not SourceKind.SYNTHETIC:
        raise NotImplementedError(
            f"create_frame_source builds only SYNTHETIC from a SourceConfig; "
            f"{kind.name} is built by pipeline.build_loop / RecordedDriver from "
            f"the full AppConfig (WEBRTC is never constructible)."
        )

    # Imported here to keep the interface module free of concrete dependencies.
    from predictivesense.camera.synthetic import SyntheticSource

    return SyntheticSource(
        width=source_config.width,
        height=source_config.height,
        target_fps=source_config.target_fps,
        seed=source_config.seed,
    )
