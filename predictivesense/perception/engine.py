"""Orchestrates the detector and the pose estimator over one sampled frame.

Both models run on the **same** frame inside the existing analysis loop and the
recorded driver - identical code for Mode A and Mode B. A single-frame inference
exception is counted and that frame's results are dropped; the loop stays alive
(Block 7). Sessions are created once, here, and reused.
"""

from __future__ import annotations

import time
from pathlib import Path

from predictivesense.config.settings import AppConfig, PerceptionConfig
from predictivesense.core.types import Frame
from predictivesense.logging_setup import get_logger
from predictivesense.perception.types import PerceptionResult

__all__ = ["PerceptionEngine", "build_perception", "MissingModelError"]

_LOG = get_logger(__name__)


class MissingModelError(FileNotFoundError):
    """A configured perception model file is not on disk (strict callers only)."""


class PerceptionEngine:
    """Holds the (optional) detector and pose sessions and runs them per frame."""

    def __init__(
        self,
        config: PerceptionConfig,
        *,
        warmup: bool = True,
    ) -> None:
        self._config = config
        self._detector = None
        self._pose = None
        self._frame_counter = -1
        self._detector_failures = 0
        self._pose_failures = 0
        self._pose_gated_skips = 0  # pose skipped: pose_requires_person, no person
        # Phase 8 pose-cadence state. Single-owner (the analysis consumer thread
        # for real-time, the recorded driver for Mode B) - no lock: the reused
        # value is an immutable tuple of frozen Pose objects, swapped by
        # reassignment, never mutated in place.
        self._pose_cadence_kind, self._pose_cadence_value = config.pose_cadence_spec()
        self._pose_max_reuse_ms = float(config.pose_max_reuse_ms)
        self._last_poses: tuple = ()
        self._last_pose_frame_id: int | None = None
        self._last_pose_capture_ts: float | None = None
        self._pose_reuses = 0

        threads = config.intra_op_threads
        if config.detection_enabled:
            from predictivesense.perception.detector import ObjectDetector

            self._detector = ObjectDetector(
                config.detector,
                provider=config.provider,
                warmup=warmup,
                intra_op_threads=threads,
            )
        if config.pose_enabled:
            from predictivesense.perception.pose import PoseEstimator

            self._pose = PoseEstimator(
                config.pose,
                provider=config.provider,
                warmup=warmup,
                intra_op_threads=threads,
            )
        _LOG.info(
            "perception engine: detection=%s pose=%s provider=%s pose_every_n=%d",
            config.detection_enabled,
            config.pose_enabled,
            config.provider,
            config.pose_every_n,
        )

    # -- inference -------------------------------------------------

    def infer(self, frame: Frame, *, frame_index: int | None = None) -> PerceptionResult:
        """Run detection and (optionally, per ``pose_every_n``) pose for a frame."""

        self._frame_counter += 1
        idx = self._frame_counter if frame_index is None else int(frame_index)

        detections: tuple = ()
        poses: tuple = ()
        detector_ms: float | None = None
        pose_ms: float | None = None
        pose_ran = False
        frame_error = False
        notes: list[str] = []

        if self._detector is not None:
            t0 = time.perf_counter()
            try:
                detections = tuple(self._detector.infer(frame))
                detector_ms = (time.perf_counter() - t0) * 1000.0
            except Exception as exc:  # noqa: BLE001 - drop this frame, keep the loop
                self._detector_failures += 1
                frame_error = True
                notes.append(f"detector: {exc!r}")
                _LOG.warning("detector inference failed on frame %s: %r", frame.frame_id, exc)

        pose_due = self._pose is not None and self._pose_cadence_due(idx, frame.capture_ts)
        if pose_due and self._config.pose_requires_person:
            has_person = any(d.class_name == "person" for d in detections)
            if not has_person:
                pose_due = False
                self._pose_gated_skips += 1
        pose_reused = False
        pose_frame_id: int | None = None
        pose_capture_ts: float | None = None
        pose_age_ms: float | None = None
        if pose_due:
            pose_ran = True
            t0 = time.perf_counter()
            try:
                poses = tuple(self._pose.infer(frame))
                pose_ms = (time.perf_counter() - t0) * 1000.0
                self._last_poses = poses
                self._last_pose_frame_id = frame.frame_id
                self._last_pose_capture_ts = frame.capture_ts
            except Exception as exc:  # noqa: BLE001
                self._pose_failures += 1
                frame_error = True
                notes.append(f"pose: {exc!r}")
                _LOG.warning("pose inference failed on frame %s: %r", frame.frame_id, exc)
        elif self._pose is not None and self._last_poses and self._last_pose_capture_ts is not None:
            # Reuse the last pose iff it is still within pose_max_reuse_ms of this
            # frame's capture time (frame-time delta -> deterministic in Mode B).
            age_ms = (frame.capture_ts - self._last_pose_capture_ts) * 1000.0
            if 0.0 <= age_ms <= self._pose_max_reuse_ms:
                poses = self._last_poses  # same frozen objects; never mutated
                pose_reused = True
                pose_frame_id = self._last_pose_frame_id
                pose_capture_ts = self._last_pose_capture_ts
                pose_age_ms = age_ms
                self._pose_reuses += 1
            # else: beyond the reuse bound -> no pose, rather than a wrong one.

        return PerceptionResult(
            detections=detections,
            poses=poses,
            detector_ms=detector_ms,
            pose_ms=pose_ms,
            pose_ran=pose_ran,
            frame_error=frame_error,
            notes=tuple(notes),
            pose_reused=pose_reused,
            pose_frame_id=pose_frame_id,
            pose_capture_ts=pose_capture_ts,
            pose_age_ms=pose_age_ms,
        )

    def _pose_cadence_due(self, idx: int, capture_ts: float) -> bool:
        """True when a fresh pose inference is due for this frame."""

        kind, value = self._pose_cadence_kind, self._pose_cadence_value
        if kind == "every_frame":
            return True
        if kind == "every_n":
            return idx % int(value) == 0
        # interval_ms: due when no pose yet, or enough capture time has passed.
        if self._last_pose_capture_ts is None:
            return True
        return (capture_ts - self._last_pose_capture_ts) * 1000.0 >= float(value)

    def detect(self, frame: Frame) -> list:
        """Raw detector output for one frame - no pose, no recognition policy.

        Used by the Phase 7 bulk-upload box proposer, which wants a rectangle,
        not a classification, and must see the class the model actually emitted
        (the policy would suppress exactly the box we want). Returns ``[]`` when
        detection is disabled. Does not touch the per-frame counters used by
        :meth:`infer`.
        """

        if self._detector is None:
            return []
        return list(self._detector.infer(frame))

    # -- introspection (Diagnostics rows, phase report) ------------

    def info(self) -> dict[str, object]:
        out: dict[str, object] = {
            "detection_enabled": self._detector is not None,
            "pose_enabled": self._pose is not None,
            "provider": self._config.provider,
            "pose_every_n": self._config.pose_every_n,
            "pose_cadence": self._config.pose_cadence,
            "pose_cadence_resolved": f"{self._pose_cadence_kind}:{self._pose_cadence_value}",
            "pose_max_reuse_ms": self._pose_max_reuse_ms,
            "pose_reuses": self._pose_reuses,
            "pose_requires_person": self._config.pose_requires_person,
            "pose_gated_skips": self._pose_gated_skips,
            "detector_failures": self._detector_failures,
            "pose_failures": self._pose_failures,
        }
        if self._detector is not None:
            out.update(
                detector_model=self._detector.model_name,
                detector_input_size=self._detector.input_size,
                detector_provider=self._detector.provider,
                detector_ep=self._detector.ep_name,
                detector_warmup_ms=round(self._detector.warmup_ms, 1),
                detector_classes=len(self._detector.class_names),
            )
        if self._pose is not None:
            out.update(
                pose_model=self._pose.model_name,
                pose_input_size=self._pose.input_size,
                pose_provider=self._pose.provider,
                pose_ep=self._pose.ep_name,
                pose_warmup_ms=round(self._pose.warmup_ms, 1),
            )
        return out


def _missing_model_paths(config: PerceptionConfig) -> list[Path]:
    missing: list[Path] = []
    if config.detection_enabled and not Path(config.detector.model_path).is_file():
        missing.append(Path(config.detector.model_path))
    if config.pose_enabled and not Path(config.pose.model_path).is_file():
        missing.append(Path(config.pose.model_path))
    return missing


def build_perception(
    config: AppConfig | PerceptionConfig,
    *,
    strict: bool = False,
    warmup: bool = True,
) -> PerceptionEngine | None:
    """Build the engine, or ``None`` when perception is off / its weights are absent.

    ``strict=True`` (scripts, ``models``-marked tests) raises
    :class:`MissingModelError` when a configured weight file is missing.
    ``strict=False`` (the analysis loop, ``POST /api/analyze``) logs one warning
    and returns ``None`` so a missing model degrades to Phase 1.6 behaviour
    rather than killing the loop.
    """

    perception = config.perception if isinstance(config, AppConfig) else config
    if not perception.any_enabled:
        return None

    missing = _missing_model_paths(perception)
    if missing:
        names = ", ".join(str(p) for p in missing)
        if strict:
            raise MissingModelError(
                f"perception is enabled but model file(s) missing: {names}. "
                f"Run `python scripts/fetch_models.py`."
            )
        _LOG.warning(
            "perception enabled but model file(s) missing (%s); running WITHOUT "
            "perception. Run `python scripts/fetch_models.py` to enable it.",
            names,
        )
        return None

    return PerceptionEngine(perception, warmup=warmup)
