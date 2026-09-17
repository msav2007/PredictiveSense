"""A tiny, real, on-disk Object Learning Studio dataset for tests and for
proving the training pipeline end-to-end before any real collection exists
(Phase 11 Part B section 12.4).

This is not a parallel/fake format: every sample is written through the real
:class:`~predictivesense.objects.registry.ObjectRegistry` /
:class:`~predictivesense.objects.samples.SampleStore` write path, so the
resulting ``data/objects``-shaped tree is byte-for-byte what the Studio itself
would produce, and the SAME export code (`training/dataset.py`) reads it back
with no special-casing.

Each class is a distinct, simple, genuinely-learnable shape (filled circle /
square / triangle in a class-specific colour) drawn with OpenCV (already a
repo dependency - no new one for a test fixture) at a small resolution -
enough for a tiny CNN to actually separate the classes within a few CPU
seconds, which is the point: this proves the pipeline trains something real,
not that it is a good vision model.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np

from predictivesense.objects.registry import ObjectRegistry
from predictivesense.objects.samples import SampleStore
from predictivesense.telemetry.manifest import utc_now_iso

__all__ = ["FIXTURE_SHAPES", "build_synthetic_object_store"]

FIXTURE_SHAPES: tuple[str, ...] = ("circle", "square", "triangle")
_IMAGE_SIZE = 64
_SESSION_GAP_S = 1200  # > the 900s session-bucket window, so sessions differ


def _draw_shape(shape: str, *, size: int, color: tuple[int, int, int], rng: np.random.Generator) -> np.ndarray:
    img = np.full((size, size, 3), 24, dtype=np.uint8)  # near-black background
    cx, cy = size // 2 + int(rng.integers(-4, 5)), size // 2 + int(rng.integers(-4, 5))
    r = size // 3
    if shape == "circle":
        cv2.circle(img, (cx, cy), r, color, thickness=-1)
    elif shape == "square":
        cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), color, thickness=-1)
    elif shape == "triangle":
        pts = np.array(
            [[cx, cy - r], [cx - r, cy + r], [cx + r, cy + r]], dtype=np.int32
        )
        cv2.fillPoly(img, [pts], color)
    else:
        raise ValueError(f"unknown fixture shape {shape!r}")
    noise = rng.integers(0, 10, size=img.shape, dtype=np.uint8)
    return cv2.add(img, noise)


def _bbox_for(shape: str, size: int) -> list[float]:
    # A generous fixed box around the shape's drawing region (rng jitter above
    # stays well inside it) - real-enough for a "one box per image" sample.
    margin = size // 6
    return [float(margin), float(margin), float(size - 2 * margin), float(size - 2 * margin)]


def build_synthetic_object_store(
    root: str | Path,
    *,
    n_per_class: int = 16,
    n_sessions_per_class: int = 3,
    seed: int = 0,
) -> list[str]:
    """Populate ``root`` (an empty or fresh directory) with one Studio object
    profile per :data:`FIXTURE_SHAPES` entry, each with real, committed
    samples spread across several distinct capture sessions. Returns the
    created ``object_id`` list (== the fixture class names, already valid
    slugs)."""

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    reg = ObjectRegistry(root)
    rng = np.random.default_rng(seed)
    colors = {
        "circle": (60, 180, 250),   # BGR: warm orange
        "square": (90, 220, 90),    # green
        "triangle": (220, 90, 200),  # magenta
    }

    object_ids: list[str] = []
    base_time = utc_now_iso()
    from datetime import datetime, timezone

    base_dt = datetime.strptime(base_time, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)

    for shape in FIXTURE_SHAPES:
        profile = reg.create(name=shape, kind="class", category="fixture")
        object_ids.append(profile.object_id)
        store = SampleStore(reg.object_dir(profile.object_id), profile.object_id)
        box = _bbox_for(shape, _IMAGE_SIZE)

        per_session = max(1, n_per_class // n_sessions_per_class)
        sample_i = 0
        for session_i in range(n_sessions_per_class):
            session_dt = base_dt + timedelta(seconds=session_i * _SESSION_GAP_S)
            for _ in range(per_session):
                sample_i += 1
                img = _draw_shape(shape, size=_IMAGE_SIZE, color=colors[shape], rng=rng)
                ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                assert ok
                captured_dt = session_dt + timedelta(milliseconds=sample_i * 200)
                store.add(
                    image_bytes=buf.tobytes(),
                    width=_IMAGE_SIZE,
                    height=_IMAGE_SIZE,
                    box=box,
                    conditions=None,
                    role="positive",
                    source="camera",
                    device_label="fixture-cam",
                    consent_ack=True,
                    box_confirmed_by_human=True,
                )
                # Monkeypatch-free timestamp override: SampleStore.add() always
                # stamps "now" (captured_utc=utc_now_iso()); overwrite it
                # in-place on the just-written manifest so the fixture can
                # simulate distinct sessions deterministically, independent of
                # real wall-clock time.
                samples = store.list()
                _overwrite_captured_utc(store, samples[-1]["sample_id"], captured_dt)

        # One hard-negative per class, cropped from the NEXT shape in the
        # cycle (a real confusable, section 11.2) - trains the classifier to
        # reject the class it is most likely to be confused with.
        other = FIXTURE_SHAPES[(FIXTURE_SHAPES.index(shape) + 1) % len(FIXTURE_SHAPES)]
        img = _draw_shape(other, size=_IMAGE_SIZE, color=colors[other], rng=rng)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        assert ok
        store.add(
            image_bytes=buf.tobytes(), width=_IMAGE_SIZE, height=_IMAGE_SIZE,
            box=_bbox_for(other, _IMAGE_SIZE), conditions=None,
            role="hard_negative", negative_for=[profile.object_id],
            source="camera", device_label="fixture-cam", consent_ack=True,
            box_confirmed_by_human=True,
        )

    return object_ids


def _overwrite_captured_utc(store: SampleStore, sample_id: str, dt) -> None:
    """Directly patch one sample's ``captured_utc`` in the manifest - there is
    no public setter (provenance fields are not meant to be edited after the
    fact in the real product), but the fixture needs deterministic session
    timestamps rather than real wall-clock time."""

    doc = store._load()  # noqa: SLF001 - fixture-only, same module family
    for s in doc["samples"]:
        if s["sample_id"] == sample_id:
            s["captured_utc"] = dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    store._save(doc)  # noqa: SLF001
