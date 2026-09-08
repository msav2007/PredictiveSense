"""Measurable per-sample quality checks and per-object coverage accounting.

No new dependency: everything here is numpy. ``blur`` is the variance of the
Laplacian of the box crop; ``phash`` is a difference hash (dHash) implemented
directly; ``coverage`` counts condition dimensions against configured targets.

These thresholds are **heuristics chosen for data-collection guidance, not
scientific quality metrics** (recorded in ``docs/decisions.md``). Nothing here
produces a composite 0-100 score, and a flagged sample is marked, never deleted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from predictivesense.objects.vocab import CONDITION_VOCAB

__all__ = [
    "laplacian_variance",
    "dhash",
    "hamming",
    "nearest_duplicate",
    "SampleQuality",
    "compute_sample_quality",
    "CoverageTargets",
    "coverage_summary",
]

# 4-neighbour discrete Laplacian.
_LAPLACIAN_KERNEL = np.array(
    [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64
)


def _to_gray(image: np.ndarray) -> np.ndarray:
    """BGR/RGB ``HxWx3`` or grayscale ``HxW`` -> float64 grayscale ``HxW``."""

    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Rec. 601 luma; channel order does not matter for a blur/hash measure.
        gray = arr[..., :3].astype(np.float64) @ np.array([0.114, 0.587, 0.299])
        return gray
    if arr.ndim == 2:
        return arr.astype(np.float64)
    raise ValueError(f"expected HxW or HxWx3, got shape {arr.shape}")


def _convolve2d_valid(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Minimal 'valid' 2-D correlation for a small kernel (no scipy)."""

    kh, kw = kernel.shape
    h, w = arr.shape
    if h < kh or w < kw:
        return np.zeros((0, 0), dtype=np.float64)
    out = np.zeros((h - kh + 1, w - kw + 1), dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * arr[i : i + out.shape[0], j : j + out.shape[1]]
    return out


def laplacian_variance(image: np.ndarray) -> float:
    """Variance of the Laplacian over ``image`` (higher = sharper).

    A tiny crop (< 3 px in either axis) returns ``0.0`` rather than raising.
    """

    gray = _to_gray(image)
    lap = _convolve2d_valid(gray, _LAPLACIAN_KERNEL)
    if lap.size == 0:
        return 0.0
    return float(np.var(lap))


def _block_reduce_mean(gray: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Area-average ``gray`` down to ``out_h x out_w`` without an image library."""

    h, w = gray.shape
    ys = np.linspace(0, h, out_h + 1).astype(int)
    xs = np.linspace(0, w, out_w + 1).astype(int)
    ys[-1] = h
    xs[-1] = w
    out = np.empty((out_h, out_w), dtype=np.float64)
    for r in range(out_h):
        y0, y1 = ys[r], max(ys[r] + 1, ys[r + 1])
        for c in range(out_w):
            x0, x1 = xs[c], max(xs[c] + 1, xs[c + 1])
            out[r, c] = float(gray[y0:y1, x0:x1].mean())
    return out


def dhash(image: np.ndarray, hash_size: int = 8) -> str:
    """Difference hash: downscale to ``hash_size x (hash_size+1)`` grayscale, then
    encode 'is each pixel brighter than its right neighbour'. Returns a hex
    string of ``hash_size * hash_size / 4`` characters (16 for the default)."""

    gray = _to_gray(image)
    small = _block_reduce_mean(gray, hash_size, hash_size + 1)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    width = (hash_size * hash_size + 3) // 4
    return f"{value:0{width}x}"


def hamming(a: str, b: str) -> int:
    """Bit-difference between two equal-length hex hashes."""

    if len(a) != len(b):
        raise ValueError(f"hash length mismatch: {len(a)} vs {len(b)}")
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def nearest_duplicate(
    phash: str, existing: Iterable[tuple[str, str]], max_hamming: int
) -> tuple[str, int] | None:
    """Closest ``(sample_id, distance)`` within ``max_hamming`` of ``phash``, or
    ``None``. ``existing`` is ``(sample_id, phash)`` pairs."""

    best: tuple[str, int] | None = None
    for sid, other in existing:
        try:
            dist = hamming(phash, other)
        except ValueError:
            continue
        if dist <= max_hamming and (best is None or dist < best[1]):
            best = (sid, dist)
    return best


@dataclass(frozen=True)
class SampleQuality:
    """The stored ``quality`` block for one sample."""

    blur_var: float
    box_area_frac: float
    phash: str
    duplicate_of: str | None = None
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "blur_var": round(self.blur_var, 3),
            "box_area_frac": round(self.box_area_frac, 5),
            "phash": self.phash,
            "duplicate_of": self.duplicate_of,
            "flags": list(self.flags),
        }


def compute_sample_quality(
    image: np.ndarray,
    box: tuple[float, float, float, float],
    *,
    existing_hashes: Iterable[tuple[str, str]] = (),
    blur_var_min: float = 60.0,
    min_box_area_frac: float = 0.01,
    duplicate_hamming_max: int = 6,
) -> SampleQuality:
    """Blur, box-area fraction, dHash and near-duplicate flag for one sample.

    ``box`` is ``[x, y, w, h]`` in pixels. The image is the full frame; blur is
    measured over the box crop only.
    """

    h_img, w_img = int(image.shape[0]), int(image.shape[1])
    x, y, bw, bh = (float(v) for v in box)
    x0 = max(0, int(round(x)))
    y0 = max(0, int(round(y)))
    x1 = min(w_img, int(round(x + bw)))
    y1 = min(h_img, int(round(y + bh)))
    crop = image[y0:y1, x0:x1] if (x1 > x0 and y1 > y0) else image
    blur = laplacian_variance(crop)
    frame_area = float(w_img * h_img) or 1.0
    area_frac = max(0.0, (bw * bh) / frame_area)
    phash = dhash(image)

    dup = nearest_duplicate(phash, existing_hashes, duplicate_hamming_max)
    flags: list[str] = []
    if blur < blur_var_min:
        flags.append("blurry")
    if area_frac < min_box_area_frac:
        flags.append("box_too_small")
    if dup is not None:
        flags.append("near_duplicate")

    return SampleQuality(
        blur_var=blur,
        box_area_frac=area_frac,
        phash=phash,
        duplicate_of=dup[0] if dup is not None else None,
        flags=flags,
    )


@dataclass(frozen=True)
class CoverageTargets:
    """Guidance thresholds (heuristics, see ``docs/decisions.md``)."""

    min_positives: int = 20
    min_views: int = 4
    min_distances: int = 2
    min_lighting: int = 2
    min_occluded: int = 2
    min_hard_negatives: int = 5

    @classmethod
    def from_config(cls, cfg: Any) -> "CoverageTargets":
        return cls(
            min_positives=int(getattr(cfg, "min_positives", 20)),
            min_views=int(getattr(cfg, "min_views", 4)),
            min_distances=int(getattr(cfg, "min_distances", 2)),
            min_lighting=int(getattr(cfg, "min_lighting", 2)),
            min_occluded=int(getattr(cfg, "min_occluded", 2)),
            min_hard_negatives=int(getattr(cfg, "min_hard_negatives", 5)),
        )


def coverage_summary(
    samples: list[dict[str, Any]], targets: CoverageTargets
) -> dict[str, Any]:
    """Counts per condition dimension plus plain-language guidance strings.

    Not a score. ``samples`` are sample records (dicts with ``role`` and
    ``conditions``). The guidance list names what is *missing*, e.g.
    "no far-distance samples", "no occluded samples", "only 2 views".
    """

    positives = [s for s in samples if s.get("role") == "positive"]
    negatives = [s for s in samples if s.get("role") == "negative"]
    hard_negatives = [s for s in samples if s.get("role") == "hard_negative"]

    per_dimension: dict[str, dict[str, int]] = {}
    for dim, allowed in CONDITION_VOCAB.items():
        counts = {value: 0 for value in allowed}
        for s in positives:
            value = (s.get("conditions") or {}).get(dim)
            if value in counts:
                counts[value] += 1
        per_dimension[dim] = counts

    def _present(dim: str) -> list[str]:
        return [v for v, n in per_dimension[dim].items() if n > 0]

    views_present = _present("view")
    distances_present = _present("distance")
    lighting_present = _present("lighting")
    occluded_count = per_dimension["occlusion"].get("partial", 0)
    blurry = sum(1 for s in samples if "blurry" in (s.get("quality") or {}).get("flags", []))
    duplicates = sum(
        1 for s in samples if "near_duplicate" in (s.get("quality") or {}).get("flags", [])
    )

    meets = {
        "positives": len(positives) >= targets.min_positives,
        "views": len(views_present) >= targets.min_views,
        "distances": len(distances_present) >= targets.min_distances,
        "lighting": len(lighting_present) >= targets.min_lighting,
        "occluded": occluded_count >= targets.min_occluded,
        "hard_negatives": len(hard_negatives) >= targets.min_hard_negatives,
    }

    guidance: list[str] = []
    if len(positives) < targets.min_positives:
        guidance.append(
            f"{len(positives)} positive sample(s); collect at least {targets.min_positives}"
        )
    if len(views_present) < targets.min_views:
        missing = [v for v in CONDITION_VOCAB["view"] if v not in views_present]
        guidance.append(
            f"{len(views_present)} view(s) so far; missing {', '.join(missing)}"
        )
    for value in CONDITION_VOCAB["distance"]:
        if per_dimension["distance"].get(value, 0) == 0:
            guidance.append(f"no {value}-distance samples")
    for value in ("bright", "dim"):
        if per_dimension["lighting"].get(value, 0) == 0:
            guidance.append(f"no {value}-lighting samples")
    if occluded_count < targets.min_occluded:
        guidance.append(
            f"{occluded_count} occluded sample(s); target {targets.min_occluded}"
        )
    if len(hard_negatives) < targets.min_hard_negatives:
        guidance.append(
            f"{len(hard_negatives)} hard negative(s); target {targets.min_hard_negatives}"
        )
    if blurry:
        guidance.append(f"{blurry} sample(s) flagged blurry - review or re-shoot")
    if duplicates:
        guidance.append(f"{duplicates} near-duplicate(s) flagged - review")

    return {
        "total_positives": len(positives),
        "total_negatives": len(negatives),
        "total_hard_negatives": len(hard_negatives),
        "distinct_views": len(views_present),
        "views_present": views_present,
        "distances_present": distances_present,
        "lighting_present": lighting_present,
        "occluded_count": occluded_count,
        "blurry_count": blurry,
        "duplicate_count": duplicates,
        "per_dimension": per_dimension,
        "targets": {
            "min_positives": targets.min_positives,
            "min_views": targets.min_views,
            "min_distances": targets.min_distances,
            "min_lighting": targets.min_lighting,
            "min_occluded": targets.min_occluded,
            "min_hard_negatives": targets.min_hard_negatives,
        },
        "meets": meets,
        "meets_all": all(meets.values()),
        "guidance": guidance,
    }
