"""Object quality checks (Phase 4): Laplacian variance ordering, dHash stability
and discrimination, duplicate detection at the configured Hamming distance, and
coverage counting on a constructed sample set. numpy only - no new dependency."""

from __future__ import annotations

import numpy as np
import pytest

from predictivesense.objects.quality import (
    CoverageTargets,
    compute_sample_quality,
    coverage_summary,
    dhash,
    hamming,
    laplacian_variance,
    nearest_duplicate,
)

pytestmark = pytest.mark.unit


def _checkerboard(n=64, cell=4) -> np.ndarray:
    ys, xs = np.mgrid[0:n, 0:n]
    board = (((xs // cell) + (ys // cell)) % 2).astype(np.uint8) * 255
    return np.stack([board] * 3, axis=-1)


def _gradient(n=64, axis=1) -> np.ndarray:
    """Monotonic ramp along ``axis`` - a well-behaved dHash input."""

    v = np.linspace(0, 255, n).astype(np.uint8)
    g = np.tile(v, (n, 1)) if axis == 1 else np.tile(v[:, None], (1, n))
    return np.stack([g] * 3, axis=-1)


def _blur3(img: np.ndarray) -> np.ndarray:
    """Cheap 3x3 box blur (numpy), repeated, to soften edges."""

    out = img.astype(np.float64)
    for _ in range(6):
        p = np.pad(out, ((1, 1), (1, 1), (0, 0)), mode="edge")
        out = sum(
            p[i : i + out.shape[0], j : j + out.shape[1]] for i in range(3) for j in range(3)
        ) / 9.0
    return out.astype(np.uint8)


def test_laplacian_variance_orders_sharp_above_blurred() -> None:
    sharp = _checkerboard()
    blurred = _blur3(sharp)
    assert laplacian_variance(sharp) > laplacian_variance(blurred) * 5
    assert laplacian_variance(np.zeros((2, 2, 3), np.uint8)) == 0.0  # tiny crop safe


def test_dhash_is_stable_under_reencode_and_differs_across_images() -> None:
    img = _gradient(axis=1)  # horizontal ramp
    h1 = dhash(img)
    assert len(h1) == 16
    # simulate a re-encode: add small noise; the ramp ordering is preserved
    noisy = np.clip(img.astype(np.int16) + np.random.randint(-4, 5, img.shape), 0, 255).astype(np.uint8)
    assert hamming(h1, dhash(noisy)) <= 4

    vertical = _gradient(axis=0)  # a genuinely different image
    assert hamming(h1, dhash(vertical)) >= 8


def test_duplicate_detection_respects_configured_hamming() -> None:
    a = _gradient(axis=1)
    b = a.copy()
    b[24:40, 24:40] = 0  # a small localised change -> a few hash bits flip
    c = _gradient(axis=0)  # clearly different
    d_a, d_b, d_c = dhash(a), dhash(b), dhash(c)
    near = hamming(d_a, d_b)
    assert 0 < near < hamming(d_a, d_c)  # b is a near-duplicate of a, c is not

    existing = [("s-a", d_a), ("s-c", d_c)]
    hit = nearest_duplicate(d_b, existing, max_hamming=near)
    assert hit is not None and hit[0] == "s-a"
    # tighten the threshold below the actual distance -> no match
    assert nearest_duplicate(d_b, existing, max_hamming=near - 1) is None


def test_compute_sample_quality_flags_blurry_and_duplicate() -> None:
    sharp = _checkerboard()
    blurred = _blur3(sharp)
    box = (8.0, 8.0, 40.0, 40.0)

    q_sharp = compute_sample_quality(sharp, box, blur_var_min=60.0)
    assert "blurry" not in q_sharp.flags
    assert 0.0 < q_sharp.box_area_frac <= 1.0

    q_blur = compute_sample_quality(blurred, box, blur_var_min=1e9)
    assert "blurry" in q_blur.flags

    q_dup = compute_sample_quality(
        sharp, box, existing_hashes=[("prev", dhash(sharp))], duplicate_hamming_max=6
    )
    assert q_dup.duplicate_of == "prev" and "near_duplicate" in q_dup.flags


def test_coverage_counts_and_guidance() -> None:
    def sample(role="positive", **cond):
        base = {"view": "front", "distance": "close", "lighting": "normal",
                "background": "plain", "occlusion": "none", "held": "on-surface",
                "frame_position": "centre"}
        base.update(cond)
        return {"role": role, "conditions": base, "quality": {"flags": []}}

    samples = [
        sample(view="front"),
        sample(view="left"),
        sample(view="right", distance="medium"),
        sample(view="top", lighting="bright"),
        sample(role="hard_negative"),
        sample(occlusion="partial"),
    ]
    cov = coverage_summary(samples, CoverageTargets())
    assert cov["total_positives"] == 5
    assert cov["total_hard_negatives"] == 1
    assert cov["distinct_views"] == 4
    assert cov["occluded_count"] == 1
    # no far-distance, no dim-lighting -> named in guidance
    joined = " ".join(cov["guidance"])
    assert "far-distance" in joined and "dim-lighting" in joined
    assert cov["meets"]["views"] is True and cov["meets"]["positives"] is False
