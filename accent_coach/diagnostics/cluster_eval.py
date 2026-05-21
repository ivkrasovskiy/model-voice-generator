"""Four-cluster acceptance criterion evaluator (Phase 0.8).

Computes the four clustering criteria C1-C4 in whatever coordinate space
the normalizer has already applied to `centroids`.

Expected speakers in centroids: fry, lindsey, bbc_male, real_BC,
modern_rp, owner, deterding.
"""
from __future__ import annotations

import math
from typing import Any

Centroids = dict[str, dict[str, dict[str, float] | None]]


def speaker_distance(centroids: Centroids, spk1: str, spk2: str) -> float:
    """Mean per-phoneme Euclidean distance between two speakers.

    Skips phonemes where either speaker has no data (None).
    """
    dists: list[float] = []
    for _phoneme, speakers in centroids.items():
        s1 = speakers.get(spk1)
        s2 = speakers.get(spk2)
        if not s1 or not s2:
            continue
        d = math.sqrt((s1["f1"] - s2["f1"]) ** 2 + (s1["f2"] - s2["f2"]) ** 2)
        dists.append(d)
    if not dists:
        return float("inf")
    return sum(dists) / len(dists)


def evaluate_clustering(centroids: Centroids) -> dict[str, Any]:
    """Evaluate the four-cluster acceptance criteria.

    Returns a dict with raw distances, ratio checks, and pass/fail booleans.

    C1: mean pairwise dist {fry, lindsey, bbc_male, real_BC} <= 50 (metric units)
    C2: dist(modern_rp, deterding) >= 1.5 * C1
    C3: dist(owner, modern_rp)    >= 2.0 * C1
    C4: dist(owner, deterding)    >= 1.5 * C1
    """
    modern_rp_speakers = ["fry", "lindsey", "bbc_male", "real_BC"]

    pair_dists: list[float] = []
    for i, s1 in enumerate(modern_rp_speakers):
        for s2 in modern_rp_speakers[i + 1 :]:
            d = speaker_distance(centroids, s1, s2)
            pair_dists.append(d)
    c1 = sum(pair_dists) / len(pair_dists)

    c2 = speaker_distance(centroids, "modern_rp", "deterding")
    c3 = speaker_distance(centroids, "owner", "modern_rp")
    c4 = speaker_distance(centroids, "owner", "deterding")

    return {
        "C1_within_rp_mean_dist": round(c1, 3),
        "C2_rp_vs_deterding": round(c2, 3),
        "C3_owner_vs_rp": round(c3, 3),
        "C4_owner_vs_deterding": round(c4, 3),
        # ratio checks (denominator is C1)
        "C2_ratio": round(c2 / c1, 3) if c1 > 0 else None,
        "C3_ratio": round(c3 / c1, 3) if c1 > 0 else None,
        "C4_ratio": round(c4 / c1, 3) if c1 > 0 else None,
        # pass/fail per criterion
        "C1_pass": c1 <= 50,
        "C2_pass": c2 >= 1.5 * c1,
        "C3_pass": c3 >= 2.0 * c1,
        "C4_pass": c4 >= 1.5 * c1,
        "overall_pass": (c1 <= 50) and (c2 >= 1.5 * c1) and (c3 >= 2.0 * c1) and (c4 >= 1.5 * c1),
        # per-pair breakdown for diagnostics
        "pairwise_within_rp": {
            f"{s1}_vs_{s2}": round(speaker_distance(centroids, s1, s2), 1)
            for i, s1 in enumerate(modern_rp_speakers)
            for s2 in modern_rp_speakers[i + 1 :]
        },
    }


def per_phoneme_sigma_rp(centroids: Centroids) -> dict[str, float]:
    """Per-phoneme within-cluster σ (mean distance from modern_rp_avg).

    Used to calibrate the piecewise linear score.
    σ(p) = mean over {fry, lindsey, bbc_male, real_BC} of dist_2d(speaker_p, modern_rp_p)
    """
    sigmas: dict[str, float] = {}
    modern_rp_speakers = ["fry", "lindsey", "bbc_male", "real_BC"]

    for phoneme, speakers in centroids.items():
        ref = speakers.get("modern_rp")
        if not ref:
            continue
        dists: list[float] = []
        for spk in modern_rp_speakers:
            s = speakers.get(spk)
            if not s:
                continue
            d = math.sqrt((s["f1"] - ref["f1"]) ** 2 + (s["f2"] - ref["f2"]) ** 2)
            dists.append(d)
        if dists:
            sigmas[phoneme] = sum(dists) / len(dists)
    return sigmas
