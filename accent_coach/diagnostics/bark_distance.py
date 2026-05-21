"""H4 — Bark/ERB-scale distance (Phase 0.8).

Converts F1/F2 from Hz to Bark before computing Euclidean distances.
Bark scale is perceptually motivated and compresses high-F2 differences
that partly reflect VTL bias rather than vowel quality.

Formula (Zwicker & Terhardt 1980):
    Bark = 13 * arctan(0.00076 * F) + 3.5 * arctan((F / 7500)^2)
"""
from __future__ import annotations

import math

Centroids = dict[str, dict[str, dict[str, float] | None]]


def hz_to_bark(f: float) -> float:
    """Convert frequency in Hz to Bark scale."""
    return 13.0 * math.atan(0.00076 * f) + 3.5 * math.atan((f / 7500.0) ** 2)


def bark_transform(centroids: Centroids) -> Centroids:
    """Return centroids with f1/f2 converted from Hz to Bark units."""
    out: Centroids = {}
    for phoneme, speakers in centroids.items():
        out[phoneme] = {}
        for spk, vals in speakers.items():
            if vals is None or vals["f1"] <= 0 or vals["f2"] <= 0:
                out[phoneme][spk] = None
                continue
            out[phoneme][spk] = {
                "f1": hz_to_bark(vals["f1"]),
                "f2": hz_to_bark(vals["f2"]),
                "n": vals.get("n", 0),
            }
    return out
