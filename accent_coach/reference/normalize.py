"""Lobanov z-score normalization for vowel formants.

Lobanov 1971: "Classification of Russian Vowels Spoken by Different
Speakers", JASA 49(4), pp. 606-608.
Each speaker's F1 and F2 are z-scored independently using that speaker's
own grand mean and SD across all their vowel tokens.
"""
from __future__ import annotations

import numpy as np

from accent_coach.models import VowelFeatures


def lobanov_normalize(vowels: list[VowelFeatures]) -> list[tuple[float, float]]:
    """Return list of (z_F1, z_F2) for each vowel token."""
    if not vowels:
        return []
    f1s = np.array([v.f1 for v in vowels])
    f2s = np.array([v.f2 for v in vowels])
    z_f1 = (f1s - f1s.mean()) / (f1s.std() + 1e-9)
    z_f2 = (f2s - f2s.mean()) / (f2s.std() + 1e-9)
    return list(zip(z_f1.tolist(), z_f2.tolist(), strict=True))


def lobanov_centroid(
    vowels: list[VowelFeatures], phoneme: str
) -> tuple[float, float] | None:
    """Lobanov centroid (mean z_F1, mean z_F2) for a specific phoneme."""
    all_z = lobanov_normalize(vowels)
    indices = [i for i, v in enumerate(vowels) if v.phoneme.phoneme == phoneme]
    if not indices:
        return None
    z_f1 = np.mean([all_z[i][0] for i in indices])
    z_f2 = np.mean([all_z[i][1] for i in indices])
    return float(z_f1), float(z_f2)
