"""Lobanov z-score normalization for vowel formants.

Lobanov 1971: "Classification of Russian Vowels Spoken by Different
Speakers", JASA 49(4), pp. 606-608.
Each speaker's F1 and F2 are z-scored independently using that speaker's
own grand mean and SD across all their vowel tokens.
"""
from __future__ import annotations

import numpy as np

from accent_coach.models import VowelFeatures


class LobanovParams:
    """Speaker-level normalisation parameters computed from all their vowel tokens."""

    def __init__(self, mean_f1: float, sd_f1: float, mean_f2: float, sd_f2: float) -> None:
        self.mean_f1 = mean_f1
        self.sd_f1   = max(sd_f1, 1.0)
        self.mean_f2 = mean_f2
        self.sd_f2   = max(sd_f2, 1.0)

    def normalise(self, f1: float, f2: float) -> tuple[float, float]:
        return (f1 - self.mean_f1) / self.sd_f1, (f2 - self.mean_f2) / self.sd_f2


def compute_lobanov_params(vowels: list[VowelFeatures]) -> LobanovParams:
    """Compute Lobanov params from a speaker's full vowel token pool."""
    f1s = np.array([v.f1 for v in vowels])
    f2s = np.array([v.f2 for v in vowels])
    return LobanovParams(
        mean_f1=float(f1s.mean()), sd_f1=float(f1s.std()),
        mean_f2=float(f2s.mean()), sd_f2=float(f2s.std()),
    )


def rp_lobanov_params(norms: dict[str, tuple[float, float]]) -> LobanovParams:
    """Compute Lobanov params from an RP norms table (one token per phoneme)."""
    f1s = np.array([v[0] for v in norms.values()])
    f2s = np.array([v[1] for v in norms.values()])
    return LobanovParams(
        mean_f1=float(f1s.mean()), sd_f1=float(f1s.std()),
        mean_f2=float(f2s.mean()), sd_f2=float(f2s.std()),
    )


def lobanov_normalize(vowels: list[VowelFeatures]) -> list[tuple[float, float]]:
    """Return list of (z_F1, z_F2) for each vowel token."""
    if not vowels:
        return []
    params = compute_lobanov_params(vowels)
    return [params.normalise(v.f1, v.f2) for v in vowels]


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
