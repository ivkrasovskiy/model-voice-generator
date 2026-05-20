from __future__ import annotations

import math

import numpy as np

from accent_coach.models import SentenceAnalysis, VowelFeatures
from accent_coach.reference.normalize import lobanov_normalize
from accent_coach.reference.rp_norms import get_rp_norms

_SCALE = 1.5  # decay constant for normalized euclidean distance


def _score_vowel_pair(
    user_f1: float, user_f2: float, ref_f1: float, ref_f2: float
) -> float:
    d = math.sqrt(((user_f1 - ref_f1) / 500) ** 2 + ((user_f2 - ref_f2) / 1000) ** 2)
    return 100.0 * math.exp(-d / _SCALE)


def score_vowels(
    user: SentenceAnalysis,
    reference_norms: dict[str, tuple[float, float]] | None = None,
    target: SentenceAnalysis | None = None,
) -> float:
    """Score vowel accuracy against reference norms or a target utterance.

    If reference_norms is given, use those. If target is given, compare
    user formants against target formants on a per-phoneme basis.
    Exactly one of the two must be provided.
    """
    if not user.vowels:
        return 0.0

    if target is not None:
        # Group target vowels by phoneme
        tgt_by_phoneme: dict[str, list[VowelFeatures]] = {}
        for v in target.vowels:
            tgt_by_phoneme.setdefault(v.phoneme.phoneme, []).append(v)
        scores: list[float] = []
        for v in user.vowels:
            tgt_list = tgt_by_phoneme.get(v.phoneme.phoneme)
            if not tgt_list:
                continue
            tgt_f1 = float(np.mean([t.f1 for t in tgt_list]))
            tgt_f2 = float(np.mean([t.f2 for t in tgt_list]))
            scores.append(_score_vowel_pair(v.f1, v.f2, tgt_f1, tgt_f2))
        return float(np.mean(scores)) if scores else 0.0

    norms = reference_norms or {}
    # Fall back to speaker f0 estimate
    if not norms and user.vowels:
        mean_f0 = float(np.mean([v.pitch_mean for v in user.vowels if v.pitch_mean > 70] or [120]))
        norms = get_rp_norms(mean_f0)

    scores = []
    for v in user.vowels:
        ref = norms.get(v.phoneme.phoneme)
        if ref is None:
            continue
        scores.append(_score_vowel_pair(v.f1, v.f2, ref[0], ref[1]))
    return float(np.mean(scores)) if scores else 0.0
