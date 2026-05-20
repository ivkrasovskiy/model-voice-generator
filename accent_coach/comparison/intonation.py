from __future__ import annotations

import math

import numpy as np
from dtw import dtw  # type: ignore[import-untyped]

from accent_coach.models import IntonationBreakdown, SentenceAnalysis
from accent_coach.reference.rp_norms import RP_PITCH_TEMPLATES

_DECAY = 1.0  # normalised DTW distance for e^{-1} decay


def _normalise_contour(contour: list[float]) -> np.ndarray:
    arr = np.array(contour, dtype=float)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def score_intonation(
    user: SentenceAnalysis,
    target: SentenceAnalysis | None = None,
) -> IntonationBreakdown:
    user_c = _normalise_contour(user.pitch_contour)

    if target is not None:
        ref_c = _normalise_contour(target.pitch_contour)
    else:
        template = RP_PITCH_TEMPLATES.get(user.sentence_type, RP_PITCH_TEMPLATES["statement"])
        ref_c = np.array(template)

    alignment = dtw(user_c.reshape(-1, 1), ref_c.reshape(-1, 1), keep_internals=False)
    norm_dist = float(alignment.normalizedDistance)
    score = 100.0 * math.exp(-norm_dist / _DECAY)
    return IntonationBreakdown(per_sentence_dtw=[norm_dist], score=score)
