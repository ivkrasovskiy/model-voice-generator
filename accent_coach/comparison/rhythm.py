from __future__ import annotations

import math

from accent_coach.models import RhythmBreakdown, SentenceAnalysis
from accent_coach.pipeline.prosody import compute_npvi
from accent_coach.reference.rp_norms import RP_NPVI_MAX, RP_NPVI_MIN

_NPVI_REF = (RP_NPVI_MIN + RP_NPVI_MAX) / 2  # 65.0
_DECAY = 20.0  # nPVI units for e^{-1} decay


def score_rhythm(
    user: SentenceAnalysis,
    target: SentenceAnalysis | None = None,
) -> RhythmBreakdown:
    user_npvi = compute_npvi(user.syllable_durations)

    if target is not None:
        ref_npvi = compute_npvi(target.syllable_durations)
        ref_min = ref_npvi - 10.0
        ref_max = ref_npvi + 10.0
    else:
        ref_npvi = _NPVI_REF
        ref_min = RP_NPVI_MIN
        ref_max = RP_NPVI_MAX

    delta = abs(user_npvi - ref_npvi)
    score = 100.0 * math.exp(-delta / _DECAY)
    return RhythmBreakdown(
        npvi=user_npvi,
        reference_npvi_min=ref_min,
        reference_npvi_max=ref_max,
        score=score,
    )
