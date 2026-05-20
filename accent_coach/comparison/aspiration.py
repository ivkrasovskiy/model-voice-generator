from __future__ import annotations

import math

import numpy as np

from accent_coach.models import AspirationBreakdown, SentenceAnalysis
from accent_coach.reference.rp_norms import RP_VOT_MEAN_MS, RP_VOT_SD_MS

_DECAY = 1.5  # z-score decay constant


def score_aspiration(
    user: SentenceAnalysis,
    target: SentenceAnalysis | None = None,
) -> AspirationBreakdown:
    if not user.stops:
        return AspirationBreakdown(per_stop={}, score=50.0)

    per_stop: dict[str, float] = {}
    if target is not None:
        tgt_by_phoneme: dict[str, list[float]] = {}
        for s in target.stops:
            tgt_by_phoneme.setdefault(s.phoneme.phoneme, []).append(s.vot_ms)
        for s in user.stops:
            p = s.phoneme.phoneme
            if p not in tgt_by_phoneme:
                continue
            tgt_mean = float(np.mean(tgt_by_phoneme[p]))
            tgt_sd = float(np.std(tgt_by_phoneme[p])) or 10.0
            z = abs(s.vot_ms - tgt_mean) / tgt_sd
            per_stop[p] = 100.0 * math.exp(-z / _DECAY)
    else:
        for s in user.stops:
            p = s.phoneme.phoneme
            mean = RP_VOT_MEAN_MS.get(p)
            sd = RP_VOT_SD_MS.get(p)
            if mean is None or sd is None:
                continue
            z = abs(s.vot_ms - mean) / sd
            per_stop[p] = 100.0 * math.exp(-z / _DECAY)

    score = float(np.mean(list(per_stop.values()))) if per_stop else 50.0
    return AspirationBreakdown(per_stop=per_stop, score=score)
