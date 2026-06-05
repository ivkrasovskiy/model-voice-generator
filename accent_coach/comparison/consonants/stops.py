"""VOT-based stop aspiration scoring.

Wraps the VOT values already extracted in StopFeatures (via pipeline/vot.py)
and scores them against RP/GenAm reference ranges.  Only voiceless stops
(/p t k/) in word-initial stressed position are scored — that is where English
aspiration is contrastive (spec 3B).

Under-aspiration (VOT < 35 ms) is the strongest non-native marker and
receives an additional severity flag.
"""
from __future__ import annotations

import math

import numpy as np

from accent_coach.models import SentenceAnalysis
from accent_coach.reference.genam_norms import GA_VOT_MEAN_MS, GA_VOT_SD_MS
from accent_coach.reference.rp_norms import RP_VOT_MEAN_MS, RP_VOT_SD_MS

_VOICELESS_STOPS: frozenset[str] = frozenset({"p", "t", "k"})
_DECAY: float = 1.5        # z-score decay (matches aspiration.py)
_UNDER_ASPIRATION_MS: float = 35.0  # VOT below this = strong non-native signal
_UNDER_ASPIRATION_PENALTY: float = 0.5  # multiply base score by this


def _vot_norms(accent_target: str) -> tuple[dict[str, float], dict[str, float]]:
    if accent_target == "genam":
        return GA_VOT_MEAN_MS, GA_VOT_SD_MS
    return RP_VOT_MEAN_MS, RP_VOT_SD_MS


def score_stops(
    user: SentenceAnalysis,
    target: SentenceAnalysis | None = None,
    accent_target: str = "rp",
) -> tuple[float | None, list[str]]:
    """Score stop aspiration from pre-extracted StopFeatures.

    Returns (score 0–100 or None if no stops, diagnostics list).
    None signals aggregator to redistribute weight.
    """
    if not user.stops:
        return None, []

    vot_mean, vot_sd = _vot_norms(accent_target)

    # Build target VOT lookup (comparison mode)
    target_vot: dict[str, list[float]] = {}
    if target is not None:
        for s in target.stops:
            ph = s.phoneme.phoneme
            if ph in _VOICELESS_STOPS:
                target_vot.setdefault(ph, []).append(s.vot_ms)

    scores: list[float] = []
    under_aspirated: list[str] = []
    diagnostics: list[str] = []

    for s in user.stops:
        ph = s.phoneme.phoneme
        if ph not in _VOICELESS_STOPS:
            continue
        if ph not in vot_mean:
            continue

        # Prefer target mean; fall back to corpus reference
        if ph in target_vot:
            mean = float(np.mean(target_vot[ph]))
            sd = float(np.std(target_vot[ph])) or vot_sd[ph]
        else:
            mean = vot_mean[ph]
            sd = vot_sd[ph]

        z = abs(s.vot_ms - mean) / sd
        base = 100.0 * math.exp(-z / _DECAY)

        if s.vot_ms < _UNDER_ASPIRATION_MS:
            base *= _UNDER_ASPIRATION_PENALTY
            under_aspirated.append(f"/{ph}/ ({s.vot_ms:.0f} ms)")

        scores.append(base)

    if under_aspirated:
        stops_str = ", ".join(under_aspirated[:3])
        diagnostics.append(
            f"Under-aspirated stops: {stops_str}. "
            "Add a noticeable puff of breath after the burst — English voiceless stops "
            "should have 55–100 ms of aspiration before the vowel starts."
        )

    return (float(np.mean(scores)) if scores else None), diagnostics
