from __future__ import annotations

import numpy as np

from accent_coach.models import SentenceAnalysis


def score_stress(
    user: SentenceAnalysis,
    target: SentenceAnalysis | None = None,
) -> float:
    """Per-syllable stress position match, averaged over the sentence."""
    if not user.stress_pattern:
        return 50.0

    if target is not None and target.stress_pattern:
        n = min(len(user.stress_pattern), len(target.stress_pattern))
        if n == 0:
            return 50.0
        matches = sum(
            u == t for u, t in zip(user.stress_pattern[:n], target.stress_pattern[:n])
        )
        return 100.0 * matches / n

    # Without target: score based on proportion of stressed syllables being reasonable
    # (~20–40% stressed is natural for English)
    pct = np.mean(user.stress_pattern)
    # Score peaks at 30% stressed, decays away
    return float(100.0 * np.exp(-((pct - 0.30) ** 2) / (2 * 0.15**2)))
