"""ConsonantScore aggregator — combines fricatives, stops, rhotics, laterals.

Weights (from spec 3E):
  fricatives    35 %
  stops         35 %
  rhotics       20 %
  laterals      10 %

Known limitation: scoring.py also calls score_aspiration() as a separate
15%-weight skill, causing stop VOT to be counted twice in the full composite
(~20% total vs the intended 15%).  This will be resolved in analyze.py when
the three comparison modules are fully integrated and score_aspiration is
retired.  Do not remove the stops weight here before that integration.

When a sub-class has no scorable tokens the weight is redistributed to the
remaining classes proportionally (same principle as rhythm function-word
weight redistribution — avoids fictional perfect sub-scores).
"""
from __future__ import annotations

import numpy as np

from accent_coach.models import ConsonantScore, SentenceAnalysis

from .fricatives import score_fricatives
from .liquids import score_liquids
from .stops import score_stops

_WEIGHTS: dict[str, float] = {
    "fricatives": 0.35,
    "stops": 0.35,
    "rhotics": 0.20,
    "laterals": 0.10,
}



def score_consonants(
    user: SentenceAnalysis,
    audio: np.ndarray,
    sr: int,
    target_audio: np.ndarray | None = None,
    target_sr: int | None = None,
    target: SentenceAnalysis | None = None,
    accent_target: str = "rp",
) -> ConsonantScore:
    """Full consonant quality assessment.

    Evaluates fricatives, stops, rhotics, and laterals from the sentence's
    phoneme list and pre-extracted StopFeatures.  Returns a ConsonantScore
    with per-class sub-scores and prioritised diagnostics.
    """
    fric_score, fric_diag = score_fricatives(
        user, audio, sr,
        target=target, target_audio=target_audio, target_sr=target_sr,
        accent_target=accent_target,
    )
    stop_score, stop_diag = score_stops(user, target=target, accent_target=accent_target)
    rhotic_score, lateral_score, liq_diag = score_liquids(user, audio, sr, accent_target=accent_target)

    sub_scores: dict[str, float | None] = {
        "fricatives": fric_score,
        "stops": stop_score,
        "rhotics": rhotic_score,
        "laterals": lateral_score,
    }

    # Redistribute weights from absent sub-classes. When NO sub-class is scorable
    # the composite is None — never a fabricated neutral that would be averaged
    # into group means as if it were a real measurement.
    available = {k: v for k, v in sub_scores.items() if v is not None}
    if not available:
        composite = None
    else:
        total_weight = sum(_WEIGHTS[k] for k in available)
        composite = float(
            np.clip(sum(_WEIGHTS[k] * available[k] for k in available) / total_weight, 0.0, 100.0)
        )

    # Collect and deduplicate diagnostics, most severe first
    all_diag = fric_diag + stop_diag + liq_diag

    return ConsonantScore(
        score=composite,
        fricative_score=fric_score,
        stop_aspiration_score=stop_score,
        rhotic_score=rhotic_score,
        lateral_score=lateral_score,
        diagnostics=all_diag[:4],
    )
