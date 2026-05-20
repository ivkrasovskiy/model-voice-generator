from __future__ import annotations

import numpy as np

from accent_coach.comparison.aspiration import score_aspiration
from accent_coach.comparison.consonants import score_consonants
from accent_coach.comparison.intonation import score_intonation
from accent_coach.comparison.rhythm import score_rhythm
from accent_coach.comparison.stress import score_stress
from accent_coach.comparison.vowels import score_vowels
from accent_coach.diagnostics.advice import build_vowel_diagnostics
from accent_coach.models import ComparisonResult, SentenceAnalysis
from accent_coach.reference.rp_norms import get_rp_norms

# Default skill weights from spec
_WEIGHTS: dict[str, float] = {
    "vowels":      25.0,
    "consonants":  15.0,
    "aspiration":  15.0,
    "rhythm":      15.0,
    "stress":      15.0,
    "intonation":  15.0,
}


def compare(
    user: SentenceAnalysis,
    user_audio: np.ndarray,
    user_sr: int,
    target: SentenceAnalysis | None = None,
    reference_norms: dict[str, tuple[float, float]] | None = None,
) -> ComparisonResult:
    """Produce a full ComparisonResult for one sentence."""

    # Infer RP norms for diagnostics if not provided
    if reference_norms is None:
        mean_f0 = float(
            np.mean([v.pitch_mean for v in user.vowels if v.pitch_mean > 70] or [120.0])
        )
        reference_norms = get_rp_norms(mean_f0)

    vowel_score = score_vowels(user, reference_norms=reference_norms, target=target)
    aspiration_bd = score_aspiration(user, target=target)
    rhythm_bd = score_rhythm(user, target=target)
    stress_score = score_stress(user, target=target)
    intonation_bd = score_intonation(user, target=target)
    consonant_score = score_consonants(user, user_audio, user_sr, target=target)

    skill_scores = {
        "vowels":      vowel_score,
        "consonants":  consonant_score,
        "aspiration":  aspiration_bd.score,
        "rhythm":      rhythm_bd.score,
        "stress":      stress_score,
        "intonation":  intonation_bd.score,
    }

    total_weight = sum(_WEIGHTS.values())
    composite = sum(_WEIGHTS[k] * v for k, v in skill_scores.items()) / total_weight

    diagnostics = build_vowel_diagnostics(user.vowels, reference_norms, target)

    return ComparisonResult(
        skill_scores=skill_scores,
        composite_score=composite,
        vowel_diagnostics=diagnostics,
        rhythm_breakdown=rhythm_bd,
        aspiration_breakdown=aspiration_bd,
        intonation_breakdown=intonation_bd,
    )
