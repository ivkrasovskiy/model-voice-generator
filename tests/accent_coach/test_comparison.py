from __future__ import annotations

from accent_coach.comparison.vowels import score_vowels
from accent_coach.models import (
    PhonemeInstance,
    SentenceAnalysis,
    VowelFeatures,
)
from accent_coach.reference.rp_norms import RP_VOWEL_F1_F2_MALE


def _phoneme(p: str) -> PhonemeInstance:
    return PhonemeInstance(
        phoneme=p, arpabet="XX", start_time=0.0, end_time=0.2,
        sentence_id=1, word="test", is_stressed=True,
    )


def _vowel(p: str, f1: float, f2: float) -> VowelFeatures:
    return VowelFeatures(phoneme=_phoneme(p), f1=f1, f2=f2, duration_ms=200.0, pitch_mean=120.0)


def _sentence(vowels: list[VowelFeatures]) -> SentenceAnalysis:
    return SentenceAnalysis(
        sentence_id=1,
        sentence_type="statement",
        duration_s=1.0,
        syllable_durations=[0.2, 0.2, 0.2],
        pitch_contour=[0.5] * 50,
        stress_pattern=[True, False, True],
        vowels=vowels,
        stops=[],
    )


def test_identity_scores_100():
    """When user formants exactly match reference norms, vowel score should be 100."""
    vowels = [_vowel(ph, f1, f2) for ph, (f1, f2) in RP_VOWEL_F1_F2_MALE.items()]
    analysis = _sentence(vowels)
    score = score_vowels(analysis, reference_norms=RP_VOWEL_F1_F2_MALE)
    assert score > 99.0, f"Identity score={score:.2f} should be ~100"


def test_large_f1_shift_drops_score():
    """Shifting every F1 by +400 Hz (strong foreign accent) should drop vowel score below 60."""
    vowels = [
        _vowel(ph, f1 + 400, f2) for ph, (f1, f2) in RP_VOWEL_F1_F2_MALE.items()
    ]
    analysis = _sentence(vowels)
    score = score_vowels(analysis, reference_norms=RP_VOWEL_F1_F2_MALE)
    assert score < 60.0, f"Shifted score={score:.2f} should be below 60"
