"""
Ordering invariant tests: scores must respect the native-vs-L2 and same-dialect-vs-cross-dialect
hierarchy.  These tests catch regressions where the scorer becomes non-discriminative.

All tests use synthetic SentenceAnalysis objects (no audio files required).
"""
from __future__ import annotations

import pytest

from accent_coach.models import PhonemeInstance, SentenceAnalysis, VowelFeatures
from accent_coach.pipeline.prosody import compute_npvi
from accent_coach.reference.genam_norms import get_genam_norms
from accent_coach.reference.rp_norms import (
    RP_NPVI_MIN,
    RP_NPVI_MAX,
    RP_VOWEL_F1_F2_MALE_MODERN,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_RP = RP_VOWEL_F1_F2_MALE_MODERN
_GA = get_genam_norms(120.0)  # male-pitch GenAm


def _ph(p: str, word: str = "test") -> PhonemeInstance:
    return PhonemeInstance(
        phoneme=p, arpabet="XX",
        start_time=0.0, end_time=0.2,
        sentence_id=1, word=word, is_stressed=True,
    )


def _vf(p: str, f1: float, f2: float) -> VowelFeatures:
    return VowelFeatures(phoneme=_ph(p), f1=f1, f2=f2, duration_ms=200.0, pitch_mean=120.0)


def _sentence(
    vowels: list[VowelFeatures],
    syllable_durations: list[float] | None = None,
) -> SentenceAnalysis:
    durs = syllable_durations or [0.2, 0.2, 0.2]
    return SentenceAnalysis(
        sentence_id=1, sentence_type="statement",
        duration_s=sum(durs), syllable_durations=durs,
        pitch_contour=[0.5] * 50,
        stress_pattern=[True, False] * (len(durs) // 2 + 1),
        vowels=vowels, stops=[], phonemes=[],
    )


# ---------------------------------------------------------------------------
# Vowel scoring: dialect discrimination
# ---------------------------------------------------------------------------


def test_rp_vowels_score_higher_on_rp_norms_than_genam():
    """A speaker with ideal RP vowels must score higher vs RP norms than vs GenAm norms.

    If score_vs_rp ≈ score_vs_ga, the norms cannot discriminate accents.
    """
    from accent_coach.comparison.vowels import score_vowels

    rp_speaker = _sentence([_vf(ph, f1, f2) for ph, (f1, f2) in _RP.items()])
    score_rp = score_vowels(rp_speaker, reference_norms=_RP)
    score_ga = score_vowels(rp_speaker, reference_norms=_GA)

    assert score_rp > score_ga + 5, (
        f"RP speaker: score_vs_rp={score_rp:.1f} vs score_vs_ga={score_ga:.1f}. "
        "Expected gap > 5 pts — norms must discriminate RP from GenAm."
    )


def test_genam_vowels_score_higher_on_genam_norms_than_rp():
    """A speaker with ideal GenAm vowels must score higher vs GenAm norms than vs RP norms.

    Symmetric counterpart to the RP test above.
    """
    from accent_coach.comparison.vowels import score_vowels

    ga_speaker = _sentence([_vf(ph, f1, f2) for ph, (f1, f2) in _GA.items()])
    score_ga = score_vowels(ga_speaker, reference_norms=_GA)
    score_rp = score_vowels(ga_speaker, reference_norms=_RP)

    assert score_ga > score_rp + 5, (
        f"GenAm speaker: score_vs_ga={score_ga:.1f} vs score_vs_rp={score_rp:.1f}. "
        "Expected gap > 5 pts — norms must discriminate GenAm from RP."
    )


def test_heavy_accent_scores_well_below_native():
    """A speaker with strong L2 formant shift (all F1+400 Hz) must score < 60 against RP.

    This is the sensitivity floor: a heavy accent must not score in the native range.
    """
    from accent_coach.comparison.vowels import score_vowels

    shifted = _sentence([_vf(ph, f1 + 400, f2) for ph, (f1, f2) in _RP.items()])
    score = score_vowels(shifted, reference_norms=_RP)
    assert score < 60, (
        f"Heavy-accent speaker (all F1+400 Hz) scored {score:.1f}. "
        "Expected < 60 — the scorer must be sensitive to large accent deviations."
    )


def test_near_native_beats_foreign_accent():
    """A 1σ-off speaker must score substantially higher than a 4σ-off speaker.

    Validates that the score is monotonically sensitive to deviation magnitude,
    not just a yes/no threshold.
    """
    from accent_coach.comparison.vowels import score_vowels

    sigma_f1 = 50.0  # typical within-RP F1 spread per phoneme
    near_native = _sentence([_vf(ph, f1 + sigma_f1, f2) for ph, (f1, f2) in _RP.items()])
    foreign = _sentence([_vf(ph, f1 + 4 * sigma_f1, f2) for ph, (f1, f2) in _RP.items()])

    score_near = score_vowels(near_native, reference_norms=_RP)
    score_far = score_vowels(foreign, reference_norms=_RP)

    assert score_near > score_far + 10, (
        f"Near-native={score_near:.1f}, foreign={score_far:.1f}. "
        "Expected gap > 10 pts — score must be proportional to deviation distance."
    )


def test_cross_accent_discrimination_is_symmetric():
    """Dialect score advantage must be symmetric: RP-vs-RP > RP-vs-GA and GA-vs-GA > GA-vs-RP.

    This test guards against the norms being accidentally identical or inverted.
    """
    from accent_coach.comparison.vowels import score_vowels

    rp_speaker = _sentence([_vf(ph, f1, f2) for ph, (f1, f2) in _RP.items()])
    ga_speaker = _sentence([_vf(ph, f1, f2) for ph, (f1, f2) in _GA.items()])

    rp_on_rp = score_vowels(rp_speaker, reference_norms=_RP)
    rp_on_ga = score_vowels(rp_speaker, reference_norms=_GA)
    ga_on_ga = score_vowels(ga_speaker, reference_norms=_GA)
    ga_on_rp = score_vowels(ga_speaker, reference_norms=_RP)

    assert rp_on_rp > rp_on_ga, (
        f"RP speaker: self-score={rp_on_rp:.1f} must exceed cross-accent={rp_on_ga:.1f}"
    )
    assert ga_on_ga > ga_on_rp, (
        f"GA speaker: self-score={ga_on_ga:.1f} must exceed cross-accent={ga_on_rp:.1f}"
    )


# ---------------------------------------------------------------------------
# Rhythm scoring: absolute-mode ordering invariants
# ---------------------------------------------------------------------------


def _alternating_durs(long: float, short: float, n: int = 8) -> list[float]:
    """Strictly alternating long/short pattern — nPVI ≈ 100 * |long-short| / mean(long, short)."""
    return [long if i % 2 == 0 else short for i in range(n)]


# Pre-computed via _alternating_durs and the nPVI formula:
# nPVI([0.25, 0.148]*4) = 100 * 0.102/0.199 = 51.3  (corpus centre = _NPVI_REF)
# nPVI([0.20, 0.133]*4) = 100 * 0.067/0.1665 = 40.2 (hybrid-detected native p25 = RP_NPVI_MIN)
# nPVI([0.15]*8) = 0.0  (perfectly syllable-timed)
_DURS_CENTRE = _alternating_durs(0.25, 0.148)       # nPVI ≈ 51  (hybrid corpus centre)
_DURS_HYBRID_P25 = _alternating_durs(0.20, 0.133)   # nPVI ≈ 40  (hybrid native p25 = RP_NPVI_MIN)
_DURS_SYLLABLE_TIMED = [0.15] * 8  # nPVI = 0  (e.g. Spanish, Japanese)
_DURS_L2 = [0.175] * 8  # nPVI ≈ 0  (flat, slightly long — typical L2)


def test_stress_timed_beats_syllable_timed_in_absolute_mode():
    """English-like stress-timing must score higher than syllable-timing in absolute mode.

    Absolute mode compares against RP_NPVI_MIN/MAX (40–62, centre 51).
    Syllable-timed nPVI ≈ 0 is far from this range; stress-timed ≈ 51 is at centre.
    This is the broadest possible sanity check on the absolute rhythm scorer.
    """
    from accent_coach.comparison.rhythm import score_rhythm

    npvi_centre = compute_npvi(_DURS_CENTRE)
    npvi_syllable = compute_npvi(_DURS_SYLLABLE_TIMED)
    assert npvi_centre > 40, f"Centre fixture nPVI={npvi_centre:.1f} unexpectedly low"
    assert npvi_syllable < 5, f"Syllable-timed fixture nPVI={npvi_syllable:.1f} unexpectedly high"

    score_stress = score_rhythm(_sentence([], _DURS_CENTRE)).score
    score_syllable = score_rhythm(_sentence([], _DURS_SYLLABLE_TIMED)).score

    assert score_stress > score_syllable + 30, (
        f"Stress-timed={score_stress:.1f} vs syllable-timed={score_syllable:.1f}. "
        "Expected gap > 30 pts — rhythm scorer must distinguish English from syllable-timed languages."
    )


def test_corpus_centre_npvi_scores_near_perfect():
    """A speaker whose nPVI is exactly at the corpus centre (51) must score >= 90 in absolute mode.

    This verifies that the _DECAY constant is not miscalibrated: a perfectly
    average native speaker should not be penalized by the absolute scorer.
    """
    from accent_coach.comparison.rhythm import score_rhythm

    npvi = compute_npvi(_DURS_CENTRE)
    assert 45 < npvi < 58, f"Fixture nPVI={npvi:.1f} not near corpus centre — check _DURS_CENTRE"

    score = score_rhythm(_sentence([], _DURS_CENTRE)).score
    assert score >= 90, (
        f"nPVI={npvi:.1f} (at corpus centre) scored {score:.1f}. Expected >= 90. "
        "Check _DECAY constant — native speakers at corpus centre must score near-perfect."
    )


def test_native_acoustic_p25_npvi_not_penalized_as_l2():
    """A speaker at hybrid-detected native nPVI p25 (≈40 = RP_NPVI_MIN) must score >= 60.

    Hybrid detection is used consistently everywhere (bench + live pipeline).
    The corpus hybrid p25 is ≈40 which equals RP_NPVI_MIN — a native at the low
    edge of the reference range must not score like an L2 learner.
    """
    from accent_coach.comparison.rhythm import score_rhythm

    npvi = compute_npvi(_DURS_HYBRID_P25)
    assert 35 < npvi < 45, f"Fixture nPVI={npvi:.1f} not in hybrid p25 range — check _DURS_HYBRID_P25"

    score = score_rhythm(_sentence([], _DURS_HYBRID_P25)).score
    assert score >= 60, (
        f"nPVI={npvi:.1f} (hybrid native p25 = RP_NPVI_MIN) scored {score:.1f}. Expected >= 60. "
        "A native speaker at the minimum of the reference range must not score like L2."
    )


def test_native_like_beats_l2_in_comparison_mode():
    """In comparison mode, a native-like timing pattern must score higher than flat L2 timing.

    When comparing against a TTS target, a speaker matching the TTS timing pattern
    should score considerably higher than one with uniform flat durations.
    """
    from accent_coach.comparison.rhythm import score_rhythm

    target = _sentence([], _DURS_CENTRE)
    native_like = _sentence([], _DURS_CENTRE)
    l2_like = _sentence([], _DURS_L2)

    score_native = score_rhythm(native_like, target=target).score
    score_l2 = score_rhythm(l2_like, target=target).score

    assert score_native > score_l2 + 15, (
        f"Native-like={score_native:.1f} vs L2-like={score_l2:.1f}. "
        "Expected gap > 15 pts in comparison mode."
    )


def test_identical_speaker_scores_100_in_comparison_mode():
    """Comparing a speaker against an identical target must yield score 100.

    Regression guard for the pattern-correlation and nPVI components —
    if either drifts from the identity case, this test catches it.
    """
    from accent_coach.comparison.rhythm import score_rhythm

    sentence = _sentence([], _DURS_CENTRE)
    score = score_rhythm(sentence, target=sentence).score
    assert score == pytest.approx(100.0, abs=1.0), (
        f"Identity comparison scored {score:.1f}. Expected 100."
    )


# ---------------------------------------------------------------------------
# Function-word inflation: edge cases
# ---------------------------------------------------------------------------


def test_no_function_words_does_not_inflate_composite():
    """A sentence with only content words must not get a free 100-pt FW bonus.

    BUG: _function_word_analysis() returns (100.0, []) when no function words are
    matched, and the 0.2 FW weight applies that fictional perfect score to the composite.
    Fix: return (None, []) when matched==0 and redistribute the FW weight.
    """
    from accent_coach.comparison.rhythm import score_rhythm

    # Build phonemes for content words only (no function words from FUNCTION_WORDS set)
    content_phonemes = [
        PhonemeInstance(
            phoneme="æ", arpabet="AE1",
            start_time=i * 0.2, end_time=(i + 1) * 0.2,
            sentence_id=1, word=word, is_stressed=True,
        )
        for i, word in enumerate(["cat", "black", "strong", "jump"])
    ]
    target_phonemes = [
        PhonemeInstance(
            phoneme="æ", arpabet="AE1",
            start_time=i * 0.15, end_time=(i + 1) * 0.15,
            sentence_id=1, word=word, is_stressed=True,
        )
        for i, word in enumerate(["cat", "black", "strong", "jump"])
    ]

    user = SentenceAnalysis(
        sentence_id=1, sentence_type="statement",
        duration_s=0.8, syllable_durations=[0.2, 0.2, 0.2, 0.2],
        pitch_contour=[0.5] * 50,
        stress_pattern=[True, False, True, False],
        vowels=[], stops=[], phonemes=content_phonemes,
    )
    target = SentenceAnalysis(
        sentence_id=1, sentence_type="statement",
        duration_s=0.6, syllable_durations=[0.15, 0.15, 0.15, 0.15],
        pitch_contour=[0.5] * 50,
        stress_pattern=[True, False, True, False],
        vowels=[], stops=[], phonemes=target_phonemes,
    )

    result = score_rhythm(user, target=target)
    # If FW returns 100 when unmatched, composite gets an artificial boost.
    # Correct behaviour: function_word_score is None and weight redistributed.
    assert result.function_word_score is None, (
        f"function_word_score={result.function_word_score} should be None when no FW matched. "
        "BUG: _function_word_analysis returns 100.0 when matched==0, inflating composite."
    )
