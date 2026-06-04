"""Unit tests for the upgraded rhythm comparison module."""
from __future__ import annotations

import pytest

from accent_coach.comparison.rhythm import (
    FUNCTION_WORDS,
    _function_word_analysis,
    _syllable_pattern_score,
    score_rhythm,
)
from accent_coach.models import PhonemeInstance, RhythmBreakdown, SentenceAnalysis

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _phoneme(word: str, start: float, end: float) -> PhonemeInstance:
    return PhonemeInstance(
        phoneme="æ", arpabet="AE1",
        start_time=start, end_time=end,
        sentence_id=1, word=word, is_stressed=True,
    )


def _sentence(syllable_durations: list[float], phonemes: list[PhonemeInstance] | None = None) -> SentenceAnalysis:
    return SentenceAnalysis(
        sentence_id=1,
        sentence_type="statement",
        duration_s=sum(syllable_durations),
        syllable_durations=syllable_durations,
        pitch_contour=[0.5] * 50,
        stress_pattern=[True, False] * (len(syllable_durations) // 2 + 1),
        vowels=[],
        stops=[],
        phonemes=phonemes or [],
    )


# ---------------------------------------------------------------------------
# _syllable_pattern_score
# ---------------------------------------------------------------------------

def test_identical_patterns_score_100():
    durs = [0.1, 0.3, 0.05, 0.25, 0.08]
    assert _syllable_pattern_score(durs, durs) == pytest.approx(100.0, abs=1.0)


def test_inverted_pattern_scores_low():
    """Long-short vs short-long → strong negative correlation → low score."""
    user = [0.3, 0.05, 0.3, 0.05, 0.3]
    target = [0.05, 0.3, 0.05, 0.3, 0.05]
    score = _syllable_pattern_score(user, target)
    assert score < 20.0, f"Inverted pattern score should be < 20, got {score:.1f}"


def test_flat_durations_returns_50():
    """If one speaker has all equal durations, correlation is undefined → neutral 50."""
    user = [0.1] * 6
    target = [0.05, 0.3, 0.05, 0.3, 0.05, 0.3]
    score = _syllable_pattern_score(user, target)
    assert score == pytest.approx(50.0, abs=1.0)


def test_short_sequence_returns_50():
    assert _syllable_pattern_score([0.1, 0.2], [0.2, 0.3]) == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# _function_word_analysis
# ---------------------------------------------------------------------------

def _fw_phonemes(word_dur_pairs: list[tuple[str, float]]) -> list[PhonemeInstance]:
    """Build a minimal phoneme list: one phoneme per word, duration as given (seconds)."""
    phonemes = []
    t = 0.0
    for word, dur in word_dur_pairs:
        phonemes.append(_phoneme(word, t, t + dur))
        t += dur
    return phonemes


def test_no_inflated_words_when_durations_match():
    words = [("the", 0.03), ("cat", 0.15), ("sat", 0.18)]
    user_ph = _fw_phonemes(words)
    target_ph = _fw_phonemes(words)
    score, inflated = _function_word_analysis(user_ph, target_ph)
    assert score == pytest.approx(100.0)
    assert inflated == []


def test_inflated_function_word_detected():
    """User says 'the' for 90ms; target says 25ms → ratio 3.6 → inflated."""
    user_ph = _fw_phonemes([("the", 0.09), ("cat", 0.15)])
    target_ph = _fw_phonemes([("the", 0.025), ("cat", 0.18)])
    score, inflated = _function_word_analysis(user_ph, target_ph)
    assert "the" in inflated
    assert score < 100.0


def test_content_words_not_flagged():
    """Content words are not in FUNCTION_WORDS — score is None (weight redistributed)."""
    user_ph = _fw_phonemes([("garden", 0.4), ("grows", 0.5)])
    target_ph = _fw_phonemes([("garden", 0.1), ("grows", 0.1)])
    score, inflated = _function_word_analysis(user_ph, target_ph)
    assert inflated == []
    assert score is None


def test_function_words_set_contains_expected():
    for w in ("the", "a", "to", "and", "is", "was", "that", "which"):
        assert w in FUNCTION_WORDS


# ---------------------------------------------------------------------------
# score_rhythm integration
# ---------------------------------------------------------------------------

def test_score_rhythm_no_target_uses_rp_reference():
    """Without a target, score is purely nPVI vs RP reference."""
    # alternating 100/180 ms → nPVI ≈ 57, inside the RP target range
    user = _sentence([0.1, 0.18, 0.1, 0.18, 0.1, 0.18])
    result = score_rhythm(user)
    assert isinstance(result, RhythmBreakdown)
    assert 50.0 <= result.npvi <= 70.0, f"expected nPVI ~57, got {result.npvi:.1f}"
    assert result.score > 60.0, f"RP-range nPVI should score >60, got {result.score:.1f}"
    assert result.pattern_correlation is None
    assert result.function_word_score is None
    assert result.inflated_function_words == []


def test_score_rhythm_even_durations_score_low_vs_rp():
    """All equal durations → nPVI ≈ 0 → very low score against RP reference."""
    user = _sentence([0.1] * 8)
    result = score_rhythm(user)
    assert result.npvi < 5.0
    assert result.score < 30.0
    assert any("even" in d or "nPVI" in d for d in result.diagnostics)


def test_score_rhythm_with_target_uses_three_signals():
    """With a matching target, all three signals should contribute."""
    durs = [0.05, 0.22, 0.04, 0.20, 0.06, 0.18]
    user = _sentence(durs)
    target = _sentence(durs)
    result = score_rhythm(user, target=target)
    assert result.score > 85.0
    assert result.pattern_correlation is not None
    assert result.pattern_correlation > 90.0


def test_score_rhythm_with_target_and_phonemes_enables_fw():
    """Function word score activates when both have phoneme data."""
    durs = [0.05, 0.2, 0.04, 0.18]
    user_ph = _fw_phonemes([("the", 0.08), ("dog", 0.2), ("and", 0.09), ("cat", 0.18)])
    target_ph = _fw_phonemes([("the", 0.025), ("dog", 0.22), ("and", 0.03), ("cat", 0.20)])
    user = _sentence(durs, phonemes=user_ph)
    target = _sentence(durs, phonemes=target_ph)
    result = score_rhythm(user, target=target)
    assert result.function_word_score is not None
    # "the" and "and" are inflated (user 3× target)
    assert len(result.inflated_function_words) >= 1
    assert result.score < 95.0  # should not be perfect


def test_diagnostics_mention_inflated_words():
    durs = [0.05, 0.2, 0.04, 0.18]
    user_ph = _fw_phonemes([("the", 0.12), ("big", 0.2), ("cat", 0.18), ("sat", 0.18)])
    target_ph = _fw_phonemes([("the", 0.025), ("big", 0.22), ("cat", 0.20), ("sat", 0.19)])
    user = _sentence(durs, phonemes=user_ph)
    target = _sentence(durs, phonemes=target_ph)
    result = score_rhythm(user, target=target)
    assert any("the" in d for d in result.diagnostics)


# ---------------------------------------------------------------------------
# Bug 1 fix: pattern correlation returns neutral when counts diverge > 30%
# ---------------------------------------------------------------------------

def test_pattern_score_neutral_on_large_count_mismatch():
    """If user detects 5 syllables and target detects 9 (>30% diff), score is 50."""
    user_durs = [0.1, 0.3, 0.05, 0.25, 0.08]         # 5 syllables
    target_durs = [0.025, 0.3, 0.05, 0.25, 0.08, 0.15, 0.2, 0.06, 0.18]  # 9 syllables
    score = _syllable_pattern_score(user_durs, target_durs)
    assert score == pytest.approx(50.0), (
        f"Count mismatch >30% should return 50 (neutral), got {score:.1f}"
    )


def test_pattern_score_computes_when_counts_close():
    """Counts within 30% diff → real correlation is computed."""
    durs_a = [0.1, 0.3, 0.05, 0.25, 0.08]   # 5
    durs_b = [0.1, 0.3, 0.05, 0.25, 0.08, 0.1]  # 6  (diff = 1/6 = 17% < 30%)
    score = _syllable_pattern_score(durs_a, durs_b)
    # Should be a real value (identical prefix → high correlation)
    assert score > 90.0, f"Similar patterns should score high, got {score:.1f}"


# ---------------------------------------------------------------------------
# Bug 2 fix: FW score omitted from composite when phoneme data unavailable
# ---------------------------------------------------------------------------

def test_composite_excludes_fw_when_no_phonemes():
    """With target but no phoneme data, composite must NOT include the default fw=100 boost."""
    durs = [0.1, 0.18, 0.1, 0.18, 0.1, 0.18]
    user = _sentence(durs, phonemes=[])      # no phonemes
    target = _sentence(durs, phonemes=[])    # no phonemes
    result = score_rhythm(user, target=target)
    assert result.function_word_score is None
    # Composite should be 0.5*nPVI + 0.5*pattern, NOT 0.4+0.4+0.2*100
    # For identical clips: nPVI delta=0 → npvi_score=100; pattern=100 → composite=100
    assert result.score == pytest.approx(100.0, abs=1.0)


def test_composite_lower_with_bad_fw_than_without_phonemes():
    """When FW analysis fires and catches inflation, composite drops vs no-phoneme baseline."""
    durs = [0.05, 0.2, 0.04, 0.18]
    user_ph = _fw_phonemes([("the", 0.12), ("cat", 0.2), ("and", 0.10), ("sat", 0.18)])
    tgt_ph  = _fw_phonemes([("the", 0.025), ("cat", 0.22), ("and", 0.025), ("sat", 0.19)])
    user_with = _sentence(durs, phonemes=user_ph)
    tgt_with  = _sentence(durs, phonemes=tgt_ph)
    user_no   = _sentence(durs, phonemes=[])
    tgt_no    = _sentence(durs, phonemes=[])
    result_with = score_rhythm(user_with, target=tgt_with)
    result_no   = score_rhythm(user_no,   target=tgt_no)
    # No-phoneme baseline uses 50/50 nPVI+pattern; with inflated FW it drops
    assert result_with.score < result_no.score, (
        f"FW inflation should lower score: {result_with.score:.1f} vs {result_no.score:.1f}"
    )
