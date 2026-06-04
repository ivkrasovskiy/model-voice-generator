"""TDD tests for Signal 4 (over-stressing) and Signal 5 (per-syllable outliers)."""
from __future__ import annotations

import pytest

from accent_coach.comparison.rhythm import score_rhythm
from accent_coach.models import PhonemeInstance, SentenceAnalysis

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TARGET_DURS = [125, 75, 120, 80, 130, 70, 125, 75, 120, 80, 125]


def _sentence(durs_ms: list[float], phonemes=None) -> SentenceAnalysis:
    durs_s = [d / 1000.0 for d in durs_ms]
    return SentenceAnalysis(
        sentence_id=0,
        sentence_type="statement",
        duration_s=sum(durs_s),
        syllable_durations=durs_s,
        pitch_contour=[],
        stress_pattern=[],
        vowels=[],
        stops=[],
        phonemes=phonemes or [],
    )


def _phonemes(word_dur_pairs: list[tuple[str, float]]) -> list[PhonemeInstance]:
    """(word, dur_ms) -> list of PhonemeInstance with sequential timestamps."""
    result, t = [], 0.0
    for word, dur_ms in word_dur_pairs:
        dur_s = dur_ms / 1000.0
        result.append(PhonemeInstance(
            phoneme="V", arpabet="AH",
            start_time=t, end_time=t + dur_s,
            sentence_id=0, word=word, is_stressed=False,
        ))
        t += dur_s
    return result


# ---------------------------------------------------------------------------
# Signal 1 — flat rhythm (French / Italian / Mandarin)
# ---------------------------------------------------------------------------

class TestSignal1FlatRhythm:
    """nPVI too flat (user_npvi < ref - 10) should fire 'too even' diagnostic."""

    @pytest.mark.parametrize("accent,durs_ms", [
        ("french",   [108, 92, 105, 95, 110, 90, 107, 93, 106, 94, 108]),
        ("italian",  [104, 96, 103, 97, 105, 95, 104, 96, 103, 97, 104]),
        ("mandarin", [100, 99, 101, 100, 102, 99, 101, 100, 100, 101, 99]),
    ])
    def test_flat_rhythm_fires_too_even(self, accent: str, durs_ms: list[float]):
        target = _sentence(TARGET_DURS)
        user = _sentence(durs_ms)
        result = score_rhythm(user, target=target)
        assert any("too even" in d or "too flat" in d for d in result.diagnostics), (
            f"[{accent}] expected 'too even'/'too flat' diagnostic, got: {result.diagnostics}"
        )

    @pytest.mark.parametrize("accent,durs_ms", [
        ("french",   [108, 92, 105, 95, 110, 90, 107, 93, 106, 94, 108]),
        ("italian",  [104, 96, 103, 97, 105, 95, 104, 96, 103, 97, 104]),
        ("mandarin", [100, 99, 101, 100, 102, 99, 101, 100, 100, 101, 99]),
    ])
    def test_flat_rhythm_npvi_below_30(self, accent: str, durs_ms: list[float]):
        user = _sentence(durs_ms)
        result = score_rhythm(user)
        assert result.npvi < 30, (
            f"[{accent}] expected nPVI < 30, got {result.npvi:.1f}"
        )


# ---------------------------------------------------------------------------
# Signal 4 (NEW) — over-stressing
# ---------------------------------------------------------------------------

# Over-stressed profile: nPVI should be well above target nPVI (~48) + 10 = 58
OVER_STRESS_DURS = [55, 145, 60, 148, 52, 145, 57, 143, 55, 145, 58]


class TestSignal4OverStressing:
    def test_over_stress_fires_diagnostic(self):
        target = _sentence(TARGET_DURS)
        user = _sentence(OVER_STRESS_DURS)
        result = score_rhythm(user, target=target)
        keywords = ("over-stressed", "over stressed", "too deliberate", "exaggerated", "over-stress")
        assert any(
            any(kw in d.lower() for kw in keywords) for d in result.diagnostics
        ), f"Expected over-stress diagnostic, got: {result.diagnostics}"

    def test_over_stress_score_is_low(self):
        target = _sentence(TARGET_DURS)
        user = _sentence(OVER_STRESS_DURS)
        result = score_rhythm(user, target=target)
        assert result.score < 70, (
            f"Over-stressed nPVI should score < 70, got {result.score:.1f}"
        )

    def test_signal_1_and_4_are_mutually_exclusive(self):
        target = _sentence(TARGET_DURS)

        # Flat user — should fire Signal 1, not Signal 4
        flat_user = _sentence([104, 96, 103, 97, 105, 95, 104, 96, 103, 97, 104])
        flat_result = score_rhythm(flat_user, target=target)
        flat_diags = " ".join(flat_result.diagnostics).lower()
        assert "too even" in flat_diags or "too flat" in flat_diags, (
            f"Flat user should fire Signal 1, got: {flat_result.diagnostics}"
        )
        over_stress_kws = ("over-stressed", "over stressed", "too deliberate", "exaggerated")
        assert not any(kw in flat_diags for kw in over_stress_kws), (
            f"Flat user should NOT fire Signal 4, but got: {flat_result.diagnostics}"
        )

        # Over-stressed user — should fire Signal 4, not Signal 1
        over_result = score_rhythm(_sentence(OVER_STRESS_DURS), target=target)
        over_diags = " ".join(over_result.diagnostics).lower()
        assert any(kw in over_diags for kw in over_stress_kws), (
            f"Over-stressed user should fire Signal 4, got: {over_result.diagnostics}"
        )
        assert "too even" not in over_diags and "too flat" not in over_diags, (
            f"Over-stressed user should NOT fire Signal 1, but got: {over_result.diagnostics}"
        )


# ---------------------------------------------------------------------------
# Signal 2 — function word inflation (Slavic profile)
# ---------------------------------------------------------------------------

class TestSignal2FunctionWordInflation:
    def test_slavic_inflated_function_words(self):
        target_phs = _phonemes([
            ("think", 125), ("of", 45), ("the", 35),
            ("most", 130), ("interesting", 180), ("things", 130),
        ])
        user_phs = _phonemes([
            ("think", 130), ("of", 90), ("the", 75),
            ("most", 135), ("interesting", 185), ("things", 135),
        ])
        target = _sentence([125, 45, 35, 130, 180, 130], phonemes=target_phs)
        user = _sentence([130, 90, 75, 135, 185, 135], phonemes=user_phs)
        result = score_rhythm(user, target=target)
        # "of" ratio = 90/45 = 2.0 and "the" ratio = 75/35 ≈ 2.14 — both > 1.7
        assert "of" in result.inflated_function_words or "the" in result.inflated_function_words, (
            f"Expected 'of' or 'the' inflated, got: {result.inflated_function_words}"
        )
        assert any(
            "of" in d or "the" in d for d in result.diagnostics
        ), f"Expected FW diagnostic, got: {result.diagnostics}"

    def test_correct_function_words_not_flagged(self):
        # "of" = 48ms user vs 45ms target → ratio 1.07 < 1.7 → not flagged
        target_phs = _phonemes([("think", 125), ("of", 45), ("things", 130)])
        user_phs = _phonemes([("think", 128), ("of", 48), ("things", 133)])
        target = _sentence([125, 45, 130], phonemes=target_phs)
        user = _sentence([128, 48, 133], phonemes=user_phs)
        result = score_rhythm(user, target=target)
        assert "of" not in result.inflated_function_words, (
            f"'of' should not be flagged at ratio 1.07, got: {result.inflated_function_words}"
        )


# ---------------------------------------------------------------------------
# Signal 3 — pattern mismatch
# ---------------------------------------------------------------------------

class TestSignal3PatternMismatch:
    def test_inverted_pattern_fires_diagnostic(self):
        target_durs = [125, 75, 120, 80, 130, 70, 125, 75]
        user_durs =   [75, 125, 80, 120, 70, 130, 75, 125]
        target = _sentence(target_durs)
        user = _sentence(user_durs)
        result = score_rhythm(user, target=target)
        assert result.pattern_correlation is not None
        assert result.pattern_correlation < 40, (
            f"Inverted pattern should have correlation < 40, got {result.pattern_correlation:.1f}"
        )
        assert any("pattern" in d.lower() or "timing" in d.lower() for d in result.diagnostics), (
            f"Expected pattern/timing diagnostic, got: {result.diagnostics}"
        )

    def test_correct_pattern_no_diagnostic(self):
        target_durs = [125, 75, 120, 80, 130, 70, 125, 75]
        user_durs =   [128, 77, 123, 82, 133, 72, 128, 77]
        target = _sentence(target_durs)
        user = _sentence(user_durs)
        result = score_rhythm(user, target=target)
        assert result.pattern_correlation is not None
        assert result.pattern_correlation > 80, (
            f"Similar pattern should score > 80, got {result.pattern_correlation:.1f}"
        )
        assert not any(
            "pattern" in d.lower() or "timing" in d.lower() for d in result.diagnostics
        ), f"Close pattern should not fire diagnostic, got: {result.diagnostics}"


# ---------------------------------------------------------------------------
# Signal 5 (NEW) — per-syllable outliers
# ---------------------------------------------------------------------------

# target: [120, 80] * 4, mean = 100
# user:   [120, 200, 120, 80, 120, 200, 120, 80], mean = 130
# user_norm pos 1: 200/130 ≈ 1.538, target_norm pos 1: 80/100 = 0.8 → diff = 0.738 > 0.5 ✓
# user_norm pos 5: same as pos 1 → diff > 0.5 ✓
OUTLIER_TARGET_DURS = [120, 80, 120, 80, 120, 80, 120, 80]
OUTLIER_USER_DURS =   [120, 200, 120, 80, 120, 200, 120, 80]


class TestSignal5PerSyllableOutliers:
    def test_specific_long_syllables_reported(self):
        target = _sentence(OUTLIER_TARGET_DURS)
        user = _sentence(OUTLIER_USER_DURS)
        result = score_rhythm(user, target=target)
        assert hasattr(result, "outlier_syllables"), (
            "RhythmBreakdown must have 'outlier_syllables' field"
        )
        assert result.outlier_syllables, (
            f"Expected non-empty outlier_syllables, got: {result.outlier_syllables}"
        )
        positions = [pos for pos, _ in result.outlier_syllables]
        assert 1 in positions, f"Position 1 should be an outlier, got: {positions}"
        assert 5 in positions, f"Position 5 should be an outlier, got: {positions}"

    def test_uniform_small_noise_no_outliers(self):
        target = _sentence(OUTLIER_TARGET_DURS)
        user = _sentence([123, 82, 122, 79, 121, 81, 123, 80])
        result = score_rhythm(user, target=target)
        assert hasattr(result, "outlier_syllables"), (
            "RhythmBreakdown must have 'outlier_syllables' field"
        )
        assert result.outlier_syllables == [], (
            f"Small noise should produce no outliers, got: {result.outlier_syllables}"
        )

    def test_outlier_diagnostic_message(self):
        target = _sentence(OUTLIER_TARGET_DURS)
        user = _sentence(OUTLIER_USER_DURS)
        result = score_rhythm(user, target=target)
        assert any(
            "syllable" in d.lower() or "position" in d.lower() for d in result.diagnostics
        ), f"Expected syllable/position diagnostic, got: {result.diagnostics}"


# ---------------------------------------------------------------------------
# Owner voice profile (real calibration data: hybrid nPVI ≈ 74, score ≈ 52)
# ---------------------------------------------------------------------------
#
# Owner's signature: correct stress placement (alternating pattern matches TTS)
# but every contrast is over-exaggerated (stressed ≈ 137 ms, unstressed ≈ 63 ms).
# TTS target has stressed ≈ 125 ms, unstressed ≈ 75 ms (nPVI ≈ 48).
#
# Predicted signals:
#   Signal 4 (over-stress): owner_nPVI (≈72) - target_nPVI (≈48) = 24 >> 10 → FIRES
#   Signal 1 (too flat):    owner_nPVI > target - 10 → does NOT fire
#   Signal 3 (pattern):     same alternating shape → high correlation → does NOT fire
#   Signal 5 (outliers):    per-position diff ≈ 0.07–0.10 < 0.5 threshold → does NOT fire
#
OWNER_DURS = [137, 63, 135, 65, 138, 62, 136, 64, 134, 66, 136]


class TestOwnerVoiceProfile:
    def test_owner_fires_only_over_stress_signal(self):
        target = _sentence(TARGET_DURS)
        owner = _sentence(OWNER_DURS)
        result = score_rhythm(owner, target=target)

        # Signal 4 must fire
        assert any(
            "over-stress" in d.lower() or "deliberate" in d.lower() or "exaggerat" in d.lower()
            for d in result.diagnostics
        ), f"Signal 4 (over-stress) should fire for owner profile, got: {result.diagnostics}"

        # Signal 1 must NOT fire (owner is over-stressed, not flat)
        assert not any("too even" in d.lower() or "too flat" in d.lower() for d in result.diagnostics), (
            f"Signal 1 (flat rhythm) should not fire for owner, got: {result.diagnostics}"
        )

        # Signal 3 must NOT fire (pattern shape matches TTS, just exaggerated)
        assert not any("pattern" in d.lower() for d in result.diagnostics), (
            f"Signal 3 (pattern mismatch) should not fire for owner, got: {result.diagnostics}"
        )

        # Signal 5 must NOT fire (no individual outlier positions, globally uniform exaggeration)
        assert result.outlier_syllables == [], (
            f"Signal 5 (per-syllable) should not fire for owner, got: {result.outlier_syllables}"
        )

    def test_owner_score_is_lower_than_native_conversational(self):
        # Synthetic data scores ~72 (pattern correlation is artificially high at 99.7
        # because the fixture is clean alternating; real audio gives ≈52 due to FW
        # analysis + natural variance). The key invariant is: below native conversational (78–86).
        target = _sentence(TARGET_DURS)
        owner = _sentence(OWNER_DURS)
        result = score_rhythm(owner, target=target)
        assert result.score < 78, (
            f"Owner should score below native conversational (78–86), got {result.score:.1f}"
        )
        assert result.npvi > 65, (
            f"Owner nPVI should be > 65 (bench measured ≈74), got {result.npvi:.1f}"
        )


# ---------------------------------------------------------------------------
# No false positives — native-quality recording
# ---------------------------------------------------------------------------

class TestNoFalsePositives:
    def test_native_quality_no_diagnostics(self):
        # Target nPVI ≈ 50; user durations are minor perturbations → user nPVI ≈ 49
        # Function word ratios: "of"=48/45=1.07, "the"=37/35=1.06 — both < 1.7 → not flagged
        target = _sentence(
            [125, 75, 120, 80, 130, 70, 125, 75],
            phonemes=_phonemes([
                ("think", 125), ("of", 45), ("the", 35),
                ("most", 130), ("and", 70), ("the", 125),
                ("great", 75), ("things", 120),
            ]),
        )
        user = _sentence(
            [128, 77, 122, 82, 133, 72, 127, 77],
            phonemes=_phonemes([
                ("think", 128), ("of", 48), ("the", 37),
                ("most", 82), ("and", 72), ("the", 72),
                ("great", 77), ("things", 122),
            ]),
        )
        result = score_rhythm(user, target=target)
        assert result.score > 75, (
            f"Native-quality user should score > 75, got {result.score:.1f}"
        )
        assert result.diagnostics == [], (
            f"Native-quality user should have no diagnostics, got: {result.diagnostics}"
        )
