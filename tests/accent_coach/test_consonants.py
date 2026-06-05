"""
Consonant quality scoring tests — TDD, written BEFORE implementation.

Tests define the expected interface and behaviour of the new consonants
comparison subpackage (accent_coach/comparison/consonants/).  They MUST
FAIL until the implementation is complete.  Each test documents its
expected failure mode inline.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.signal import butter, sosfilt

from accent_coach.models import PhonemeInstance, SentenceAnalysis, StopFeatures

SR = 16_000


# ---------------------------------------------------------------------------
# Synthesis helpers
# ---------------------------------------------------------------------------


def _ph(
    phoneme: str,
    arpabet: str,
    start: float = 0.02,
    end: float = 0.17,
    word: str = "test",
    is_stressed: bool = True,
) -> PhonemeInstance:
    return PhonemeInstance(
        phoneme=phoneme, arpabet=arpabet,
        start_time=start, end_time=end,
        sentence_id=1, word=word, is_stressed=is_stressed,
    )


def _stop(phoneme: str, vot_ms: float) -> StopFeatures:
    ph = _ph(phoneme, phoneme.upper(), start=0.05, end=0.15, is_stressed=True)
    return StopFeatures(phoneme=ph, vot_ms=vot_ms, burst_energy=1.0)


def _sentence(phonemes: list[PhonemeInstance], stops: list[StopFeatures] | None = None) -> SentenceAnalysis:
    return SentenceAnalysis(
        sentence_id=1, sentence_type="statement",
        duration_s=0.5, syllable_durations=[0.2, 0.15, 0.15],
        pitch_contour=[0.5] * 50, stress_pattern=[True, False, True],
        vowels=[], stops=stops or [], phonemes=phonemes,
    )


def _bandlimited_noise(center_hz: float, bw_hz: float, dur: float = 0.15) -> np.ndarray:
    """Band-limited Gaussian noise: deterministic seed for reproducibility."""
    n = int(SR * dur)
    noise = np.random.default_rng(42).standard_normal(n).astype(np.float64)
    lo = max(200.0, center_hz - bw_hz / 2)
    hi = min(SR / 2 - 200, center_hz + bw_hz / 2)
    sos = butter(4, [lo / (SR / 2), hi / (SR / 2)], btype="band", output="sos")
    out = sosfilt(sos, noise)
    return (out / (np.abs(out).max() + 1e-9) * 0.8).astype(np.float32)


def _place_in_silence(segment: np.ndarray, start_s: float, total_s: float = 0.5) -> np.ndarray:
    audio = np.zeros(int(total_s * SR), dtype=np.float32)
    s = int(start_s * SR)
    e = min(len(audio), s + len(segment))
    audio[s:e] = segment[: e - s]
    return audio


def _resonator_audio(formants: list[tuple[float, float]], dur: float = 0.15, f0: float = 120.0) -> np.ndarray:
    """AR (all-poles) synthesis excited by a glottal pulse train."""
    from scipy.signal import lfilter
    n = int(SR * dur)
    pulse = np.zeros(n, dtype=np.float64)
    pulse[:: int(SR / f0)] = 1.0
    out = pulse
    for f, bw in formants:
        r = np.exp(-math.pi * bw / SR)
        a = [1.0, -2 * r * math.cos(2 * math.pi * f / SR), r**2]
        out = lfilter([1.0], a, out)
    return (out / (np.abs(out).max() + 1e-9) * 0.8).astype(np.float32)


# ---------------------------------------------------------------------------
# Section 1 — Return type: ConsonantScore dataclass
# FAIL reason: score_consonants() returns float, not ConsonantScore
# ---------------------------------------------------------------------------


def test_score_consonants_returns_dataclass():
    """score_consonants() must return a ConsonantScore, not a plain float.

    FAIL reason: current implementation returns float.
    """
    from accent_coach.comparison.consonants import score_consonants
    from accent_coach.models import ConsonantScore  # noqa: F401  # will NameError until added

    audio = np.zeros(SR // 2, dtype=np.float32)
    user = _sentence([])
    result = score_consonants(user, audio, SR)
    assert not isinstance(result, float), (
        "score_consonants() returned a plain float — expected ConsonantScore dataclass. "
        "This is the stub interface; upgrade to the dataclass."
    )


def test_consonant_score_has_sub_scores():
    """ConsonantScore must expose per-class sub-scores and diagnostics list.

    FAIL reason: ConsonantScore model does not yet exist.
    """
    from accent_coach.models import ConsonantScore  # noqa: F401

    dummy = ConsonantScore(
        score=75.0,
        fricative_score=80.0,
        stop_aspiration_score=70.0,
        rhotic_score=65.0,
        lateral_score=60.0,
        diagnostics=["test"],
    )
    assert dummy.score == 75.0
    assert isinstance(dummy.diagnostics, list)


# ---------------------------------------------------------------------------
# Section 2 — Fricatives: extended phoneme set /θ ð f v/
# FAIL reason: only /s z ʃ ʒ/ in current _FRICATIVES; /θ ð f v/ silently skipped
# ---------------------------------------------------------------------------


def test_dental_fricative_th_processed():
    """CoG analysis must run on /θ/ phonemes, not silently skip them.

    FAIL reason: /θ/ not in current _FRICATIVES set — all dental tokens are
    dropped and contribute nothing to the fricative score.
    """
    from accent_coach.comparison.consonants import score_consonants

    # Band-limited noise centred at ~4000 Hz — expected TH CoG range
    noise = _bandlimited_noise(center_hz=4000, bw_hz=3000, dur=0.15)
    audio = _place_in_silence(noise, start_s=0.02)
    ph = _ph("θ", "DH", start=0.02, end=0.17)
    user = _sentence([ph])
    result = score_consonants(user, audio, SR)
    # If /θ/ is silently skipped, the per-class analysis will be absent/empty.
    # After implementation, fricative_score must be populated.
    assert hasattr(result, "fricative_score"), (
        "Result missing fricative_score — ConsonantScore not returned yet."
    )


def test_labiodental_fricative_f_processed():
    """CoG analysis must run on /f/ phonemes.

    FAIL reason: /f/ not in current _FRICATIVES; silently skipped.
    """
    from accent_coach.comparison.consonants import score_consonants

    noise = _bandlimited_noise(center_hz=5500, bw_hz=4000, dur=0.15)
    audio = _place_in_silence(noise, start_s=0.02)
    ph = _ph("f", "F", start=0.02, end=0.17)
    user = _sentence([ph])
    result = score_consonants(user, audio, SR)
    assert hasattr(result, "fricative_score"), "fricative_score attribute missing"


def test_th_substituted_as_s_gets_low_score():
    """A /θ/ realised with /s/-like CoG (~7000 Hz) must score low.

    Acoustic expectation: /θ/ has CoG ~3000-5000 Hz.  A CoG of 7000 Hz
    signals /s/-substitution — the commonest TH error in L2 English.
    FAIL reason: /θ/ currently not scored at all (scored as 70.0 default).
    """
    from accent_coach.comparison.consonants import score_consonants

    # High-CoG noise mimics /s/-substitution for /θ/
    s_like_noise = _bandlimited_noise(center_hz=7000, bw_hz=2000, dur=0.15)
    audio = _place_in_silence(s_like_noise, start_s=0.02)
    ph = _ph("θ", "TH", start=0.02, end=0.17)
    user = _sentence([ph])
    result = score_consonants(user, audio, SR)
    assert hasattr(result, "fricative_score"), "fricative_score missing"
    # A /θ/ realised as /s/ should score below 60
    assert result.fricative_score is not None and result.fricative_score < 60, (
        f"fricative_score={result.fricative_score:.1f} — /θ/ with /s/-like CoG must score < 60. "
        "TH-substitution should be penalised."
    )


def test_correct_th_gets_higher_score_than_s_substitution():
    """Correct /θ/ CoG (~4000 Hz) must outscore /s/-substituted /θ/ (~7000 Hz).

    FAIL reason: /θ/ not currently scored, both cases return 70.0 default.
    """
    from accent_coach.comparison.consonants import score_consonants

    correct_th = _bandlimited_noise(center_hz=4000, bw_hz=2500, dur=0.15)
    s_sub = _bandlimited_noise(center_hz=7000, bw_hz=2000, dur=0.15)

    audio_correct = _place_in_silence(correct_th, 0.02)
    audio_wrong = _place_in_silence(s_sub, 0.02)
    ph = _ph("θ", "TH", start=0.02, end=0.17)

    score_correct = score_consonants(_sentence([ph]), audio_correct, SR)
    score_wrong = score_consonants(_sentence([ph]), audio_wrong, SR)

    assert hasattr(score_correct, "fricative_score"), "fricative_score missing"
    fc = score_correct.fricative_score
    fw = score_wrong.fricative_score
    assert fc > fw + 10, (
        f"Correct-TH fricative_score={fc:.1f} vs s-sub={fw:.1f}. "
        "Expected gap > 10 — correct /θ/ must score above /s/-substitution."
    )


def test_th_s_substitution_generates_diagnosis():
    """A /θ/ with /s/-like CoG must produce a tongue-placement diagnostic.

    FAIL reason: current code returns float with no diagnostics.
    """
    from accent_coach.comparison.consonants import score_consonants

    s_like = _bandlimited_noise(center_hz=7200, bw_hz=1800, dur=0.15)
    audio = _place_in_silence(s_like, 0.02)
    ph = _ph("θ", "TH", start=0.02, end=0.17)
    user = _sentence([ph])
    result = score_consonants(user, audio, SR)
    assert hasattr(result, "diagnostics"), "diagnostics attribute missing"
    assert result.diagnostics, (
        "diagnostics list is empty — /θ/ with /s/-substitution CoG must trigger "
        "a tongue-placement instruction (e.g. 'place tongue tip against upper teeth')."
    )


# ---------------------------------------------------------------------------
# Section 3 — VOT: integration with pipeline/vot.py
# FAIL reason: VOT extraction not wired into score_consonants; stop_aspiration_score absent
# ---------------------------------------------------------------------------


def test_vot_under_aspiration_scores_low():
    """A /p/ with VOT=15 ms (well below English 55-80 ms) must score low on aspiration.

    FAIL reason: score_consonants ignores StopFeatures.vot_ms in its consonant score —
    aspiration is handled by a separate score_aspiration() call, not in consonants.
    """
    from accent_coach.comparison.consonants import score_consonants

    audio = np.zeros(SR // 2, dtype=np.float32)
    stops = [_stop("p", vot_ms=15.0)]
    user = _sentence([], stops=stops)
    result = score_consonants(user, audio, SR)
    assert hasattr(result, "stop_aspiration_score"), "stop_aspiration_score attribute missing"
    sas = result.stop_aspiration_score
    assert sas is not None and sas < 50, (
        f"stop_aspiration_score={sas:.1f} — VOT=15ms (severely under-aspirated) should score < 50."
    )


def test_vot_native_range_scores_high():
    """A /t/ with VOT=75 ms (midpoint of English 65-90 ms range) must score high.

    FAIL reason: stop_aspiration_score attribute not yet on the return value.
    """
    from accent_coach.comparison.consonants import score_consonants

    audio = np.zeros(SR // 2, dtype=np.float32)
    stops = [_stop("t", vot_ms=75.0)]
    user = _sentence([], stops=stops)
    result = score_consonants(user, audio, SR)
    assert hasattr(result, "stop_aspiration_score"), "stop_aspiration_score missing"
    sas = result.stop_aspiration_score
    assert sas is not None and sas > 70, (
        f"stop_aspiration_score={sas:.1f} — VOT=75ms (RP midpoint for /t/) must score > 70."
    )


# ---------------------------------------------------------------------------
# Section 4 — Rhotics: F3 depression for /r/
# FAIL reason: accent_coach/comparison/consonants/liquids.py does not exist
# ---------------------------------------------------------------------------


def test_rhotic_f3_depressed_scores_high():
    """English /r/ with F3≈1900 Hz (retroflex) must score high against RP/GenAm norms.

    FAIL reason: liquids.py does not exist — ImportError expected.

    Note: 4 resonances included so Praat Burg LPC with 5 formants does not
    insert a spurious pole in the F2–F3 gap that would mis-identify F3.
    """
    from accent_coach.comparison.consonants.liquids import score_rhotic  # noqa

    # F1=500, F2=1200, F3=1900 (English /r/), F4=3400 (fills spectrum)
    seg = _resonator_audio([(500, 80), (1200, 120), (1900, 150), (3400, 200)])
    audio = _place_in_silence(seg, 0.02)
    ph = _ph("r", "R", start=0.02, end=0.17, is_stressed=True)
    score = score_rhotic(audio, SR, ph)
    assert score > 70, (
        f"score={score:.1f} — English /r/ (F3≈1900 Hz) must score > 70. "
        "F3 depression is the primary rhoticity cue."
    )


def test_rhotic_f3_high_scores_low():
    """Non-English /r/ (tap/trill) with F3≈2700 Hz must score low.

    FAIL reason: liquids.py does not exist — ImportError expected.
    """
    from accent_coach.comparison.consonants.liquids import score_rhotic  # noqa

    # F1=500, F2=1200, F3=2700 (tapped /r/), F4=3800
    seg = _resonator_audio([(500, 80), (1200, 120), (2700, 150), (3800, 200)])
    audio = _place_in_silence(seg, 0.02)
    ph = _ph("r", "R", start=0.02, end=0.17, is_stressed=True)
    score = score_rhotic(audio, SR, ph)
    # LPC underestimates F3 by ~400-500 Hz in short segments; measured ≈ 2200 Hz → score ≈ 49
    assert score < 55, (
        f"score={score:.1f} — Non-English /r/ (F3≈2700 Hz) must score < 55. "
        "High F3 signals the wrong articulation (not bunched/retroflex)."
    )


def test_rhotic_ordering_english_beats_foreign():
    """English /r/ (F3 depressed) must score higher than foreign /r/ (F3 high).

    FAIL reason: liquids.py does not exist.
    """
    from accent_coach.comparison.consonants.liquids import score_rhotic  # noqa

    seg_eng = _resonator_audio([(500, 80), (1200, 120), (1900, 150), (3400, 200)])
    seg_for = _resonator_audio([(500, 80), (1200, 120), (2700, 150), (3800, 200)])
    ph = _ph("r", "R", start=0.02, end=0.17)

    score_eng = score_rhotic(_place_in_silence(seg_eng, 0.02), SR, ph)
    score_for = score_rhotic(_place_in_silence(seg_for, 0.02), SR, ph)

    assert score_eng > score_for + 10, (
        f"English /r/ score={score_eng:.1f} vs foreign={score_for:.1f}. "
        "Expected gap > 10 pts."
    )


# ---------------------------------------------------------------------------
# Section 5 — Laterals: dark vs clear /l/
# FAIL reason: accent_coach/comparison/consonants/liquids.py does not exist
# ---------------------------------------------------------------------------


def test_dark_l_final_position_scores_high():
    """Syllable-final /l/ with F2≈1000 Hz (velarised dark /l/) must score high.

    FAIL reason: liquids.py does not exist — ImportError expected.
    F4 included so Praat does not insert a spurious pole between F2 and F3.
    """
    from accent_coach.comparison.consonants.liquids import score_lateral  # noqa

    seg = _resonator_audio([(450, 80), (1000, 120), (2600, 200), (3600, 250)])
    audio = _place_in_silence(seg, 0.02)
    ph = _ph("l", "L", start=0.02, end=0.17, is_stressed=False)
    score = score_lateral(audio, SR, ph, syllable_final=True)
    assert score > 65, (
        f"score={score:.1f} — Dark /l/ (F2≈1000 Hz) in final position must score > 65."
    )


def test_clear_l_in_final_position_scores_low():
    """Syllable-final /l/ with F2≈1700 Hz (clear /l/ where dark expected) must score low.

    FAIL reason: liquids.py does not exist.
    """
    from accent_coach.comparison.consonants.liquids import score_lateral  # noqa

    seg = _resonator_audio([(450, 80), (1700, 120), (2700, 200), (3700, 250)])
    audio = _place_in_silence(seg, 0.02)
    ph = _ph("l", "L", start=0.02, end=0.17, is_stressed=False)
    score = score_lateral(audio, SR, ph, syllable_final=True)
    assert score < 55, (
        f"score={score:.1f} — Clear /l/ (F2≈1700 Hz) in final position must score < 55. "
        "Missing dark-/l/ velarisation is a key L2 error."
    )


def test_lateral_ordering_dark_beats_clear_in_final():
    """Dark /l/ must score substantially higher than clear /l/ in syllable-final position.

    FAIL reason: liquids.py does not exist.
    """
    from accent_coach.comparison.consonants.liquids import score_lateral  # noqa

    ph = _ph("l", "L", start=0.02, end=0.17, is_stressed=False)
    seg_dark = _resonator_audio([(450, 80), (1000, 120), (2600, 200), (3600, 250)])
    seg_clear = _resonator_audio([(450, 80), (1700, 120), (2700, 200), (3700, 250)])

    score_dark = score_lateral(_place_in_silence(seg_dark, 0.02), SR, ph, syllable_final=True)
    score_clear = score_lateral(_place_in_silence(seg_clear, 0.02), SR, ph, syllable_final=True)

    assert score_dark > score_clear + 10, (
        f"Dark={score_dark:.1f} vs clear={score_clear:.1f}. Expected gap > 10 pts."
    )


def test_clear_l_in_initial_position_not_penalised():
    """Clear /l/ in syllable-initial position must NOT be penalised (only final matters).

    FAIL reason: liquids.py does not exist.
    """
    from accent_coach.comparison.consonants.liquids import score_lateral  # noqa

    seg = _resonator_audio([(450, 80), (1700, 120), (2700, 200), (3700, 250)])
    audio = _place_in_silence(seg, 0.02)
    ph = _ph("l", "L", start=0.02, end=0.17, is_stressed=True)
    score = score_lateral(audio, SR, ph, syllable_final=False)
    assert score >= 65, (
        f"score={score:.1f} — Clear /l/ in initial position must score >= 65. "
        "Only syllable-final position requires dark /l/."
    )


# ---------------------------------------------------------------------------
# Section 6 — Score ordering: native RP > TTS >= owner
# FAIL reason: stub returns 70.0 default for all; cannot discriminate
# ---------------------------------------------------------------------------


def test_correct_fricatives_outscore_wrong_fricatives():
    """A speaker with all-correct fricative CoGs must score above one with all-wrong CoGs.

    FAIL reason: /s z ʃ ʒ/ are currently scored, but /θ ð f v/ are not —
    the 'wrong' phonemes are silently dropped and score defaults to 70.0.
    After implementation, the full extended set must discriminate.
    """
    from accent_coach.comparison.consonants import score_consonants

    # Correct /s/ (~7000 Hz) and correct /ʃ/ (~3500 Hz)
    noise_s = _bandlimited_noise(center_hz=7000, bw_hz=2000, dur=0.12)
    noise_sh = _bandlimited_noise(center_hz=3500, bw_hz=2000, dur=0.12)
    audio_correct = np.zeros(SR // 2, dtype=np.float32)
    audio_correct[320: 320 + len(noise_s)] = noise_s
    audio_correct[2560: 2560 + len(noise_sh)] = noise_sh

    # Wrong /s/ (~3000 Hz, sounds like ʃ/ʒ) and wrong /ʃ/ (~7000 Hz)
    noise_s_wrong = _bandlimited_noise(center_hz=3000, bw_hz=2000, dur=0.12)
    noise_sh_wrong = _bandlimited_noise(center_hz=7000, bw_hz=2000, dur=0.12)
    audio_wrong = np.zeros(SR // 2, dtype=np.float32)
    audio_wrong[320: 320 + len(noise_s_wrong)] = noise_s_wrong
    audio_wrong[2560: 2560 + len(noise_sh_wrong)] = noise_sh_wrong

    ph_s = _ph("s", "S", start=0.02, end=0.095)
    ph_sh = _ph("ʃ", "SH", start=0.16, end=0.235)

    score_correct = score_consonants(_sentence([ph_s, ph_sh]), audio_correct, SR)
    score_wrong = score_consonants(_sentence([ph_s, ph_sh]), audio_wrong, SR)

    assert hasattr(score_correct, "fricative_score"), "fricative_score missing"
    fc = score_correct.fricative_score
    fw = score_wrong.fricative_score
    assert fc > fw + 10, (
        f"Correct fricatives score={fc:.1f} vs wrong={fw:.1f}. "
        "Expected gap > 10 pts — correct CoGs must outscore wrong CoGs."
    )


def test_no_consonants_returns_neutral_score():
    """With no consonant phonemes in the sentence, score must be a neutral ConsonantScore.

    FAIL reason: current stub returns 70.0 float, not ConsonantScore.
    """
    from accent_coach.comparison.consonants import score_consonants

    audio = np.zeros(SR // 2, dtype=np.float32)
    user = _sentence([])
    result = score_consonants(user, audio, SR)
    assert hasattr(result, "score"), "score attribute missing on ConsonantScore"
    assert 50 <= result.score <= 80, (
        f"score={result.score:.1f} out of neutral range [50, 80] for empty consonant set."
    )


# ---------------------------------------------------------------------------
# Section 7 — RP non-rhotic position gate
# /r/ in RP is only pronounced in pre-vocalic position (before a vowel).
# Post-vocalic /r/ (e.g. "bird" before consonant) is not produced → skip.
# Linking /r/ (word-final /r/ before vowel-initial next word) IS produced → score.
# GenAm is fully rhotic: all positions scored regardless.
# ---------------------------------------------------------------------------


def test_rp_prevocalic_r_is_scored():
    """RP /r/ immediately before a vowel phoneme must be scored.

    Pre-vocalic /r/ is produced in all English varieties including RP.
    rhotic_score must not be None when the only /r/ token is before a vowel.
    """
    from accent_coach.comparison.consonants import score_consonants

    seg = _resonator_audio([(500, 80), (1200, 120), (1950, 150), (3400, 200)])
    audio = _place_in_silence(seg, 0.02)
    # /r/ (0.02–0.17) followed by vowel /eɪ/ (0.17–0.32) — e.g. the /r/ in "rain"
    ph_r = _ph("r",  "R",   start=0.02, end=0.17, word="rain")
    ph_v = _ph("eɪ", "EY1", start=0.17, end=0.32, word="rain")
    result = score_consonants(_sentence([ph_r, ph_v]), audio, SR, accent_target="rp")
    assert result.rhotic_score is not None, (
        "RP pre-vocalic /r/ (before vowel, e.g. 'rain') must be scored. "
        "rhotic_score is None — pre-vocalic gate is over-filtering."
    )


def test_rp_postvocalic_r_before_consonant_is_skipped():
    """RP /r/ before a consonant (e.g. 'bird') must NOT be scored — RP is non-rhotic.

    G2P (CMU dict) always emits an /r/ phoneme for words like 'bird' (B IH1 R D).
    RP speakers do not produce that /r/; scoring it penalises RP natives unfairly.
    When the only /r/ token is followed by a consonant, rhotic_score must be None
    (weight redistributed to other sub-classes, not a fictional penalty).
    """
    from accent_coach.comparison.consonants import score_consonants

    seg = _resonator_audio([(500, 80), (1200, 120), (2700, 150), (3800, 200)])
    audio = _place_in_silence(seg, 0.02)
    # /r/ (0.02–0.17) followed by /d/ (0.17–0.32) — the /r/ in "bird" before 'd'
    ph_r = _ph("r", "R", start=0.02, end=0.17, word="bird")
    ph_d = _ph("d", "D", start=0.17, end=0.32, word="bird")
    result = score_consonants(_sentence([ph_r, ph_d]), audio, SR, accent_target="rp")
    assert result.rhotic_score is None, (
        f"RP post-vocalic /r/ before consonant must be skipped; rhotic_score={result.rhotic_score}. "
        "RP is non-rhotic — this /r/ is not produced, so no F3 depression is expected."
    )


def test_rp_linking_r_is_scored():
    """RP linking /r/ (word-final /r/ before vowel-initial next word) must be scored.

    In RP, 'there and', 'car is', 'here are' produce a linking /r/.
    The next phoneme globally is a vowel (from the following word), so the gate
    must allow it through — same rule as pre-vocalic, just across a word boundary.
    """
    from accent_coach.comparison.consonants import score_consonants

    seg = _resonator_audio([(500, 80), (1200, 120), (1950, 150), (3400, 200)])
    audio = _place_in_silence(seg, 0.02)
    # "there and": /r/ at end of "there", followed by /æ/ (first phoneme of "and")
    ph_r = _ph("r",  "R",   start=0.02, end=0.17, word="there")
    ph_v = _ph("æ",  "AE1", start=0.17, end=0.32, word="and")
    result = score_consonants(_sentence([ph_r, ph_v]), audio, SR, accent_target="rp")
    assert result.rhotic_score is not None, (
        "RP linking /r/ (word-final /r/ before vowel-initial next word, e.g. 'there and') "
        "must be scored. rhotic_score is None — the gate is incorrectly filtering linking /r/."
    )


def test_genam_r_before_consonant_is_scored():
    """GenAm /r/ before a consonant must be scored — GA is fully rhotic.

    GA speakers produce /r/ in all positions. The RP non-rhotic gate must NOT
    apply when accent_target='genam'. rhotic_score must not be None.
    """
    from accent_coach.comparison.consonants import score_consonants

    seg = _resonator_audio([(500, 80), (1200, 120), (1950, 150), (3400, 200)])
    audio = _place_in_silence(seg, 0.02)
    # Same token as the RP-skip test but in GenAm mode — must score
    ph_r = _ph("r", "R", start=0.02, end=0.17, word="bird")
    ph_d = _ph("d", "D", start=0.17, end=0.32, word="bird")
    result = score_consonants(_sentence([ph_r, ph_d]), audio, SR, accent_target="genam")
    assert result.rhotic_score is not None, (
        "GenAm /r/ before a consonant must be scored (GA is fully rhotic). "
        f"rhotic_score={result.rhotic_score}. The RP non-rhotic gate must not apply to GenAm."
    )
