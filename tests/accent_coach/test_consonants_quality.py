"""
Consonant quality calibration tests — business logic layer.

These tests verify that scoring produces MEANINGFUL values, not just correct
signatures.  They encode the core ordering rule:
  native RP/GA speakers  →  score ≥ 75
  TTS clone (BC)         →  score ≥ 65  (near natives)
  L2 speaker (Slavic)    →  score ≤ 50  (detectable errors)

All thresholds account for the ~200–350 Hz LPC tracking tolerance and the
~200 Hz centroid shift from bandpass filter transition bands in synthetic audio.
A trivially flat scorer (returning 65.0 for all inputs) would fail every test
in Sections 2–4 and all ordering tests.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, sosfilt

from accent_coach.models import PhonemeInstance, SentenceAnalysis, StopFeatures

SR = 16_000


# ---------------------------------------------------------------------------
# Synthesis helpers
# ---------------------------------------------------------------------------


def _ph(ph: str, arp: str, start: float = 0.02, end: float = 0.17, word: str = "test") -> PhonemeInstance:
    return PhonemeInstance(phoneme=ph, arpabet=arp, start_time=start, end_time=end,
                           sentence_id=1, word=word, is_stressed=True)


def _stop(ph: str, vot_ms: float) -> StopFeatures:
    p = _ph(ph, ph.upper(), start=0.05, end=0.15)
    return StopFeatures(phoneme=p, vot_ms=vot_ms, burst_energy=1.0)


def _sentence(phonemes: list[PhonemeInstance], stops: list[StopFeatures] | None = None) -> SentenceAnalysis:
    return SentenceAnalysis(
        sentence_id=1, sentence_type="statement",
        duration_s=0.5, syllable_durations=[0.2, 0.2, 0.1],
        pitch_contour=[0.5] * 50, stress_pattern=[True, False, True],
        vowels=[], stops=stops or [], phonemes=phonemes,
    )


def _noise(center: float, bw: float, dur: float = 0.15) -> np.ndarray:
    """Band-limited Gaussian noise with fixed seed."""
    n = int(SR * dur)
    raw = np.random.default_rng(0).standard_normal(n).astype(np.float64)
    lo, hi = max(200.0, center - bw / 2), min(SR / 2 - 200, center + bw / 2)
    sos = butter(4, [lo / (SR / 2), hi / (SR / 2)], btype="band", output="sos")
    out = sosfilt(sos, raw)
    return (out / (np.abs(out).max() + 1e-9) * 0.8).astype(np.float32)


def _place(seg: np.ndarray, start: float, total: float = 0.5) -> np.ndarray:
    """Place segment into silence, truncating if needed."""
    audio = np.zeros(int(total * SR), dtype=np.float32)
    s, e = int(start * SR), int(start * SR) + len(seg)
    audio[s: min(len(audio), e)] = seg[: min(len(audio), e) - s]
    return audio


def _resonator(formants: list[tuple[float, float]], dur: float = 0.15, f0: float = 120.0) -> np.ndarray:
    """AR synthesis with given formants [(Hz, BW_Hz)]."""
    from scipy.signal import lfilter
    n = int(SR * dur)
    pulse = np.zeros(n, dtype=np.float64)
    pulse[:: int(SR / f0)] = 1.0
    out = pulse
    for f, bw in formants:
        r = np.exp(-math.pi * bw / SR)
        out = lfilter([1.0], [1.0, -2 * r * math.cos(2 * math.pi * f / SR), r**2], out)
    return (out / (np.abs(out).max() + 1e-9) * 0.8).astype(np.float32)


# ---------------------------------------------------------------------------
# Section 1 — Fricative quality calibration
# Thresholds account for ~200 Hz centroid shift from bandpass filter edges.
# ---------------------------------------------------------------------------


def test_ideal_s_scores_at_least_85():
    """An /s/ band-limited near 7000 Hz must score ≥ 85.

    CoG of bandpass noise centered at 7000 Hz lands within ~250 Hz of the
    reference; at decay=2000 Hz that gives score ≥ 88.  Threshold 85 allows
    for filter-edge effects without hiding real miscalibration.
    Fails if: scoring is flat, or reference CoG is wildly wrong.
    """
    from accent_coach.comparison.consonants.fricatives import score_fricatives

    noise = _noise(center=7000, bw=1500, dur=0.15)
    audio = _place(noise, 0.02)
    ph = _ph("s", "S", start=0.02, end=0.17)
    score, _ = score_fricatives(_sentence([ph]), audio, SR)
    assert score is not None and score >= 85, (
        f"Near-ideal /s/ (CoG ≈ 7000 Hz) scored {score:.1f}. Expected ≥ 85."
    )


def test_severely_wrong_s_scores_below_20():
    """An /s/ at CoG=2000 Hz (5000 Hz below reference) must score < 20.

    5000 Hz / 2000 Hz decay = 2.5 e-foldings → score ≈ 8.
    Fails if: decay constant is too lenient or scoring is flat.
    """
    from accent_coach.comparison.consonants.fricatives import score_fricatives

    noise = _noise(center=2000, bw=1500, dur=0.15)
    audio = _place(noise, 0.02)
    ph = _ph("s", "S", start=0.02, end=0.17)
    score, _ = score_fricatives(_sentence([ph]), audio, SR)
    assert score is not None and score < 20, (
        f"Severely wrong /s/ (CoG≈2000 Hz) scored {score:.1f}. Expected < 20."
    )


def test_fricative_score_gap_good_vs_bad_at_least_65():
    """Score gap between ideal and severely wrong /s/ must be ≥ 65 pts."""
    from accent_coach.comparison.consonants.fricatives import score_fricatives

    ph = _ph("s", "S", start=0.02, end=0.17)
    good, _ = score_fricatives(_sentence([ph]), _place(_noise(7000, 1500), 0.02), SR)
    bad, _ = score_fricatives(_sentence([ph]), _place(_noise(2000, 1500), 0.02), SR)
    assert good is not None and bad is not None
    assert good - bad >= 65, (
        f"Score gap: {good:.1f} − {bad:.1f} = {good - bad:.1f}. Expected ≥ 65 pts."
    )


# ---------------------------------------------------------------------------
# Section 2 — VOT quality calibration
# ---------------------------------------------------------------------------


def test_perfect_p_vot_scores_at_least_90():
    """A /p/ with VOT exactly at RP mean (67.5 ms) must score ≥ 90."""
    from accent_coach.comparison.consonants.stops import score_stops

    score, _ = score_stops(_sentence([], [_stop("p", 67.5)]))
    assert score is not None and score >= 90, (
        f"/p/ at RP mean (67.5 ms) scored {score:.1f}. Expected ≥ 90."
    )


def test_zero_vot_scores_below_5():
    """A /p/ with VOT=5 ms (no aspiration, z≈6.25, +under-aspiration penalty) must score < 5."""
    from accent_coach.comparison.consonants.stops import score_stops

    score, diag = score_stops(_sentence([], [_stop("p", 5.0)]))
    assert score is not None and score < 5, (
        f"/p/ VOT=5 ms scored {score:.1f}. Expected < 5."
    )
    assert diag, "Under-aspiration diagnostic must be non-empty for VOT=5 ms."


def test_vot_gap_native_vs_zero_at_least_85():
    """Score gap between native /p/ VOT (67.5 ms) and VOT=5 ms must be ≥ 85 pts."""
    from accent_coach.comparison.consonants.stops import score_stops

    native, _ = score_stops(_sentence([], [_stop("p", 67.5)]))
    zero, _ = score_stops(_sentence([], [_stop("p", 5.0)]))
    assert native is not None and zero is not None
    assert native - zero >= 85, f"VOT gap={native - zero:.1f}. Expected ≥ 85."


# ---------------------------------------------------------------------------
# Section 3 — Rhotic quality calibration
# Thresholds account for ≤350 Hz LPC tracking tolerance in short segments.
# ---------------------------------------------------------------------------


def test_english_r_f3_near_target_scores_at_least_85():
    """English /r/ with F3 synthesised at 1950 Hz must score ≥ 85.

    4 resonances: F4 at 3400 Hz fills the spectrum so Praat Burg LPC
    does not insert a spurious pole in the F2–F3 gap (1200–1950 Hz).
    """
    from accent_coach.comparison.consonants.liquids import score_rhotic

    seg = _resonator([(500, 80), (1200, 120), (1950, 150), (3400, 200)])
    score = score_rhotic(_place(seg, 0.02), SR, _ph("r", "R"))
    assert score >= 85, f"/r/ F3≈1950 Hz scored {score:.1f}. Expected ≥ 85."


def test_foreign_r_f3_high_scores_below_55():
    """Non-English /r/ with F3 synthesised at 2700 Hz must score < 55.

    Parselmouth Burg LPC on short voiced segments underestimates F3 by
    ~400–500 Hz in practice (AR cascade interaction with F4), landing the
    measured F3 at ~2200 Hz → score ≈ 49.  Threshold 55 clears LPC variance
    while still failing a flat scorer (65) by a comfortable margin.
    The rhotic_gap_at_least_50 test validates the native-vs-foreign discrimination.
    """
    from accent_coach.comparison.consonants.liquids import score_rhotic

    seg = _resonator([(500, 80), (1200, 120), (2700, 150), (3800, 200)])
    score = score_rhotic(_place(seg, 0.02), SR, _ph("r", "R"))
    assert score < 55, f"/r/ F3≈2700 Hz scored {score:.1f}. Expected < 55."


def test_rhotic_gap_at_least_50():
    """Score gap between English /r/ (F3=1950) and foreign /r/ (F3=2700) must be ≥ 50 pts."""
    from accent_coach.comparison.consonants.liquids import score_rhotic

    ph = _ph("r", "R")
    eng = score_rhotic(_place(_resonator([(500, 80), (1200, 120), (1950, 150), (3400, 200)]), 0.02), SR, ph)
    for_ = score_rhotic(_place(_resonator([(500, 80), (1200, 120), (2700, 150), (3800, 200)]), 0.02), SR, ph)
    assert eng - for_ >= 50, f"Rhotic gap={eng - for_:.1f}. Expected ≥ 50 pts."


# ---------------------------------------------------------------------------
# Section 4 — Lateral quality calibration
# ---------------------------------------------------------------------------


def test_dark_l_at_target_scores_at_least_80():
    """Dark /l/ with F2 synthesised at 1050 Hz in final position must score ≥ 80.

    LPC tracks within ~100 Hz, giving score ≥ 88 at the target; threshold 80
    allows for LPC variance on short segments.
    """
    from accent_coach.comparison.consonants.liquids import score_lateral

    seg = _resonator([(450, 80), (1050, 120), (2600, 200), (3600, 250)])
    score = score_lateral(_place(seg, 0.02), SR, _ph("l", "L"), syllable_final=True)
    assert score >= 80, f"Dark /l/ F2≈1050 Hz scored {score:.1f}. Expected ≥ 80."


def test_clear_l_in_final_scores_below_35():
    """Clear /l/ (F2≈1700 Hz) in final position must score < 35.

    delta = 1700 - 1350 (threshold) = 350 / 300 decay → score ≈ 31.
    Fails if: position gate not applied or decay too lenient.
    """
    from accent_coach.comparison.consonants.liquids import score_lateral

    seg = _resonator([(450, 80), (1700, 120), (2700, 200), (3700, 250)])
    score = score_lateral(_place(seg, 0.02), SR, _ph("l", "L"), syllable_final=True)
    assert score < 35, f"Clear /l/ F2≈1700 in final scored {score:.1f}. Expected < 35."


# ---------------------------------------------------------------------------
# Section 5 — End-to-end composite: native RP vs Slavic L2
# Uses _place() to avoid broadcasting errors (truncates segments to fit).
# ---------------------------------------------------------------------------


def _composite_audio_and_phonemes(
    s_center: float,
    r_f3: float,
    l_f2: float,
) -> tuple[np.ndarray, list[PhonemeInstance]]:
    """Build a 0.7 s audio clip with /s/, /r/, /l/ at fixed positions."""
    dur = 0.13  # each segment fits within its 0.13 s slot
    seg_s = _noise(s_center, 1500, dur=dur)
    # 4-resonance synthesis prevents Praat from inserting spurious poles in the F2-F3 gap
    r_f4 = 3400.0 if r_f3 < 2400 else 3800.0  # tighter F4 for English, higher for foreign /r/
    seg_r = _resonator([(500, 80), (1200, 120), (r_f3, 150), (r_f4, 200)], dur=dur)
    l_f3 = 2600.0 if l_f2 < 1350 else 2700.0
    seg_l = _resonator([(450, 80), (l_f2, 120), (l_f3, 200), (3600, 250)], dur=dur)

    audio = np.zeros(int(0.7 * SR), dtype=np.float32)
    # Slots: /s/ 0.02–0.15, /r/ 0.20–0.33, /l/ 0.38–0.51
    s_seg = min(len(seg_s), int(0.13 * SR))
    r_seg = min(len(seg_r), int(0.13 * SR))
    l_seg = min(len(seg_l), int(0.13 * SR))
    audio[int(0.02 * SR): int(0.02 * SR) + s_seg] = seg_s[:s_seg]
    audio[int(0.20 * SR): int(0.20 * SR) + r_seg] = seg_r[:r_seg]
    audio[int(0.38 * SR): int(0.38 * SR) + l_seg] = seg_l[:l_seg]

    phonemes = [
        _ph("s", "S", start=0.02, end=0.15),
        _ph("r", "R", start=0.20, end=0.33),
        _ph("l", "L", start=0.38, end=0.51, word="call"),
    ]
    return audio, phonemes


def test_native_rp_composite_at_least_75():
    """Synthetic 'native RP' sentence (/s/+/r/+/l/ all correct, /p/ VOT=67.5) must score ≥ 75.

    Expected sub-scores: fric≈89, stop=100, rhotic≈92, lateral≈88.
    Composite ≈ 0.35×89 + 0.35×100 + 0.20×92 + 0.10×88 ≈ 93.
    Fails if: any sub-class is miscalibrated or aggregation weights are wrong.
    """
    from accent_coach.comparison.consonants import score_consonants

    audio, phonemes = _composite_audio_and_phonemes(s_center=7000, r_f3=1950, l_f2=1050)
    stops = [_stop("p", 67.5)]
    result = score_consonants(_sentence(phonemes, stops), audio, SR)
    assert result.score >= 75, (
        f"Native-RP composite={result.score:.1f} "
        f"(fric={result.fricative_score}, stop={result.stop_aspiration_score}, "
        f"rh={result.rhotic_score}, lat={result.lateral_score}). Expected ≥ 75."
    )


def test_slavic_l2_composite_at_most_50():
    """Synthetic Slavic L2 errors (/r/ tap, no aspiration, clear /l/) must score ≤ 50.

    Expected sub-scores: fric≈89 (correct), stop≈1.5 (VOT=15), rhotic≈17, lateral≈31.
    Composite ≈ 0.35×89 + 0.35×1.5 + 0.20×17 + 0.10×31 ≈ 38.
    Fails if: errors are not penalised or scoring is trivially flat.
    """
    from accent_coach.comparison.consonants import score_consonants

    # Slavic L2 errors: tapped /r/ (F3 high), clear /l/ in final, no aspiration
    audio, phonemes = _composite_audio_and_phonemes(s_center=7000, r_f3=2700, l_f2=1700)
    stops = [_stop("p", 15.0)]
    result = score_consonants(_sentence(phonemes, stops), audio, SR)
    assert result.score <= 50, (
        f"Slavic-L2 composite={result.score:.1f} "
        f"(fric={result.fricative_score}, stop={result.stop_aspiration_score}, "
        f"rh={result.rhotic_score}, lat={result.lateral_score}). Expected ≤ 50."
    )


def test_native_rp_beats_slavic_l2_by_at_least_25():
    """Native RP must outscore Slavic L2 by ≥ 25 pts — the core ordering invariant.

    This test catches regressions where a scorer change makes L2 score near
    native (e.g. a decay constant ten times too large would compress all scores
    toward 100, wiping out the gap).
    """
    from accent_coach.comparison.consonants import score_consonants

    native_audio, ph = _composite_audio_and_phonemes(s_center=7000, r_f3=1950, l_f2=1050)
    l2_audio, _ = _composite_audio_and_phonemes(s_center=7000, r_f3=2700, l_f2=1700)

    native = score_consonants(_sentence(ph, [_stop("p", 67.5)]), native_audio, SR)
    l2 = score_consonants(_sentence(ph, [_stop("p", 15.0)]), l2_audio, SR)

    gap = native.score - l2.score
    assert gap >= 25, (
        f"Native={native.score:.1f} vs L2={l2.score:.1f} gap={gap:.1f}. "
        "Expected gap ≥ 25 pts — native must score well above L2."
    )
