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


def test_severely_wrong_s_scores_below_35():
    """An /s/ realised like /ʃ/ (CoG≈3500 Hz, 3500 Hz below the 7000 ref) scores < 35.

    Uses a still-FRICATION signal (energy above the 3 kHz frication gate) but with
    a badly wrong CoG — the realistic "wrong /s/" (an /s/→/ʃ/ shift), not a
    sub-3 kHz buzz (which the frication gate correctly rejects as non-fricative).
    At the distribution-calibrated decay (2000 Hz) a 3500 Hz error is 1.75
    e-foldings → score ≈ 17, far below 35. Decay-robust.
    Fails if: decay is widened toward a "believable band" until the scorer goes flat.
    """
    from accent_coach.comparison.consonants.fricatives import score_fricatives

    noise = _noise(center=3500, bw=1500, dur=0.15)
    audio = _place(noise, 0.02)
    ph = _ph("s", "S", start=0.02, end=0.17)
    score, _ = score_fricatives(_sentence([ph]), audio, SR)
    assert score is not None and score < 35, (
        f"Severely wrong /s/ (CoG≈3500 Hz) scored {score:.1f}. Expected < 35."
    )


def test_fricative_score_gap_good_vs_bad_at_least_65():
    """Score gap between ideal and severely wrong /s/ must be ≥ 65 pts."""
    from accent_coach.comparison.consonants.fricatives import score_fricatives

    ph = _ph("s", "S", start=0.02, end=0.17)
    good, _ = score_fricatives(_sentence([ph]), _place(_noise(7000, 1500), 0.02), SR)
    # /ʃ/-like 3500 Hz mis-production: still frication (passes the HF gate), wrong CoG.
    bad, _ = score_fricatives(_sentence([ph]), _place(_noise(3500, 1500), 0.02), SR)
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


# ---------------------------------------------------------------------------
# Section 6 — §0: RP/GA fricative table provenance
# ---------------------------------------------------------------------------


def test_fricative_cog_tables_are_accent_neutral():
    """RP_FRICATIVE_COG_HZ must equal GA_FRICATIVE_COG_HZ (same Jongman 2000 source).

    Both tables cite American English Jongman et al. (2000). Different values
    (z: 6500 vs 6400, ʃ: 3800 vs 3700, ʒ: 3300 vs 3200) are false precision
    from copying the same corpus with slight rounding differences.
    Fix: update RP table to match GA (the correctly-cited source).
    Fails as long as the two tables are separate copies with different values.
    """
    from accent_coach.reference.rp_norms import RP_FRICATIVE_COG_HZ
    from accent_coach.reference.genam_norms import GA_FRICATIVE_COG_HZ

    differing = [(k, RP_FRICATIVE_COG_HZ[k], GA_FRICATIVE_COG_HZ[k])
                 for k in RP_FRICATIVE_COG_HZ if RP_FRICATIVE_COG_HZ[k] != GA_FRICATIVE_COG_HZ[k]]
    assert not differing, (
        f"Tables differ at: {differing}. Both cite Jongman 2000 (American English). "
        "Declare fricative CoG accent-neutral: update RP values to match GA."
    )


# ---------------------------------------------------------------------------
# Section 7 — §4A: coda-cluster /l/ heuristic
# ---------------------------------------------------------------------------


def test_coda_cluster_l_scored_as_syllable_final():
    """Coda-cluster /l/ in 'milk' (M IH1 L K) must be treated as syllable-final.

    Current heuristic: word_phones[-1] == 'l' → False for 'milk' (ends in 'k').
    So the /l/ is scored as initial-position (no dark-/l/ check), and a clear
    /l/ (F2=1700 Hz) gets a high score when it should get a low one.
    Fails as long as word_phones[-1] == 'l' is the only syllable-final test.
    """
    from accent_coach.comparison.consonants.liquids import score_liquids

    seg = _resonator([(450, 80), (1700, 120), (2700, 200), (3700, 250)])
    audio = _place(seg, 0.30)
    phonemes = [
        _ph("m",  "M",   0.00, 0.10, word="milk"),
        _ph("ɪ",  "IH1", 0.10, 0.20, word="milk"),
        _ph("l",  "L",   0.30, 0.43, word="milk"),
        _ph("k",  "K",   0.43, 0.53, word="milk"),
    ]
    _, lateral, _ = score_liquids(_sentence(phonemes), audio, SR)
    assert lateral is not None and lateral < 55, (
        f"lateral={lateral:.1f} — clear /l/ (F2=1700 Hz) in 'milk' coda must score < 55. "
        "word_phones[-1]=='l' misses pre-consonant coda /l/: fix to check next phoneme."
    )


# ---------------------------------------------------------------------------
# Section 8 — §4B: accent-aware initial /l/ clear target
# ---------------------------------------------------------------------------


def test_lateral_initial_clear_target_differs_by_accent():
    """Initial /l/ score must differ between 'rp' and 'genam' accent targets.

    Current code: clear_target = 1550.0 is hardcoded for both accents → identical
    scores regardless of accent_target parameter.
    Fix: add RP_LATERAL_CLEAR_F2_TARGET_HZ / GA_LATERAL_CLEAR_F2_TARGET_HZ
    to the norms files and branch on accent_target.
    """
    from accent_coach.comparison.consonants.liquids import score_lateral

    # F2 = 1550 Hz — exactly the RP clear target; GA target differs
    seg = _resonator([(450, 80), (1550, 120), (2600, 200), (3600, 250)])
    audio = _place(seg, 0.02)
    ph = _ph("l", "L", start=0.02, end=0.17)
    s_rp  = score_lateral(audio, SR, ph, syllable_final=False, accent_target="rp")
    s_ga  = score_lateral(audio, SR, ph, syllable_final=False, accent_target="genam")
    assert s_rp != s_ga, (
        f"rp={s_rp:.1f} == genam={s_ga:.1f}: clear_target=1550.0 is hardcoded for both. "
        "Branch on accent_target using RP_LATERAL_CLEAR_F2_TARGET_HZ / GA_LATERAL_CLEAR_F2_TARGET_HZ."
    )


# ---------------------------------------------------------------------------
# Section 9 — §1A: power spectrum for CoG
# ---------------------------------------------------------------------------


def test_power_spectrum_cog_matches_f_reference():
    """/f/ bimodal signal whose power-spectrum CoG = 5500 Hz (reference) must score ≥ 95.

    Signal: tone at 5000 Hz (amplitude 2) + tone at 7500 Hz (amplitude 1).
      Magnitude CoG = (5000×2 + 7500×1) / 3 ≈ 5833 Hz → score ≈ 85
      Power CoG    = (5000×4 + 7500×1) / 5  = 5500 Hz → score = 100

    Both frequencies are well above the 2000 Hz HP cutoff so the HP method
    (hard mask or Butterworth) does not affect the result — this test isolates
    the magnitude-vs-power spectrum bug (§1A).
    Fails with current magnitude spectrum (score ≈ 85 < 95).
    """
    from accent_coach.comparison.consonants.fricatives import score_fricatives

    n = int(0.15 * SR)
    t = np.arange(n, dtype=np.float64) / SR
    # amplitude 2 at 5000 Hz + amplitude 1 at 7500 Hz
    signal = (2.0 * np.sin(2 * np.pi * 5000 * t)
              + 1.0 * np.sin(2 * np.pi * 7500 * t)).astype(np.float32)
    signal = signal / (np.abs(signal).max() + 1e-9) * 0.8
    audio = _place(signal, 0.02)
    ph = _ph("f", "F", start=0.02, end=0.17)
    score, _ = score_fricatives(_sentence([ph]), audio, SR)
    assert score is not None and score >= 95, (
        f"/f/ bimodal signal (power CoG = 5500 Hz = reference) scored {score:.1f}. "
        "Expected ≥ 95. Magnitude spectrum gives CoG ≈ 5833 Hz → score ≈ 85. "
        "Fix: spectrum = np.abs(np.fft.rfft(segment)) ** 2"
    )


# ---------------------------------------------------------------------------
# Section 9b — fricative token must be frication, not a mis-aligned vowel/closure
# (data-cleaning: alignment sometimes lands an /s z/ window on the adjacent vowel)
# ---------------------------------------------------------------------------


def test_fricative_token_on_vowel_is_rejected():
    """A fricative window that actually contains a VOWEL (low HF energy) must be
    skipped, not scored — otherwise mis-aligned tokens pollute the group mean.

    A vowel resonator (F1 500, F2 1500, F3 2500) has almost no energy above
    3 kHz, so it is not frication and must be dropped (score None, no tokens).
    """
    from accent_coach.comparison.consonants.fricatives import score_fricatives

    seg = _resonator([(500, 80), (1500, 120), (2500, 200)], dur=0.15)
    audio = _place(seg, 0.02)
    ph = _ph("s", "S", start=0.02, end=0.17)
    score, _ = score_fricatives(_sentence([ph]), audio, SR)
    assert score is None, (
        f"Vowel mis-aligned as /s/ scored {score} — must be rejected (not frication)."
    )


def test_real_frication_is_accepted():
    """A genuine /s/ (band-limited noise near 7 kHz) must still be scored."""
    from accent_coach.comparison.consonants.fricatives import score_fricatives

    audio = _place(_noise(center=7000, bw=1500, dur=0.15), 0.02)
    ph = _ph("s", "S", start=0.02, end=0.17)
    score, _ = score_fricatives(_sentence([ph]), audio, SR)
    assert score is not None, "Real frication must be accepted, not rejected by the HF gate."


# ---------------------------------------------------------------------------
# Section 10 — §1B: Butterworth HP not hard mask
# ---------------------------------------------------------------------------


def test_butterworth_hp_not_hard_mask():
    """Butterworth HP gives accurate CoG; hard mask lets float-noise dominate.

    Signal: equal-amplitude tones at 1500 Hz and 4000 Hz for /ʒ/ (GA ref 3200 Hz).
    The 1500 Hz component helps pull the true CoG toward the 3200 Hz reference.

    Hard mask at 2000 Hz: zeroes bin 225 (1500 Hz) → only 4000 Hz remains →
      CoG = 4000 Hz → score vs 3200 Hz = exp(-800/2000) ≈ 67 < 73.
    Butterworth HP at 2000 Hz: 1500 Hz attenuated (-5 dB) but present →
      CoG pulled toward reference → score ≈ 77–94 ≥ 73.

    Both 1500 Hz and 4000 Hz map to exact FFT bins for n=2400 (no leakage).
    Fails with current hard-mask implementation (score ≈ 67 < 73).
    """
    from accent_coach.comparison.consonants.fricatives import score_fricatives

    n = int(0.15 * SR)  # = 2400: 1500 Hz → bin 225, 4000 Hz → bin 600 (exact)
    t = np.arange(n, dtype=np.float64) / SR
    sig = (np.sin(2 * np.pi * 1500 * t) + np.sin(2 * np.pi * 4000 * t)).astype(np.float32)
    audio = _place(sig, 0.02)
    ph = _ph("ʒ", "ZH", start=0.02, end=0.17)
    score, _ = score_fricatives(_sentence([ph]), audio, SR, accent_target="genam")
    assert score is not None and score >= 73, (
        f"score={score:.1f} — bimodal /ʒ/ (1500+4000 Hz) scored below 73. "
        "Hard mask zeroes the 1500 Hz component → CoG=4000 Hz → score≈67. "
        "Butterworth HP (-5 dB at 1500 Hz) preserves it → CoG~3500 Hz → score≥73. "
        "Fix §1B: apply butter(4, 2000/nyq, 'high') to segment in time domain before FFT."
    )


# ---------------------------------------------------------------------------
# Section 11 — §2B: VOT median on search window
# ---------------------------------------------------------------------------


def test_vot_median_on_search_window_detects_burst():
    """VOT burst detection must use search-window median, not full-window median.

    Scenario: moderate burst at stop boundary + loud broadband noise AFTER the
    search window.  Full-window median is elevated by the loud post-search noise,
    raising 3× threshold above the burst energy → burst not detected (vot = None).
    Search-window median is near the background level → 3× threshold is low →
    burst detected → vot ≈ 70 ms.

    Fix: `median_e = np.median(search)` (pipeline/vot.py line ~55)
    """
    import numpy as np
    from accent_coach.models import PhonemeInstance
    from accent_coach.pipeline.vot import extract_vot

    _SR = 16_000
    n = int(0.35 * _SR)
    rng = np.random.default_rng(0)

    audio = (rng.standard_normal(n) * 0.01).astype(np.float32)  # low background

    # Moderate burst at 0.02 s (within search window)
    b_s = int(0.02 * _SR)
    audio[b_s: b_s + int(0.005 * _SR)] += (
        rng.standard_normal(int(0.005 * _SR)) * 0.25
    ).astype(np.float32)

    # Voicing onset at 0.09 s — amplitude 0.8 sine (autocorr ≈ 0.58 > 0.5 for
    # a fully-voiced frame, ensuring voicing is detectable with the current
    # threshold = 0.5 so the §2B test is independent of §2C).
    v_s = int(0.09 * _SR)
    t_v = np.arange(n - v_s, dtype=np.float64) / _SR
    audio[v_s:] += (0.8 * np.sin(2 * np.pi * 120 * t_v)).astype(np.float32)

    # Loud broadband noise starting at 0.11 s (frame 22 of the 44-frame segment).
    # Frames 21-43 are loud (23/44 = 52 %), so the full-window median is in the
    # loud range and suppresses the burst under the 3× threshold.
    # Frame 18's 4-hop autocorr window (samples 1440-1759) ends before loud at
    # sample 1760, so the voicing detection at frame 18 is uncontaminated.
    loud_s = int(0.11 * _SR)
    audio[loud_s:] += (rng.standard_normal(n - loud_s) * 0.9).astype(np.float32)

    stop = PhonemeInstance(
        phoneme="p", arpabet="P", start_time=0.02, end_time=0.07,
        sentence_id=1, word="pop", is_stressed=True,
    )
    vot = extract_vot(audio, _SR, stop)
    assert vot is not None and 40 < vot < 120, (
        f"vot={vot} — burst at 0.02 s, voicing at 0.09 s, loud noise after search window. "
        "Full-window median is elevated above the burst by the loud post-search noise. "
        "Fix: median_e = np.median(search)  (search window only)."
    )


# ---------------------------------------------------------------------------
# Section 12 — §2C: voicing onset threshold 0.5 → 0.35
# ---------------------------------------------------------------------------


def test_voicing_threshold_catches_partial_onset():
    """Voicing onset at autocorr ≈ 0.39 must be detected at threshold 0.35.

    Signal: sine (amplitude 1.0) + noise (amplitude 0.5) starts at 0.09 s.
    Fully-voiced frame autocorr ≈ (0.5 × 0.58) / (0.5 + 0.25) ≈ 0.39
      threshold 0.35: detects at the first fully-voiced frame → VOT ≈ 70 ms
      threshold 0.50: 0.39 < 0.5 → NEVER detected → vot = None

    Fails with current threshold 0.5 (vot = None).
    Fix: change `> 0.5` to `> 0.35` in pipeline/vot.py autocorr check.
    """
    import numpy as np
    from accent_coach.models import PhonemeInstance
    from accent_coach.pipeline.vot import extract_vot

    _SR = 16_000
    n = int(0.35 * _SR)
    rng = np.random.default_rng(9)

    audio = np.zeros(n, dtype=np.float32)

    # Clear burst at 0.02 s
    b_s = int(0.02 * _SR)
    audio[b_s: b_s + int(0.005 * _SR)] += (
        rng.standard_normal(int(0.005 * _SR)) * 0.6
    ).astype(np.float32)

    # Partial voicing onset at 0.09 s: sine + moderate noise → autocorr ≈ 0.39
    v_s = int(0.09 * _SR)
    t_v = np.arange(n - v_s, dtype=np.float64) / _SR
    audio[v_s:] += (
        np.sin(2 * np.pi * 120 * t_v)
        + 0.5 * rng.standard_normal(n - v_s)
    ).astype(np.float32)

    stop = PhonemeInstance(
        phoneme="p", arpabet="P", start_time=0.02, end_time=0.07,
        sentence_id=1, word="pop", is_stressed=True,
    )
    vot = extract_vot(audio, _SR, stop)
    assert vot is not None and 40 < vot < 110, (
        f"vot={vot} — voicing onset at autocorr ≈ 0.39 must give VOT 40-110 ms. "
        "Threshold 0.5 never detects this onset (0.39 < 0.5) → vot = None. "
        "Fix: lower autocorr gate from 0.5 to 0.35 in pipeline/vot.py."
    )


# ---------------------------------------------------------------------------
# Section 13 — §3A: minimum-F3 sampling for rhotics
# ---------------------------------------------------------------------------


def test_rhotic_min_f3_finds_constriction_trough():
    """/r/ segment with F3 trough in first half must score ≥ 70.

    Construct /r/ where the first half has F3=1950 Hz (English /r/ constriction)
    and the second half has F3=2800 Hz (F3 has risen as the following vowel begins).
    With midpoint-only sampling the formant tracker is near the transition region
    and may measure F3 ≈ 2400–2600 Hz → score ≈ 30–50.
    With multi-point minimum (25/33/50/67 % of segment), it finds the 1950 Hz
    trough → score ≥ 70.

    Fails as long as score_rhotic uses only the segment midpoint for F3.
    """
    from accent_coach.comparison.consonants.liquids import score_rhotic

    half = 0.08  # 80 ms each half
    seg_r = _resonator([(500, 80), (1200, 120), (1950, 150), (3400, 200)], dur=half)
    seg_v = _resonator([(500, 80), (1500, 120), (2800, 200), (3600, 250)], dur=half)
    seg = np.concatenate([seg_r, seg_v]).astype(np.float32)
    audio = _place(seg, 0.02)
    ph = _ph("r", "R", start=0.02, end=0.02 + 2 * half)
    score = score_rhotic(audio, SR, ph)
    assert score >= 70, (
        f"score={score:.1f} — /r/ with F3 trough at 1950 Hz (first half) must score ≥ 70. "
        "Midpoint-only sampling can land in the F3-rising second half. "
        "Fix: sample at 25/33/50/67 % and return the minimum F3 for scoring."
    )
