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
import pytest
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
    """An /s/ at the IN-DOMAIN native CoG (≈5200 Hz) must score ≥ 85.

    The reference is the native /s/ mean measured through this 16 kHz pipeline
    (5200 Hz), NOT Jongman's 22 kHz-band 7000 Hz. A synthetic /s/ at 5200 lands
    on the reference → ~100. Fails if scoring is flat or the reference is wrong.
    """
    from accent_coach.comparison.consonants.fricatives import score_fricatives

    noise = _noise(center=5200, bw=1500, dur=0.15)
    audio = _place(noise, 0.02)
    ph = _ph("s", "S", start=0.02, end=0.17)
    score, _ = score_fricatives(_sentence([ph]), audio, SR)
    assert score is not None and score >= 85, (
        f"Near-ideal /s/ (CoG ≈ 5200 Hz = in-domain native) scored {score:.1f}. Expected ≥ 85."
    )


def test_sh_substitution_scores_below_55():
    """An /s/ realised like /ʃ/ (CoG≈3500 Hz) must score clearly below a good /s/.

    With the in-domain /s/ reference (5200 Hz, not 7000) the dynamic range is
    compressed — /s/ now sits nearer the other fricatives — so an /s/→/ʃ/ shift
    (~1600 Hz off, ~2 native SD) scores ~45, not <35. The invariant that matters
    is discrimination: a substitution scores far below the ~100 of a correct /s/.
    Still-frication signal (passes the HF gate), so it tests CoG, not the gate.
    """
    from accent_coach.comparison.consonants.fricatives import score_fricatives

    noise = _noise(center=3500, bw=1500, dur=0.15)
    audio = _place(noise, 0.02)
    ph = _ph("s", "S", start=0.02, end=0.17)
    score, _ = score_fricatives(_sentence([ph]), audio, SR)
    assert score is not None and score < 55, (
        f"/s/→/ʃ/ substitution (CoG≈3500 Hz) scored {score:.1f}. Expected < 55."
    )


def test_fricative_score_gap_good_vs_bad_at_least_45():
    """Discrimination: ideal /s/ (5200) minus /ʃ/-substitution (3500) ≥ 45 pts.

    Gap target lowered 65→45: the in-domain reference (5200, not 7000) compresses
    the achievable range, but the metric must still strongly separate a correct
    /s/ from a substitution. (Validate fine calibration on the bench gap, not here.)
    """
    from accent_coach.comparison.consonants.fricatives import score_fricatives

    ph = _ph("s", "S", start=0.02, end=0.17)
    good, _ = score_fricatives(_sentence([ph]), _place(_noise(5200, 1500), 0.02), SR)
    bad, _ = score_fricatives(_sentence([ph]), _place(_noise(3500, 1500), 0.02), SR)
    assert good is not None and bad is not None
    assert good - bad >= 45, (
        f"Score gap: {good:.1f} − {bad:.1f} = {good - bad:.1f}. Expected ≥ 45 pts."
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
# Section 2b — In-domain VOT reference (Open item #2)
#
# RP_VOT_MEAN_MS/SD (Lisker & Abramson 1964, 67.5-87.5 ms) describe TEXTBOOK
# isolated-word VOT, not what extract_vot measures through this pipeline on
# connected speech. Real in-domain VOT (scripts/tools/consonant_per_phoneme.py,
# n=24/group, see docs/prosody_consonant_upgrade.md) is 0-22 ms even for
# natives:
#   /p/ native=9.8ms (n=16)  owner=1.7ms (n=8)  raw gap +8.0ms
#   /t/ native=16.0ms (n=41) owner=3.5ms (n=13) raw gap +12.5ms
#   /k/ native=9.5ms (n=48)  owner=1.6ms (n=8)  raw gap +7.9ms
#
# Scoring these against the textbook reference puts BOTH groups 6-9 SD away
# -> scores < 2/100 for everyone, score-gap < 1 point (same bandwidth-confound
# pattern as the pre-fix fricative CoG, see Section 1). Re-deriving
# RP_VOT_MEAN_MS/SD in-domain (mirroring measure_fricative_cog.py /
# scripts/tools/measure_vot_reference.py) must turn these SAME raw-VOT
# numbers into a real score gap, per the native-owner-gap validation
# principle (CLAUDE.md: validate by the gap, never the absolute level).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ph, native_vot, owner_vot, min_native_score, min_gap",
    [
        ("p", 9.8, 1.7, 20.0, 10.0),
        ("t", 16.0, 3.5, 20.0, 10.0),
        ("k", 9.5, 1.6, 20.0, 10.0),
    ],
)
def test_in_domain_vot_gap_meaningful(ph, native_vot, owner_vot, min_native_score, min_gap):
    """In-domain native VOT must score non-negligibly, and beat owner by a real gap.

    With the Lisker & Abramson textbook RP_VOT_MEAN_MS (67.5-87.5 ms, sd=10),
    both native (9.5-16 ms) and owner (1.6-3.5 ms) land 6-9 SD away, so both
    score < 2/100 and the gap is < 1 point — no discrimination at all. After
    in-domain re-derivation, the native value (which is now near the
    reference mean) must score meaningfully, and must beat the owner value by
    a real margin.
    """
    from accent_coach.comparison.consonants.stops import score_stops

    native, _ = score_stops(_sentence([], [_stop(ph, native_vot)]))
    owner, _ = score_stops(_sentence([], [_stop(ph, owner_vot)]))
    assert native is not None and owner is not None
    assert native >= min_native_score, (
        f"In-domain native /{ph}/ VOT={native_vot}ms scored {native:.1f}. "
        f"Expected >= {min_native_score}. RP_VOT_MEAN_MS['{ph}'] is still the "
        "Lisker & Abramson textbook value — re-derive in-domain."
    )
    gap = native - owner
    assert gap >= min_gap, (
        f"In-domain /{ph}/ gap: native({native:.1f}) - owner({owner:.1f}) = "
        f"{gap:.1f}. Expected >= {min_gap}."
    )


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

    audio, phonemes = _composite_audio_and_phonemes(s_center=5200, r_f3=1950, l_f2=1050)
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
    audio, phonemes = _composite_audio_and_phonemes(s_center=5200, r_f3=2700, l_f2=1700)
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

    native_audio, ph = _composite_audio_and_phonemes(s_center=5200, r_f3=1950, l_f2=1050)
    l2_audio, _ = _composite_audio_and_phonemes(s_center=5200, r_f3=2700, l_f2=1700)

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
    from accent_coach.reference.genam_norms import GA_FRICATIVE_COG_HZ
    from accent_coach.reference.rp_norms import RP_FRICATIVE_COG_HZ

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
    """/f/ bimodal signal whose power-spectrum CoG = the /f/ reference (4700 Hz) scores ≥ 95.

    Tones chosen so the POWER CoG lands on the in-domain /f/ reference (4700 Hz):
      tone 4200 Hz (amplitude 2) + tone 6700 Hz (amplitude 1).
      Power CoG     = (4200×4 + 6700×1) / 5  = 4700 Hz → score ≈ 100
      Magnitude CoG = (4200×2 + 6700×1) / 3 ≈ 5033 Hz → score ≈ 85
    Both above the 2 kHz HP and below the 7.6 kHz ceiling, so this isolates the
    magnitude-vs-power spectrum bug (§1A). Fails with a magnitude spectrum.
    """
    from accent_coach.comparison.consonants.fricatives import score_fricatives

    n = int(0.15 * SR)
    t = np.arange(n, dtype=np.float64) / SR
    # amplitude 2 at 4200 Hz + amplitude 1 at 6700 Hz → power CoG = 4700 (/f/ ref)
    signal = (2.0 * np.sin(2 * np.pi * 4200 * t)
              + 1.0 * np.sin(2 * np.pi * 6700 * t)).astype(np.float32)
    signal = signal / (np.abs(signal).max() + 1e-9) * 0.8
    audio = _place(signal, 0.02)
    ph = _ph("f", "F", start=0.02, end=0.17)
    score, _ = score_fricatives(_sentence([ph]), audio, SR)
    assert score is not None and score >= 95, (
        f"/f/ bimodal signal (power CoG = 4700 Hz = reference) scored {score:.1f}. "
        "Expected ≥ 95. Magnitude spectrum gives CoG ≈ 5033 Hz → score ≈ 85. "
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
# VOT extraction (burst → voicing onset) — business-logic tests moved to
# tests/accent_coach/test_vot.py. The old §2B/§2C tests pinned the deleted
# median-window / autocorr-threshold internals of the previous greedy detector;
# the rewrite (closure → burst → F0-constrained voicing) is covered there on
# hand-checkable ground truth (connected-speech VOT, short-vs-long, English range).
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Section 14 — /θ ð/ phoneme-aware frication gate
#
# Root cause of the /θ ð/ gate bug (docs/prosody_consonant_upgrade.md §Open-1):
# The HF>3 kHz gate uses a fixed 0.20 ratio threshold designed for sibilants /s z/.
# Real dental frication has two properties that push the ratio below 0.20:
#   (a) Voiced /ð/: strong harmonic carrier below 1 kHz dwarfs the weak turbulence
#       above 3 kHz → the F0/harmonics energy pulls the HF ratio to ~0.10–0.15.
#   (b) Any dental in connected speech: the aligned window mixes the frication
#       segment with adjacent-vowel coarticulation energy (below 3 kHz), further
#       diluting the ratio to ~0.10–0.15.
# Result: ~100% of /θ ð/ tokens are rejected for ALL speaker groups, so the
# phoneme contributes nothing to anyone's score.
#
# Fix: phoneme-aware gate — /θ ð/ (and /f v/) use a relaxed HF ratio threshold.
#
# Each test below FAILS with the current uniform 0.20 gate and PASSES after the
# phoneme-aware fix.
# ---------------------------------------------------------------------------


def _voiced_dental(dur: float = 0.15) -> np.ndarray:
    """Voiced /ð/: harmonic carrier below 840 Hz + weak dental frication 2–6 kHz.

    Models real voiced dental frication.  Measured ratios:
      HF>3 kHz ≈ 0.053  →  below the 0.20 sibilant gate AND the old 0.06 fallback
      HF>2 kHz ≈ 0.069  →  above the 0.05 relaxed-dental gate (fix target)
    The voiced harmonic energy dominates total power; frication is weak but present.
    """
    n = int(SR * dur)
    t = np.arange(n, dtype=np.float64) / SR
    voiced = sum((1.0 / k) * np.sin(2 * np.pi * 120 * k * t) for k in range(1, 8))
    voiced = (voiced / (np.abs(voiced).max() + 1e-9) * 0.8).astype(np.float32)
    raw = np.random.default_rng(7).standard_normal(n).astype(np.float64)
    sos = butter(4, [2000 / (SR / 2), 6000 / (SR / 2)], btype="band", output="sos")
    fric = sosfilt(sos, raw)
    fric = (fric / (np.abs(fric).max() + 1e-9) * 0.4).astype(np.float32)
    return voiced + fric


def _th_with_coarticulation(dur: float = 0.15) -> np.ndarray:
    """Unvoiced /θ/ + strong adjacent-vowel coarticulation energy (below 3 kHz).

    Mimics a real-speech aligned /θ/ window: the vowel resonances (F1=600,
    F2=1500, F3=2500) dominate energy below 3 kHz; actual dental frication
    (2–6 kHz, amplitude 0.35) is much weaker.
      vowel power  ≈ 0.32 (resonator peak 0.8; energy almost all below 3 kHz)
      fric power   ≈ 0.061 (0.35 peak, 75% above 3 kHz → 0.046)
      ratio        ≈ 0.046 / (0.32 + 0.061) ≈ 0.12   →  below 0.20 sibilant gate
    """
    n = int(SR * dur)
    vowel = _resonator([(600, 100), (1500, 150), (2500, 200)], dur=dur)
    raw = np.random.default_rng(3).standard_normal(n).astype(np.float64)
    sos = butter(4, [2000 / (SR / 2), 6000 / (SR / 2)], btype="band", output="sos")
    fric = sosfilt(sos, raw)
    fric = (fric / (np.abs(fric).max() + 1e-9) * 0.35).astype(np.float32)
    return vowel + fric


def test_voiced_dh_weak_frication_is_scored():
    """Voiced /ð/ (harmonic carrier + weak frication) must not be gate-rejected.

    The HF>3 kHz ratio for this signal is ~0.15 — below the 0.20 sibilant gate.
    This is the same reason real /ð/ tokens score nothing for ALL speaker groups:
    the voiced energy pulls the ratio below threshold.

    Before fix: score=None  (gate rejects voiced dental frication).
    After fix:  score is not None  (phoneme-aware relaxed gate for /ð/).
    """
    from accent_coach.comparison.consonants.fricatives import score_fricatives

    audio = _place(_voiced_dental(), 0.02)
    ph = _ph("ð", "DH", start=0.02, end=0.17)
    score, _ = score_fricatives(_sentence([ph]), audio, SR)
    assert score is not None, (
        "Voiced /ð/ with weak frication (HF>3 kHz ratio ~0.15) must not be gate-rejected. "
        "Current uniform 0.20 threshold discards real dental frication for ALL speakers. "
        "Fix: phoneme-aware relaxed HF threshold for /θ ð/ (and /f v/)."
    )


def test_th_frication_with_adjacent_vowel_energy_is_scored():
    """/θ/ frication mixed with vowel coarticulation energy must not be gate-rejected.

    In connected speech the aligned /θ/ window includes adjacent-vowel energy
    (strong F1/F2 below 3 kHz) that dilutes the HF ratio to ~0.12.  The signal
    still contains real dental frication at 2–6 kHz.

    Before fix: score=None.  After fix: score is not None.
    """
    from accent_coach.comparison.consonants.fricatives import score_fricatives

    audio = _place(_th_with_coarticulation(), 0.02)
    ph = _ph("θ", "TH", start=0.02, end=0.17)
    score, _ = score_fricatives(_sentence([ph]), audio, SR)
    assert score is not None, (
        "/θ/ frication mixed with strong vowel coarticulation (HF>3 kHz ratio ~0.12) "
        "must not be gate-rejected. Fix: phoneme-aware relaxed threshold for /θ ð/."
    )


def test_dental_gate_rejects_stop_closure():
    """A voiced stop closure (LP-filtered voicing only) aligned to /ð/ must be rejected.

    /ð/→/d/ substitution: during the closure there is only low-frequency voiced
    energy (voice bar below 500 Hz).  The relaxed dental gate must still reject
    this — it is not frication, even at the relaxed threshold.
    Passes before and after the fix (regression guard against over-relaxation).
    """
    from accent_coach.comparison.consonants.fricatives import score_fricatives

    n = int(0.5 * SR)
    raw = np.random.default_rng(13).standard_normal(n).astype(np.float64)
    sos = butter(4, 400 / (SR / 2), btype="low", output="sos")
    closure = (sosfilt(sos, raw) * 0.5).astype(np.float32)
    ph = _ph("ð", "DH", start=0.02, end=0.17)
    score, _ = score_fricatives(_sentence([ph]), closure, SR)
    assert score is None, (
        "Stop closure (voice bar below 400 Hz, no frication) must be gate-rejected "
        "even with the relaxed dental gate."
    )


def test_dental_gate_yield_covers_at_least_90_percent():
    """Gate must score ≥ 90% of a realistic batch of dental /θ ð/ tokens.

    English dental tokens in real speech are mostly voiced /ð/ ("the", "that",
    "with", "their") plus a minority of unvoiced /θ/ ("think", "through").  We
    model this realistic 2:1 voiced-to-unvoiced ratio with 12 varied tokens.

    Before fix: voiced /ð/ tokens all return None → yield ≈ 33% (only unvoiced
                pass the 0.20 sibilant gate at 3 kHz).
    After fix:  phoneme-aware gate → yield ≥ 90%.
    """
    from accent_coach.comparison.consonants.fricatives import score_fricatives

    # ------------------------------------------------------------------
    # Build 12 synthetic dental tokens and place them at non-overlapping
    # positions in a 2-second audio clip.
    # ------------------------------------------------------------------
    total_dur = 2.0
    slot_dur = 0.14   # 140 ms per slot (≥ the 150 ms segment used by scoring)
    audio = np.zeros(int(total_dur * SR), dtype=np.float32)
    phonemes: list[PhonemeInstance] = []
    rng = np.random.default_rng(42)

    def _place_at(seg: np.ndarray, slot: int) -> tuple[float, float]:
        start_s = slot * slot_dur + 0.01
        end_s = start_s + len(seg) / SR
        s = int(start_s * SR)
        e = min(len(audio), s + len(seg))
        audio[s:e] = seg[:e - s]
        return start_s, start_s + len(seg) / SR

    def _voiced_dh_variant(f0: float, voiced_amp: float, fric_amp: float) -> np.ndarray:
        """Voiced dental: harmonic carrier at f0 + weak frication 2–6 kHz."""
        n = int(0.12 * SR)
        t = np.arange(n, dtype=np.float64) / SR
        voiced = sum((1.0/k) * np.sin(2*np.pi*f0*k*t) for k in range(1, 8))
        voiced = (voiced / (np.abs(voiced).max() + 1e-9) * voiced_amp).astype(np.float32)
        raw = rng.standard_normal(n).astype(np.float64)
        sos = butter(4, [2000/(SR/2), 6000/(SR/2)], btype="band", output="sos")
        fric = sosfilt(sos, raw)
        fric = (fric / (np.abs(fric).max() + 1e-9) * fric_amp).astype(np.float32)
        return voiced + fric

    def _unvoiced_th_variant(center: float, bw: float) -> np.ndarray:
        """Unvoiced dental: band-limited frication noise."""
        return _noise(center, bw, dur=0.12)

    # 8 voiced /ð/ tokens — varied F0 (100–200 Hz) and voiced/frication amplitude
    dh_params = [
        (120, 0.8, 0.40), (150, 0.7, 0.35), (180, 0.9, 0.45), (100, 0.6, 0.30),
        (130, 0.8, 0.38), (160, 0.75, 0.40), (200, 0.85, 0.42), (110, 0.65, 0.32),
    ]
    for i, (f0, va, fa) in enumerate(dh_params):
        seg = _voiced_dh_variant(f0, va, fa)
        start, end = _place_at(seg, i)
        phonemes.append(_ph("ð", "DH", start=start, end=end))

    # 4 unvoiced /θ/ tokens — varied center frequency (3 kHz–5 kHz range)
    th_params = [(3500, 2500), (4200, 2000), (4800, 3000), (3800, 2200)]
    for j, (center, bw) in enumerate(th_params):
        seg = _unvoiced_th_variant(center, bw)
        start, end = _place_at(seg, len(dh_params) + j)
        phonemes.append(_ph("θ", "TH", start=start, end=end))

    # ------------------------------------------------------------------
    # Score all tokens; count how many are not gate-rejected (not None).
    # ------------------------------------------------------------------
    n_expected = len(phonemes)   # 12
    user = _sentence(phonemes)
    score, _ = score_fricatives(user, audio, SR)

    # score_fricatives returns a mean over all non-None tokens; to count
    # individual yields we call the internal centroid per phoneme manually.
    from accent_coach.comparison.consonants.fricatives import _spectral_centroid  # noqa: PLC0415
    n_scored = sum(
        1 for p in phonemes
        if _spectral_centroid(audio, SR, p) is not None
    )

    yield_pct = 100 * n_scored / n_expected
    assert yield_pct >= 90, (
        f"Dental gate yield: {n_scored}/{n_expected} = {yield_pct:.0f}%. "
        "Expected ≥ 90%. Before fix: voiced /ð/ tokens all gate-rejected → ~33% yield. "
        "Fix: phoneme-aware relaxed gate for /θ ð/."
    )


def test_dental_gate_yield_before_fix_was_below_50_pct():
    """Documents the pre-fix yield: uniform 0.20 gate rejects ALL voiced /ð/.

    This test asserts the old broken behaviour using the same 8 voiced /ð/ tokens.
    It passes only if the old gate is restored — it exists to make the regression
    obvious if someone removes the phoneme-aware gate.

    The assertion is: using the strict sibilant gate on /ð/ tokens gives < 50%
    yield (because all voiced dentals fall below 0.20 at 3 kHz).
    """
    n = int(0.12 * SR)
    rng = np.random.default_rng(42)

    def _voiced_dh(f0, va, fa):
        t = np.arange(n, dtype=np.float64) / SR
        voiced = sum((1.0/k)*np.sin(2*np.pi*f0*k*t) for k in range(1, 8))
        voiced = (voiced / (np.abs(voiced).max()+1e-9) * va).astype(np.float32)
        raw = rng.standard_normal(n).astype(np.float64)
        sos = butter(4, [2000/(SR/2), 6000/(SR/2)], btype="band", output="sos")
        fric = sosfilt(sos, raw)
        fric = (fric / (np.abs(fric).max()+1e-9) * fa).astype(np.float32)
        return voiced + fric

    params = [
        (120, 0.8, 0.40), (150, 0.7, 0.35), (180, 0.9, 0.45), (100, 0.6, 0.30),
        (130, 0.8, 0.38), (160, 0.75, 0.40), (200, 0.85, 0.42), (110, 0.65, 0.32),
    ]

    # Compute HF>3 kHz ratio for each voiced /ð/ signal (the old gate)
    old_gate_hz = 3000.0
    old_gate_threshold = 0.20
    n_pass_old = 0
    for f0, va, fa in params:
        sig = _voiced_dh(f0, va, fa).astype(np.float64)
        spec = np.abs(np.fft.rfft(sig))**2
        freqs = np.fft.rfftfreq(len(sig), d=1.0/SR)
        ratio = float(spec[freqs >= old_gate_hz].sum() / spec.sum())
        if ratio >= old_gate_threshold:
            n_pass_old += 1

    # All 8 voiced /ð/ tokens fail the old sibilant gate
    old_yield = 100 * n_pass_old / len(params)
    assert old_yield < 50, (
        f"Voiced /ð/ HF>3 kHz yield under old gate: {n_pass_old}/8 = {old_yield:.0f}%. "
        "Expected < 50% — the old gate should reject all or almost all voiced dentals. "
        "If this assertion fails, the voiced dental signals no longer model real speech."
    )


def test_voiced_dh_correct_cog_outscores_s_substitution():
    """After gate fix: correct /ð/ CoG (~4000 Hz) must outscore /ð/→/s/ substitution.

    Uses the voiced-carrier signal (the one the current gate rejects) for the
    correct-/ð/ case, so this test exercises the gate AND the CoG scoring.

    Before fix: score_correct=None  (gate rejects the voiced signal).
    After fix:  score_correct > score_wrong + 10.
    """
    from accent_coach.comparison.consonants.fricatives import score_fricatives

    ph = _ph("ð", "DH", start=0.02, end=0.17)
    # Correct /ð/: voiced carrier + frication CoG ~4000 Hz
    audio_correct = _place(_voiced_dental(), 0.02)
    # /ð/→/s/ substitution: high-frequency sibilant noise, CoG ~7000 Hz
    audio_wrong = _place(_noise(7000, 2000, dur=0.15), 0.02)

    score_c, _ = score_fricatives(_sentence([ph]), audio_correct, SR)
    score_w, _ = score_fricatives(_sentence([ph]), audio_wrong, SR)

    assert score_c is not None, (
        "Correct /ð/ (voiced carrier + CoG ~4000 Hz) must be scored after gate fix."
    )
    assert score_w is not None, "/ð/→/s/ substitution (high CoG) must be scored."
    assert score_c > score_w + 10, (
        f"Correct /ð/ (score={score_c:.1f}) must outscore s-substitution "
        f"({score_w:.1f}) by ≥ 10 pts."
    )
