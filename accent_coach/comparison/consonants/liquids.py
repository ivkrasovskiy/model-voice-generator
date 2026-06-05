"""Rhotic /r/ and lateral /l/ quality scoring.

/r/ — F3 depression (spec 3C):
  English retroflex/bunched /r/ drops F3 to 1800–2200 Hz.
  Non-English /r/ (tapped, trilled, uvular) leaves F3 at 2400–2800 Hz.
  F3 measured at phoneme midpoint via parselmouth Burg formant tracker.

/l/ — dark vs clear allophone (spec 3D):
  Syllable-final /l/ (dark) requires velarisation: F2 drops to 800–1300 Hz.
  Clear /l/ everywhere (F2 stays ~1400–1800 Hz) is the common L2 error.
  Only penalised in syllable-final position — that is where it is perceptually
  distinctive in RP/GenAm.

References:
  Ladefoged & Johnson (2011), "A Course in Phonetics", 7th ed., Ch. 9.
  Wells (1982), "Accents of English", Vol. 1, p. 259.
  Sproat & Fujimura (1993), "Allophonic variation in English /l/ and its
    implications for phonetic implementation", J. Phonetics 21, 291–311.
"""
from __future__ import annotations

import math

import numpy as np
import parselmouth

from accent_coach.models import PhonemeInstance, SentenceAnalysis
from accent_coach.pipeline.alignment import IPA_VOWELS
from accent_coach.reference.genam_norms import (
    GA_LATERAL_CLEAR_F2_THRESHOLD_HZ,
    GA_LATERAL_DARK_F2_TARGET_HZ,
    GA_LATERAL_F2_DECAY_HZ,
    GA_RHOTIC_F3_DECAY_HZ,
    GA_RHOTIC_F3_TARGET_HZ,
)
from accent_coach.reference.rp_norms import (
    RP_LATERAL_CLEAR_F2_THRESHOLD_HZ,
    RP_LATERAL_DARK_F2_TARGET_HZ,
    RP_LATERAL_F2_DECAY_HZ,
    RP_RHOTIC_F3_DECAY_HZ,
    RP_RHOTIC_F3_TARGET_HZ,
)

_MIN_SEGMENT_S: float = 0.020   # 20 ms minimum for reliable formant estimation
# Burg parameters — 5 formants is the Praat-recommended minimum to track F3
# reliably.  With 4 formants a spurious pole can crowd out F3 in short windows.
# Ceiling: 5000 Hz for male (mean F0 <165 Hz), 5500 Hz for female — mirrors
# the approach in pipeline/formants.py._estimate_max_formant().
_MAX_FORMANTS: int = 5
_MAX_FORMANT_MALE_HZ: float = 5000.0
_MAX_FORMANT_FEMALE_HZ: float = 5500.0
_WINDOW_LENGTH_S: float = 0.025
_MALE_F0_THRESHOLD_HZ: float = 165.0  # F0 below this → male ceiling


def _formant_at_midpoint(
    audio: np.ndarray,
    sr: int,
    p: PhonemeInstance,
    formant_n: int,
    mean_f0: float = 120.0,
) -> float | None:
    """Measure Fn at phoneme midpoint using parselmouth Burg LPC.

    Uses the full-signal formant object to exploit surrounding context (Praat
    documentation recommends tracking on the full signal rather than
    per-segment to avoid edge artefacts at short windows).

    mean_f0 selects male (≤165 Hz) or female (>165 Hz) Burg ceiling so that
    Praat does not mistake upper harmonics for F3/F4.
    """
    dur = p.end_time - p.start_time
    if dur < _MIN_SEGMENT_S:
        return None

    max_formant = _MAX_FORMANT_MALE_HZ if mean_f0 <= _MALE_F0_THRESHOLD_HZ else _MAX_FORMANT_FEMALE_HZ
    sound = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=sr)
    formant = sound.to_formant_burg(
        time_step=0.01,
        max_number_of_formants=_MAX_FORMANTS,
        maximum_formant=max_formant,
        window_length=_WINDOW_LENGTH_S,
        pre_emphasis_from=50.0,
    )
    mid = (p.start_time + p.end_time) / 2
    value = formant.get_value_at_time(formant_n, mid)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)


def score_rhotic(
    audio: np.ndarray,
    sr: int,
    phoneme: PhonemeInstance,
    accent_target: str = "rp",
) -> float:
    """Score 0–100 for English /r/ quality based on F3 depression.

    100 = F3 at or below the English target (≤ ~2000 Hz).
    Score falls exponentially as F3 rises above the rhotic target.
    Returns 50.0 (neutral) when F3 cannot be measured.
    """
    if accent_target == "genam":
        target_hz = GA_RHOTIC_F3_TARGET_HZ
        decay = GA_RHOTIC_F3_DECAY_HZ
    else:
        target_hz = RP_RHOTIC_F3_TARGET_HZ
        decay = RP_RHOTIC_F3_DECAY_HZ

    f3 = _formant_at_midpoint(audio, sr, phoneme, formant_n=3)
    if f3 is None:
        return 50.0

    # Score: perfect when F3 ≤ target; exponential penalty for F3 > target
    if f3 <= target_hz:
        return 100.0
    delta = f3 - target_hz
    return float(100.0 * math.exp(-delta / decay))


def score_lateral(
    audio: np.ndarray,
    sr: int,
    phoneme: PhonemeInstance,
    syllable_final: bool,
    accent_target: str = "rp",
) -> float:
    """Score 0–100 for /l/ quality based on F2 in its syllable position.

    Syllable-final position: dark /l/ expected (F2 ~800–1300 Hz in RP).
      F2 above clear-/l/ threshold → penalty (missing velarisation).
    Syllable-initial position: clear /l/ is correct → F2 near clear target
      is fine; no penalty regardless of F2 value.
    Returns 50.0 (neutral) when F2 cannot be measured.
    """
    if accent_target == "genam":
        dark_target = GA_LATERAL_DARK_F2_TARGET_HZ
        clear_threshold = GA_LATERAL_CLEAR_F2_THRESHOLD_HZ
        decay = GA_LATERAL_F2_DECAY_HZ
    else:
        dark_target = RP_LATERAL_DARK_F2_TARGET_HZ
        clear_threshold = RP_LATERAL_CLEAR_F2_THRESHOLD_HZ
        decay = RP_LATERAL_F2_DECAY_HZ

    f2 = _formant_at_midpoint(audio, sr, phoneme, formant_n=2)
    if f2 is None:
        return 50.0

    if not syllable_final:
        # Initial position: clear /l/ is target — score based on F2 closeness
        # to a typical clear /l/ (1500–1700 Hz); mild tolerance.
        clear_target = 1550.0
        delta = abs(f2 - clear_target)
        return float(min(100.0, 100.0 * math.exp(-delta / (decay * 2))))

    # Final position: dark /l/ required
    if f2 <= clear_threshold:
        # F2 is in the dark or transitional range — score by proximity to dark target
        delta = abs(f2 - dark_target)
        return float(100.0 * math.exp(-delta / decay))
    else:
        # F2 above clear threshold in final position = missing velarisation
        delta = f2 - clear_threshold
        return float(100.0 * math.exp(-delta / decay))


def score_liquids(
    user: SentenceAnalysis,
    audio: np.ndarray,
    sr: int,
    accent_target: str = "rp",
) -> tuple[float | None, float | None, list[str]]:
    """Score rhotics and laterals. Returns (rhotic_score, lateral_score, diagnostics).

    None signals aggregator to redistribute weight for that class.
    Phoneme position (initial/final) is inferred from whether the phoneme's
    word position has the /l/ at the end — using a heuristic: is_stressed=False
    in a final phoneme position implies coda. When WhisperX phoneme-level
    alignment is available in SentenceAnalysis.phonemes, this is exact.
    """
    rhotic_scores: list[float] = []
    lateral_scores: list[float] = []
    diagnostics: list[str] = []
    clear_l_errors: list[str] = []
    rhotic_errors: list[str] = []

    # Group phonemes by word to determine position within word
    word_groups: dict[str, list[PhonemeInstance]] = {}
    for p in user.phonemes:
        word_groups.setdefault(p.word, []).append(p)

    # Build sorted word phoneme lists for position inference
    word_phoneme_order: dict[str, list[str]] = {
        w: [p.phoneme for p in sorted(ps, key=lambda x: x.start_time)]
        for w, ps in word_groups.items()
    }

    # Build a next-phoneme lookup (by start_time) for pre-vocalic /r/ gate.
    # Key on start_time float (not id()) so Pydantic object copies don't break lookup.
    sorted_all = sorted(user.phonemes, key=lambda x: x.start_time)
    next_ph_by_time: dict[float, str | None] = {}
    for i, ph_inst in enumerate(sorted_all):
        next_ph_by_time[ph_inst.start_time] = (
            sorted_all[i + 1].phoneme if i + 1 < len(sorted_all) else None
        )

    for p in user.phonemes:
        ph = p.phoneme

        if ph == "r":
            # RP is non-rhotic: /r/ at post-vocalic coda positions (e.g. "over",
            # "bird") is not pronounced. G2P always emits an /r/ phoneme there,
            # so scoring that position penalises RP natives unfairly. Gate to
            # pre-vocalic tokens only (next phoneme is a vowel) for RP.
            # Linking /r/ (word-final /r/ before vowel-initial next word) IS
            # pronounced in RP and is correctly included — the check is purely
            # on whether the next phoneme globally is a vowel.
            if accent_target == "rp" and next_ph_by_time.get(p.start_time) not in IPA_VOWELS:
                continue
            score = score_rhotic(audio, sr, p, accent_target=accent_target)
            rhotic_scores.append(score)
            # Flag if F3 measured above non-native threshold
            if score < 50:
                rhotic_errors.append(p.word)

        elif ph == "l":
            # Heuristic for syllable-final: is this /l/ the last phoneme in the word?
            word_phones = word_phoneme_order.get(p.word, [])
            is_final = bool(word_phones) and word_phones[-1] == "l"
            score = score_lateral(audio, sr, p, syllable_final=is_final, accent_target=accent_target)
            lateral_scores.append(score)
            if is_final and score < 55:
                clear_l_errors.append(p.word)

    if rhotic_errors:
        words = ", ".join(f"'{w}'" for w in rhotic_errors[:3])
        diagnostics.append(
            f"Your /r/ in {words} sounds non-English (F3 too high). "
            "Bunch the sides of your tongue against the upper molars, or curl the tip back — "
            "neither touching the roof of the mouth."
        )
    if clear_l_errors:
        words = ", ".join(f"'{w}'" for w in clear_l_errors[:3])
        diagnostics.append(
            f"Word-final /l/ in {words} sounds too 'clear'. "
            "For dark /l/, raise the back of your tongue toward the soft palate "
            "while keeping the tip behind the upper teeth."
        )

    return (
        float(np.mean(rhotic_scores)) if rhotic_scores else None,
        float(np.mean(lateral_scores)) if lateral_scores else None,
        diagnostics,
    )
