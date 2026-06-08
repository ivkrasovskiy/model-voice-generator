"""CoG-based fricative quality scoring.

Covers /s z ʃ ʒ θ ð f v/ — the full inventory of English obstruent fricatives.
Primary comparison is user CoG vs target CoG when a target is available;
absolute mode uses corpus-derived reference values.

Reference: Jongman et al. (2000), "Acoustic characteristics of English
fricatives", JASA 108(3), 1252–1263.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, sosfilt

from accent_coach.models import PhonemeInstance, SentenceAnalysis
from accent_coach.reference.genam_norms import (
    GA_FRICATIVE_COG_DECAY_HZ,
    GA_FRICATIVE_COG_HZ,
    GA_TH_S_SUBSTITUTION_THRESHOLD_HZ,
)
from accent_coach.reference.rp_norms import (
    RP_FRICATIVE_COG_DECAY_HZ,
    RP_FRICATIVE_COG_HZ,
    RP_TH_S_SUBSTITUTION_THRESHOLD_HZ,
)

_FRICATIVES: frozenset[str] = frozenset(RP_FRICATIVE_COG_HZ)

# /HH/ (aspiration) skipped — not a true fricative CoG signal (spec 3A)
_SKIP = frozenset({"h", "hh"})

# Butterworth HP cutoff: removes voicing fundamental, F1, and F2 contamination
# from adjacent vowels.  A 4th-order Butterworth gives a gradual rolloff that
# preserves the left shoulder of broad-spectrum fricatives (/θ ð f v/), unlike
# the former brick-wall bin-mask that over-suppressed those phonemes.
# Matches the approximate HP conditioning used in Jongman et al. (2000).
_COG_HP_HZ: float = 2000.0

# Frication gate: a true fricative concentrates energy in the high band; a window
# that mis-aligned onto the adjacent vowel/closure does not.  Require ≥ this
# fraction of raw-segment energy above _FRICATION_HF_HZ, else the token is NOT
# frication and is dropped (return None) instead of polluting the group mean.
_FRICATION_HF_HZ: float = 3000.0
_FRICATION_HF_MIN_RATIO: float = 0.20


def _spectral_centroid(audio: np.ndarray, sr: int, p: PhonemeInstance) -> float | None:
    """Spectral centre of gravity (CoG) using power spectrum with Butterworth HP.

    Power spectrum per Jongman et al. (2000): spectrum = |FFT|².
    4th-order Butterworth HP applied in time domain before FFT so that the
    gradual rolloff does not abruptly zero energy near the cutoff frequency.

    Returns None when the segment is not frication (a mis-aligned vowel/closure):
    we refuse to emit a CoG for a token that does not look like a fricative.
    """
    start = max(0, int(p.start_time * sr))
    end = min(len(audio), int(p.end_time * sr))
    if end - start < 32:
        return None
    raw = audio[start:end].astype(np.float64)

    # Frication gate on the RAW segment (before HP): is most energy high-band?
    raw_spec = np.abs(np.fft.rfft(raw)) ** 2
    raw_freqs = np.fft.rfftfreq(len(raw), d=1.0 / sr)
    raw_total = raw_spec.sum()
    if raw_total < 1e-12:
        return None
    hf_ratio = float(raw_spec[raw_freqs >= _FRICATION_HF_HZ].sum() / raw_total)
    if hf_ratio < _FRICATION_HF_MIN_RATIO:
        return None  # not frication — likely a mis-aligned vowel/closure window

    nyq = sr / 2.0
    sos = butter(4, _COG_HP_HZ / nyq, btype="high", output="sos")
    segment = sosfilt(sos, raw)
    spectrum = np.abs(np.fft.rfft(segment)) ** 2
    freqs = np.fft.rfftfreq(len(segment), d=1.0 / sr)
    total = spectrum.sum()
    if total < 1e-12:
        return None
    return float(np.dot(freqs, spectrum) / total)


def _cog_norms(accent_target: str) -> tuple[dict[str, float], float, float]:
    if accent_target == "genam":
        return GA_FRICATIVE_COG_HZ, GA_FRICATIVE_COG_DECAY_HZ, GA_TH_S_SUBSTITUTION_THRESHOLD_HZ
    return RP_FRICATIVE_COG_HZ, RP_FRICATIVE_COG_DECAY_HZ, RP_TH_S_SUBSTITUTION_THRESHOLD_HZ


def score_fricatives(
    user: SentenceAnalysis,
    audio: np.ndarray,
    sr: int,
    target: SentenceAnalysis | None = None,
    target_audio: np.ndarray | None = None,
    target_sr: int | None = None,
    accent_target: str = "rp",
) -> tuple[float | None, list[str]]:
    """Score fricative quality. Returns (score 0–100 or None, diagnostics).

    None is returned when no scorable fricative tokens are found so the
    aggregator can redistribute that weight rather than applying a fictional score.
    """
    ref_cog, decay, th_s_threshold = _cog_norms(accent_target)

    # Build target CoG per phoneme (comparison mode)
    target_cog: dict[str, list[float]] = {}
    if target is not None and target_audio is not None and target_sr is not None:
        for p in target.phonemes:
            if p.phoneme not in _FRICATIVES or p.phoneme in _SKIP:
                continue
            cog = _spectral_centroid(target_audio, target_sr, p)
            if cog is not None:
                target_cog.setdefault(p.phoneme, []).append(cog)

    scores: list[float] = []
    diagnostics: list[str] = []
    th_s_errors: list[str] = []

    for p in user.phonemes:
        ph = p.phoneme
        if ph not in _FRICATIVES or ph in _SKIP:
            continue
        cog = _spectral_centroid(audio, sr, p)
        if cog is None:
            continue

        # Prefer target CoG; fall back to corpus reference
        ref = float(np.mean(target_cog[ph])) if ph in target_cog else ref_cog[ph]

        delta = abs(cog - ref)
        scores.append(100.0 * math.exp(-delta / decay))

        # TH/DH substitution check: CoG too high for a dental fricative
        if ph in ("θ", "ð") and cog > th_s_threshold:
            th_s_errors.append(p.word)

    if th_s_errors:
        words = ", ".join(f"'{w}'" for w in th_s_errors[:3])
        diagnostics.append(
            f"In {words}: your /θ/ or /ð/ sounds like /s/ or /z/ (spectral energy too high). "
            "Place your tongue tip lightly against the upper front teeth — not the alveolar ridge."
        )

    return (float(np.mean(scores)) if scores else None), diagnostics
