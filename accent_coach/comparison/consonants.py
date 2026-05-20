"""Fricative spectral centroid comparison (lightweight).

Only covers /s z ʃ ʒ/. Uses spectral centroid averaged over each fricative
segment and compares against approximate RP reference centroids.
"""
from __future__ import annotations

import math

import numpy as np

from accent_coach.models import PhonemeInstance, SentenceAnalysis

# Approximate RP spectral centroid (Hz) per fricative — TODO(cite)
_RP_FRICATIVE_CENTROID: dict[str, float] = {
    "s":  6500.0,
    "z":  6000.0,
    "ʃ":  3500.0,
    "ʒ":  3000.0,
}

_FRICATIVES = frozenset(_RP_FRICATIVE_CENTROID)
_DECAY = 2000.0  # Hz decay constant


def _spectral_centroid(audio: np.ndarray, sr: int, p: PhonemeInstance) -> float | None:
    start = max(0, int(p.start_time * sr))
    end = min(len(audio), int(p.end_time * sr))
    if end - start < 32:
        return None
    segment = audio[start:end].astype(np.float64)
    spectrum = np.abs(np.fft.rfft(segment))
    freqs = np.fft.rfftfreq(len(segment), d=1.0 / sr)
    total = spectrum.sum()
    if total < 1e-12:
        return None
    return float(np.dot(freqs, spectrum) / total)


def score_consonants(
    user: SentenceAnalysis,
    audio: np.ndarray,
    sr: int,
    target_audio: np.ndarray | None = None,
    target_sr: int | None = None,
    target: SentenceAnalysis | None = None,
) -> float:
    scores: list[float] = []
    all_phonemes: list[PhonemeInstance] = (
        [v.phoneme for v in user.vowels] + [s.phoneme for s in user.stops]
    )
    for p in all_phonemes:
        if p.phoneme not in _FRICATIVES:
            continue
        centroid = _spectral_centroid(audio, sr, p)
        if centroid is None:
            continue
        ref = _RP_FRICATIVE_CENTROID[p.phoneme]
        delta = abs(centroid - ref)
        scores.append(100.0 * math.exp(-delta / _DECAY))

    return float(np.mean(scores)) if scores else 70.0  # neutral default
