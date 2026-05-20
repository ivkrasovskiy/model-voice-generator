"""Synthetic audio fixtures — no binary files committed."""
from __future__ import annotations

import numpy as np
import pytest

SR = 16000


@pytest.fixture
def sr() -> int:
    return SR


@pytest.fixture
def synthetic_vowel_audio():
    """Synthesise /æ/ at F1≈748, F2≈1710 Hz using proper AR (all-poles) synthesis.

    Two 2nd-order resonators with bandwidths 80 Hz (F1) and 100 Hz (F2) excited
    by a pulse train at 120 Hz. Parselmouth's LPC should recover the formants
    within ±200 Hz.
    """
    from scipy.signal import lfilter

    duration = 0.5
    n = int(SR * duration)

    def _resonator_coeffs(f: float, bw: float) -> tuple[list[float], list[float]]:
        r = np.exp(-np.pi * bw / SR)
        b_coef = [1.0]
        a_coef = [1.0, -2 * r * np.cos(2 * np.pi * f / SR), r**2]
        return b_coef, a_coef

    # Glottal pulse train at 120 Hz
    pulse = np.zeros(n)
    pulse[:: int(SR / 120)] = 1.0

    b1, a1 = _resonator_coeffs(748, 80)
    b2, a2 = _resonator_coeffs(1710, 100)
    out = lfilter(b1, a1, lfilter(b2, a2, pulse))
    out = (out / (np.abs(out).max() + 1e-9) * 0.8).astype(np.float32)
    return out, SR


@pytest.fixture
def synthetic_burst_voice_audio():
    """Synthesise a burst at t=0.02s, voicing onset at t=0.09s → VOT ~70 ms."""
    duration = 0.3
    samples = int(SR * duration)
    audio = np.zeros(samples, dtype=np.float32)

    burst_start = int(0.02 * SR)
    burst_end = int(0.025 * SR)
    audio[burst_start:burst_end] = 0.8 * np.random.randn(burst_end - burst_start).astype(np.float32)

    voice_start = int(0.09 * SR)
    t_voice = np.linspace(0, duration - 0.09, samples - voice_start)
    audio[voice_start:] += 0.5 * np.sin(2 * np.pi * 120 * t_voice).astype(np.float32)

    return audio, SR
