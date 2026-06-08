"""Canonical audio normalization — bandwidth-equalization contract.

The point: the same articulation recorded at different sample rates / bandwidths
must produce the SAME measurement after normalization, so the scorer can never
again reward wider recording bandwidth (the fricative inversion root cause).
"""
from __future__ import annotations

import numpy as np

from accent_coach.pipeline.audio_io import STANDARD_LP_HZ, STANDARD_SR, normalize_audio


def _tone(freq: float, sr: int, dur: float = 0.3, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(sr * dur)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _cog(audio: np.ndarray, sr: int) -> float:
    spec = np.abs(np.fft.rfft(audio.astype(np.float64))) ** 2
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / sr)
    return float(np.dot(freqs, spec) / (spec.sum() + 1e-12))


def test_normalize_returns_standard_sr_and_mono():
    stereo = np.stack([_tone(3000, 44100), _tone(3000, 44100)], axis=1)
    out, sr = normalize_audio(stereo, 44100)
    assert sr == STANDARD_SR
    assert out.ndim == 1


def test_energy_above_ceiling_is_removed_not_aliased():
    """A 12 kHz tone (above the 8 kHz Nyquist) must be filtered out, not aliased
    down into the analysis band."""
    out, sr = normalize_audio(_tone(12000, 44100), 44100)
    assert float(np.sqrt(np.mean(out**2))) < 0.05, (
        "Above-Nyquist content must be removed by anti-aliased resampling, "
        "not folded into the band."
    )


def test_within_ceiling_tone_is_preserved():
    out, sr = normalize_audio(_tone(5000, 16000), 16000)
    assert float(np.sqrt(np.mean(out**2))) > 0.2, "In-band content must survive normalization."


def test_bandwidth_equalization_makes_cog_invariant():
    """Two signals identical below the ceiling but differing in HF bandwidth must
    measure (almost) the same CoG after normalization.

    base  = 3 kHz + 6 kHz (both in band)
    wide  = base + 11 kHz  (extra HF a wider-band recorder would capture)
    Without equalization the 11 kHz term pulls `wide`'s CoG far above `base`.
    After normalize() both are capped at STANDARD_LP_HZ → 11 kHz gone → CoGs match.
    """
    sr_hi = 32000
    base = _tone(3000, sr_hi) + _tone(6000, sr_hi)
    wide = base + _tone(11000, sr_hi)

    cog_base_raw = _cog(base, sr_hi)
    cog_wide_raw = _cog(wide, sr_hi)
    assert cog_wide_raw - cog_base_raw > 1000, "sanity: extra HF should raise raw CoG a lot"

    nb, _ = normalize_audio(base, sr_hi)
    nw, _ = normalize_audio(wide, sr_hi)
    cog_nb, cog_nw = _cog(nb, STANDARD_SR), _cog(nw, STANDARD_SR)
    assert abs(cog_nw - cog_nb) < 250, (
        f"After equalization CoGs must match: base={cog_nb:.0f} wide={cog_nw:.0f}. "
        "Extra recording bandwidth must NOT change the measured CoG."
    )
    assert STANDARD_LP_HZ < STANDARD_SR / 2  # ceiling is below Nyquist
