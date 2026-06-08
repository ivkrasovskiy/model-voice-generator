"""Canonical audio normalization for the accent coach.

EVERY audio source — corpus clips, the BC target, and any future user
recording — must pass through `normalize_audio` / `load_standard_audio` before
analysis. Standardizing the sample rate (and capping the bandwidth to a band all
sources actually contain) prevents the bandwidth confound that once inverted the
fricative score: clips recorded at 44.1 kHz studio measured a higher spectral
centre of gravity than 16 kHz-sourced corpora, so CoG measured *recording
bandwidth* rather than articulation. See docs/prosody_consonant_upgrade.md.

Design choice: standardize DOWN to 16 kHz. The native corpora are 16 kHz at
source (8 kHz Nyquist); high-frequency frication they never captured cannot be
recovered, so the only fair common ground is the lowest common bandwidth. A
common anti-alias / brick-wall low-pass at `STANDARD_LP_HZ` removes the top
sliver where wider-band sources would otherwise have an unfair CoG advantage.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

# Single source of truth for the analysis sample rate.
STANDARD_SR: int = 16_000
# Common bandwidth ceiling (Hz). Below the 8 kHz Nyquist so the transition band
# of the anti-alias filter is shared identically by every source.
STANDARD_LP_HZ: float = 7_600.0


def normalize_audio(audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """Return (audio, STANDARD_SR): mono, resampled to STANDARD_SR, band-limited.

    Steps: downmix to mono → resample to STANDARD_SR (librosa, anti-aliased) →
    common low-pass at STANDARD_LP_HZ so all sources share one bandwidth.
    Idempotent for audio already at STANDARD_SR within the ceiling.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if sr != STANDARD_SR:
        import librosa

        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=STANDARD_SR)
        sr = STANDARD_SR

    # Common bandwidth cap so a wider-band source cannot win on bandwidth alone.
    nyq = STANDARD_SR / 2.0
    if nyq > STANDARD_LP_HZ:
        sos = butter(8, STANDARD_LP_HZ / nyq, btype="low", output="sos")
        audio = sosfiltfilt(sos, audio.astype(np.float64)).astype(np.float32)

    return audio, STANDARD_SR


def load_standard_audio(path: str | Path) -> tuple[np.ndarray, int] | None:
    """Load a WAV and return (audio, STANDARD_SR) normalized; None on read failure.

    The single entry point for getting analysis-ready audio from disk.
    """
    import soundfile as sf

    try:
        audio, sr = sf.read(str(path), always_2d=False)
    except Exception:  # noqa: BLE001 — unreadable/corrupt clip: caller drops it
        return None
    return normalize_audio(audio, sr)
