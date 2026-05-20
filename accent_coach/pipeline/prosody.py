from __future__ import annotations

import numpy as np


def extract_pitch_contour(audio: np.ndarray, sr: int, n_points: int = 50) -> list[float]:
    import librosa

    f0 = librosa.yin(audio.astype(np.float64), fmin=50, fmax=400, sr=sr)
    voiced = np.where(f0 > 50, f0, np.nan)
    # Interpolate over unvoiced gaps then downsample to n_points
    indices = np.arange(len(voiced))
    valid = ~np.isnan(voiced)
    if valid.sum() < 2:
        return [0.0] * n_points
    interp = np.interp(indices, indices[valid], voiced[valid])
    xs = np.linspace(0, len(interp) - 1, n_points)
    contour = np.interp(xs, indices, interp)
    return contour.tolist()


def extract_syllable_durations(
    phonemes: list, audio_duration: float
) -> list[float]:
    """Collapse phoneme sequence into syllable durations using vowel nuclei."""
    from accent_coach.pipeline.alignment import IPA_VOWELS

    vowel_spans: list[tuple[float, float]] = []
    for p in phonemes:
        if p.phoneme in IPA_VOWELS:
            vowel_spans.append((p.start_time, p.end_time))

    if not vowel_spans:
        return [audio_duration]

    # Syllable boundary = midpoint between consecutive vowel nuclei ends and starts
    durations: list[float] = []
    prev_end = 0.0
    for start, end in vowel_spans:
        syl_end = (end + vowel_spans[vowel_spans.index((start, end)) + 1][0]) / 2 if (
            vowel_spans.index((start, end)) + 1 < len(vowel_spans)
        ) else audio_duration
        durations.append(syl_end - prev_end)
        prev_end = syl_end

    return durations


def compute_npvi(syllable_durations: list[float]) -> float:
    """Normalised Pairwise Variability Index.

    Grabe & Low 2002 formula: 100 * mean(|d_k - d_{k+1}| / ((d_k + d_{k+1}) / 2))
    """
    d = np.array(syllable_durations, dtype=float)
    if len(d) < 2:
        return 0.0
    pairs = np.abs(d[:-1] - d[1:]) / ((d[:-1] + d[1:]) / 2 + 1e-9)
    return float(100.0 * np.mean(pairs))


def extract_stress_pattern(
    phonemes: list, audio: np.ndarray, sr: int
) -> list[bool]:
    """Per-syllable stress: True if duration + energy + pitch above syllable mean."""
    import librosa

    from accent_coach.pipeline.alignment import IPA_VOWELS

    hop = 512
    rms = librosa.feature.rms(y=audio.astype(np.float64), hop_length=hop)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    f0 = librosa.yin(audio.astype(np.float64), fmin=50, fmax=400, sr=sr)
    f0_times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop)

    vowels = [(p.start_time, p.end_time) for p in phonemes if p.phoneme in IPA_VOWELS]
    if not vowels:
        return []

    def _mean_in(arr: np.ndarray, times: np.ndarray, t0: float, t1: float) -> float:
        mask = (times >= t0) & (times <= t1)
        vals = arr[mask]
        return float(np.mean(vals)) if len(vals) else 0.0

    energies = [_mean_in(rms, rms_times, s, e) for s, e in vowels]
    pitches = [_mean_in(f0, f0_times, s, e) for s, e in vowels]
    durs = [(e - s) for s, e in vowels]

    e_mean = float(np.mean(energies)) + 1e-9
    p_mean = float(np.mean([p for p in pitches if p > 70] or [0.0])) + 1e-9
    d_mean = float(np.mean(durs)) + 1e-9

    stressed = [
        (en / e_mean + du / d_mean + (pi / p_mean if pi > 70 else 0.0)) / 3 > 0.85
        for en, du, pi in zip(energies, durs, pitches, strict=True)
    ]
    return stressed
