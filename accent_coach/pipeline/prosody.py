from __future__ import annotations

import numpy as np


def extract_syllable_durations_acoustic(audio: np.ndarray, sr: int) -> list[float]:
    """Bandpass-energy vowel-nucleus detector → syllable duration list.

    Uses a 300–3000 Hz bandpass to isolate vowel energy, then finds local maxima
    (syllable nuclei) separated by ≥ 150 ms. Returns inter-nucleus intervals.

    More accurate than G2P-uniform distribution for nPVI: captures actual acoustic
    duration contrast between stressed and unstressed syllables.
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import butter, find_peaks, sosfilt

    audio_f = audio.astype(np.float64)

    # Trim leading/trailing silence so rms.mean() isn't diluted by silent frames
    import librosa
    audio_f, _ = librosa.effects.trim(audio_f, top_db=25)
    if len(audio_f) < sr * 0.2:
        return [len(audio_f) / sr]

    nyq = sr / 2
    sos = butter(4, [300 / nyq, min(0.99, 3000 / nyq)], btype="bandpass", output="sos")
    filtered = sosfilt(sos, audio_f)

    hop = max(1, int(0.010 * sr))  # 10 ms hop
    frame_len = max(hop, int(0.025 * sr))  # 25 ms frame
    frames = librosa.util.frame(filtered, frame_length=frame_len, hop_length=hop)
    rms = np.sqrt(np.mean(frames**2, axis=0))
    rms = gaussian_filter1d(rms.astype(float), sigma=8)  # ~80 ms smoothing

    min_dist = max(1, int(0.150 * sr / hop))
    peaks, _ = find_peaks(rms, distance=min_dist, height=rms.mean() * 0.25)

    # Need ≥ 3 peaks to get ≥ 2 intervals (nPVI requires ≥ 2).
    # The 2-peak case would return 1 interval → compute_npvi returns 0.0 → false
    # "your rhythm is too even" diagnostic. Use the rate-based estimate instead.
    if len(peaks) < 3:
        est = max(2, round(len(audio_f) / sr * 5))
        return [float(len(audio_f) / sr / est)] * est

    times = peaks * hop / sr
    durs = list(np.diff(times).astype(float))
    # Drop noise-level intervals (< 10 ms), which would inflate nPVI toward infinity
    durs = [d for d in durs if d >= 0.010]
    return durs if len(durs) >= 2 else [float(len(audio_f) / sr)]


def _syllable_durs_within_word(
    word_audio: np.ndarray,
    sr: int,
    n_syllables: int,
    stressed_indices: list[int],
) -> list[float]:
    """Acoustic nucleus detection within one word slice; falls back to stress-weighted uniform.

    Only called for words with ≥ 2 syllables. Uses tighter parameters (50 ms min distance,
    lower threshold) than the global detector to catch within-word contrasts.
    """
    import librosa
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import butter, find_peaks, sosfilt

    word_dur = len(word_audio) / sr

    if len(word_audio) >= int(0.040 * sr):
        nyq = sr / 2
        sos = butter(4, [300 / nyq, min(0.99, 3000 / nyq)], btype="bandpass", output="sos")
        filtered = sosfilt(sos, word_audio.astype(np.float64))
        hop = max(1, int(0.005 * sr))
        frame_len = max(hop, int(0.020 * sr))
        frames = librosa.util.frame(filtered, frame_length=frame_len, hop_length=hop)
        rms = np.sqrt(np.mean(frames**2, axis=0))
        rms = gaussian_filter1d(rms.astype(float), sigma=4)
        min_dist = max(1, int(0.050 * sr / hop))
        peaks, _ = find_peaks(rms, distance=min_dist, height=rms.mean() * 0.15)
        if len(peaks) >= 2:
            durs = [d for d in np.diff(peaks * hop / sr).tolist() if d >= 0.015]
            if durs:
                return durs

    # Stress-weighted fallback: stressed syllable gets 2× the duration of an unstressed one.
    stressed_set = set(stressed_indices)
    weights = [2.0 if i in stressed_set else 1.0 for i in range(n_syllables)]
    total = sum(weights)
    return [w * word_dur / total for w in weights]


def extract_syllable_durations_from_words(
    phonemes: list,
    audio: np.ndarray,
    sr: int,
) -> list[float]:
    """Hybrid syllable duration extraction: word-level timestamps + per-word acoustic detection.

    Combines WhisperX word boundaries (reliable for function words) with per-word acoustic
    nucleus detection (captures within-word stress contrast). Falls back to
    extract_syllable_durations_acoustic if phonemes are empty.

    Fixes two failure modes of acoustic-only detection:
    - Short function words ('the', 'a', 'in') now get their actual durations from alignment
    - Within-word stress contrast uses per-word peak detection, not uniform distribution
    """
    from itertools import groupby

    from accent_coach.pipeline.alignment import IPA_VOWELS

    if not phonemes:
        return extract_syllable_durations_acoustic(audio, sr)

    all_durs: list[float] = []

    for word_key, group in groupby(phonemes, key=lambda p: p.word.lower().strip(".,!?;:")):
        if not word_key:
            continue
        phs = list(group)
        word_start = phs[0].start_time
        word_end = phs[-1].end_time
        word_dur = word_end - word_start

        if word_dur < 0.015:
            continue

        vowel_ph_indices = [i for i, p in enumerate(phs) if p.phoneme in IPA_VOWELS]
        n_syl = len(vowel_ph_indices)
        if n_syl == 0:
            all_durs.append(word_dur)
            continue

        if n_syl == 1:
            all_durs.append(word_dur)
            continue

        # Multi-syllable: determine stressed positions and run per-word detector
        stressed = [syl_pos for syl_pos, vi in enumerate(vowel_ph_indices) if phs[vi].is_stressed]

        start_sample = int(word_start * sr)
        end_sample = int(min(word_end * sr, len(audio)))
        if start_sample >= end_sample:
            all_durs.append(word_dur)
            continue

        durs = _syllable_durs_within_word(audio[start_sample:end_sample], sr, n_syl, stressed)
        all_durs.extend(durs)

    if len(all_durs) < 2:
        return extract_syllable_durations_acoustic(audio, sr)
    return all_durs


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
