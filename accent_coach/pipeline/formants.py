from __future__ import annotations

import numpy as np
import parselmouth

from accent_coach.models import PhonemeInstance, VowelFeatures
from accent_coach.pipeline.alignment import IPA_VOWELS

_MIN_VOWEL_MS = 30.0


def _estimate_max_formant(audio: np.ndarray, sr: int) -> float:
    import librosa

    # Use a 5-second window from the middle of the signal for f0 estimation
    mid = len(audio) // 2
    window = audio[max(0, mid - sr * 5 // 2) : mid + sr * 5 // 2]
    f0 = librosa.yin(window.astype(np.float64), fmin=70, fmax=400, sr=sr)
    f0_voiced = f0[f0 > 70]
    if len(f0_voiced) == 0:
        return 5500.0
    mean_f0 = float(np.mean(f0_voiced))
    # Why max_formant=5000 for male (f0<165), 5500 for female: Praat docs recommend
    # 5000 Hz ceiling for adult male to avoid harmonics being mistaken for F3/F4.
    return 5000.0 if mean_f0 < 165 else 5500.0


def extract_vowel_features(
    audio: np.ndarray, sr: int, phonemes: list[PhonemeInstance]
) -> list[VowelFeatures]:
    import librosa

    sound = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=sr)
    max_formant = _estimate_max_formant(audio, sr)

    formant = sound.to_formant_burg(
        time_step=0.01,
        max_number_of_formants=5,
        maximum_formant=max_formant,
        window_length=0.025,
        pre_emphasis_from=50.0,
    )

    f0_full = librosa.yin(audio.astype(np.float64), fmin=70, fmax=400, sr=sr)
    hop = 512
    frame_times = librosa.frames_to_time(np.arange(len(f0_full)), sr=sr, hop_length=hop)

    results: list[VowelFeatures] = []
    for p in phonemes:
        if p.phoneme not in IPA_VOWELS:
            continue
        dur_ms = (p.end_time - p.start_time) * 1000
        if dur_ms < _MIN_VOWEL_MS:
            continue
        mid = (p.start_time + p.end_time) / 2
        f1 = formant.get_value_at_time(1, mid)
        f2 = formant.get_value_at_time(2, mid)
        f3 = formant.get_value_at_time(3, mid)
        if np.isnan(f1) or np.isnan(f2):
            continue

        # Pitch mean over vowel region
        mask = (frame_times >= p.start_time) & (frame_times <= p.end_time)
        f0_region = f0_full[mask]
        voiced = f0_region[f0_region > 70]
        pitch_mean = float(np.mean(voiced)) if len(voiced) else 0.0

        results.append(
            VowelFeatures(
                phoneme=p,
                f1=float(f1),
                f2=float(f2),
                f3=float(f3) if not np.isnan(f3) else None,
                duration_ms=dur_ms,
                pitch_mean=pitch_mean,
            )
        )
    return results
