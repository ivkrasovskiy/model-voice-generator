from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

from accent_coach.models import PhonemeInstance, StopFeatures
from accent_coach.pipeline.alignment import filter_stops

_PRE_MS = 20.0
_POST_MS = 200.0    # extended: voicing onset can be 150+ ms after burst when G2P timestamps are off
_BURST_SEARCH_MS = 100.0  # extended: G2P uniform-split timestamps can be 40-80 ms off from actual burst
_HF_LO = 2000
_HF_HI = 7500  # Why: must stay < nyquist (8000 Hz at SR=16kHz) to avoid scipy boundary error
_HOP_MS = 5.0


def _bandpass_energy(audio: np.ndarray, sr: int, lo: int, hi: int) -> np.ndarray:
    nyq = sr / 2.0
    sos = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
    filtered = sosfilt(sos, audio)
    hop = int(_HOP_MS / 1000 * sr)
    n_frames = max(1, len(filtered) // hop)
    energy = np.array(
        [np.sum(filtered[i * hop : (i + 1) * hop] ** 2) for i in range(n_frames)]
    )
    return energy


def _autocorr_peak(frame: np.ndarray) -> float:
    if len(frame) < 2:
        return 0.0
    corr = np.correlate(frame, frame, mode="full")
    corr = corr[len(corr) // 2 :]
    if corr[0] == 0:
        return 0.0
    corr /= corr[0]
    # Look for first peak beyond lag ~2 ms (skip zero-lag peak)
    min_lag = max(1, len(corr) // 20)
    return float(np.max(corr[min_lag:]))


def extract_vot(audio: np.ndarray, sr: int, stop: PhonemeInstance) -> float | None:
    pre = int(_PRE_MS / 1000 * sr)
    post = int(_POST_MS / 1000 * sr)
    start_sample = max(0, int(stop.start_time * sr) - pre)
    end_sample = min(len(audio), int(stop.start_time * sr) + post)
    segment = audio[start_sample:end_sample]

    if len(segment) < int(0.02 * sr):
        return None

    hf_energy = _bandpass_energy(segment, sr, _HF_LO, _HF_HI)
    burst_search_frames = int(_BURST_SEARCH_MS / _HOP_MS)
    search = hf_energy[: min(burst_search_frames, len(hf_energy))]
    median_e = np.median(hf_energy)
    if median_e == 0:
        return None
    burst_candidates = np.where(search > 3 * median_e)[0]
    if len(burst_candidates) == 0:
        return None
    burst_frame = int(burst_candidates[0])
    burst_time = start_sample / sr + burst_frame * _HOP_MS / 1000

    # Voicing onset: first frame where autocorrelation peak > 0.5 (post-burst)
    hop = int(_HOP_MS / 1000 * sr)
    voice_time: float | None = None
    for i in range(burst_frame, len(hf_energy)):
        frame_start = i * hop
        frame_end = min(frame_start + hop * 4, len(segment))
        frame = segment[frame_start:frame_end]
        if _autocorr_peak(frame) > 0.5:
            voice_time = start_sample / sr + i * _HOP_MS / 1000
            break

    if voice_time is None:
        return None

    vot_ms = (voice_time - burst_time) * 1000
    if vot_ms < 0:
        return None
    return vot_ms


def extract_stop_features(
    audio: np.ndarray, sr: int, phonemes: list[PhonemeInstance]
) -> list[StopFeatures]:
    stops = filter_stops(phonemes, stressed_only=True)
    results: list[StopFeatures] = []
    for stop in stops:
        vot = extract_vot(audio, sr, stop)
        if vot is None:
            continue
        # Burst energy: peak HF energy in the 40 ms search window
        pre = int(_PRE_MS / 1000 * sr)
        post = int(_BURST_SEARCH_MS / 1000 * sr)
        seg = audio[max(0, int(stop.start_time * sr) - pre) : int(stop.start_time * sr) + post]
        burst_e = float(np.max(_bandpass_energy(seg, sr, _HF_LO, _HF_HI)))
        results.append(StopFeatures(phoneme=stop, vot_ms=vot, burst_energy=burst_e))
    return results
