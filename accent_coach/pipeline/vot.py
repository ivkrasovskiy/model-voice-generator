"""Voice Onset Time (VOT) extraction — Lisker & Abramson (1964) definition.

VOT = time from the stop BURST release to the onset of voicing (quasi-periodicity).

The previous greedy two-threshold heuristic collapsed to ~0 ms on connected
speech: it anchored the burst on a high-frequency energy median that included the
following vowel, and read the PRECEDING vowel's periodicity as the stop's voicing
onset. This implementation follows validated methods (AutoVOT / Praat):

  1. CLOSURE — find the low-energy minimum near the aligned boundary; its energy
     is the baseline (NOT a vowel-straddling median).
  2. BURST — first sharp energy rise out of the closure (a broadband transient),
     searched only AFTER the closure minimum (so the preceding vowel is excluded).
  3. VOICING ONSET — first F0-constrained voiced frame (parselmouth pitch, floor
     75 / ceiling 400 Hz) STRICTLY AFTER the burst, persisting ≥ 20 ms. This is
     what prevents the preceding vowel from being read as the onset.

See docs/prosody_consonant_upgrade.md.
"""
from __future__ import annotations

import numpy as np
import parselmouth
from scipy.signal import butter, sosfilt

from accent_coach.models import PhonemeInstance, StopFeatures
from accent_coach.pipeline.alignment import filter_aspirating_stops

# Search window around the aligned stop boundary (char-aligned start_time ≈ closure
# onset). Generous enough to contain closure → burst → aspiration → voicing.
_PRE_MS = 40.0
# Long enough to contain closure (≤110 ms) + an aspirated VOT (≤125 ms) + the
# voicing-persistence check (≥20 ms) of the following vowel, plus pitch edge trim.
_POST_MS = 300.0
_HOP_MS = 2.0           # fine grid for burst/voicing localisation
_FRAME_MS = 10.0
# Closure must lie within this much of the boundary (before the following vowel).
_CLOSURE_WINDOW_MS = 110.0
# After the closure minimum, skip this many ms before searching for the burst.
# Only needs to be > 0 frames: thresh = e_closure + frac*(e_max-e_closure) is
# strictly greater than rms[closure_idx] = e_closure, so the closure-minimum
# frame itself can never satisfy the burst threshold — a 1-frame gate is
# sufficient to avoid a degenerate search range.
#
# Was 20 ms (rationale: skip a "noise transient in the preceding vowel, 8-24 ms
# after the closure minimum"). On real connected speech this FIXED 20 ms offset
# was the dominant error: an instrumented trace found burst_idx ==
# closure_idx + gate_frames in 6/6 tokens (the threshold was already exceeded
# at the very first frame the gate allowed), and a closure_ms x vot_ms sweep on
# synthetic stops confirmed the mechanism — for closures shorter than the gate
# (plausible: in-domain VOT itself is 0-22 ms, so closures are short too), the
# fixed offset overshoots PAST the true burst and into/at voicing onset, eating
# `(20 - closure_ms)` ms of the true VOT and collapsing it toward 0. See
# docs/prosody_consonant_upgrade.md.
_MIN_CLOSURE_GATE_MS = 2.0
# Burst = first frame this fraction of the dynamic range above the closure baseline.
_BURST_RISE_FRAC = 0.15
# Voicing onset must persist at least this long to count (rejects transient blips).
_MIN_VOICE_MS = 20.0
# Pitch (periodicity) range for a male/female voice — F0-constrained so aspiration
# noise and formant-band ringing are NOT mistaken for voicing.
_PITCH_FLOOR_HZ = 75.0
_PITCH_CEIL_HZ = 400.0
# Plausibility bounds (ms): outside this, treat as a detection failure (None).
# Aspirated English /p t k/ peak ~125 ms; >150 ms is a spurious detection
# (e.g. an unreleased utterance-final stop whose "voicing" is the next word).
_VOT_MIN_MS = -150.0
_VOT_MAX_MS = 150.0

_HF_LO = 2000
_HF_HI = 7500  # < Nyquist at 16 kHz


def _bandpass_energy(audio: np.ndarray, sr: int, lo: int, hi: int) -> np.ndarray:
    nyq = sr / 2.0
    sos = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
    filtered = sosfilt(sos, audio)
    hop = int(_HOP_MS / 1000 * sr)
    n_frames = max(1, len(filtered) // hop)
    return np.array([np.sum(filtered[i * hop : (i + 1) * hop] ** 2) for i in range(n_frames)])


def _frame_rms(x: np.ndarray, hop: int, win: int) -> np.ndarray:
    n = max(1, (len(x) - win) // hop + 1)
    return np.array([np.sqrt(np.mean(x[i * hop : i * hop + win] ** 2)) for i in range(n)])


def extract_vot(audio: np.ndarray, sr: int, stop: PhonemeInstance) -> float | None:
    """VOT in ms (burst → voicing onset), or None when not measurable.

    Returns None rather than a fabricated 0 when no burst or no post-burst voicing
    can be found, so the caller drops the token instead of scoring garbage.
    """
    t0 = stop.start_time
    win_start = max(0, int((t0 - _PRE_MS / 1000) * sr))
    win_end = min(len(audio), int((t0 + _POST_MS / 1000) * sr))
    seg = audio[win_start:win_end].astype(np.float64)
    if len(seg) < int(0.04 * sr):
        return None

    hop = max(1, int(_HOP_MS / 1000 * sr))
    win = max(hop, int(_FRAME_MS / 1000 * sr))
    rms = _frame_rms(seg, hop, win)
    if len(rms) < 5:
        return None

    # 1. Closure: min-energy frame within the closure window (before the vowel).
    closure_n = min(len(rms), int(_CLOSURE_WINDOW_MS / _HOP_MS))
    closure_idx = int(np.argmin(rms[:closure_n]))
    e_closure = float(rms[closure_idx])
    e_max = float(rms.max())
    if e_max <= e_closure:
        return None

    # 2. Burst: first frame AFTER a minimum closure gate whose energy rises out of the
    #    closure baseline.  Starting the search at closure_idx+gate_frames skips the
    #    8–24 ms artefact window where a noise transient in the preceding vowel context
    #    fools the detector into firing before the real burst.
    thresh = e_closure + _BURST_RISE_FRAC * (e_max - e_closure)
    gate_frames = max(1, int(_MIN_CLOSURE_GATE_MS / _HOP_MS))
    burst_idx = next(
        (i for i in range(closure_idx + gate_frames, len(rms)) if rms[i] > thresh), None
    )
    if burst_idx is None:
        return None
    t_burst = (win_start + burst_idx * hop) / sr

    # 3. Voicing onset: first F0-constrained voiced run, strictly AFTER the burst.
    snd = parselmouth.Sound(seg, sampling_frequency=sr)
    try:
        pitch = snd.to_pitch_ac(
            time_step=_HOP_MS / 1000, pitch_floor=_PITCH_FLOOR_HZ, pitch_ceiling=_PITCH_CEIL_HZ
        )
    except Exception:  # noqa: BLE001 — pitch analysis failed → not measurable
        return None
    freqs = pitch.selected_array["frequency"]  # 0.0 where unvoiced
    times = pitch.xs()
    burst_t_local = t_burst - win_start / sr
    min_voiced = max(1, int(_MIN_VOICE_MS / _HOP_MS))

    voice_t_local: float | None = None
    # Praat's AC pitch window spans ≈ 1/pitch_floor seconds; the first frame
    # whose centre is detected as voiced sits ~1/pitch_floor after the actual
    # onset.  Subtracting this latency estimate aligns the measurement with the
    # physical voice onset (Lisker & Abramson definition).
    _onset_correction_s = 1.0 / _PITCH_FLOOR_HZ
    i = 0
    n = len(freqs)
    while i < n:
        if freqs[i] > 0 and times[i] > burst_t_local:
            j = i
            while j < n and freqs[j] > 0:
                j += 1
            if (j - i) >= min_voiced:
                voice_t_local = max(burst_t_local, float(times[i]) - _onset_correction_s)
                break
            i = j
        else:
            i += 1
    if voice_t_local is None:
        return None

    t_voice = win_start / sr + voice_t_local
    vot_ms = (t_voice - t_burst) * 1000.0
    if vot_ms < _VOT_MIN_MS or vot_ms > _VOT_MAX_MS:
        return None
    return vot_ms


def extract_stop_features(
    audio: np.ndarray, sr: int, phonemes: list[PhonemeInstance]
) -> list[StopFeatures]:
    # Only aspirating-context stops carry the English-vs-L2 VOT contrast.
    stops = filter_aspirating_stops(phonemes)
    results: list[StopFeatures] = []
    for stop in stops:
        vot = extract_vot(audio, sr, stop)
        if vot is None:
            continue
        pre = int(_PRE_MS / 1000 * sr)
        post = int(_POST_MS / 1000 * sr)
        seg = audio[max(0, int(stop.start_time * sr) - pre) : int(stop.start_time * sr) + post]
        burst_e = float(np.max(_bandpass_energy(seg, sr, _HF_LO, _HF_HI))) if len(seg) else 0.0
        results.append(StopFeatures(phoneme=stop, vot_ms=vot, burst_energy=burst_e))
    return results
