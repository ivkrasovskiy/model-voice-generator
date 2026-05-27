"""WORLD-vocoder-based formant shift for Phase 0.13 Lever B.

Only frames inside target phoneme segments are modified; outside frames
are passed through unchanged.  A 10 ms cosine-fade is applied at segment
boundaries to avoid audible splices.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import parselmouth
import pyworld
import soundfile as sf
from scipy.interpolate import interp1d

# pyworld's default frame period in seconds
_FRAME_PERIOD_MS = 5.0  # pyworld default
_BOUNDARY_FADE_S = 0.010  # 10 ms overlap-add smoothing
# Splice mode: run WORLD only on the segment window to avoid global round-trip
# vocoder loss on unshifted phonemes (Phase 0.13 finding: −0.37 DNSMOS hit).
_CONTEXT_MS = 30.0   # ms context added on each side for pitch-period coherence
_XFADE_MS = 20.0     # ms equal-power crossfade at window boundaries


def _time_to_frame(t_s: float, sr: int) -> int:
    """Convert time in seconds to a pyworld frame index."""
    frame_period_s = _FRAME_PERIOD_MS / 1000.0
    return int(round(t_s / frame_period_s))


def _measure_f1f2(audio: np.ndarray, sr: int, start_s: float, end_s: float) -> tuple[float, float]:
    """Return mean F1, F2 Hz of the segment using parselmouth Burg."""
    start_s = max(0.0, start_s)
    end_s = min(len(audio) / sr, end_s)
    if end_s <= start_s:
        return float("nan"), float("nan")

    start_i = int(start_s * sr)
    end_i = int(end_s * sr)
    seg = audio[start_i:end_i].astype(np.float64)
    if len(seg) < 64:
        return float("nan"), float("nan")

    sound = parselmouth.Sound(seg, sampling_frequency=float(sr))
    formant = sound.to_formant_burg(
        time_step=0.01,
        max_number_of_formants=5,
        maximum_formant=5500.0,
        window_length=0.025,
        pre_emphasis_from=50.0,
    )
    dur = end_s - start_s
    times = np.arange(0.025, dur, 0.010)
    f1s, f2s = [], []
    for t in times:
        v1 = formant.get_value_at_time(1, t)
        v2 = formant.get_value_at_time(2, t)
        if not (np.isnan(v1) or np.isnan(v2)):
            f1s.append(v1)
            f2s.append(v2)
    if not f1s:
        return float("nan"), float("nan")
    return float(np.mean(f1s)), float(np.mean(f2s))


def _warp_sp_frame(
    sp_frame: np.ndarray,
    sr: int,
    f1_old: float, f1_new: float,
    f2_old: float, f2_new: float,
) -> np.ndarray:
    """Apply a piecewise-linear frequency warp to one spectral envelope frame.

    The warp map is anchored at:
      (0, 0), (f1_old, f1_new), (f2_old, f2_new), (sr/2, sr/2)
    We warp the *source* frequency axis — i.e. for each output bin we find
    the fractional input bin via the inverse map and interpolate sp.
    """
    n_bins = len(sp_frame)
    freqs = np.linspace(0, sr / 2, n_bins)

    # Control points on the frequency axis (sorted by old freq)
    ctrl_old = np.array([0.0, f1_old, f2_old, sr / 2.0])
    ctrl_new = np.array([0.0, f1_new, f2_new, sr / 2.0])

    # Ensure strictly increasing (in case f1_old > f2_old or identical)
    if f1_old >= f2_old or f1_new >= f2_new:
        return sp_frame  # degenerate — skip warp

    # Inverse map: for each output freq, find the input freq it came from
    # inv_map(f_out) = f_in  →  we resample sp at those f_in positions
    inv_ctrl_old = ctrl_new  # input to the inverse map
    inv_ctrl_new = ctrl_old  # output of the inverse map
    inv_warp = interp1d(inv_ctrl_old, inv_ctrl_new,
                        kind="linear", bounds_error=False,
                        fill_value=(0.0, sr / 2.0))

    source_freqs = inv_warp(freqs)
    # Interpolate sp at source_freqs
    interp_sp = interp1d(freqs, sp_frame, kind="linear",
                         bounds_error=False, fill_value=(sp_frame[0], sp_frame[-1]))
    return interp_sp(source_freqs).astype(np.float64)


def _splice_one_segment(
    audio: np.ndarray,
    sr: int,
    start_s: float,
    end_s: float,
    f1_meas: float,
    f1_tgt: float,
    f2_meas: float,
    f2_tgt: float,
) -> np.ndarray:
    """Return audio with one segment splice-warped; all other samples original."""
    ctx_s = _CONTEXT_MS / 1000.0
    xfade_s = _XFADE_MS / 1000.0

    win_start = max(0.0, start_s - ctx_s)
    win_end = min(len(audio) / sr, end_s + ctx_s)
    ws_i = int(win_start * sr)
    we_i = int(win_end * sr)
    window = audio[ws_i:we_i].copy()
    win_len = we_i - ws_i
    if win_len < 64:
        return audio

    f0_w, sp_w, ap_w = pyworld.wav2world(window, sr)
    sp_out = sp_w.copy()

    # Warp only core (non-context) frames toward target centroid
    fr_core_start = max(0, _time_to_frame(start_s - win_start, sr))
    fr_core_end = min(sp_w.shape[0], _time_to_frame(end_s - win_start, sr))
    for fi in range(fr_core_start, fr_core_end):
        sp_out[fi] = _warp_sp_frame(sp_w[fi], sr, f1_meas, f1_tgt, f2_meas, f2_tgt)

    synth = pyworld.synthesize(f0_w, sp_out, ap_w, float(sr))

    # RMS-match synthesized window to original to avoid level jumps
    rms_o = np.sqrt(np.mean(window ** 2)) + 1e-8
    rms_s = np.sqrt(np.mean(synth ** 2)) + 1e-8
    synth = synth * (rms_o / rms_s)

    # Equal-power (sqrt) crossfade at window boundaries; core replaced outright
    xfade_n = max(1, min(int(xfade_s * sr), win_len // 4))
    blend = np.ones(win_len)
    ramp = np.sqrt(np.linspace(0.0, 1.0, xfade_n))
    blend[:xfade_n] = ramp
    blend[win_len - xfade_n:] = ramp[::-1]

    slen = min(len(synth), win_len)
    out = audio.copy()
    out[ws_i:ws_i + slen] = (
        blend[:slen] * synth[:slen] + (1.0 - blend[:slen]) * window[:slen]
    )
    return out


def shift_vowels_to_centroid(
    wav_in: Path,
    wav_out: Path,
    segments: list[dict],
    target_phonemes: set[str],
    target_centroid: dict,
    splice: bool = False,
) -> dict:
    """Shift F1/F2 of target phoneme segments toward target_centroid using WORLD.

    Parameters
    ----------
    wav_in : Path  — source WAV (24 kHz mono float32 from IndexTTS-2)
    wav_out : Path — destination WAV (same sample rate)
    segments : list[dict] — rows from extract_formants CSV, each with keys:
               phoneme, start_s, end_s, F1, F2  (F1/F2 may be "" → skipped)
    target_phonemes : set[str] — IPA phonemes to edit, e.g. {"ʌ", "ʊ"}
    target_centroid : dict — {phoneme: {"f1": float, "f2": float}}
    splice : bool — when True, run WORLD only on each segment window (avoids
               global round-trip vocoder loss on unshifted phonemes).

    Returns
    -------
    dict with keys edits_applied (int) and segments_skipped (int).
    """
    wav_in = Path(wav_in)
    wav_out = Path(wav_out)

    audio_f32, sr = sf.read(str(wav_in))
    if audio_f32.ndim > 1:
        audio_f32 = audio_f32.mean(axis=1)
    audio_f32 = audio_f32.astype(np.float32)
    audio = audio_f32.astype(np.float64)

    edits_applied = 0
    segments_skipped = 0

    if splice:
        out_audio64 = audio.copy()
        for seg in segments:
            phoneme = seg.get("phoneme", "")
            if phoneme not in target_phonemes:
                continue
            if phoneme not in target_centroid:
                segments_skipped += 1
                continue
            try:
                start_s = float(seg["start_s"])
                end_s = float(seg["end_s"])
            except (KeyError, ValueError, TypeError):
                segments_skipped += 1
                continue
            f1_meas, f2_meas = _measure_f1f2(audio_f32, sr, start_s, end_s)
            if np.isnan(f1_meas) or np.isnan(f2_meas):
                segments_skipped += 1
                continue
            f1_tgt = target_centroid[phoneme]["f1"]
            f2_tgt = target_centroid[phoneme]["f2"]
            out_audio64 = _splice_one_segment(
                out_audio64, sr, start_s, end_s, f1_meas, f1_tgt, f2_meas, f2_tgt
            )
            edits_applied += 1
        out_audio_f32 = np.clip(out_audio64, -1.0, 1.0).astype(np.float32)
    else:
        f0, sp, ap = pyworld.wav2world(audio, sr)
        sp_out = sp.copy()

        frame_period_s = _FRAME_PERIOD_MS / 1000.0
        fade_frames = max(1, int(_BOUNDARY_FADE_S / frame_period_s))  # 2 frames at 5 ms
        n_frames = sp.shape[0]

        for seg in segments:
            phoneme = seg.get("phoneme", "")
            if phoneme not in target_phonemes:
                continue
            if phoneme not in target_centroid:
                segments_skipped += 1
                continue
            try:
                start_s = float(seg["start_s"])
                end_s = float(seg["end_s"])
            except (KeyError, ValueError, TypeError):
                segments_skipped += 1
                continue
            f1_meas, f2_meas = _measure_f1f2(audio_f32, sr, start_s, end_s)
            if np.isnan(f1_meas) or np.isnan(f2_meas):
                segments_skipped += 1
                continue
            f1_tgt = target_centroid[phoneme]["f1"]
            f2_tgt = target_centroid[phoneme]["f2"]
            fr_start = max(0, _time_to_frame(start_s, sr) - fade_frames)
            fr_end   = min(n_frames, _time_to_frame(end_s, sr) + fade_frames)
            seg_len  = fr_end - fr_start
            if seg_len <= 0:
                segments_skipped += 1
                continue
            weights = np.ones(seg_len)
            ramp = (1 - np.cos(np.pi * np.arange(fade_frames) / fade_frames)) / 2.0
            weights[:fade_frames] = ramp
            tail = weights[max(0, seg_len - fade_frames):]
            tail[:] = ramp[::-1][:len(tail)]
            for i, fi in enumerate(range(fr_start, fr_end)):
                w = weights[i]
                if w <= 0:
                    continue
                warped = _warp_sp_frame(sp[fi], sr, f1_meas, f1_tgt, f2_meas, f2_tgt)
                sp_out[fi] = (1 - w) * sp[fi] + w * warped
            edits_applied += 1

        out_audio = pyworld.synthesize(f0, sp_out, ap, float(sr))
        out_audio_f32 = np.clip(out_audio, -1.0, 1.0).astype(np.float32)

    wav_out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav_out), out_audio_f32, sr, subtype="FLOAT")
    return {"edits_applied": edits_applied, "segments_skipped": segments_skipped}
