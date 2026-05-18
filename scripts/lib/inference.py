"""F5-TTS generation helpers: text splitting, retry on NaN/silence, segment stitching.

Why this exists:
  F5-TTS on MPS produces NaN audio on ~30-50% of generations longer than ~58 chars.
  This module wraps `tts.infer(...)` with three protections:

    1. split_to_short_segments() pre-splits the prompt so each generation is single-batch
       (≤ 50 chars by default).
    2. gen_segment() retries up to N times, varying the seed per attempt.
    3. gen_phrase() concatenates segments with 150 ms inter-segment silence.

All inference calls go through gen_phrase() so behaviour is consistent across scripts.
"""

import gc
import re

import numpy as np
import torch


def split_to_short_segments(text: str, max_chars: int = 50) -> list[str]:
    """Split a phrase into ≤max_chars segments, preferring sentence/clause boundaries.

    Splits hierarchically: sentence → clause (,;:) → words. Keeps every segment
    under max_chars, which is the F5-TTS single-batch ceiling for typical refs.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    segments: list[str] = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(s) <= max_chars:
            segments.append(s)
            continue
        parts = re.split(r"(?<=[,;:])\s+", s)
        buf = ""
        for p in parts:
            p = p.strip()
            if not p:
                continue
            joined = (buf + " " + p).strip() if buf else p
            if len(joined) <= max_chars:
                buf = joined
            else:
                if buf:
                    segments.append(buf)
                if len(p) <= max_chars:
                    buf = p
                else:
                    words = p.split()
                    buf2 = ""
                    for w in words:
                        joined2 = (buf2 + " " + w).strip() if buf2 else w
                        if len(joined2) <= max_chars:
                            buf2 = joined2
                        else:
                            if buf2:
                                segments.append(buf2)
                            buf2 = w
                    buf = buf2
        if buf:
            segments.append(buf)
    return segments


def mps_reset():
    """Flush MPS memory pool to prevent fragmentation-induced NaN across segments."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def gen_segment(tts, ref_audio: str, ref_text: str, segment: str, *,
                cfg_strength: float = 2.0, nfe_step: int = 32,
                speed: float = 1.0, seed: int = 42, max_retries: int = 5,
                selective_cfg_threshold: float | None = None,
                verbose: bool = False) -> tuple[np.ndarray | None, int | None]:
    """Generate a single ≤50-char segment with retry. Each retry varies seed.

    Returns (wav_np, sr) on success, (None, None) on persistent failure.
    """
    for attempt in range(1, max_retries + 1):
        wav, sr, _ = tts.infer(
            ref_file=ref_audio, ref_text=ref_text, gen_text=segment,
            cfg_strength=cfg_strength, nfe_step=nfe_step,
            speed=speed,
            seed=seed + (attempt - 1),
            selective_cfg_threshold=selective_cfg_threshold,
        )
        w = wav.squeeze() if hasattr(wav, "squeeze") else wav
        w = np.asarray(w, dtype=np.float32)
        finite = np.isfinite(w).all()
        peak = float(np.abs(w[np.isfinite(w)]).max()) if finite else 0.0
        mps_reset()
        if finite and peak > 0.01:
            return w, sr
        if verbose:
            print(f"      attempt {attempt}: {'NaN/Inf' if not finite else f'SILENT peak={peak:.4f}'}")
    return None, None


def gen_phrase(tts, ref_audio: str, ref_text: str, text: str, *,
               cfg_strength: float = 2.0, nfe_step: int = 32,
               speed: float = 1.0, seed: int = 42, max_retries: int = 5,
               selective_cfg_threshold: float | None = None,
               max_chars: int = 50, inter_segment_silence_sec: float = 0.15,
               verbose: bool = False) -> tuple[np.ndarray | None, int | None]:
    """Generate audio for arbitrary-length text by pre-splitting + per-segment retry.

    Each segment is generated separately, then concatenated with 150 ms silence
    between them. Returns (wav, sr) or (None, None) if any segment fails after retries.
    """
    segments = split_to_short_segments(text, max_chars=max_chars)
    if len(segments) == 1:
        return gen_segment(tts, ref_audio, ref_text, segments[0],
                           cfg_strength=cfg_strength, nfe_step=nfe_step,
                           speed=speed, seed=seed, max_retries=max_retries,
                           selective_cfg_threshold=selective_cfg_threshold,
                           verbose=verbose)
    if verbose:
        print(f"    split into {len(segments)} segments (≤{max_chars} chars each)")
    pieces = []
    sr_out = None
    for i, seg in enumerate(segments, 1):
        wav_np, sr = gen_segment(tts, ref_audio, ref_text, seg,
                                 cfg_strength=cfg_strength, nfe_step=nfe_step,
                                 speed=speed, seed=seed, max_retries=max_retries,
                                 selective_cfg_threshold=selective_cfg_threshold,
                                 verbose=verbose)
        if wav_np is None:
            if verbose:
                print(f"    segment {i}/{len(segments)} FAILED — abort")
            return None, None
        if verbose:
            print(f"    segment {i}/{len(segments)} ok  ({len(wav_np)/sr:.1f}s, peak={float(np.abs(wav_np).max()):.3f})")
        pieces.append(wav_np)
        if sr_out is None:
            sr_out = sr
        pieces.append(np.zeros(int(inter_segment_silence_sec * sr), dtype=np.float32))
    return np.concatenate(pieces[:-1]), sr_out  # drop trailing silence
