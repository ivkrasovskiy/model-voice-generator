"""WAV I/O helpers — read mono float32, resample, sample-rate-agnostic safety."""

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


def read_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """Load a WAV as float32 mono. Returns (wav, sample_rate)."""
    wav, sr = sf.read(str(path))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav.astype(np.float32), sr


def resample(wav: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample 1-D float32 wav. No-op if rates already match."""
    if sr_in == sr_out:
        return wav
    t = torch.from_numpy(wav).float().unsqueeze(0)
    return torchaudio.functional.resample(t, sr_in, sr_out).squeeze(0).numpy()


def read_wav_at(path: str | Path, target_sr: int) -> np.ndarray:
    """Convenience: read + mono + resample to target_sr."""
    wav, sr = read_wav_mono(path)
    return resample(wav, sr, target_sr)
