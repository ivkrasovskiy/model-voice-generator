"""Whisper transcription helpers — single model load + per-clip transcribe."""

from pathlib import Path

import numpy as np

from .audio_io import read_wav_at

_WHISPER = None  # lazy singleton


def load_whisper(model_size: str = "large-v3", device: str = "cpu"):
    """Load (and cache) the Whisper model. Default large-v3 on CPU (~3 GB RAM)."""
    global _WHISPER
    if _WHISPER is not None:
        return _WHISPER
    import whisper
    _WHISPER = whisper.load_model(model_size, device=device)
    return _WHISPER


def transcribe_wav(wav: np.ndarray, sr: int, model=None, language: str = "en") -> str:
    """Transcribe a 1-D float32 wav. Resamples to 16 kHz internally."""
    if model is None:
        model = load_whisper()
    from .audio_io import resample
    wav16 = resample(wav, sr, 16000)
    result = model.transcribe(wav16, language=language, fp16=False, verbose=False)
    return result["text"].strip()


def transcribe_file(path: str | Path, model=None, language: str = "en") -> str:
    """Convenience: read a WAV file and transcribe."""
    wav16 = read_wav_at(path, 16000)
    return transcribe_wav(wav16, 16000, model, language)
