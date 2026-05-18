"""ECAPA-TDNN speaker identity helpers — embed, cosine, robust centroid."""

from pathlib import Path

import numpy as np
import torch

from .audio_io import read_wav_at

_ECAPA = None  # lazy singleton


def load_ecapa(device: str = "cpu"):
    """Load (and cache) the SpeechBrain ECAPA-TDNN VoxCeleb model."""
    global _ECAPA
    if _ECAPA is not None:
        return _ECAPA
    from speechbrain.inference.speaker import EncoderClassifier
    savedir = Path.home() / ".cache" / "speechbrain-ecapa"
    savedir.mkdir(parents=True, exist_ok=True)
    _ECAPA = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(savedir),
        run_opts={"device": device},
    )
    return _ECAPA


def embed_wav(wav: np.ndarray, sr: int, ecapa=None) -> np.ndarray:
    """Embed a 1-D float32 wav. Resamples to 16 kHz internally."""
    if ecapa is None:
        ecapa = load_ecapa()
    from .audio_io import resample
    wav16 = resample(wav, sr, 16000)
    t = torch.from_numpy(wav16).float().unsqueeze(0)
    with torch.no_grad():
        emb = ecapa.encode_batch(t).squeeze().cpu().numpy()
    return emb


def embed_file(path: str | Path, ecapa=None) -> np.ndarray | None:
    """Embed a WAV file. Returns None on read failure."""
    try:
        wav16 = read_wav_at(path, 16000)
        return embed_wav(wav16, 16000, ecapa)
    except Exception:
        return None


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for ECAPA embeddings (or any 1-D vectors)."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def robust_centroid(embs: np.ndarray, n_iter: int = 3, drop_frac: float = 0.1) -> np.ndarray:
    """Iteratively compute centroid, dropping the lowest-sim drop_frac fraction each iteration.

    Useful for getting a clean speaker centroid when the dataset has noise/outliers.
    """
    keep = np.arange(len(embs))
    for _ in range(n_iter):
        centroid = embs[keep].mean(axis=0)
        sims = np.array([cosine(centroid, embs[i]) for i in keep])
        thresh = np.quantile(sims, drop_frac)
        keep = keep[sims >= thresh]
    return embs[keep].mean(axis=0)
