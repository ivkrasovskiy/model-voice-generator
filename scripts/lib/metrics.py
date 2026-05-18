"""Evaluation metrics: WER (intelligibility) and DNSMOS (audio quality).

Both take WAV arrays and return scalar scores. Import ONNX/jiwer lazily
so scripts that don't need these don't pay the load cost.
"""

import re
import urllib.request
from pathlib import Path

import numpy as np

from .audio_io import resample

DNSMOS_CACHE_DIR = Path.home() / ".cache" / "dnsmos"
DNSMOS_URL = (
    "https://raw.githubusercontent.com/microsoft/DNS-Challenge"
    "/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
)

_DNSMOS_SESSION = None  # lazy singleton


def _normalize_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[^\w\s']", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def compute_wer(ref_text: str, wav: np.ndarray, sr: int, whisper_model) -> tuple[float, str]:
    """Transcribe wav with Whisper and return (WER, hypothesis_string).

    Lower WER = better intelligibility. ref_text is the ground-truth prompt.
    """
    import jiwer

    wav16 = resample(wav, sr, 16000)
    result = whisper_model.transcribe(wav16, language="en", fp16=False, verbose=False)
    hyp = result["text"]
    wer = float(jiwer.wer(_normalize_text(ref_text), _normalize_text(hyp)))
    return wer, hyp


def load_dnsmos() -> object:
    """Load (and cache) the Microsoft DNSMOS ONNX model."""
    global _DNSMOS_SESSION
    if _DNSMOS_SESSION is not None:
        return _DNSMOS_SESSION
    import onnxruntime as ort

    DNSMOS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = DNSMOS_CACHE_DIR / "sig_bak_ovr.onnx"
    if not model_path.exists():
        print(f"Downloading DNSMOS ONNX → {model_path}")
        urllib.request.urlretrieve(DNSMOS_URL, str(model_path))
    _DNSMOS_SESSION = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    return _DNSMOS_SESSION


def compute_dnsmos(wav: np.ndarray, sr: int, session=None) -> dict:
    """Return {sig, bak, ovr} MOS scores in [1, 5]. All ↑ = better.

    sig = speech quality, bak = background noise, ovr = overall.
    """
    if session is None:
        session = load_dnsmos()
    wav16 = resample(wav, sr, 16000)
    expected_len = session.get_inputs()[0].shape[1]
    if len(wav16) < expected_len:
        wav16 = np.pad(wav16, (0, expected_len - len(wav16)), mode="constant")
    else:
        wav16 = wav16[:expected_len]
    inp = wav16.astype(np.float32).reshape(1, -1)
    out = session.run(None, {session.get_inputs()[0].name: inp})
    sig, bak, ovr = out[0][0]
    return {"sig": float(sig), "bak": float(bak), "ovr": float(ovr)}
