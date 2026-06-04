"""WhisperX transcription + ECAPA sliding-window speaker diarization.

Extracted from accent_coach_build_real_bc.py to hold that script under 400 lines.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf


def transcribe_diarize(wav_path: Path, hf_token: str) -> list[dict]:
    """Transcribe + assign speakers. Returns word-dicts with 'speaker' field."""
    import functools

    import torch
    import whisperx

    _orig_load = torch.load
    @functools.wraps(_orig_load)
    def _patched_load(*args, **kwargs):
        if kwargs.get("weights_only") is None:
            kwargs["weights_only"] = False
        return _orig_load(*args, **kwargs)
    torch.load = _patched_load

    device = "cpu"
    words_cache = wav_path.parent / f"{wav_path.stem}_words.json"

    if words_cache.exists():
        print(f"  [whisperx] loading cached words from {words_cache.name} …", flush=True)
        words_flat = json.load(words_cache.open())
    else:
        audio = whisperx.load_audio(str(wav_path))
        print("  [whisperx] loading model medium (int8) …", flush=True)
        model = whisperx.load_model("medium", device=device, language="en", compute_type="int8")
        result = model.transcribe(audio, batch_size=8, verbose=True)

        print("  [whisperx] aligning …", flush=True)
        model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
        result = whisperx.align(
            result["segments"], model_a, metadata, audio, device=device,
            return_char_alignments=False,
        )

        words_flat = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                if "start" not in w or "end" not in w:
                    continue
                words_flat.append({
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                    "word": w.get("word", "").strip(),
                    "speaker": "UNKNOWN",
                })
        words_cache.write_text(json.dumps(words_flat))
        print(f"  [whisperx] cached {len(words_flat)} words → {words_cache.name}", flush=True)

    print("  [ecapa] sliding-window speaker ID …", flush=True)
    wav_np, wav_sr = sf.read(str(wav_path))
    if wav_np.ndim > 1:
        wav_np = wav_np.mean(axis=1)
    wav_np = wav_np.astype(np.float32)
    return ecapa_assign_speakers(words_flat, wav_np, wav_sr)


def ecapa_assign_speakers(
    words: list[dict],
    wav: np.ndarray,
    sr: int,
    ref_wav_path: Path | None = None,
    window_s: float = 3.0,
    stride_s: float = 1.5,
    threshold: float = 0.45,
) -> list[dict]:
    """Label each word dict with 'speaker' = 'BC' or 'OTHER' via ECAPA windows."""
    from .identity import embed_wav, load_ecapa

    ecapa = load_ecapa()
    if ref_wav_path is not None:
        ref_wav, ref_sr = sf.read(str(ref_wav_path))
        if ref_wav.ndim > 1:
            ref_wav = ref_wav.mean(axis=1)
        ref_emb = embed_wav(ref_wav.astype(np.float32), ref_sr, ecapa)
    else:
        raise ValueError("ref_wav_path is required")

    total_s = len(wav) / sr
    window_sims: dict[float, float] = {}
    t = 0.0
    while t + window_s <= total_s:
        chunk = wav[int(t * sr):int((t + window_s) * sr)]
        emb = embed_wav(chunk, sr, ecapa)
        sim = float(np.dot(emb.flatten(), ref_emb.flatten()) /
                    (np.linalg.norm(emb) * np.linalg.norm(ref_emb) + 1e-9))
        window_sims[t] = sim
        t += stride_s

    window_starts = sorted(window_sims)

    def _sim_at(word_mid: float) -> float:
        if not window_starts:
            return 0.0
        closest = min(window_starts, key=lambda s: abs(s + window_s / 2 - word_mid))
        return window_sims[closest]

    result = [{**w, "speaker": "BC" if _sim_at((w["start"] + w["end"]) / 2) >= threshold else "OTHER"}
              for w in words]
    bc_n = sum(1 for w in result if w["speaker"] == "BC")
    print(f"    ECAPA: {bc_n}/{len(result)} words assigned to BC", flush=True)
    return result
