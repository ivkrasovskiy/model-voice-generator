"""F5-TTS generation and aggregation helpers for posthoc_eval.py.

These functions depend on F5-TTS model objects and inference settings;
they are separated here to keep posthoc_eval.py under the 400 code-line cap.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import soundfile as sf


class GenSettings(NamedTuple):
    ref_audio: str
    ref_text: str
    cfg_strength: float = 2.0
    nfe_step: int = 32
    seed: int = 42
    speed_fix: bool = False
    selective_cfg: float = 0.0


def apply_checkpoint(tts, ckpt_path: Path) -> int:
    """Load a partial fine-tune checkpoint into tts.ema_model.transformer."""
    import torch
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = checkpoint["model"]
    model = tts.ema_model.transformer
    own = model.state_dict()
    n_loaded = 0
    for k, v in sd.items():
        key = k.replace("transformer.", "", 1) if k.startswith("transformer.") else k
        if key in own:
            own[key] = v.to(own[key].device)
            n_loaded += 1
    model.load_state_dict(own, strict=False)
    return n_loaded


def _gen_segment(tts, segment: str, settings: GenSettings, max_retries: int = 5):
    """Single-segment generation with retry."""
    from lib.inference import mps_reset
    speed = 0.3 if settings.speed_fix and len(segment.encode("utf-8")) < 10 else 1.0
    if speed != 1.0:
        print(f"      speed-fix: len={len(segment.encode('utf-8'))} bytes → speed={speed}")
    for attempt in range(1, max_retries + 1):
        wav, sr, _ = tts.infer(
            ref_file=settings.ref_audio, ref_text=settings.ref_text, gen_text=segment,
            cfg_strength=settings.cfg_strength, nfe_step=settings.nfe_step,
            speed=speed,
            seed=settings.seed + (attempt - 1),
            selective_cfg_threshold=settings.selective_cfg,
        )
        w = np.asarray(wav.squeeze() if hasattr(wav, "squeeze") else wav, dtype=np.float32)
        finite = np.isfinite(w).all()
        peak = float(np.abs(w[np.isfinite(w)]).max()) if finite else 0.0
        mps_reset()
        if finite and peak > 0.01:
            return w, sr
        print(f"      attempt {attempt}: {'NaN/Inf' if not finite else f'SILENT peak={peak:.4f}'}")
    return None, None


def gen_with_retry(tts, text: str, settings: GenSettings, max_retries: int = 5):
    """Split text → generate per-segment → concatenate with 150 ms silence."""
    from lib.inference import split_to_short_segments
    segments = split_to_short_segments(text, max_chars=50)
    if len(segments) == 1:
        return _gen_segment(tts, segments[0], settings, max_retries=max_retries)
    print(f"    split into {len(segments)} segments (≤50 chars each)")
    pieces, sr_out = [], None
    for i, seg in enumerate(segments, 1):
        wav_np, sr = _gen_segment(tts, seg, settings, max_retries=max_retries)
        if wav_np is None:
            print(f"    segment {i}/{len(segments)} FAILED — abort")
            return None, None
        print(f"    segment {i}/{len(segments)} ok  ({len(wav_np)/sr:.1f}s, "
              f"peak={float(np.abs(wav_np).max()):.3f})")
        pieces.append(wav_np)
        if sr_out is None:
            sr_out = sr
        pieces.append(np.zeros(int(0.15 * sr), dtype=np.float32))
    return np.concatenate(pieces[:-1]), sr_out


def compute_ref_centroid(centroid_dir: Path, n_samples: int, ecapa, seed: int) -> np.ndarray:
    """Embed N random WAVs from centroid_dir and average → speaker identity centroid."""
    import random

    from lib.audio_io import read_wav_mono
    from lib.identity import embed_wav

    wav_dir = centroid_dir / "wavs" if (centroid_dir / "wavs").is_dir() else centroid_dir
    wavs = sorted(wav_dir.glob("*.wav"))
    if len(wavs) < n_samples:
        raise RuntimeError(f"Only {len(wavs)} WAVs in {wav_dir}, need {n_samples}")
    picked = random.Random(seed).sample(wavs, n_samples)
    print(f"  centroid: averaging {n_samples} ECAPA embeddings from {wav_dir.name}/")
    embs = [embed_wav(*read_wav_mono(p), ecapa) for p in picked]
    centroid = np.mean(np.stack(embs, axis=0), axis=0)
    print(f"  centroid norm={np.linalg.norm(centroid):.3f}  "
          f"(single-clip avg norm={np.mean([np.linalg.norm(e) for e in embs]):.3f})")
    return centroid


def evaluate_one(
    label: str,
    phrases,
    tts,
    whisper_model,
    ecapa,
    ecapa_ref_emb: np.ndarray,
    dnsmos_session,
    out_dir: Path,
    save_wavs: bool,
    settings: GenSettings,
    project_root: Path | None = None,
):
    """Generate + score one config. Returns (rows, saved_paths)."""
    from lib.scoring import score_single_wav

    rows, saved = [], []
    for slug, prompt in phrases:
        wav_np, sr = gen_with_retry(tts, prompt, settings)
        if wav_np is None:
            print(f"  {label}/{slug}: GENERATION FAILED")
            rows.append({"slug": slug, "wer": np.nan, "ecapa_sim": np.nan,
                         "dnsmos_sig": np.nan, "dnsmos_bak": np.nan, "dnsmos_ovr": np.nan,
                         "transcript": ""})
            continue

        if save_wavs:
            wav_path = out_dir / f"{label}_{slug}.wav"
            sf.write(str(wav_path), wav_np, sr)
            rel = str(wav_path.relative_to(project_root)) if project_root else str(wav_path)
            saved.append(rel)
            print(f"  saved → {wav_path.name}")

        m = score_single_wav(wav_np, sr, prompt, whisper_model, ecapa, dnsmos_session, ecapa_ref_emb)
        wer, hyp = m["wer"], m["transcript"]
        ecapa_sim = m["ecapa_sim"]
        mos = {"sig": m["dnsmos_sig"], "bak": m["dnsmos_bak"], "ovr": m["dnsmos_ovr"]}
        print(f"  {label}/{slug}: wer={wer:.3f} ecapa={ecapa_sim:.4f} "
              f"sig={mos['sig']:.2f} bak={mos['bak']:.2f} ovr={mos['ovr']:.2f}")
        if wer > 0.1:
            print(f"    (whisper heard: \"{hyp[:80]}\")")
        rows.append({"slug": slug, "wer": wer, "ecapa_sim": ecapa_sim,
                     "dnsmos_sig": mos["sig"], "dnsmos_bak": mos["bak"],
                     "dnsmos_ovr": mos["ovr"], "transcript": hyp})

    return rows, saved


def aggregate(rows: list[dict]) -> dict:
    """Aggregate per-clip metric rows → mean/std summary."""
    def m(key):
        vals = [r[key] for r in rows if not np.isnan(r[key])]
        if not vals:
            return float("nan"), float("nan"), 0
        return float(np.mean(vals)), float(np.std(vals)), len(vals)

    w_mean, w_std, n = m("wer")
    e_mean, e_std, _ = m("ecapa_sim")
    ec_mean, ec_std, _ = m("ecapa_centroid_sim")
    sig_mean, _, _ = m("dnsmos_sig")
    bak_mean, _, _ = m("dnsmos_bak")
    ovr_mean, _, _ = m("dnsmos_ovr")
    return {
        "n_scored": n,
        "wer_mean": w_mean, "wer_std": w_std,
        "ecapa_sim_mean": e_mean, "ecapa_sim_std": e_std,
        "ecapa_centroid_mean": ec_mean, "ecapa_centroid_std": ec_std,
        "dnsmos_sig_mean": sig_mean, "dnsmos_bak_mean": bak_mean, "dnsmos_ovr_mean": ovr_mean,
    }
