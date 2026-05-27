"""Shared WER / ECAPA / DNSMOS scoring utilities.

Single source of truth for manifest-based post-hoc scoring and model loading.
All callers should use score_clips() or (for live scoring) load_scoring_models()
+ score_single_wav() — never load Whisper/ECAPA/DNSMOS primitives directly.
"""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np

_WHISPER = None
_ECAPA = None
_DNSMOS = None
_ECAPA_REF_EMB: np.ndarray | None = None
_ECAPA_REF_PATH: str | None = None


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def load_scoring_models(device: str = "cpu") -> tuple:
    """Load and cache Whisper, ECAPA, DNSMOS models. Returns (whisper, ecapa, dnsmos)."""
    global _WHISPER, _ECAPA, _DNSMOS
    if _WHISPER is None:
        from lib.transcribe import load_whisper
        _log("Loading Whisper large-v3…")
        _WHISPER = load_whisper("large-v3", device=device)
    if _ECAPA is None:
        from lib.identity import load_ecapa
        _log("Loading ECAPA-TDNN…")
        _ECAPA = load_ecapa(device=device)
    if _DNSMOS is None:
        from lib.metrics import load_dnsmos
        _log("Loading DNSMOS…")
        _DNSMOS = load_dnsmos()
    return _WHISPER, _ECAPA, _DNSMOS


def _get_ref_emb(ecapa_ref: Path, ecapa) -> np.ndarray:
    global _ECAPA_REF_EMB, _ECAPA_REF_PATH
    ref_key = str(ecapa_ref)
    if _ECAPA_REF_EMB is not None and ref_key == _ECAPA_REF_PATH:
        return _ECAPA_REF_EMB
    from lib.identity import embed_file
    emb = embed_file(str(ecapa_ref), ecapa)
    if emb is None:
        raise RuntimeError(f"Failed to embed ECAPA reference: {ecapa_ref}")
    _ECAPA_REF_EMB = emb
    _ECAPA_REF_PATH = ref_key
    _log(f"ECAPA ref: {Path(ecapa_ref).name}  dim={emb.shape[0]}")
    return _ECAPA_REF_EMB


def score_single_wav(
    wav_np: np.ndarray,
    sr: int,
    prompt: str,
    whisper,
    ecapa,
    dnsmos,
    ref_emb: np.ndarray,
) -> dict:
    """Score one WAV against a reference embedding.

    Returns dict with keys: wer, transcript, ecapa_sim, dnsmos_sig, dnsmos_bak, dnsmos_ovr.
    Any failed metric is NaN.
    """
    from lib.identity import cosine, embed_wav
    from lib.metrics import compute_dnsmos, compute_wer

    wer = float("nan")
    transcript = ""
    try:
        wer, transcript = compute_wer(prompt, wav_np, sr, whisper)
    except Exception as e:
        _log(f"  WER failed: {e}")

    ecapa_sim = float("nan")
    try:
        gen_emb = embed_wav(wav_np, sr, ecapa)
        ecapa_sim = cosine(ref_emb, gen_emb)
    except Exception as e:
        _log(f"  ECAPA failed: {e}")

    dnsmos_sig = dnsmos_bak = dnsmos_ovr = float("nan")
    try:
        mos = compute_dnsmos(wav_np, sr, dnsmos)
        dnsmos_sig, dnsmos_bak, dnsmos_ovr = mos["sig"], mos["bak"], mos["ovr"]
    except Exception as e:
        _log(f"  DNSMOS failed: {e}")

    return {
        "wer": wer,
        "transcript": transcript,
        "ecapa_sim": ecapa_sim,
        "dnsmos_sig": dnsmos_sig,
        "dnsmos_bak": dnsmos_bak,
        "dnsmos_ovr": dnsmos_ovr,
    }


def score_clips(
    manifest: Path,
    ecapa_ref: Path,
    out_csv: Path,
    cache: bool = True,
) -> dict:
    """Score all clips in a manifest JSON for WER, ECAPA cosine sim, and DNSMOS.

    If cache=True and out_csv already exists, returns cached aggregates without
    re-scoring. Returns {'WER', 'ECAPA', 'DNSMOS_OVR'} as mean floats.
    """
    import soundfile as sf

    manifest = Path(manifest)
    ecapa_ref = Path(ecapa_ref)
    out_csv = Path(out_csv)

    if cache and out_csv.exists():
        _log(f"score_clips: cached at {out_csv.name}, skipping")
        existing = list(csv.DictReader(out_csv.open()))
        if existing:
            wers   = [float(r["wer"])       for r in existing if r.get("wer")       not in ("", "nan")]
            ecapas = [float(r["ecapa_sim"])  for r in existing if r.get("ecapa_sim") not in ("", "nan")]
            dnsmos = [float(r["dnsmos_ovr"]) for r in existing if r.get("dnsmos_ovr") not in ("", "nan")]
            return {
                "WER":        round(float(np.mean(wers)),   4) if wers   else float("nan"),
                "ECAPA":      round(float(np.mean(ecapas)), 4) if ecapas else float("nan"),
                "DNSMOS_OVR": round(float(np.mean(dnsmos)), 4) if dnsmos else float("nan"),
            }

    whisper, ecapa, dnsmos = load_scoring_models()
    ref_emb = _get_ref_emb(ecapa_ref, ecapa)

    clips = json.loads(manifest.read_text())
    _log(f"score_clips: scoring {len(clips)} clips…")

    rows = []
    for i, entry in enumerate(clips):
        wav_path = Path(entry["wav_path"])
        if not wav_path.exists():
            _log(f"  SKIP missing: {wav_path.name}")
            rows.append({"slug": entry["slug"], "prompt": entry["prompt"],
                         "wer": float("nan"), "ecapa_sim": float("nan"),
                         "dnsmos_ovr": float("nan"), "wav_path": str(wav_path)})
            continue

        audio, sr = sf.read(str(wav_path))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        m = score_single_wav(audio, sr, entry["prompt"], whisper, ecapa, dnsmos, ref_emb)
        wer, ecapa_sim, dnsmos_ovr = m["wer"], m["ecapa_sim"], m["dnsmos_ovr"]

        if not math.isnan(wer) and wer > 0.1:
            _log(f"  WER={wer:.3f} heard: {m['transcript'][:80]!r}")
        rows.append({
            "slug":       entry["slug"],
            "prompt":     entry["prompt"],
            "wer":        round(wer, 4),
            "ecapa_sim":  round(ecapa_sim, 4),
            "dnsmos_ovr": round(dnsmos_ovr, 4),
            "wav_path":   str(wav_path),
        })
        _log(f"  [{i+1}/{len(clips)}] {entry['slug']}: "
             f"wer={wer:.3f} ecapa={ecapa_sim:.4f} dnsmos={dnsmos_ovr:.2f}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else ["slug"])
        w.writeheader()
        w.writerows(rows)

    def _mean(key):
        vals = [r[key] for r in rows if not (isinstance(r[key], float) and math.isnan(r[key]))]
        return round(float(np.mean(vals)), 4) if vals else float("nan")

    return {"WER": _mean("wer"), "ECAPA": _mean("ecapa_sim"), "DNSMOS_OVR": _mean("dnsmos_ovr")}
