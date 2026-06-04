"""ECAPA audit of Lindsey + Fry modern_rp corpus clips.

For each clip in modern_rp_corpus/{speaker}/clips/, compute ECAPA cosine
similarity to a clean reference embedding (the 14s emo clip we extracted).
Classify clips as 'target' (sim >= threshold) or 'other' (sim < threshold).

Does NOT rebuild centroids or re-extract formants. Outputs a JSON audit per
speaker showing which clips are likely contaminated, so a future Phase 0.12
can decide what to filter out before rebuilding the corpus.

Usage:
    .venv/bin/python scripts/accent_coach_corpus_audit.py

Outputs:
    tts_output/accent_coach/corpus_audit/{speaker}_audit.json
    tts_output/accent_coach/corpus_audit/summary.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import soundfile as sf

OUT_DIR  = PROJECT_ROOT / "tts_output/accent_coach/corpus_audit"
CORPUS   = PROJECT_ROOT / "tts_output/modern_rp_corpus"

# Each speaker → (ref_clip_path, clips_dir)
# Use the same 14s emo refs we extracted — verified clean monologue.
SPEAKERS = {
    "lindsey": {
        "ref":   PROJECT_ROOT / "tts_output/refs/rp_ceiling/ref_lindsey_emo.wav",
        "clips": CORPUS / "lindsey/clips",
    },
    "fry": {
        "ref":   PROJECT_ROOT / "tts_output/refs/rp_ceiling/ref_fry_emo.wav",
        "clips": CORPUS / "fry/clips",
    },
}

# Threshold for classifying as "target speaker" — same as build_real_bc default (0.45 there);
# being conservative here at 0.55 because we want clean centroid data, not maximum recall.
SIM_THRESHOLDS = [0.40, 0.50, 0.60]


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def _embed_wav(wav: np.ndarray, sr: int, ecapa) -> np.ndarray:
    from scripts.lib.identity import embed_wav
    return embed_wav(wav.astype(np.float32), sr, ecapa)


def audit_speaker(name: str, ref_path: Path, clips_dir: Path, ecapa) -> dict:
    from scripts.lib.identity import cosine

    _log(f"\n=== {name}: ref={ref_path.name}, clips={clips_dir} ===")
    if not ref_path.exists():
        _log(f"  ERROR: reference clip missing: {ref_path}")
        return {"speaker": name, "error": "ref missing"}
    if not clips_dir.exists():
        _log(f"  ERROR: clips dir missing: {clips_dir}")
        return {"speaker": name, "error": "clips dir missing"}

    ref_wav, ref_sr = sf.read(str(ref_path))
    if ref_wav.ndim > 1:
        ref_wav = ref_wav.mean(axis=1)
    ref_emb = _embed_wav(ref_wav, ref_sr, ecapa)
    _log(f"  ref embed computed (dur={len(ref_wav)/ref_sr:.1f}s, dim={ref_emb.shape})")

    clips = sorted(clips_dir.glob("*.wav"))
    _log(f"  found {len(clips)} clips to audit")

    per_clip = []
    t0 = time.time()
    for i, clip_path in enumerate(clips, 1):
        try:
            wav, sr = sf.read(str(clip_path))
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            dur = len(wav) / sr
            if dur < 0.5:
                per_clip.append({"clip": clip_path.name, "sim": None,
                                 "dur_s": round(dur, 2), "note": "too short"})
                continue
            emb = _embed_wav(wav, sr, ecapa)
            sim = cosine(emb, ref_emb)
            per_clip.append({"clip": clip_path.name, "sim": round(sim, 4),
                             "dur_s": round(dur, 2)})
        except Exception as e:
            per_clip.append({"clip": clip_path.name, "sim": None,
                             "dur_s": None, "note": f"error: {e}"})

        if i % 200 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (len(clips) - i) / rate
            _log(f"  [{i}/{len(clips)}] elapsed={elapsed:.0f}s, rate={rate:.1f}/s, eta={eta:.0f}s")

    sims = [c["sim"] for c in per_clip if c["sim"] is not None]
    by_threshold = {}
    for thr in SIM_THRESHOLDS:
        kept = sum(1 for s in sims if s >= thr)
        by_threshold[f"thr_{thr:.2f}"] = {
            "kept": kept,
            "dropped": len(sims) - kept,
            "kept_pct": round(100 * kept / max(len(sims), 1), 1),
        }

    summary = {
        "speaker": name,
        "ref_clip": str(ref_path.relative_to(PROJECT_ROOT)),
        "n_clips_total": len(clips),
        "n_clips_scored": len(sims),
        "sim_mean": round(float(np.mean(sims)), 3) if sims else None,
        "sim_std":  round(float(np.std(sims)),  3) if sims else None,
        "sim_p5":   round(float(np.percentile(sims, 5)), 3) if sims else None,
        "sim_p25":  round(float(np.percentile(sims, 25)), 3) if sims else None,
        "sim_p50":  round(float(np.percentile(sims, 50)), 3) if sims else None,
        "sim_p75":  round(float(np.percentile(sims, 75)), 3) if sims else None,
        "sim_p95":  round(float(np.percentile(sims, 95)), 3) if sims else None,
        "by_threshold": by_threshold,
        "elapsed_s": round(time.time() - t0, 1),
    }
    _log(f"  {name}: sim_mean={summary['sim_mean']}  "
         f"p25={summary['sim_p25']}  p50={summary['sim_p50']}  p75={summary['sim_p75']}")
    for thr_key, stats in by_threshold.items():
        _log(f"    {thr_key}: kept {stats['kept']}/{summary['n_clips_scored']} "
             f"({stats['kept_pct']}%)")
    return {**summary, "per_clip": per_clip}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _log("=== Phase 0.11+ corpus ECAPA audit ===")

    from scripts.lib.identity import load_ecapa
    _log("Loading ECAPA model…")
    ecapa = load_ecapa()
    _log("  loaded.")

    overall = {"speakers": {}}
    for name, info in SPEAKERS.items():
        audit = audit_speaker(name, info["ref"], info["clips"], ecapa)
        (OUT_DIR / f"{name}_audit.json").write_text(json.dumps(audit, indent=2))
        overall["speakers"][name] = {k: v for k, v in audit.items() if k != "per_clip"}
        _log(f"  wrote: {OUT_DIR / f'{name}_audit.json'}")

    (OUT_DIR / "summary.json").write_text(json.dumps(overall, indent=2))
    _log(f"\nAudit complete. Summary: {OUT_DIR / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
