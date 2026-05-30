"""Identity-preserving accent shift via DSP formant-warping (Lever B).

Shifts the F1/F2 of target vowels in every clip toward a target accent centroid,
using the WORLD vocoder (splice mode: only the vowel window is resynthesised,
everything else passes through untouched). Reports per-clip and aggregate:
  - ECAPA identity(shifted, original)    → still the same voice?
  - mean Δ Bark toward target            → did accent move?

Two modes:
  F2 own-voice → RP   (owner recordings → RP vowel centroids)
  F1 BC voice  → GA   (IndexTTS-2 gen_base clips → GenAm vowel centroids)

Usage:
  .venv/bin/python scripts/accent_coach_dsp_shift.py --mode f2-own-rp
  .venv/bin/python scripts/accent_coach_dsp_shift.py --mode f1-bc-ga
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from accent_coach.diagnostics.bark_distance import hz_to_bark  # noqa: E402
from accent_coach.dsp.formant_shift import _measure_f1f2, shift_vowels_to_centroid  # noqa: E402
from accent_coach.reference.genam_norms import get_genam_norms  # noqa: E402
from accent_coach.reference.rp_norms import get_rp_norms  # noqa: E402
from scripts.lib.identity import cosine, embed_file, load_ecapa  # noqa: E402

MEAN_F0 = 110.0

MODES = {
    "f2-own-rp": {
        "desc": "F2: owner recordings → RP vowel centroids",
        "formants_csv": "tts_output/owner_cal_50/formants.csv",
        "manifest":     "tts_output/owner_cal_50/manifest.json",
        "wav_key":      "path",
        "clip_key":     "clip_id",
        "out_dir":      "tts_output/accent_coach/phase0_16/dsp_own_voice_rp",
        "suffix":       "_rpshift.wav",
        "target":       "rp",
        # owner's worst RP vowels + TRAP (the coaching priority vowels)
        "phonemes":     {"əʊ", "ɑː", "uː", "ɔː", "æ"},
    },
    "f1-bc-ga": {
        "desc": "F1: IndexTTS-2 gen_base (BC voice, RP accent) → GenAm vowel centroids",
        "formants_csv": "tts_output/cross_eval_50/gen_base/formants_gen_base_raw.csv",
        "manifest":     "tts_output/cross_eval_50/gen_base/manifest.json",
        "wav_key":      "wav_path",
        "clip_key":     "slug",
        "out_dir":      "tts_output/accent_coach/phase0_16/dsp_bc_voice_ga",
        "suffix":       "_gashift.wav",
        "target":       "genam",
        # vowels where RP and GenAm differ most (TRAP, GOAT, DRESS, FLEECE, STRUT)
        "phonemes":     {"æ", "əʊ", "ɛ", "iː", "ʌ", "ɑː"},
    },
}


def _bark_dist(f1: float, f2: float, tgt: tuple[float, float]) -> float:
    return float(np.hypot(hz_to_bark(f1) - hz_to_bark(tgt[0]),
                          hz_to_bark(f2) - hz_to_bark(tgt[1])))


def run_mode(cfg: dict, max_clips: int) -> None:
    norms = get_rp_norms(MEAN_F0) if cfg["target"] == "rp" else get_genam_norms(MEAN_F0)
    target_phonemes: set[str] = cfg["phonemes"]
    centroid = {ph: {"f1": norms[ph][0], "f2": norms[ph][1]}
                for ph in target_phonemes if ph in norms}

    # Load per-token formants, keyed by clip_id
    by_clip: dict[str, list[dict]] = defaultdict(list)
    with (PROJECT_ROOT / cfg["formants_csv"]).open() as fh:
        for r in csv.DictReader(fh):
            if r.get("F1") and r.get("phoneme") in target_phonemes:
                by_clip[r["clip_id"]].append(r)

    # Load manifest → clip_id → wav path
    manifest_rows = json.loads((PROJECT_ROOT / cfg["manifest"]).read_text())
    wav_of: dict[str, Path] = {}
    for m in manifest_rows:
        cid = m[cfg["clip_key"]]
        p = m[cfg["wav_key"]]
        wav_of[cid] = Path(p) if Path(p).is_absolute() else PROJECT_ROOT / p

    out_dir = PROJECT_ROOT / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    ecapa = load_ecapa()

    # Clips that actually have target-phoneme tokens, sorted by clip_id
    clip_ids = sorted([cid for cid in by_clip if cid in wav_of])[:max_clips]

    print(f"\n{cfg['desc']}")
    print(f"Target phonemes: {', '.join(sorted(target_phonemes))}")
    print(f"Processing {len(clip_ids)} clips → {out_dir.relative_to(PROJECT_ROOT)}/\n")
    print(f"{'clip':16}{'edits':7}{'identity':10}{'Δ Bark':10}")
    print("-" * 48)

    ids, deltas = [], []
    for cid in clip_ids:
        src = wav_of[cid]
        dst = out_dir / (cid + cfg["suffix"])
        segs = by_clip[cid]

        res = shift_vowels_to_centroid(src, dst, segs, target_phonemes, centroid, splice=True)

        id_cos = cosine(embed_file(dst, ecapa), embed_file(src, ecapa))
        ids.append(id_cos)

        a, sr = sf.read(str(dst))
        if a.ndim > 1:
            a = a.mean(axis=1)
        orig_d, new_d = [], []
        for s in segs:
            ph = s["phoneme"]
            if ph not in norms:
                continue
            f1o, f2o = float(s["F1"]), float(s["F2"])
            f1n, f2n = _measure_f1f2(a.astype(np.float32), sr,
                                     float(s["start_s"]), float(s["end_s"]))
            if np.isnan(f1n):
                continue
            orig_d.append(_bark_dist(f1o, f2o, norms[ph]))
            new_d.append(_bark_dist(f1n, f2n, norms[ph]))
        delta = float(np.mean(orig_d) - np.mean(new_d)) if orig_d else float("nan")
        deltas.append(delta)
        print(f"{cid:16}{res['edits_applied']:<7}{id_cos:<10.3f}{delta:+.3f}")

    print("-" * 48)
    print(f"mean identity  = {np.nanmean(ids):.3f}  (>0.70 = same speaker)")
    print(f"mean Δ→target  = {np.nanmean(deltas):+.3f} Bark  (+ = moved toward target accent)")
    identity_label = "self" if cfg["target"] == "rp" else "BC-original"
    print(f"\n>>> Listen: {out_dir.relative_to(PROJECT_ROOT)}/*.wav"
          f"  (identity compared to {identity_label})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(MODES), default="f2-own-rp")
    ap.add_argument("--max-clips", type=int, default=50)
    args = ap.parse_args()
    run_mode(MODES[args.mode], args.max_clips)
    return 0


if __name__ == "__main__":
    sys.exit(main())
