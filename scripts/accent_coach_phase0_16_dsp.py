"""B2(ii) — identity-preserving accent shift via DSP (Phase 0.16, Lever B).

Demonstrates Feature F2 (own voice → target accent): take owner clips and warp
the formants of the owner's worst vowels toward the RP centroid, WITHOUT touching
timbre or pitch. Reports, per clip:
  - ECAPA identity(shifted, original)  → is it still the owner's voice?
  - vowel movement toward the RP target → did the accent shift?

Pure DSP (WORLD vocoder), no TTS model.
  .venv/bin/python scripts/accent_coach_phase0_16_dsp.py
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from accent_coach.diagnostics.bark_distance import hz_to_bark  # noqa: E402
from accent_coach.dsp.formant_shift import shift_vowels_to_centroid  # noqa: E402
from accent_coach.reference.rp_norms import get_rp_norms  # noqa: E402
from scripts.lib.identity import cosine, embed_file, load_ecapa  # noqa: E402

OWNER_FORMANTS = PROJECT_ROOT / "tts_output/owner_cal_50/formants.csv"
OWNER_MANIFEST = PROJECT_ROOT / "tts_output/owner_cal_50/manifest.json"
OUT_DIR = PROJECT_ROOT / "tts_output/accent_coach/phase0_16/dsp_own_voice_rp"
# Owner's worst RP vowels (Phase 0.15/0.16 gate): GOAT, BATH, GOOSE
TARGET_PHONEMES = {"əʊ", "ɑː", "uː"}
N_CLIPS = 4


def _segments_by_clip() -> dict[str, list[dict]]:
    by_clip: dict[str, list[dict]] = defaultdict(list)
    with OWNER_FORMANTS.open() as fh:
        for r in csv.DictReader(fh):
            if r.get("F1") and r.get("phoneme") in TARGET_PHONEMES:
                by_clip[r["clip_id"]].append(r)
    return by_clip


def _bark_dist(f1, f2, tgt) -> float:
    return float(np.hypot(hz_to_bark(f1) - hz_to_bark(tgt[0]), hz_to_bark(f2) - hz_to_bark(tgt[1])))


def main() -> int:
    import json
    rp = get_rp_norms(110.0)
    centroid = {ph: {"f1": rp[ph][0], "f2": rp[ph][1]} for ph in TARGET_PHONEMES if ph in rp}
    manifest = {m["clip_id"]: m["path"] for m in json.loads(OWNER_MANIFEST.read_text())}
    by_clip = _segments_by_clip()
    ecapa = load_ecapa()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Shifting owner vowels {TARGET_PHONEMES} toward RP centroid (identity-preserving)\n")
    print(f"{'clip':12}{'edits':7}{'id(shift,orig)':16}{'mean Δ→RP (Bark)':18}")
    print("-" * 60)
    ids, deltas = [], []
    for clip_id in list(by_clip)[:N_CLIPS]:
        segs = by_clip[clip_id]
        src = PROJECT_ROOT / manifest[clip_id]
        dst = OUT_DIR / f"{clip_id}_rpshift.wav"
        res = shift_vowels_to_centroid(src, dst, segs, TARGET_PHONEMES, centroid, splice=True)

        # identity: ECAPA(shifted, original)
        id_cos = cosine(embed_file(dst, ecapa), embed_file(src, ecapa))
        ids.append(id_cos)

        # movement: re-measure shifted segments, compare dist-to-RP before/after
        orig_d, new_d = [], []
        a, sr = sf.read(str(dst))
        if a.ndim > 1:
            a = a.mean(axis=1)
        from accent_coach.dsp.formant_shift import _measure_f1f2
        for s in segs:
            ph = s["phoneme"]
            f1o, f2o = float(s["F1"]), float(s["F2"])
            f1n, f2n = _measure_f1f2(a.astype(np.float32), sr, float(s["start_s"]), float(s["end_s"]))
            if np.isnan(f1n):
                continue
            orig_d.append(_bark_dist(f1o, f2o, rp[ph]))
            new_d.append(_bark_dist(f1n, f2n, rp[ph]))
        delta = float(np.mean(orig_d) - np.mean(new_d)) if orig_d else float("nan")
        deltas.append(delta)
        print(f"{clip_id:12}{res['edits_applied']:<7}{id_cos:<16.3f}{delta:<18.3f}")

    print("-" * 60)
    print(f"mean identity(shifted,original) = {np.nanmean(ids):.3f}  "
          f"(1.0 = identical voice; >0.7 = strongly same speaker)")
    print(f"mean Δ→RP = {np.nanmean(deltas):+.3f} Bark  (positive = moved TOWARD RP)")
    print(f"\nOutputs: {OUT_DIR.relative_to(PROJECT_ROOT)}/ — listen-check own-voice-in-RP")
    return 0


if __name__ == "__main__":
    sys.exit(main())
