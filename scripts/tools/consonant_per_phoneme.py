"""Per-phoneme consonant discrimination diagnostic (Phase 1).

Breaks the consonant score down to INDIVIDUAL phonemes and reports, per group,
each phoneme's mean score + token count, plus the native−owner gap. This answers:
which consonants actually separate the owner from natives, which tie (because the
sound exists ~identically in Russian), and which are starved of tokens or
mismeasured. See docs/prosody_consonant_upgrade.md.

It also computes TWO composites per group:
  - class_weighted : the current production weighting (fric .35 / stop .35 /
                     rhotic .20 / lateral .10) — preserved for comparison.
  - equal_phoneme  : every phoneme TYPE weighted the same (mean over phonemes of
                     each phoneme's mean score) — surfaces rare discriminators.

    .venv/bin/python scripts/tools/consonant_per_phoneme.py --n 16
Writes tts_output/accent_coach/consonant_per_phoneme.csv + prints a summary.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from accent_coach.comparison.consonants import score_consonants
from accent_coach.comparison.consonants.fricatives import (
    _FRICATIVES,
    _SKIP,
    _cog_norms,
    _spectral_centroid,
)
from accent_coach.comparison.consonants.liquids import score_liquids
from accent_coach.comparison.consonants.stops import _DECAY as _STOP_DECAY
from accent_coach.comparison.consonants.stops import _UNDER_ASPIRATION_MS, _UNDER_ASPIRATION_PENALTY
from accent_coach.models import SentenceAnalysis
from accent_coach.pipeline.alignment import align_audio
from accent_coach.pipeline.audio_io import load_standard_audio
from accent_coach.pipeline.vot import extract_stop_features
from accent_coach.reference.rp_norms import RP_VOT_MEAN_MS, RP_VOT_SD_MS
from scripts.lib.manifest import load_manifest, resolve_path

# (label, manifest, accent_family) — owner first, then natives + TTS.
GROUPS = [
    ("owner",        "tts_output/owner_cal_50/manifest.json", "rp"),
    ("rp_fry",       "tts_output/modern_rp_corpus/fry_manifest.json", "rp"),
    ("rp_lindsey",   "tts_output/modern_rp_corpus/lindsey_manifest.json", "rp"),
    ("real_bc",      "tts_output/real_bc_corpus/manifest.json", "rp"),
    ("genam_harris", "tts_output/genam_lecture_corpus/manifest.json", "rp"),
    ("tts_bc",       "tts_output/eval_indextts_interview_short/manifest.json", "rp"),
]
NATIVE_LABELS = {"rp_fry", "rp_lindsey", "real_bc", "genam_harris"}
PHONEME_ORDER = ["s", "z", "ʃ", "ʒ", "θ", "ð", "f", "v", "p", "t", "k", "r", "l"]


def _fricative_scores(user, audio, sr, accent) -> dict[str, list[float]]:
    """Per-token fricative score, grouped by phoneme — mirrors score_fricatives'
    absolute path: 100·exp(-|CoG-ref|/decay)."""
    ref_cog, decay, _ = _cog_norms(accent)
    out: dict[str, list[float]] = defaultdict(list)
    for p in user.phonemes:
        if p.phoneme not in _FRICATIVES or p.phoneme in _SKIP:
            continue
        cog = _spectral_centroid(audio, sr, p)
        if cog is None:
            continue
        out[p.phoneme].append(100.0 * math.exp(-abs(cog - ref_cog[p.phoneme]) / decay))
    return out


def _stop_scores(user, accent) -> dict[str, list[float]]:
    """Per-token stop score from pre-extracted StopFeatures — mirrors score_stops."""
    out: dict[str, list[float]] = defaultdict(list)
    for s in user.stops:
        ph = s.phoneme.phoneme
        if ph not in RP_VOT_MEAN_MS:
            continue
        z = abs(s.vot_ms - RP_VOT_MEAN_MS[ph]) / RP_VOT_SD_MS[ph]
        base = 100.0 * math.exp(-z / _STOP_DECAY)
        if s.vot_ms < _UNDER_ASPIRATION_MS:
            base *= _UNDER_ASPIRATION_PENALTY
        out[ph].append(base)
    return out


def _score_clip(wav: Path, transcript: str, accent: str) -> dict[str, list[float]] | None:
    loaded = load_standard_audio(wav)
    if loaded is None:
        return None
    audio, sr = loaded
    if len(audio) < sr * 0.8:
        return None
    try:
        phon = align_audio(wav, transcript, 0, accent)
    except Exception:  # noqa: BLE001
        return None
    if not phon:
        return None
    stops = extract_stop_features(audio, sr, phon)
    sa = SentenceAnalysis(sentence_id=0, sentence_type="statement", duration_s=len(audio) / sr,
                          syllable_durations=[0.2], pitch_contour=[0.5] * 50, stress_pattern=[True],
                          vowels=[], stops=stops, phonemes=phon)
    per: dict[str, list[float]] = defaultdict(list)
    for ph, vals in _fricative_scores(sa, audio, sr, accent).items():
        per[ph].extend(vals)
    for ph, vals in _stop_scores(sa, accent).items():
        per[ph].extend(vals)
    rho, lat, _ = score_liquids(sa, audio, sr, accent)
    if rho is not None:
        per["r"].append(rho)
    if lat is not None:
        per["l"].append(lat)
    # class-weighted composite (production), for comparison
    cs = score_consonants(sa, audio, sr, accent_target=accent)
    if cs.score is not None:
        per["__composite_class__"].append(cs.score)
    return per


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    # group -> phoneme -> [scores]
    data: dict[str, dict[str, list[float]]] = {g: defaultdict(list) for g, _, _ in GROUPS}
    for label, mani, accent in GROUPS:
        mp = REPO_ROOT / mani
        if not mp.exists():
            continue
        entries = load_manifest(mp)
        entries = sorted(entries, key=lambda e: (e.get("end_s", 3) - e.get("start_s", 0)), reverse=True)
        sample = random.sample(entries[: max(args.n * 3, 40)], min(args.n, len(entries)))
        for e in sample:
            wav = resolve_path(e, mp, REPO_ROOT)
            tr = e.get("transcript") or e.get("prompt", "")
            if wav is None or not tr:
                continue
            per = _score_clip(wav, tr, accent)
            if per is None:
                continue
            for ph, vals in per.items():
                data[label][ph].extend(vals)
        print(f"  scored {label}", file=sys.stderr)

    def mean(label: str, ph: str) -> float | None:
        v = data[label].get(ph, [])
        return float(np.mean(v)) if v else None

    def equal_phoneme(label: str) -> float | None:
        ms = [mean(label, ph) for ph in PHONEME_ORDER]
        ms = [m for m in ms if m is not None]
        return float(np.mean(ms)) if ms else None

    def native_pooled(ph: str) -> float | None:
        vals = [x for lbl in NATIVE_LABELS for x in data[lbl].get(ph, [])]
        return float(np.mean(vals)) if vals else None

    # ---- print per-phoneme table ----
    print("\n" + "=" * 92)
    print("  PER-PHONEME consonant score  (mean; n in parens). gap = native_pooled − owner")
    print("=" * 92)
    hdr = f"{'ph':<4} " + "".join(f"{g:>12}" for g, _, _ in GROUPS) + f"{'nat_pool':>10}{'gap':>7}"
    print(hdr)
    print("-" * len(hdr))
    rows_csv = []
    for ph in PHONEME_ORDER:
        cells = ""
        for g, _, _ in GROUPS:
            m = mean(g, ph)
            n = len(data[g].get(ph, []))
            cells += f"{(f'{m:.0f}({n})' if m is not None else '-'):>12}"
        npool = native_pooled(ph)
        owner = mean("owner", ph)
        gap = (npool - owner) if (npool is not None and owner is not None) else None
        print(f"{ph:<4} {cells}{(f'{npool:.0f}' if npool is not None else '-'):>10}"
              f"{(f'{gap:+.0f}' if gap is not None else '-'):>7}")
        rows_csv.append({"phoneme": ph, "owner": owner, "native_pooled": npool, "gap": gap,
                         **{g: mean(g, ph) for g, _, _ in GROUPS},
                         **{f"n_{g}": len(data[g].get(ph, [])) for g, _, _ in GROUPS}})

    # ---- composites ----
    print("\n" + "-" * 60)
    print(f"  {'group':<14}{'class_weighted':>16}{'equal_phoneme':>16}")
    for g, _, _ in GROUPS:
        cw = mean(g, "__composite_class__")
        ep = equal_phoneme(g)
        print(f"  {g:<14}{(f'{cw:.1f}' if cw else '-'):>16}{(f'{ep:.1f}' if ep else '-'):>16}")

    # ---- save CSV ----
    out_csv = REPO_ROOT / "tts_output/accent_coach/consonant_per_phoneme.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
        w.writeheader()
        w.writerows(rows_csv)
    print(f"\n  saved {out_csv}")


if __name__ == "__main__":
    main()
