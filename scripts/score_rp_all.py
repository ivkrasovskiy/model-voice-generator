"""Compare all sources against modern RP norms.

Sources scored:
  gen_base       — IndexTTS-2 base model, BC reference
  gen_lora_best  — IndexTTS-2 + LoRA best (step_400), BC reference
  owner          — owner voice recordings (no RP corrections — raw measurement)
  fry            — Stephen Fry corpus clips (RP corrections applied)
  lindsey        — John Lindsey corpus clips (RP corrections applied)
  bbc            — BBC male corpus clips (RP corrections applied)
  real_bc        — real Benedict Cumberbatch recordings (RP corrections applied)

Usage:
  .venv/bin/python scripts/score_rp_all.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from accent_coach_extract_formants import extract_formants  # noqa: E402

from accent_coach.diagnostics.bark_distance import hz_to_bark  # noqa: E402
from accent_coach.pipeline.centroids import build_centroids_from_formants  # noqa: E402
from accent_coach.reference.rp_norms import get_rp_norms  # noqa: E402

RP = get_rp_norms(110.0)  # BC male F0 ~110 Hz

OUT_ROOT = PROJECT_ROOT / "tts_output/accent_coach/phase0_14/lora"
CORPUS_ROOT = PROJECT_ROOT / "tts_output/modern_rp_corpus"
TRAIN_JSONL = OUT_ROOT / "dataset/train.jsonl"

FOCUS_VOWELS = [
    ("ɑː", "BATH/PALM"), ("æ", "TRAP"),   ("ɛ", "DRESS"),
    ("əʊ", "GOAT"),       ("ə", "SCHWA"),  ("ɔː", "THOUGHT"),
    ("ɪ",  "KIT"),        ("ɜː", "NURSE"), ("aɪ", "PRICE"),
    ("uː", "GOOSE"),
]

# Max clips per corpus speaker (keep scoring fast)
MAX_CORPUS_CLIPS = 150


def _bark_dist(f1a: float, f2a: float, f1b: float, f2b: float) -> float:
    return float(np.sqrt((hz_to_bark(f1a) - hz_to_bark(f1b)) ** 2
                         + (hz_to_bark(f2a) - hz_to_bark(f2b)) ** 2))


def _avg_dist_to_rp(centroids: dict) -> float:
    dists = [_bark_dist(v["f1"], v["f2"], *RP[ph])
             for ph, v in centroids.items() if ph in RP]
    return float(np.mean(dists)) if dists else float("nan")


def _build_corpus_manifest(speaker: str, max_clips: int) -> Path:
    """Build a temporary manifest JSON for a corpus speaker from training JSONL."""
    out_path = PROJECT_ROOT / f"tts_output/modern_rp_corpus/{speaker}_manifest.json"
    if out_path.exists():
        return out_path
    with open(TRAIN_JSONL) as fh:
        items = [json.loads(line) for line in fh]
    clips = [{"clip_id": Path(i["wav"]).stem,
              "path": i["wav"],
              "transcript": i["text"]}
             for i in items if i["speaker"] == speaker][:max_clips]
    out_path.write_text(json.dumps(clips, indent=2))
    print(f"  built manifest: {out_path.name} ({len(clips)} clips)")
    return out_path


def score_source(
    label: str,
    manifest_or_csv: Path,
    out_csv: Path,
    apply_rp_corrections: bool = True,
) -> dict | None:
    """Extract formants if needed, return centroids."""
    if manifest_or_csv.suffix == ".csv" and manifest_or_csv.exists():
        # Pre-existing formants CSV (e.g. owner) — use as-is
        print(f"  {label}: using existing {manifest_or_csv.name}")
        return build_centroids_from_formants(manifest_or_csv)

    if not manifest_or_csv.exists():
        print(f"  [SKIP] {label}: manifest not found at {manifest_or_csv}")
        return None

    if out_csv.exists():
        print(f"  {label}: reusing {out_csv.name}")
    else:
        print(f"  {label}: extracting formants (rp_corrections={apply_rp_corrections})…")
        extract_formants(manifest_or_csv, out_csv, label,
                         apply_rp_corrections=apply_rp_corrections)
    return build_centroids_from_formants(out_csv)


def main() -> int:
    sources: list[tuple[str, Path, Path, bool]] = [
        # (label, manifest_or_formants_csv, out_formants_csv, apply_rp_corrections)
        ("gen_base",
         PROJECT_ROOT / "tts_output/cross_eval_50/gen_base/manifest.json",
         PROJECT_ROOT / "tts_output/cross_eval_50/gen_base/formants_gen_base.csv",
         True),
        ("gen_lora_best",
         PROJECT_ROOT / "tts_output/cross_eval_50/gen_lora_best/manifest.json",
         PROJECT_ROOT / "tts_output/cross_eval_50/gen_lora_best/formants_gen_lora_best.csv",
         True),
        ("owner",
         PROJECT_ROOT / "tts_output/owner_cal_50/formants.csv",  # pre-existing, no corrections
         PROJECT_ROOT / "tts_output/owner_cal_50/formants.csv",  # not re-extracted
         False),
        ("fry",
         _build_corpus_manifest("fry", MAX_CORPUS_CLIPS),
         CORPUS_ROOT / "formants_fry.csv",
         True),
        ("lindsey",
         _build_corpus_manifest("lindsey", MAX_CORPUS_CLIPS),
         CORPUS_ROOT / "formants_lindsey.csv",
         True),
        ("bbc",
         _build_corpus_manifest("bbc", MAX_CORPUS_CLIPS),
         CORPUS_ROOT / "formants_bbc.csv",
         True),
        ("real_bc",
         PROJECT_ROOT / "tts_output/real_bc_corpus/manifest.json",
         PROJECT_ROOT / "tts_output/real_bc_corpus/formants_real_bc.csv",
         True),
    ]

    runs: list[tuple[str, dict]] = []
    for label, manifest_or_csv, out_csv, rp_flag in sources:
        cent = score_source(label, manifest_or_csv, out_csv, rp_flag)
        if cent is not None:
            runs.append((label, cent))

    if not runs:
        print("No sources scored.")
        return 1

    labels = [name for name, _ in runs]
    col = 14

    print("\n" + "=" * (36 + col * len(labels)))
    header = f"{'Vowel':<36}" + "".join(f"{name:>{col}}" for name in labels)
    print(header)
    print("-" * (36 + col * len(labels)))

    overall = {name: _avg_dist_to_rp(cent) for name, cent in runs}
    row = f"{'dist_to_RP  (overall, Bark)':<36}"
    row += "".join(f"{overall[name]:>{col}.3f}" for name in labels)
    print(row)
    print("-" * (36 + col * len(labels)))

    for ph, vowel_name in FOCUS_VOWELS:
        if ph not in RP:
            continue
        f1t, f2t = RP[ph]
        dists = {}
        for name, cent in runs:
            v = cent.get(ph)
            dists[name] = _bark_dist(v["f1"], v["f2"], f1t, f2t) if v else float("nan")
        best = min((d for d in dists.values() if not np.isnan(d)), default=float("nan"))
        row = f"  {vowel_name + ' (' + ph + ')':<34}"
        for name in labels:
            dist = dists[name]
            marker = "*" if (not np.isnan(dist) and abs(dist - best) < 0.001) else " "
            row += f"{dist:>{col - 1}.3f}{marker}"
        print(row)

    print("=" * (36 + col * len(labels)))

    ranked = sorted(overall.items(), key=lambda x: x[1])
    print("\nRanked by dist_to_RP (lower = closer to modern RP):")
    for rank, (name, dist) in enumerate(ranked, 1):
        print(f"  {rank}. {name:<20} {dist:.3f} Bark")

    print("\nRaw BATH (ɑː)  [RP target F1=518 F2=1215]:")
    for name, cent in runs:
        v = cent.get("ɑː")
        if v:
            print(f"  {name:<20} F1={v['f1']:.0f}  F2={v['f2']:.0f}  n={v.get('n','?')}")
        else:
            print(f"  {name:<20} no ɑː tokens")

    return 0


if __name__ == "__main__":
    sys.exit(main())
