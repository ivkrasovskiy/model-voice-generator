"""Compare base vs LoRA adapters dist-to-RP on BATH-heavy phrases.

Extracts vowel formants (with RP corrections: coda-R strip, BATH relabel,
LOT relabel) and computes Bark-distance to modern RP norms (male).

Usage:
  .venv/bin/python scripts/score_bath_rp.py \
      --runs base:bath_base ep1:bath_lora ep2_300:bath_step300
"""
from __future__ import annotations

import argparse
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

# BC male F0 ~110 Hz → male modern RP norms
RP = get_rp_norms(110.0)

# Vowels to highlight in the per-row table
FOCUS_VOWELS = [
    ("ɑː", "BATH/PALM"),
    ("æ",  "TRAP"),
    ("ɛ",  "DRESS"),
    ("əʊ", "GOAT"),
    ("ə",  "SCHWA"),
    ("ɔː", "THOUGHT"),
    ("ɪ",  "KIT"),
]


def _bark_dist(f1a: float, f2a: float, f1b: float, f2b: float) -> float:
    return float(np.sqrt((hz_to_bark(f1a) - hz_to_bark(f1b)) ** 2
                         + (hz_to_bark(f2a) - hz_to_bark(f2b)) ** 2))


def _avg_dist_to_rp(centroids: dict[str, dict]) -> float:
    dists = [
        _bark_dist(v["f1"], v["f2"], *RP[ph])
        for ph, v in centroids.items() if ph in RP
    ]
    return float(np.mean(dists)) if dists else float("nan")


def score_run(label: str, out_dir: Path) -> dict[str, dict] | None:
    manifest = out_dir / "manifest.json"
    if not manifest.exists():
        print(f"  [SKIP] {label}: no manifest at {out_dir}")
        return None
    formants_csv = out_dir / f"formants_{label}.csv"
    if formants_csv.exists():
        print(f"  [cache] {label}: reusing {formants_csv.name}")
    else:
        print(f"  {label}: extracting formants…")
        extract_formants(manifest, formants_csv, source_label=label)
    return build_centroids_from_formants(formants_csv)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs", nargs="+",
        default=["base:bath_base", "ep1_final:bath_lora", "ep2_step300:bath_step300"],
        help="label:subdir pairs under tts_output/accent_coach/phase0_14/lora/",
    )
    args = parser.parse_args()

    lora_root = PROJECT_ROOT / "tts_output/accent_coach/phase0_14/lora"
    runs: list[tuple[str, dict]] = []
    for spec in args.runs:
        label, subdir = spec.split(":", 1)
        cent = score_run(label, lora_root / subdir)
        if cent is not None:
            runs.append((label, cent))

    if not runs:
        print("No runs scored.")
        return 1

    labels = [r[0] for r in runs]
    col = 10

    print("\n" + "=" * (36 + col * len(labels)))
    print(f"{'Metric':<36}" + "".join(f"{lbl:>{col}}" for l in labels))
    print("-" * (36 + col * len(labels)))

    # Overall dist-to-RP
    overall = {lbl: _avg_dist_to_rp(c) for lbl, c in runs}
    row = f"{'dist_to_RP  (all vowels, Bark)':<36}"
    row += "".join(f"{overall[lbl]:>{col}.3f}" for l in labels)
    print(row)

    # Per-vowel rows
    for ph, name in FOCUS_VOWELS:
        if ph not in RP:
            continue
        f1t, f2t = RP[ph]
        dists = {}
        for lbl, cent in runs:
            v = cent.get(ph)
            dists[lbl] = _bark_dist(v["f1"], v["f2"], f1t, f2t) if v else float("nan")
        best = min((d for d in dists.values() if not np.isnan(d)), default=float("nan"))
        row = f"  {name+' ('+ph+')':<34}"
        for lbl in labels:
            d = dists[lbl]
            marker = " *" if (not np.isnan(d) and abs(d - best) < 0.001) else "  "
            row += f"{d:>{col-2}.3f}{marker}"
        print(row)

    print("=" * (36 + col * len(labels)))

    # Winner
    best_label = min(overall, key=lambda l: overall[lbl])
    print(f"\nClosest to modern RP: {best_label} (dist={overall[best_label]:.3f} Bark)")

    # Raw BATH centroids
    print("\nRaw BATH (ɑː) centroids  [RP target F1=518 F2=1215]:")
    for lbl, cent in runs:
        v = cent.get("ɑː")
        if v:
            print(f"  {l:<20} F1={v['f1']:.0f}  F2={v['f2']:.0f}  n={v.get('n', '?')}")
        else:
            print(f"  {l:<20} ɑː not found")

    print("\nRaw TRAP (æ) centroids   [RP target F1=545 F2=1496]:")
    for lbl, cent in runs:
        v = cent.get("æ")
        if v:
            print(f"  {l:<20} F1={v['f1']:.0f}  F2={v['f2']:.0f}  n={v.get('n', '?')}")
        else:
            print(f"  {l:<20} æ not found")

    return 0


if __name__ == "__main__":
    sys.exit(main())
