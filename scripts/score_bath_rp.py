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


def _avg_dist_to_rp(centroids: dict) -> float:
    dists = [
        _bark_dist(v["f1"], v["f2"], *RP[ph])
        for ph, v in centroids.items() if ph in RP
    ]
    return float(np.mean(dists)) if dists else float("nan")


def score_run(label: str, out_dir: Path) -> dict | None:
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

    labels = [name for name, _ in runs]
    col = 12

    print("\n" + "=" * (36 + col * len(labels)))
    print(f"{'Metric':<36}" + "".join(f"{name:>{col}}" for name in labels))
    print("-" * (36 + col * len(labels)))

    overall = {name: _avg_dist_to_rp(cent) for name, cent in runs}
    row = f"{'dist_to_RP  (all vowels, Bark)':<36}"
    row += "".join(f"{overall[name]:>{col}.3f}" for name in labels)
    print(row)

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
            marker = " *" if (not np.isnan(dist) and abs(dist - best) < 0.001) else "  "
            row += f"{dist:>{col - 2}.3f}{marker}"
        print(row)

    print("=" * (36 + col * len(labels)))
    best_name = min(overall, key=lambda name: overall[name])
    print(f"\nClosest to modern RP: {best_name} (dist={overall[best_name]:.3f} Bark)")

    print("\nRaw BATH (ɑː) centroids  [RP target F1=518 F2=1215]:")
    for name, cent in runs:
        v = cent.get("ɑː")
        if v:
            print(f"  {name:<22} F1={v['f1']:.0f}  F2={v['f2']:.0f}  n={v.get('n', '?')}")
        else:
            print(f"  {name:<22} ɑː not found")

    print("\nRaw TRAP (æ) centroids   [RP target F1=545 F2=1496]:")
    for name, cent in runs:
        v = cent.get("æ")
        if v:
            print(f"  {name:<22} F1={v['f1']:.0f}  F2={v['f2']:.0f}  n={v.get('n', '?')}")
        else:
            print(f"  {name:<22} æ not found")

    return 0


if __name__ == "__main__":
    sys.exit(main())
