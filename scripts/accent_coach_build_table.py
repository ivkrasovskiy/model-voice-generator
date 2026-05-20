"""Aggregate per-token formant CSVs into the 5-source comparison table.

Usage:
    .venv/bin/python scripts/accent_coach_build_table.py \\
        --modern-rp-csv tts_output/modern_rp_corpus/formants.csv \\
        --real-bc-csv   tts_output/real_bc_corpus/formants.csv \\
        --synth-bc-csv  tts_output/bc_cal_50/formants.csv \\
        --owner-csv     tts_output/owner_cal_50/formants.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

PHONEMES = ["iː", "ɪ", "ʊ", "uː", "eɪ", "ɛ", "æ", "ʌ", "ɔː"]
DEFAULT_OUT = "docs/accent_coach_phase0_5_table.csv"


def build_table(
    modern_rp_csv: Path | None,
    real_bc_csv: Path | None,
    synth_bc_csv: Path | None,
    owner_csv: Path | None,
    out_csv: Path,
) -> None:
    from accent_coach.reference.rp_norms import RP_VOWEL_F1_F2_MALE

    rows = []

    # Deterding 1997 — from rp_norms.py (male norms)
    for ph, (f1, f2) in RP_VOWEL_F1_F2_MALE.items():
        if ph in PHONEMES:
            rows.append({
                "phoneme": ph, "source": "deterding_rp",
                "F1_mean": f1, "F2_mean": f2,
                "F1_std": None, "F2_std": None, "n_tokens": None,
            })

    # Per-source aggregates from CSVs
    csv_inputs: list[tuple[Path | None, str]] = [
        (modern_rp_csv, "from_manifest"),  # source_label comes from rows
        (real_bc_csv, "from_manifest"),
        (synth_bc_csv, "from_manifest"),
        (owner_csv, "from_manifest"),
    ]

    modern_rp_frames = []

    for csv_path, _label in csv_inputs:
        if csv_path is None or not csv_path.exists():
            print(f"  SKIP (not found): {csv_path}", flush=True)
            continue

        df = pd.read_csv(csv_path)
        df = df[df.phoneme.isin(PHONEMES)]

        for source_label, sub in df.groupby("source_label"):
            for ph in PHONEMES:
                ph_sub = sub[sub.phoneme == ph]
                if len(ph_sub) == 0:
                    continue
                rows.append({
                    "phoneme": ph,
                    "source": str(source_label),
                    "F1_mean": round(ph_sub.F1.mean(), 1),
                    "F2_mean": round(ph_sub.F2.mean(), 1),
                    "F1_std": round(ph_sub.F1.std(), 1) if len(ph_sub) > 1 else None,
                    "F2_std": round(ph_sub.F2.std(), 1) if len(ph_sub) > 1 else None,
                    "n_tokens": len(ph_sub),
                })
            # Collect modern_rp_* rows for combined pooling
            if str(source_label).startswith("modern_rp"):
                modern_rp_frames.append(sub)

    # Combined "modern_rp" row (BBC + Lindsey pooled)
    if modern_rp_frames:
        combined = pd.concat(modern_rp_frames, ignore_index=True)
        for ph in PHONEMES:
            ph_sub = combined[combined.phoneme == ph]
            if len(ph_sub) == 0:
                continue
            rows.append({
                "phoneme": ph,
                "source": "modern_rp",
                "F1_mean": round(ph_sub.F1.mean(), 1),
                "F2_mean": round(ph_sub.F2.mean(), 1),
                "F1_std": round(ph_sub.F1.std(), 1) if len(ph_sub) > 1 else None,
                "F2_std": round(ph_sub.F2.std(), 1) if len(ph_sub) > 1 else None,
                "n_tokens": len(ph_sub),
            })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows, columns=["phoneme", "source", "F1_mean", "F2_mean",
                                          "F1_std", "F2_std", "n_tokens"])
    out_df.to_csv(out_csv, index=False)

    print(f"\nTable written: {out_csv}  ({len(out_df)} rows)")
    print(out_df.pivot_table(values="F1_mean", index="phoneme", columns="source",
                              aggfunc="first").to_string())


def main() -> int:
    parser = argparse.ArgumentParser(description="Build phoneme comparison table")
    parser.add_argument("--modern-rp-csv", type=Path, default=None)
    parser.add_argument("--real-bc-csv", type=Path, default=None)
    parser.add_argument("--synth-bc-csv", type=Path, default=None)
    parser.add_argument("--owner-csv", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    def _resolve(p: Path | None) -> Path | None:
        if p is None:
            return None
        return p if p.is_absolute() else PROJECT_ROOT / p

    build_table(
        modern_rp_csv=_resolve(args.modern_rp_csv),
        real_bc_csv=_resolve(args.real_bc_csv),
        synth_bc_csv=_resolve(args.synth_bc_csv),
        owner_csv=_resolve(args.owner_csv),
        out_csv=_resolve(args.out),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
