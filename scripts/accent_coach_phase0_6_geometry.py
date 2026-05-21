"""Phase A: Vowel-space geometry metrics for Phase 0.6."""

import pathlib
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull

REPO = pathlib.Path(__file__).parent.parent
CSV_IN = REPO / "docs" / "accent_coach_phase0_5_table.csv"
CSV_OUT = REPO / "docs" / "accent_coach_phase0_6_geometry.csv"

PHONEMES = ["iː", "ɪ", "ɛ", "æ", "ɔː", "ʊ", "uː", "ʌ", "eɪ"]

AGGREGATED = {
    "modern_rp_full": ["modern_rp_bbc", "modern_rp_fry", "modern_rp_lindsey"],
    "modern_rp_no_bbc": ["modern_rp_fry", "modern_rp_lindsey"],
    "native_wide": ["modern_rp_fry", "modern_rp_lindsey", "real_bc", "synth_bc"],
}

NATIVE_SOURCES = ["modern_rp_fry", "modern_rp_lindsey", "real_bc", "synth_bc"]


def convex_hull_area(f1: np.ndarray, f2: np.ndarray) -> float:
    pts = np.column_stack([f1, f2])
    if len(pts) < 3:
        return float("nan")
    try:
        return ConvexHull(pts).volume  # 2D: volume == area
    except Exception:
        return float("nan")


def geometry_for_source(rows: pd.DataFrame) -> dict:
    rows = rows[rows["phoneme"].isin(PHONEMES)].copy()
    f1 = rows["F1_mean"].values
    f2 = rows["F2_mean"].values
    return {
        "n_vowels": len(rows),
        "f1_range_hz": float(np.max(f1) - np.min(f1)),
        "f2_range_hz": float(np.max(f2) - np.min(f2)),
        "area_hz2": convex_hull_area(f1, f2),
    }


def main():
    df = pd.read_csv(CSV_IN)
    df = df[df["phoneme"].isin(PHONEMES)]

    # Build aggregated baseline rows (equal-weight per speaker)
    agg_frames = []
    for agg_name, members in AGGREGATED.items():
        subset = df[df["source"].isin(members)]
        per_phoneme = (
            subset.groupby("phoneme")[["F1_mean", "F2_mean"]]
            .mean()
            .reset_index()
        )
        per_phoneme["source"] = agg_name
        agg_frames.append(per_phoneme)
    agg_df = pd.concat(agg_frames, ignore_index=True)
    full_df = pd.concat([df, agg_df], ignore_index=True)

    all_sources = list(df["source"].unique()) + list(AGGREGATED.keys())
    rows_out = []
    for src in all_sources:
        subset = full_df[full_df["source"] == src]
        g = geometry_for_source(subset)
        g["source"] = src
        rows_out.append(g)

    result = pd.DataFrame(rows_out, columns=["source", "n_vowels", "f1_range_hz", "f2_range_hz", "area_hz2"])

    native_areas = result[result["source"].isin(NATIVE_SOURCES)]["area_hz2"]
    native_median = native_areas.median()
    result["articulation_idx"] = result["area_hz2"] / native_median

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(CSV_OUT, index=False, float_format="%.4f")
    print(result.to_string(index=False))
    print(f"\nWritten: {CSV_OUT}")
    print(f"Row count: {len(result)}")
    owner_ai = result[result["source"] == "owner"]["articulation_idx"].values[0]
    print(f"articulation_idx(owner) = {owner_ai:.4f}  {'PASS (<1.0)' if owner_ai < 1.0 else 'FAIL'}")


if __name__ == "__main__":
    main()
