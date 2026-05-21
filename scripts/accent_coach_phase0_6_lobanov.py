"""Phase B: Lobanov normalisation, distance matrix, dendrogram, vowel-space plot."""

import pathlib
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

REPO = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from accent_coach.diagnostics.lobanov import lobanov_normalize

CSV_IN = REPO / "docs" / "accent_coach_phase0_5_table.csv"
CSV_LOBANOV = REPO / "docs" / "accent_coach_phase0_6_lobanov_table.csv"
CSV_DIST = REPO / "docs" / "accent_coach_phase0_6_lobanov_distances.csv"
IMG_DEND = REPO / "docs" / "img" / "accent_coach_phase0_6_dendrogram.png"
IMG_VOWEL = REPO / "docs" / "img" / "accent_coach_phase0_6_lobanov_vowel_space.png"

PHONEMES = ["iː", "ɪ", "ɛ", "æ", "ɔː", "ʊ", "uː", "ʌ", "eɪ"]

AGGREGATED = {
    "modern_rp_full": ["modern_rp_bbc", "modern_rp_fry", "modern_rp_lindsey"],
    "modern_rp_no_bbc": ["modern_rp_fry", "modern_rp_lindsey"],
    "native_wide": ["modern_rp_fry", "modern_rp_lindsey", "real_bc", "synth_bc"],
}

SOURCE_COLORS = {
    "deterding_rp": "#888888",
    "modern_rp_bbc": "#2196F3",
    "modern_rp_fry": "#4CAF50",
    "modern_rp_lindsey": "#009688",
    "real_bc": "#FF5722",
    "synth_bc": "#FF9800",
    "owner": "#E91E63",
    "modern_rp": "#1565C0",
    "modern_rp_full": "#0D47A1",
    "modern_rp_no_bbc": "#1976D2",
    "native_wide": "#6A1B9A",
}


def build_aggregated(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for agg_name, members in AGGREGATED.items():
        subset = df[df["source"].isin(members)]
        per_ph = subset.groupby("phoneme")[["F1_mean", "F2_mean"]].mean().reset_index()
        per_ph["source"] = agg_name
        frames.append(per_ph)
    return pd.concat(frames, ignore_index=True)


def source_to_vector(lob_df: pd.DataFrame, src: str) -> np.ndarray:
    """Flatten 9 phonemes × 2 formants into 18-D vector, ordered by PHONEMES list."""
    rows = lob_df[lob_df["source"] == src].set_index("phoneme")
    vec = []
    for ph in PHONEMES:
        vec.append(rows.loc[ph, "F1_lobanov"])
        vec.append(rows.loc[ph, "F2_lobanov"])
    return np.array(vec, dtype=float)


def main():
    df = pd.read_csv(CSV_IN)
    df = df[df["phoneme"].isin(PHONEMES)]

    agg_df = build_aggregated(df)
    full_df = pd.concat([df[["phoneme", "source", "F1_mean", "F2_mean"]], agg_df], ignore_index=True)

    lob_df = lobanov_normalize(full_df)

    CSV_LOBANOV.parent.mkdir(parents=True, exist_ok=True)
    lob_df.to_csv(CSV_LOBANOV, index=False, float_format="%.6f")
    print(f"Written: {CSV_LOBANOV}")

    # 18-D vectors per source
    all_sources = list(df["source"].unique()) + list(AGGREGATED.keys())
    vectors = {src: source_to_vector(lob_df, src) for src in all_sources}

    # Pairwise distance matrix
    n = len(all_sources)
    dist_mat = np.zeros((n, n))
    for i, s1 in enumerate(all_sources):
        for j, s2 in enumerate(all_sources):
            dist_mat[i, j] = np.linalg.norm(vectors[s1] - vectors[s2])

    dist_df = pd.DataFrame(dist_mat, index=all_sources, columns=all_sources)
    dist_df.to_csv(CSV_DIST, float_format="%.6f")
    print(f"Written: {CSV_DIST}")

    # Dendrogram
    condensed = squareform(dist_mat)
    Z = linkage(condensed, method="ward")
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(Z, labels=all_sources, ax=ax, leaf_rotation=45, leaf_font_size=9)
    ax.set_title("Ward dendrogram — Lobanov 18-D space")
    ax.set_ylabel("Distance")
    fig.tight_layout()
    IMG_DEND.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMG_DEND, dpi=150)
    plt.close(fig)
    print(f"Written: {IMG_DEND}")

    # Vowel-space plot (Lobanov F1 vs F2, all sources, vowel labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    for src in all_sources:
        rows = lob_df[lob_df["source"] == src].set_index("phoneme")
        color = SOURCE_COLORS.get(src, "#333333")
        linestyle = "--" if src in AGGREGATED else "-"
        ax.plot(rows.loc[PHONEMES, "F2_lobanov"], rows.loc[PHONEMES, "F1_lobanov"],
                "o", color=color, alpha=0.7, markersize=5)
        for ph in PHONEMES:
            ax.annotate(
                f"{ph}\n{src}",
                (rows.loc[ph, "F2_lobanov"], rows.loc[ph, "F1_lobanov"]),
                fontsize=5, alpha=0.6, color=color,
            )

    # Draw convex hulls for key groups
    from scipy.spatial import ConvexHull
    native_srcs = ["modern_rp_fry", "modern_rp_lindsey", "real_bc", "synth_bc"]
    for group, srcs, color in [
        ("natives", native_srcs, "#4CAF50"),
        ("owner", ["owner"], "#E91E63"),
        ("deterding", ["deterding_rp"], "#888888"),
    ]:
        pts = []
        for src in srcs:
            rows = lob_df[lob_df["source"] == src]
            pts.extend(zip(rows["F2_lobanov"], rows["F1_lobanov"]))
        pts = np.array(pts)
        if len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                for simplex in hull.simplices:
                    ax.plot(pts[simplex, 0], pts[simplex, 1], color=color, alpha=0.3, lw=1.5)
            except Exception:
                pass

    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_xlabel("F2 (Lobanov z-score)")
    ax.set_ylabel("F1 (Lobanov z-score)")
    ax.set_title("Lobanov-normalised vowel space — all sources")

    # Legend
    from matplotlib.lines import Line2D
    legend_els = [Line2D([0], [0], marker="o", color="w", markerfacecolor=SOURCE_COLORS.get(s, "#333"),
                         label=s, markersize=7) for s in all_sources]
    ax.legend(handles=legend_els, fontsize=7, loc="upper right", ncol=2)

    fig.tight_layout()
    fig.savefig(IMG_VOWEL, dpi=150)
    plt.close(fig)
    print(f"Written: {IMG_VOWEL}")

    # Print key distances for Phase C
    print("\n--- Key Lobanov distances ---")
    for baseline in ["modern_rp_full", "modern_rp_no_bbc", "native_wide"]:
        bi = all_sources.index(baseline)
        oi = all_sources.index("owner")
        di = all_sources.index("deterding_rp")
        d_owner = dist_mat[oi, bi]
        d_det = dist_mat[di, bi]
        verdict = "PASS" if d_owner > d_det else "FAIL"
        print(f"  {baseline}: d(owner)={d_owner:.4f}, d(deterding)={d_det:.4f}  V3={verdict}")


if __name__ == "__main__":
    main()
