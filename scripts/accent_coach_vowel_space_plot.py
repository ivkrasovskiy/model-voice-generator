"""F1/F2 scatter plot — all sources overlaid on a single vowel space.

Usage:
    .venv/bin/python scripts/accent_coach_vowel_space_plot.py
    .venv/bin/python scripts/accent_coach_vowel_space_plot.py \\
        --table docs/accent_coach_phase0_5_table.csv \\
        --out   docs/img/accent_coach_phase0_5_vowel_space.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_TABLE = "docs/accent_coach_phase0_5_table.csv"
DEFAULT_OUT = "docs/img/accent_coach_phase0_5_vowel_space.png"

SOURCE_STYLE = {
    "deterding_rp": {"color": "black",     "marker": "o", "s": 100, "zorder": 5},
    "modern_rp":    {"color": "tab:blue",   "marker": "s", "s": 80,  "zorder": 4},
    "modern_rp_bbc":     {"color": "tab:blue",   "marker": "s", "s": 50,  "zorder": 3, "alpha": 0.4},
    "modern_rp_lindsey": {"color": "tab:cyan",   "marker": "s", "s": 50,  "zorder": 3, "alpha": 0.4},
    "real_bc":      {"color": "tab:green",  "marker": "D", "s": 80,  "zorder": 4},
    "synth_bc":     {"color": "tab:orange", "marker": "^", "s": 80,  "zorder": 4},
    "owner":        {"color": "tab:red",    "marker": "v", "s": 80,  "zorder": 4},
}

LABEL_ORDER = ["deterding_rp", "modern_rp", "real_bc", "synth_bc", "owner"]


def plot(table_csv: Path, out_png: Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(table_csv)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot sub-sources (lighter) first, then main sources on top
    for src, style in SOURCE_STYLE.items():
        sub = df[df.source == src]
        if sub.empty:
            continue
        kw = {k: v for k, v in style.items() if k != "alpha"}
        alpha = style.get("alpha", 0.85)
        ax.scatter(sub.F2_mean, sub.F1_mean, label=src, alpha=alpha, **kw)
        for _, row in sub.iterrows():
            ax.annotate(
                row.phoneme,
                (row.F2_mean, row.F1_mean),
                fontsize=10,
                xytext=(6, 4),
                textcoords="offset points",
                color=style["color"],
                alpha=alpha,
            )

    # Connect same phoneme across sources with thin grey lines
    main_sources = [s for s in LABEL_ORDER if s in df.source.values]
    for ph in df.phoneme.unique():
        pts = []
        for src in main_sources:
            row = df[(df.source == src) & (df.phoneme == ph)]
            if not row.empty:
                pts.append((float(row.F2_mean.iloc[0]), float(row.F1_mean.iloc[0])))
        if len(pts) >= 2:
            xs, ys = zip(*pts, strict=False)
            ax.plot(xs, ys, color="grey", lw=0.5, alpha=0.3, zorder=1)

    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel("F2 (Hz)", fontsize=12)
    ax.set_ylabel("F1 (Hz)", fontsize=12)
    ax.set_title(
        "Vowel space: Deterding RP vs Modern RP vs Real BC vs Synth BC vs Owner",
        fontsize=13,
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=150)
    plt.close(fig)
    print(f"Plot saved: {out_png}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Vowel space scatter plot")
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    table = args.table if args.table.is_absolute() else PROJECT_ROOT / args.table
    out_png = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out

    if not table.exists():
        print(f"ERROR: table not found: {table}", file=sys.stderr)
        return 1

    plot(table, out_png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
