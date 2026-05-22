"""Phase 0.9 vowel space plots — Bark-normalised F1/F2.

Two figures:
  Track A  — A1-A6 synth_BC variants vs RP cluster vs owner
  Track B  — B1-B2 synth_BC variants vs RP cluster vs owner

Design:
  - Bark scale (H4 normalisation from Phase 0.8)
  - RP cluster shown as a shaded convex hull + individual speaker points (muted)
  - modern_rp centroid has large bold IPA phoneme labels (single labelling pass)
  - real_BC, owner, synth_BC variants shown as clean per-phoneme rings
  - Only 10 core monophthongs (removes schwa / diphthongs clutter)
  - Standard phonetics orientation: F2 inverted left→right, F1 inverted top→bottom

Usage:
    uv run python scripts/accent_coach_phase0_9_vowel_plot.py
    uv run python scripts/accent_coach_phase0_9_vowel_plot.py --out-dir docs/img
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from accent_coach.diagnostics.bark_distance import bark_transform
from accent_coach.reference.rp_norms import RP_VOWEL_F1_F2_MALE_LEGACY

BASELINE_CENTROIDS = PROJECT_ROOT / "tts_output/accent_coach/bench/phase0_7/speaker_centroids.json"
VARIANTS_DIR = PROJECT_ROOT / "tts_output/accent_coach/phase0_9/variants"

# Only core monophthongs — enough diagnostic power, much less clutter
CORE_VOWELS = ["iː", "ɪ", "ɛ", "æ", "ɑː", "ɔː", "ʊ", "uː", "ʌ", "ɜː"]

RP_SPEAKERS = ["fry", "lindsey", "bbc_male", "real_BC"]

TRACK_A_VARIANTS = ["A1", "A2", "A3", "A4", "A5", "A6"]
TRACK_B_VARIANTS = ["B1", "B2"]

# Colors: synth_BC variants
VARIANT_COLORS = {
    "A1": ("#999999", "A1 baseline (74.6)"),
    "A2": ("#ff7f0e", "A2"),
    "A3": ("#ffc473", "A3"),
    "A4": ("#d62728", "A4 ★"),
    "A5": ("#9467bd", "A5"),
    "A6": ("#8c564b", "A6"),
    "B1": ("#e377c2", "B1 (Fry ref)"),
    "B2": ("#7f7f7f", "B2 (Lindsey ref)"),
}


def load_baseline() -> dict:
    with BASELINE_CENTROIDS.open() as f:
        raw = json.load(f)
    for phoneme, f1f2 in RP_VOWEL_F1_F2_MALE_LEGACY.items():
        raw.setdefault(phoneme, {})["deterding"] = {"f1": f1f2[0], "f2": f1f2[1], "n": 0}
    return raw


def bark_xy(bark: dict, spk: str, vowels: list[str] | None = None) -> dict[str, tuple[float, float]]:
    """Return {phoneme: (F2_bark, F1_bark)} for a speaker, filtered to vowel set."""
    target = set(vowels) if vowels else None
    pts: dict[str, tuple[float, float]] = {}
    for ph, spks in bark.items():
        if target and ph not in target:
            continue
        v = spks.get(spk)
        if v and v.get("f1", 0) > 0 and v.get("f2", 0) > 0:
            pts[ph] = (v["f2"], v["f1"])
    return pts


def load_variant_synth_bc(variant_id: str) -> dict | None:
    p = VARIANTS_DIR / variant_id / "centroids.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    result = {ph: spks["synth_BC"] for ph, spks in data.items() if spks.get("synth_BC")}
    return result or None


def load_variant_score(variant_id: str) -> float | None:
    p = VARIANTS_DIR / variant_id / "bark_scores.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["scores"]["synth_BC"]["composite"]


def convex_hull_patch(pts_list: list[dict[str, tuple[float, float]]],
                      vowels: list[str]) -> tuple[list, list] | None:
    """Collect all F2/F1 points across a speaker list and return hull vertices."""
    import numpy as np
    from scipy.spatial import ConvexHull

    all_pts: list[tuple[float, float]] = []
    for pts in pts_list:
        for ph in vowels:
            if ph in pts:
                all_pts.append(pts[ph])
    if len(all_pts) < 3:
        return None
    arr = np.array(all_pts)
    try:
        hull = ConvexHull(arr)
        hull_pts = arr[hull.vertices]
        # Close the hull
        xs = list(hull_pts[:, 0]) + [hull_pts[0, 0]]
        ys = list(hull_pts[:, 1]) + [hull_pts[0, 1]]
        return xs, ys
    except Exception:
        return None


def _label_pts(ax, pts: dict[str, tuple], color: str, fontsize: int,
               dx: int, dy: int, bold: bool = False) -> None:
    """Annotate each phoneme point with its IPA symbol, offset by (dx, dy) points."""
    for ph, (x, y) in pts.items():
        ax.annotate(ph, (x, y), fontsize=fontsize,
                    fontweight="bold" if bold else "normal",
                    color=color, alpha=0.85,
                    xytext=(dx, dy), textcoords="offset points", zorder=10)


def make_plot(ax, title: str, bark_base: dict,
              synth_variants: list[str],
              variant_colors: dict[str, tuple[str, str]]) -> None:
    vowels = CORE_VOWELS

    # ── modern_rp mean — primary RP anchor ──
    pts_rp = bark_xy(bark_base, "modern_rp", vowels)
    ax.scatter([p[0] for p in pts_rp.values()], [p[1] for p in pts_rp.values()],
               color="#1f77b4", marker="*", s=220, alpha=0.95, zorder=8,
               label="Modern RP mean")
    _label_pts(ax, pts_rp, "#003d82", fontsize=11, dx=6, dy=5, bold=True)

    # ── Lindsey ──
    pts_lin = bark_xy(bark_base, "lindsey", vowels)
    ax.scatter([p[0] for p in pts_lin.values()], [p[1] for p in pts_lin.values()],
               color="#17becf", marker="s", s=65, alpha=0.85, zorder=6,
               label="Lindsey (RP)")
    _label_pts(ax, pts_lin, "#0e8a9e", fontsize=8, dx=-14, dy=5)

    # ── real_BC ──
    pts_rbc = bark_xy(bark_base, "real_BC", vowels)
    ax.scatter([p[0] for p in pts_rbc.values()], [p[1] for p in pts_rbc.values()],
               color="#2ca02c", marker="D", s=85, alpha=0.92, zorder=7,
               label="Real BC (89.8)")
    _label_pts(ax, pts_rbc, "#1a6e1a", fontsize=8, dx=5, dy=-12)

    # ── Owner ──
    pts_own = bark_xy(bark_base, "owner", vowels)
    ax.scatter([p[0] for p in pts_own.values()], [p[1] for p in pts_own.values()],
               color="#d62728", marker="v", s=110, alpha=0.95, zorder=8,
               label="Owner (56.5)")
    _label_pts(ax, pts_own, "#9e1c1c", fontsize=8, dx=-14, dy=-12)

    # Per-phoneme connector lines across fixed speakers (solid, visible)
    for ph in vowels:
        chain = [d[ph] for d in [pts_rp, pts_lin, pts_rbc, pts_own] if ph in d]
        if len(chain) >= 2:
            xs, ys = zip(*chain, strict=False)
            ax.plot(xs, ys, color="#888888", lw=1.2, alpha=0.45, zorder=1,
                    linestyle="--")

    # ── synth_BC variants — arrows pointing toward modern_rp ──
    for vid in synth_variants:
        synth_bc = load_variant_synth_bc(vid)
        if synth_bc is None:
            continue
        score = load_variant_score(vid)
        import copy
        c = copy.deepcopy(load_baseline())
        for ph, vals in synth_bc.items():
            c.setdefault(ph, {})["synth_BC"] = vals
        bark_v = bark_transform(c)
        pts_v = bark_xy(bark_v, "synth_BC", vowels)
        color, base_label = variant_colors[vid]
        label = f"{base_label} ({score:.1f})" if score else base_label
        ax.scatter([p[0] for p in pts_v.values()], [p[1] for p in pts_v.values()],
                   color=color, marker="o", s=65, alpha=0.88, zorder=7, label=label)
        _label_pts(ax, pts_v, color, fontsize=7, dx=4, dy=4)
        # Arrows from each synth_BC phoneme point → modern_rp
        for ph, (x, y) in pts_v.items():
            if ph in pts_rp:
                rx, ry = pts_rp[ph]
                ax.annotate("", xy=(rx, ry), xytext=(x, y),
                            arrowprops=dict(
                                arrowstyle="->",
                                color=color,
                                lw=1.1,
                                alpha=0.45,
                                shrinkA=4, shrinkB=5,
                            ), zorder=3)

    # ── Axes / decoration ──
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel("F2 (Bark)", fontsize=11)
    ax.set_ylabel("F1 (Bark)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.93)
    ax.grid(alpha=0.18)
    ax.text(0.02, 0.03,
            "★ Modern RP  ·  ◆ Real BC  ·  ▼ Owner  ·  ○ Synth BC (arrows → RP target)  ·  □ Lindsey",
            transform=ax.transAxes, fontsize=7.5, color="#555555")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 0.9 vowel space plots (Bark)")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "docs/img")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    base = load_baseline()
    bark_base = bark_transform(base)

    # Collect score summary for title
    done_a = [v for v in TRACK_A_VARIANTS if load_variant_synth_bc(v)]
    scores_a = [s for v in done_a if (s := load_variant_score(v))]
    range_str_a = f"{min(scores_a):.1f}–{max(scores_a):.1f}" if scores_a else "pending"

    done_b = [v for v in TRACK_B_VARIANTS if load_variant_synth_bc(v)]
    scores_b = [s for v in done_b if (s := load_variant_score(v))]
    range_str_b = f"{min(scores_b):.1f}–{max(scores_b):.1f}" if scores_b else "pending"

    # ── Track A ──
    fig, ax = plt.subplots(figsize=(11, 8))
    title_a = (f"Track A: synth_BC reference variants  [Bark F1/F2, 10 monophthongs]\n"
               f"real_BC=89.8  |  owner=56.5  |  synth_BC range: {range_str_a}")
    make_plot(ax, title_a, bark_base, TRACK_A_VARIANTS, VARIANT_COLORS)
    fig.tight_layout()
    path_a = out_dir / "phase0_9_vowel_space_trackA.png"
    fig.savefig(str(path_a), dpi=150)
    plt.close(fig)
    print(f"Saved: {path_a}  ({len(done_a)}/{len(TRACK_A_VARIANTS)} variants)")

    # ── Track B ──
    fig, ax = plt.subplots(figsize=(11, 8))
    title_b = (f"Track B: RP ceiling (Fry/Lindsey reference)  [Bark F1/F2, 10 monophthongs]\n"
               f"real_BC=89.8  |  owner=56.5  |  synth_BC range: {range_str_b}")
    make_plot(ax, title_b, bark_base, TRACK_B_VARIANTS, VARIANT_COLORS)
    fig.tight_layout()
    path_b = out_dir / "phase0_9_vowel_space_trackB.png"
    fig.savefig(str(path_b), dpi=150)
    plt.close(fig)
    print(f"Saved: {path_b}  ({len(done_b)}/{len(TRACK_B_VARIANTS)} variants)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
