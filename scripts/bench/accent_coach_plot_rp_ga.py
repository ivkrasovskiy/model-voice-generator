"""Phase 0.16 — RP vs GenAm scatter plot (projection + rhoticity).

Two axes that actually discriminate modern RP from GenAm:

  X — vowel-space projection score (Bark), averaged over DISC_VOWELS
        Each token is projected onto the RP→GA axis in 2D Bark space.
        score = 2 * dot(bark(token)-bark(rp), bark(ga)-bark(rp))
                    / |bark(ga)-bark(rp)|^2  − 1
        -1 = exactly at RP centroid; +1 = exactly at GA centroid; 0 = equidistant.
        Using signed projection (not abs distance) removes orthogonal noise that
        caused all points to cluster on y=x in the previous absolute-distance plot.

  Y — rhoticity: mean F3 Bark for NURSE (ɜː) + pre-/r/ vowels per clip
        High = non-rhotic (RP, ~14.3 Bark); Low = rhotic (GA, ~12.4 Bark).
        F3 data already in all formant CSVs; pre-/r/ detected via next_phoneme col.

Expected clusters:
  RP speakers  → negative X (RP vowels) + high Y (non-rhotic) → top-left
  GA speakers  → positive X (GA vowels) + low  Y (rhotic)     → bottom-right
  Owner (L2)   → near-zero X            + high Y (non-rhotic)
  gen_base     → negative X             + high Y
  clone_hub    → positive X             + low  Y

Two figures:
  Fig 1 — real recordings (fry, lindsey, huberman, harris, sapolsky, owner)
  Fig 2 — generated clips (gen_base, clone_hub) + real-speaker ellipses

Usage:
  .venv/bin/python scripts/accent_coach_plot_rp_ga.py
  .venv/bin/python scripts/accent_coach_plot_rp_ga.py --out docs/accent_rp_ga
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from accent_coach.diagnostics.bark_distance import hz_to_bark
from accent_coach.reference.genam_norms import get_genam_norms
from accent_coach.reference.rp_norms import get_rp_norms

# Vowels where RP and GA centroids differ by > 0.7 Bark — real accent discriminators.
DISC_VOWELS = {"əʊ", "ɔː", "uː", "aʊ", "æ", "iː"}
VOWEL_NAMES = {"əʊ": "GOAT", "ɔː": "THOUGHT", "uː": "GOOSE",
               "aʊ": "MOUTH", "æ": "TRAP", "iː": "FLEECE"}

RP = get_rp_norms(110.0)
GA = get_genam_norms(110.0)

SPEAKER_STYLE = {
    "fry":       {"color": "#1565C0", "marker": "o", "label": "Fry (RP)",                  "zorder": 4},
    "lindsey":   {"color": "#42A5F5", "marker": "s", "label": "Lindsey (RP)",              "zorder": 4},
    "huberman":  {"color": "#E65100", "marker": "o", "label": "Huberman (GA)",             "zorder": 4},
    "harris":    {"color": "#FF8F00", "marker": "s", "label": "Harris (GA)",               "zorder": 4},
    "sapolsky":  {"color": "#FFCA28", "marker": "^", "label": "Sapolsky (GA)",             "zorder": 3},
    "owner":     {"color": "#6A1B9A", "marker": "D", "label": "Owner (L2)",                "zorder": 5},
    "gen_base":  {"color": "#2E7D32", "marker": "o", "label": "gen_base (BC, RP ref)",     "zorder": 4},
    "clone_hub": {"color": "#F57F17", "marker": "^", "label": "clone_hub (Huberman voice, GA ref)", "zorder": 4},
}


def _project_score(f1: float, f2: float, rp: tuple, ga: tuple) -> float:
    """Signed projection of token onto RP→GA axis, normalised to [-1, +1].

    -1 = exactly at RP centroid, +1 = exactly at GA centroid, 0 = equidistant.
    """
    b = np.array([hz_to_bark(f1), hz_to_bark(f2)])
    brp = np.array([hz_to_bark(rp[0]), hz_to_bark(rp[1])])
    bga = np.array([hz_to_bark(ga[0]), hz_to_bark(ga[1])])
    d = bga - brp
    norm_sq = float(np.dot(d, d))
    if norm_sq < 1e-9:
        return 0.0
    proj = float(np.dot(b - brp, d)) / norm_sq
    return 2.0 * proj - 1.0


def _per_clip_projection(csv_path: Path, spk_prefix: str | None = None,
                         ) -> dict[str, float]:
    """Per clip_id → mean projection score over DISC_VOWELS (min 2 tokens)."""
    by_clip: dict[str, list[float]] = defaultdict(list)
    with csv_path.open() as fh:
        for r in csv.DictReader(fh):
            ph = r.get("phoneme", "")
            if ph not in DISC_VOWELS or not r.get("F1"):
                continue
            cid = r["clip_id"]
            if spk_prefix and not cid.startswith(spk_prefix):
                continue
            try:
                f1, f2 = float(r["F1"]), float(r["F2"])
                dur = float(r.get("duration_s", 0))
            except ValueError:
                continue
            if dur < 0.04 or np.isnan(f1) or np.isnan(f2):
                continue
            if ph not in RP or ph not in GA:
                continue
            by_clip[cid].append(_project_score(f1, f2, RP[ph], GA[ph]))
    return {cid: float(np.mean(scores))
            for cid, scores in by_clip.items() if len(scores) >= 2}


def _per_clip_rhoticity(csv_path: Path, spk_prefix: str | None = None,
                        ) -> dict[str, float]:
    """Per clip_id → mean F3 Bark for NURSE (ɜː) + pre-/r/ vowels (min 1 token).

    High value = non-rhotic RP (~14.3 Bark); Low = rhotic GA (~12.4 Bark).
    Pre-/r/ context detected via next_phoneme column (absent in old CSVs → NURSE only).
    """
    by_clip: dict[str, list[float]] = defaultdict(list)
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        has_next = "next_phoneme" in (reader.fieldnames or [])
        for r in reader:
            ph = r.get("phoneme", "")
            is_nurse = ph == "ɜː"
            is_pre_r = has_next and r.get("next_phoneme", "") == "r"
            if not (is_nurse or is_pre_r):
                continue
            cid = r["clip_id"]
            if spk_prefix and not cid.startswith(spk_prefix):
                continue
            f3_raw = r.get("F3", "")
            if not f3_raw:
                continue
            try:
                f3 = float(f3_raw)
                dur = float(r.get("duration_s", 0))
            except ValueError:
                continue
            if np.isnan(f3) or f3 < 500 or dur < 0.04:
                continue
            by_clip[cid].append(hz_to_bark(f3))
    return {cid: float(np.mean(vals)) for cid, vals in by_clip.items() if vals}


def _merge_xy(proj: dict[str, float], rhot: dict[str, float],
              ) -> tuple[list[float], list[float]]:
    """Merge projection (X) and rhoticity (Y) dicts; keep only clips with both."""
    common = sorted(set(proj) & set(rhot))
    return [proj[c] for c in common], [rhot[c] for c in common]


def _confidence_ellipse(xs, ys, ax, color, n_std=1.5, alpha=0.12):
    if len(xs) < 3:
        return
    cov = np.cov(xs, ys)
    evals, evecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(*evecs[:, 1][::-1]))
    w, h = 2 * n_std * np.sqrt(np.maximum(evals, 0))
    ell = Ellipse(xy=(np.mean(xs), np.mean(ys)), width=w, height=h, angle=angle,
                  facecolor=color, alpha=alpha, edgecolor=color, linewidth=1.5,
                  linestyle="--")
    ax.add_patch(ell)


def _add_quadrant_labels(ax, xlim, ylim):
    kw = dict(fontsize=7, color="#aaaaaa", ha="center", va="center", style="italic")
    mx = (xlim[0] + xlim[1]) / 2
    my = (ylim[0] + ylim[1]) / 2
    ax.text(xlim[0] * 0.55 + mx * 0.45, ylim[1] * 0.75 + my * 0.25,
            "RP vowels\nnon-rhotic", **kw)
    ax.text(xlim[1] * 0.55 + mx * 0.45, ylim[0] * 0.75 + my * 0.25,
            "GA vowels\nrhotic", **kw)
    ax.text(xlim[1] * 0.55 + mx * 0.45, ylim[1] * 0.75 + my * 0.25,
            "GA vowels\nnon-rhotic", **kw)
    ax.text(xlim[0] * 0.55 + mx * 0.45, ylim[0] * 0.75 + my * 0.25,
            "RP vowels\nrhotic", **kw)


def _scatter_speaker(ax, spk, xs, ys, alpha=0.55, size=28):
    if not xs:
        return
    s = SPEAKER_STYLE[spk]
    ax.scatter(xs, ys, c=s["color"], marker=s["marker"], s=size,
               alpha=alpha, zorder=s["zorder"], linewidths=0.3, edgecolors="white")
    _confidence_ellipse(np.array(xs), np.array(ys), ax, s["color"])
    mx, my = float(np.mean(xs)), float(np.mean(ys))
    ax.annotate(s["label"], xy=(mx, my), xytext=(4, 4),
                textcoords="offset points", fontsize=7.5, color=s["color"],
                fontweight="bold", zorder=10)


def make_plots(out_prefix: str) -> None:
    lecture_csv = PROJECT_ROOT / "tts_output/genam_lecture_corpus/formants_genam_lecture.csv"
    rp_fry    = PROJECT_ROOT / "tts_output/modern_rp_corpus/formants_fry.csv"
    rp_lin    = PROJECT_ROOT / "tts_output/modern_rp_corpus/formants_lindsey.csv"
    owner_csv = PROJECT_ROOT / "tts_output/owner_cal_50/formants.csv"
    genbase_csv = PROJECT_ROOT / "tts_output/cross_eval_50/gen_base/formants_gen_base_raw.csv"
    clone_csv = PROJECT_ROOT / "tts_output/accent_coach/phase0_16/clone_huberman/formants_none.csv"

    def _load(csv_path, prefix=None):
        proj = _per_clip_projection(csv_path, prefix)
        rhot = _per_clip_rhoticity(csv_path, prefix)
        return _merge_xy(proj, rhot)

    data: dict[str, tuple[list[float], list[float]]] = {
        "fry":      _load(rp_fry),
        "lindsey":  _load(rp_lin),
        "huberman": _load(lecture_csv, "huberman"),
        "harris":   _load(lecture_csv, "harris"),
        "sapolsky": _load(lecture_csv, "sapolsky"),
        "owner":    _load(owner_csv),
        "gen_base": _load(genbase_csv),
    }
    if clone_csv.exists():
        data["clone_hub"] = _load(clone_csv)

    # Axis limits
    all_x = [x for xs, _ in data.values() for x in xs]
    all_y = [y for _, ys in data.values() for y in ys]
    pad_x, pad_y = 0.10, 0.25
    xlim = (min(all_x) - pad_x, max(all_x) + pad_x)
    ylim = (min(all_y) - pad_y, max(all_y) + pad_y)

    xlabel = ("Vowel projection score  (← RP side  |  GA side →)\n"
              "Normalised: −1 = at RP centroid, 0 = equidistant, +1 = at GA centroid")
    ylabel = ("F3 Bark — NURSE + pre-/r/ vowels\n"
              "↑ Non-rhotic (RP ~14.3 Bark)       ↓ Rhotic (GA ~12.4 Bark)")

    # ── Figure 1: real recordings ──────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(8, 7))
    for spk in ["fry", "lindsey", "huberman", "harris", "sapolsky", "owner"]:
        xs, ys = data.get(spk, ([], []))
        _scatter_speaker(ax1, spk, xs, ys)

    ax1.axvline(0, color="#dddddd", lw=0.8, ls=":", zorder=1)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    _add_quadrant_labels(ax1, xlim, ylim)
    ax1.set_xlabel(xlabel, fontsize=10)
    ax1.set_ylabel(ylabel, fontsize=10)
    ax1.set_title("Real recordings — each point = one clip\n"
                  f"X: vowel projection ({', '.join(sorted(VOWEL_NAMES.values()))})"
                  f"  |  Y: rhoticity F3",
                  fontsize=10)
    ax1.legend(handles=[mpatches.Patch(color=SPEAKER_STYLE[s]["color"],
                                       label=SPEAKER_STYLE[s]["label"])
                        for s in ["fry", "lindsey", "huberman", "harris", "sapolsky", "owner"]
                        if data.get(s) and data[s][0]],
               fontsize=8, loc="upper right")
    fig1.tight_layout()
    p1 = f"{out_prefix}_real.png"
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    print(f"Saved: {p1}")

    # ── Figure 2: generated + real ellipses ───────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    for spk in ["fry", "lindsey", "huberman", "harris", "owner"]:
        xs, ys = data.get(spk, ([], []))
        if not xs:
            continue
        _confidence_ellipse(np.array(xs), np.array(ys), ax2,
                            SPEAKER_STYLE[spk]["color"], n_std=1.5, alpha=0.07)
        ax2.annotate(SPEAKER_STYLE[spk]["label"] + " zone",
                     xy=(float(np.mean(xs)), float(np.mean(ys))),
                     fontsize=7, color=SPEAKER_STYLE[spk]["color"],
                     alpha=0.6, ha="center", style="italic")

    for spk in ["gen_base", "clone_hub"]:
        xs, ys = data.get(spk, ([], []))
        _scatter_speaker(ax2, spk, xs, ys, alpha=0.75, size=40)

    ax2.axvline(0, color="#dddddd", lw=0.8, ls=":", zorder=1)
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    _add_quadrant_labels(ax2, xlim, ylim)
    ax2.set_xlabel(xlabel, fontsize=10)
    ax2.set_ylabel(ylabel, fontsize=10)
    ax2.set_title("Generated clips — each point = one clip\n"
                  "Shaded zones = real-speaker reference ellipses",
                  fontsize=10)
    ax2.legend(handles=[mpatches.Patch(color=SPEAKER_STYLE[s]["color"],
                                       label=SPEAKER_STYLE[s]["label"])
                        for s in ["gen_base", "clone_hub"]
                        if data.get(s) and data[s][0]],
               fontsize=8, loc="upper right")
    fig2.tight_layout()
    p2 = f"{out_prefix}_generated.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    print(f"Saved: {p2}")

    # Summary stats
    print("\nPer-speaker means and clip counts:")
    print(f"  {'speaker':12}  {'n':>4}  {'proj_X':>7}  {'F3_Y':>7}  side")
    for spk in ["fry", "lindsey", "huberman", "harris", "sapolsky", "owner",
                "gen_base", "clone_hub"]:
        xs, ys = data.get(spk, ([], []))
        if not xs:
            continue
        side = "GA" if np.mean(xs) > 0 else "RP"
        rhotic = "rhotic" if np.mean(ys) < 13.3 else "non-rhotic"
        print(f"  {spk:12}  {len(xs):4}  {np.mean(xs):+.3f}±{np.std(xs):.3f}"
              f"  {np.mean(ys):.2f}±{np.std(ys):.2f}  {side}/{rhotic}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/accent_coach_rp_ga_scatter")
    args = ap.parse_args()
    make_plots(str(PROJECT_ROOT / args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
