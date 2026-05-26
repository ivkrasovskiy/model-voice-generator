"""Phase 0.10: 4-speaker Bark-normalised F1/F2 vowel space chart.

Plots exactly 4 speakers: owner, real_BC, synth_BC[best], lindsey.
IPA labels placed at lindsey's centroid (RP exemplar).

Phonemes: 12 monophthongs + sonorants (l, r, m, n, ŋ) if available in data.
Colors: owner=red, real_BC=blue, synth_BC=green, lindsey=gray.

Usage:
    # After B3 completes — auto-detect best trial from optuna.db
    .venv/bin/python scripts/accent_coach_phase0_10_plot.py

    # Provide centroids explicitly (for testing before B3)
    .venv/bin/python scripts/accent_coach_phase0_10_plot.py \
        --synth-centroids tts_output/accent_coach/phase0_10/cell_smoke/rep0/centroids.json

    --out-dir docs/img  (default)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from accent_coach.diagnostics.bark_distance import bark_transform

BASELINE_CENTROIDS = PROJECT_ROOT / "tts_output/accent_coach/bench/phase0_7/speaker_centroids.json"
PHASE10_DIR  = PROJECT_ROOT / "tts_output/accent_coach/phase0_10"
STORAGE_PATH = PHASE10_DIR / "optuna.db"
TRIALS_DIR   = PHASE10_DIR / "trials"

# 12 monophthongs from the spec
MONOPHTHONGS = ["iː", "ɪ", "ɛ", "æ", "ɑː", "ɒ", "ɔː", "ʊ", "uː", "ʌ", "ɜː", "ə"]
# Sonorants — included if in data, silently skipped if absent
SONORANTS    = ["l", "r", "m", "n", "ŋ"]
SPEC_PHONEMES = MONOPHTHONGS + SONORANTS

SPEAKER_COLORS = {
    "owner":    ("#d62728", "v", 80,  "Owner"),
    "real_BC":  ("#1f77b4", "D", 80,  "Real BC"),
    "synth_BC": ("#2ca02c", "o", 65,  "Synth BC (best)"),
    "lindsey":  ("#888888", "s", 65,  "Lindsey (RP target)"),
}


def load_baseline() -> dict:
    return json.loads(BASELINE_CENTROIDS.read_text())


def find_best_centroids() -> Path | None:
    """Return path to the best trial's rep0 centroids.json, or None."""
    if not STORAGE_PATH.exists():
        return None
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.load_study(
            study_name="phase0_10",
            storage=f"sqlite:///{STORAGE_PATH}",
        )
        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            return None
        best = min(completed, key=lambda t: t.value)
        p = TRIALS_DIR / f"trial_{best.number:03d}" / "rep0" / "centroids.json"
        return p if p.exists() else None
    except Exception as e:
        print(f"[warn] Could not load optuna study: {e}")
        return None


def bark_xy(bark: dict, spk: str, vowels: list[str]) -> dict[str, tuple[float, float]]:
    """Return {phoneme: (F2_bark, F1_bark)} — only phonemes with valid data."""
    pts: dict[str, tuple[float, float]] = {}
    for ph in vowels:
        v = bark.get(ph, {}).get(spk)
        if v and v.get("f1", 0) > 0 and v.get("f2", 0) > 0:
            pts[ph] = (v["f2"], v["f1"])
    return pts


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 0.10 4-speaker vowel space plot")
    parser.add_argument("--synth-centroids", type=Path, default=None,
                        help="Path to synth_BC centroids.json (default: auto from optuna best)")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "docs/img")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = (args.out_dir if args.out_dir.is_absolute()
               else PROJECT_ROOT / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load baseline (has owner, real_BC, lindsey, modern_rp, fry, bbc_male)
    baseline = load_baseline()

    # Load synth_BC centroids
    synth_path: Path | None = args.synth_centroids
    if synth_path is None:
        synth_path = find_best_centroids()
    if synth_path is None:
        # Fall back to cell_smoke if present
        fallback = PHASE10_DIR / "cell_smoke" / "rep0" / "centroids.json"
        if fallback.exists():
            synth_path = fallback
            print(f"[info] Using cell_smoke centroids (no optuna best found): {synth_path}")
        else:
            print("ERROR: no synth_BC centroids found. "
                  "Run A6 cell smoke or B3 full study first.")
            return 1

    synth_centroids: dict = json.loads(synth_path.read_text())
    print(f"[info] synth_BC centroids: {synth_path}  ({len(synth_centroids)} phonemes)")

    # Patch synth_BC into a copy of the baseline
    import copy
    merged = copy.deepcopy(baseline)
    for ph, vals in synth_centroids.items():
        merged.setdefault(ph, {})["synth_BC"] = vals

    bark = bark_transform(merged)

    # Only plot phonemes present in lindsey (RP exemplar for labels)
    available = [ph for ph in SPEC_PHONEMES if bark.get(ph, {}).get("lindsey")]
    missing   = [ph for ph in SPEC_PHONEMES if ph not in available]
    if missing:
        print(f"[info] Phonemes missing from lindsey (not plotted): {missing}")
    phonemes = available
    print(f"[info] Plotting {len(phonemes)} phonemes: {phonemes}")

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(12, 9))

    for spk, (color, marker, size, label) in SPEAKER_COLORS.items():
        pts = bark_xy(bark, spk, phonemes)
        if not pts:
            print(f"[warn] No data for speaker '{spk}' — skipped")
            continue
        xs = [p[0] for p in pts.values()]
        ys = [p[1] for p in pts.values()]
        ax.scatter(xs, ys, color=color, marker=marker, s=size,
                   alpha=0.90, zorder=7, label=label)

    # IPA labels placed at lindsey's positions
    lindsey_pts = bark_xy(bark, "lindsey", phonemes)
    for ph, (x, y) in lindsey_pts.items():
        ax.annotate(
            ph, (x, y),
            fontsize=10, fontweight="bold",
            color="#444444", alpha=0.90,
            xytext=(7, 6), textcoords="offset points",
            zorder=10,
        )

    # Thin connector lines between the 4 speakers per phoneme
    speaker_order = ["owner", "real_BC", "synth_BC", "lindsey"]
    for ph in phonemes:
        chain = []
        for spk in speaker_order:
            v = bark.get(ph, {}).get(spk)
            if v and v.get("f1", 0) > 0 and v.get("f2", 0) > 0:
                chain.append((v["f2"], v["f1"]))
        if len(chain) >= 2:
            xs, ys = zip(*chain)
            ax.plot(xs, ys, color="#aaaaaa", lw=0.9, alpha=0.35,
                    linestyle="--", zorder=1)

    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel("F2 (Bark)", fontsize=12)
    ax.set_ylabel("F1 (Bark)", fontsize=12)
    ax.set_title(
        "Phase 0.10 — Vowel space: 4 speakers  [Bark F1/F2]\n"
        "owner  ·  real_BC  ·  synth_BC (best Optuna trial)  ·  lindsey (RP target)",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.93)
    ax.grid(alpha=0.18)
    ax.text(
        0.02, 0.03,
        "IPA labels placed at Lindsey (RP exemplar) centroids  ·  "
        "dashed lines connect same phoneme across speakers",
        transform=ax.transAxes, fontsize=8, color="#666666",
    )

    out_path = out_dir / "phase0_10_4speaker_plot.png"
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
