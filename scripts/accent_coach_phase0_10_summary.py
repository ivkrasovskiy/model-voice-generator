"""Phase 0.10 summary: top trials from Optuna study.

Reads optuna.db, prints top 10 by mean loss with mean±std for each objective,
marks the closest match to Phase 0.9 config, and writes
docs/accent_coach_phase0_10_top_trials.csv.

Usage:
    .venv/bin/python scripts/accent_coach_phase0_10_summary.py
    .venv/bin/python scripts/accent_coach_phase0_10_summary.py --top 20
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import optuna
from optuna.importance import get_param_importances
import scipy.stats as stats
import numpy as np

PHASE10_DIR = PROJECT_ROOT / "tts_output/accent_coach/phase0_10"
STORAGE     = f"sqlite:///{PHASE10_DIR}/optuna.db"
STUDY_NAME  = "phase0_10"
OUT_CSV     = PROJECT_ROOT / "docs/accent_coach_phase0_10_top_trials.csv"

# Phase 0.9 winning params for sanity check
PHASE09_PARAMS = {
    "num_beams": 5,
    "temperature": 0.8,
    "top_p": 0.8,
    "cfg_rate": 0.7,
    "diffusion_steps": 25,
}


def _param_distance(p1: dict, p2: dict) -> float:
    """Simple Euclidean distance after normalising each dimension to its range."""
    ranges = {
        "num_beams":       (3, 7),
        "temperature":     (0.4, 0.9),
        "top_p":           (0.7, 0.95),
        "cfg_rate":        (0.3, 1.5),
        "diffusion_steps": (15, 60),
    }
    d = 0.0
    for k, (lo, hi) in ranges.items():
        v1 = (p1.get(k, lo) - lo) / (hi - lo)
        v2 = (p2.get(k, lo) - lo) / (hi - lo)
        d += (v1 - v2) ** 2
    return d ** 0.5


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 0.10 Optuna study summary")
    parser.add_argument("--top", type=int, default=10, help="Number of top trials (default 10)")
    args = parser.parse_args()

    if not (PHASE10_DIR / "optuna.db").exists():
        print(f"ERROR: optuna.db not found at {PHASE10_DIR}/optuna.db")
        print("Run the Optuna study first (B2 pilot or B3 full).")
        return 1

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed    = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    print(f"\n=== Phase 0.10 Study Summary ===")
    print(f"Total trials: {len(study.trials)}  "
          f"complete: {len(completed)}  pruned: {len(pruned)}  failed: {len(failed)}")

    if not completed:
        print("No completed trials yet.")
        return 0

    # Sort by value (loss — lower is better)
    sorted_trials = sorted(completed, key=lambda t: t.value)
    top_n = sorted_trials[:args.top]

    # Phase 0.9 closest trial
    closest = min(
        completed,
        key=lambda t: _param_distance(t.params, PHASE09_PARAMS),
    )
    closest_rank = sorted_trials.index(closest) + 1

    print(f"\nTop {len(top_n)} trials (lower loss = better):")
    print(f"{'Rank':>4}  {'Trial':>5}  {'loss':>8}  "
          f"{'lindsey_mean':>13}  {'lindsey_std':>11}  "
          f"{'BC_mean':>8}  {'BC_std':>7}  "
          f"{'n_beams':>7}  {'temp':>6}  {'top_p':>6}  "
          f"{'cfg_rate':>8}  {'diff_steps':>10}")
    print("-" * 120)

    rows: list[dict] = []
    for rank, t in enumerate(top_n, 1):
        lm  = t.user_attrs.get("composite_lindsey_mean", float("nan"))
        ls  = t.user_attrs.get("composite_lindsey_std",  float("nan"))
        bm  = t.user_attrs.get("composite_BC_mean",      float("nan"))
        bs  = t.user_attrs.get("composite_BC_std",       float("nan"))
        p   = json.loads(t.user_attrs.get("params", json.dumps(t.params)))
        tag = " ← Phase0.9 closest" if t.number == closest.number else ""
        print(f"{rank:>4}  {t.number:>5}  {t.value:>8.4f}  "
              f"{lm:>13.2f}  {ls:>11.2f}  "
              f"{bm:>8.2f}  {bs:>7.2f}  "
              f"{p.get('num_beams','?'):>7}  {p.get('temperature',0):>6.3f}  {p.get('top_p',0):>6.3f}  "
              f"{p.get('cfg_rate',0):>8.3f}  {p.get('diffusion_steps','?'):>10}{tag}")
        rows.append({
            "rank": rank,
            "trial_id": t.number,
            "loss": t.value,
            "num_beams": p.get("num_beams"),
            "temperature": p.get("temperature"),
            "top_p": p.get("top_p"),
            "cfg_rate": p.get("cfg_rate"),
            "diffusion_steps": p.get("diffusion_steps"),
            "lindsey_mean": lm,
            "lindsey_std": ls,
            "BC_mean": bm,
            "BC_std": bs,
            "phase09_closest": int(t.number == closest.number),
        })

    print(f"\nPhase 0.9 params closest match: trial #{closest.number} (rank {closest_rank})")
    print(f"  params: {closest.params}")
    print(f"  loss: {closest.value:.4f}")

    # Param importances
    try:
        importance = get_param_importances(study)
        print(f"\nParam importances (fANOVA):")
        for pname, imp in sorted(importance.items(), key=lambda x: -x[1]):
            print(f"  {pname:<20s}: {imp:.4f}")
    except Exception as e:
        print(f"\nParam importances: unavailable ({e})")

    # Spearman correlation lindsey vs BC
    if len(completed) >= 3:
        lm_all = [t.user_attrs.get("composite_lindsey_mean", float("nan")) for t in completed]
        bm_all = [t.user_attrs.get("composite_BC_mean",      float("nan")) for t in completed]
        valid = [(l, b) for l, b in zip(lm_all, bm_all)
                 if not (np.isnan(l) or np.isnan(b))]
        if len(valid) >= 3:
            rho, pval = stats.spearmanr([v[0] for v in valid], [v[1] for v in valid])
            print(f"\nSpearman ρ (composite_lindsey vs composite_BC): {rho:.3f}  p={pval:.3f}")

    # Write CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["rank", "trial_id", "loss", "num_beams", "temperature", "top_p",
                  "cfg_rate", "diffusion_steps", "lindsey_mean", "lindsey_std",
                  "BC_mean", "BC_std", "phase09_closest"]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote: {OUT_CSV}")

    # Stop criteria check
    best = sorted_trials[0]
    lm_best = best.user_attrs.get("composite_lindsey_mean", 0)
    ls_best = best.user_attrs.get("composite_lindsey_std",  0)
    bm_best = best.user_attrs.get("composite_BC_mean",      0)
    green = lm_best >= 82 and (lm_best - ls_best) >= 80 and bm_best >= 75
    yellow = (lm_best >= 82 or bm_best >= 75) and not green
    verdict = "GREEN" if green else ("YELLOW" if yellow else "RED")
    print(f"\nStop criteria (best trial #{best.number}):")
    print(f"  composite_lindsey_mean={lm_best:.2f}  (≥82 {'✓' if lm_best >= 82 else '✗'})")
    print(f"  lindsey_mean-std={lm_best-ls_best:.2f}  (≥80 {'✓' if (lm_best-ls_best) >= 80 else '✗'})")
    print(f"  composite_BC_mean={bm_best:.2f}  (≥75 {'✓' if bm_best >= 75 else '✗'})")
    print(f"  Verdict: {verdict}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
