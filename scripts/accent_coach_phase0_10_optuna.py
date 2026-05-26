"""Phase 0.10 Optuna TPE sweep over IndexTTS-2 generation params.

5 free dims: num_beams {3,5,7}, temperature [0.4,0.9], top_p [0.7,0.95],
             cfg_rate [0.3,1.5], diffusion_steps {15,25,40,60}

Objective: mean of N replicates of:
    loss = 0.5*(100 - composite_lindsey) + 0.5*(100 - composite_BC)

Usage:
    # Pilot (N=1, 5 trials)
    .venv/bin/python scripts/accent_coach_phase0_10_optuna.py --n-trials 5 --replicates 1 --pilot

    # Full study (N=3, 30 trials, resume)
    .venv/bin/python scripts/accent_coach_phase0_10_optuna.py --n-trials 30 --replicates 3 --resume
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import optuna

from scripts.accent_coach_phase0_10_run_cell import run_cell

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PHASE10_DIR = PROJECT_ROOT / "tts_output/accent_coach/phase0_10"
STORAGE     = f"sqlite:///{PHASE10_DIR}/optuna.db"
STUDY_NAME  = "phase0_10"

CAL_CSVS = {
    "cal_25": PROJECT_ROOT / "tts_output/accent_coach/cal_25.csv",
    "cal_50": PROJECT_ROOT / "tts_output/accent_coach/cal_50.csv",
}


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def make_objective(n_replicates: int, cal_csv: Path, trials_dir: Path):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_beams":       trial.suggest_categorical("num_beams",      [3, 5, 7]),
            "temperature":     trial.suggest_float("temperature",          0.4, 0.9),
            "top_p":           trial.suggest_float("top_p",               0.7, 0.95),
            "cfg_rate":        trial.suggest_float("cfg_rate",            0.3, 1.5),
            "diffusion_steps": trial.suggest_categorical("diffusion_steps", [15, 25, 40, 60]),
        }

        _log(f"Trial {trial.number}: params={params}  n_replicates={n_replicates}")
        trial_dir = trials_dir / f"trial_{trial.number:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        losses:          list[float] = []
        lindsey_scores:  list[float] = []
        bc_scores:       list[float] = []

        for rep in range(n_replicates):
            _log(f"  Trial {trial.number} rep {rep+1}/{n_replicates}: starting")
            t0 = time.time()
            result = run_cell(
                params=params,
                replicate=rep,
                out_dir=trial_dir,
                cal_csv=cal_csv,
            )
            elapsed = time.time() - t0
            _log(f"  Trial {trial.number} rep {rep+1}/{n_replicates}: "
                 f"loss={result['loss']:.4f}  "
                 f"lindsey={result['composite_lindsey']:.2f}  "
                 f"BC={result['composite_BC']:.2f}  "
                 f"elapsed={elapsed:.0f}s")

            losses.append(result["loss"])
            lindsey_scores.append(result["composite_lindsey"])
            bc_scores.append(result["composite_BC"])

            # Allow Optuna to prune slow/bad trials early
            trial.report(float(np.mean(losses)), step=rep)
            if trial.should_prune():
                _log(f"  Trial {trial.number}: pruned at rep {rep+1}")
                raise optuna.TrialPruned()

        mean_loss    = float(np.mean(losses))
        mean_lindsey = float(np.mean(lindsey_scores))
        std_lindsey  = float(np.std(lindsey_scores))
        mean_bc      = float(np.mean(bc_scores))
        std_bc       = float(np.std(bc_scores))

        trial.set_user_attr("composite_lindsey_mean", mean_lindsey)
        trial.set_user_attr("composite_lindsey_std",  std_lindsey)
        trial.set_user_attr("composite_BC_mean",      mean_bc)
        trial.set_user_attr("composite_BC_std",       std_bc)
        trial.set_user_attr("n_replicates",           n_replicates)
        trial.set_user_attr("params",                 json.dumps(params))

        # Persist per-trial summary
        summary = {
            "trial_number": trial.number,
            "params": params,
            "loss_mean": mean_loss,
            "composite_lindsey_mean": mean_lindsey,
            "composite_lindsey_std": std_lindsey,
            "composite_BC_mean": mean_bc,
            "composite_BC_std": std_bc,
        }
        (trial_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        _log(f"Trial {trial.number} COMPLETE: loss={mean_loss:.4f}  "
             f"lindsey={mean_lindsey:.2f}±{std_lindsey:.2f}  "
             f"BC={mean_bc:.2f}±{std_bc:.2f}")
        return mean_loss

    return objective


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 0.10 Optuna TPE sweep")
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Total trials to run in this session (default 30)")
    parser.add_argument("--replicates", type=int, default=3,
                        help="Replicates per trial (default 3; use 1 for pilot)")
    parser.add_argument("--cal", default="cal_25", choices=list(CAL_CSVS.keys()),
                        help="Calibration phrase set (default cal_25)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume existing study (load_if_exists)")
    parser.add_argument("--pilot", action="store_true",
                        help="Flag for pilot run (recorded in log; no functional change)")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Optuna n_jobs (default 1; try 2 only after B2 pilot passes)")
    args = parser.parse_args()

    PHASE10_DIR.mkdir(parents=True, exist_ok=True)
    trials_dir = PHASE10_DIR / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    cal_csv = CAL_CSVS[args.cal]
    if not cal_csv.exists():
        print(f"ERROR: cal CSV not found: {cal_csv}")
        return 1

    load_if_exists = args.resume or args.pilot  # pilot also loads so runs accumulate

    _log(f"=== Phase 0.10 Optuna sweep ===")
    _log(f"  study={STUDY_NAME}  storage={STORAGE}")
    _log(f"  n_trials={args.n_trials}  replicates={args.replicates}  cal={args.cal}")
    _log(f"  pilot={args.pilot}  resume={args.resume}  n_jobs={args.n_jobs}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        load_if_exists=load_if_exists,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=8),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )

    existing = len(study.trials)
    _log(f"  Study has {existing} existing trials. Running {args.n_trials} more.")

    objective = make_objective(
        n_replicates=args.replicates,
        cal_csv=cal_csv,
        trials_dir=trials_dir,
    )

    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        gc_after_trial=True,
    )

    # Final summary
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    _log(f"\n=== Study summary ===")
    _log(f"  Total trials: {len(study.trials)}  complete: {len(completed)}  pruned: {len(pruned)}")

    if completed:
        best = study.best_trial
        _log(f"  Best trial: #{best.number}  loss={best.value:.4f}")
        _log(f"    params:             {json.loads(best.user_attrs.get('params', '{}'))}")
        _log(f"    composite_lindsey:  {best.user_attrs.get('composite_lindsey_mean', 'N/A'):.2f}"
             f" ± {best.user_attrs.get('composite_lindsey_std', 'N/A'):.2f}")
        _log(f"    composite_BC:       {best.user_attrs.get('composite_BC_mean', 'N/A'):.2f}"
             f" ± {best.user_attrs.get('composite_BC_std', 'N/A'):.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
