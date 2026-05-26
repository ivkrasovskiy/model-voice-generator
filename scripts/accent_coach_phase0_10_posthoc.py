"""Phase 0.10 C3: posthoc WER/ECAPA/DNSMOS on the best trial's params.

Generates eval_short.csv (15 phrases) × N=3 replicates with the best Optuna
params, scores each replicate, and writes docs/accent_coach_phase0_10_posthoc.csv.

Usage:
    .venv/bin/python scripts/accent_coach_phase0_10_posthoc.py
    .venv/bin/python scripts/accent_coach_phase0_10_posthoc.py --trial 25
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

from accent_coach.pipeline.experiment import generate_clips, score_posthoc_clips

PHASE10_DIR  = PROJECT_ROOT / "tts_output/accent_coach/phase0_10"
STORAGE      = f"sqlite:///{PHASE10_DIR}/optuna.db"
STUDY_NAME   = "phase0_10"
EVAL_CSV     = PROJECT_ROOT / "tts_output/cross_eval_50/eval_short.csv"
REF_AUDIO    = PROJECT_ROOT / "tts_output/ref_interview.wav"
ECAPA_REF    = PROJECT_ROOT / "tts_output/ref_narrator.wav"
OUT_CSV      = PROJECT_ROOT / "docs/accent_coach_phase0_10_posthoc.csv"
N_REPLICATES = 3


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 0.10 C3 posthoc scoring")
    parser.add_argument("--trial", type=int, default=None,
                        help="Trial number to use (default: best by loss)")
    parser.add_argument("--n-replicates", type=int, default=N_REPLICATES)
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
    complete = [t for t in study.trials if t.state.name == "COMPLETE"]
    if not complete:
        print("ERROR: no complete trials in study")
        return 1

    if args.trial is not None:
        trial = next((t for t in study.trials if t.number == args.trial), None)
        if trial is None:
            print(f"ERROR: trial #{args.trial} not found")
            return 1
    else:
        trial = min(complete, key=lambda t: t.value)

    params = json.loads(trial.user_attrs.get("params", json.dumps(trial.params)))
    print(f"Using trial #{trial.number}  loss={trial.value:.4f}")
    print(f"  params: {params}")
    print(f"  lindsey={trial.user_attrs.get('composite_lindsey_mean','?'):.2f}"
          f"  BC={trial.user_attrs.get('composite_BC_mean','?'):.2f}")

    out_dir = PHASE10_DIR / f"posthoc_trial{trial.number:03d}"
    rows: list[dict] = []

    for rep in range(args.n_replicates):
        rep_dir = out_dir / f"rep{rep}"
        print(f"\n[rep {rep+1}/{args.n_replicates}] generating {EVAL_CSV.name}...")
        manifest = generate_clips(
            phrases_csv=EVAL_CSV,
            ref_audio=REF_AUDIO,
            out_dir=rep_dir / "clips",
            gen_params=params,
            label="synth_BC",
        )

        scores_csv = rep_dir / "posthoc_scores.csv"
        print(f"[rep {rep+1}/{args.n_replicates}] scoring WER/ECAPA/DNSMOS...")
        result = score_posthoc_clips(
            manifest=manifest,
            ecapa_ref=ECAPA_REF,
            out_csv=scores_csv,
        )
        print(f"  WER={result['WER']:.4f}  ECAPA={result['ECAPA']:.4f}  DNSMOS={result['DNSMOS_OVR']:.3f}")
        rows.append({
            "trial": trial.number,
            "replicate": rep,
            "WER": result["WER"],
            "ECAPA": result["ECAPA"],
            "DNSMOS_OVR": result["DNSMOS_OVR"],
        })

    import numpy as np
    wers   = [r["WER"]        for r in rows]
    ecapas = [r["ECAPA"]      for r in rows]
    dnsmos = [r["DNSMOS_OVR"] for r in rows]

    print(f"\n=== Posthoc summary (trial #{trial.number}, N={args.n_replicates}) ===")
    print(f"  WER    = {np.mean(wers):.4f} ± {np.std(wers):.4f}  (Phase0.9 baseline: 0.037)")
    print(f"  ECAPA  = {np.mean(ecapas):.4f} ± {np.std(ecapas):.4f}  (Phase0.9 baseline: 0.276)")
    print(f"  DNSMOS = {np.mean(dnsmos):.3f} ± {np.std(dnsmos):.3f}  (Phase0.9 baseline: 2.67)")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trial", "replicate", "WER", "ECAPA", "DNSMOS_OVR"])
        w.writeheader()
        w.writerows(rows)
        w.writerow({
            "trial": f"mean±std",
            "replicate": "",
            "WER":        f"{np.mean(wers):.4f}±{np.std(wers):.4f}",
            "ECAPA":      f"{np.mean(ecapas):.4f}±{np.std(ecapas):.4f}",
            "DNSMOS_OVR": f"{np.mean(dnsmos):.3f}±{np.std(dnsmos):.3f}",
        })
    print(f"\nWrote: {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
