"""Phase 0.13b — Lever A driver: phoneme-dense emo clip sweep (Task A2/A3).

Only run after Lever B verdict is GREEN or YELLOW (§4 of plan).

For each of the 3 dense candidates × 3 emo_alpha values × N=3 reps,
generates cal_25 clips, extracts formants, scores against modern_rp.

Usage:
    .venv/bin/python scripts/accent_coach_phase0_13_lever_a.py --replicates 3

    # dry-run: check emo candidates exist and print planned cells
    .venv/bin/python scripts/accent_coach_phase0_13_lever_a.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))  # needed by score_posthoc_clips → lib.*

import numpy as np

from accent_coach.pipeline.experiment import (
    build_centroids_from_formants,
    extract_formants,
    generate_clips,
    load_baseline_centroids,
    score_against,
    score_posthoc_clips,
)

PHASE13_DIR   = PROJECT_ROOT / "tts_output/accent_coach/phase0_13"
CELLS_DIR     = PHASE13_DIR / "cells"
EMO_DENSE_DIR = PHASE13_DIR / "emo_dense"
CAL_CSV       = PROJECT_ROOT / "tts_output/accent_coach/cal_25.csv"
SPK_REF       = PROJECT_ROOT / "tts_output/refs/production/ref_interview.wav"
ECAPA_REF     = PROJECT_ROOT / "tts_output/refs/indextts_baseline/ref_narrator.wav"

BASE_PARAMS = {
    "num_beams":       5,
    "temperature":     0.8,
    "top_p":           0.8,
    "cfg_rate":        0.7,
    "diffusion_steps": 25,
}

EMO_ALPHAS = [0.3, 0.5, 0.7]
ALL_5_PHONEMES = {"ʌ", "ʊ", "ɔː", "aʊ", "ɜː"}


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def _load_candidates() -> list[dict]:
    index_path = EMO_DENSE_DIR / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"emo_dense/index.json not found. Run accent_coach_phase0_13_mine_emo.py first.\n"
            f"  Expected: {index_path}"
        )
    return json.loads(index_path.read_text())


def _cell_id(cand_idx: int, alpha: float) -> str:
    alpha_str = str(alpha).replace(".", "")
    return f"cell_a_cand{cand_idx}_alpha{alpha_str}"


def run_lever_a_cell(
    cell_id: str,
    cand_idx: int,
    emo_wav: Path,
    emo_alpha: float,
    replicates: int,
    baseline_cents: dict,
) -> dict:
    """Generate clips with emo conditioning, extract, score."""
    cell_dir = CELLS_DIR / cell_id

    rep_composites: list[float] = []
    ph_scores: dict[str, list[float]] = defaultdict(list)
    wers, ecapas, dnsmos_list = [], [], []
    t0 = time.time()

    params = {**BASE_PARAMS, "emo_audio": str(emo_wav), "emo_alpha": emo_alpha}

    for rep in range(replicates):
        rep_dir = cell_dir / f"rep_{rep}"
        _log(f"{cell_id} rep_{rep}: generating with emo_alpha={emo_alpha}…")
        manifest_path = generate_clips(
            phrases_csv=CAL_CSV,
            ref_audio=SPK_REF,
            out_dir=rep_dir,
            gen_params=params,
            label="synth_BC",
        )

        formants_csv = rep_dir / "formants.csv"
        extract_formants(manifest_path, formants_csv, source_label="synth_BC")

        synth_cents = build_centroids_from_formants(formants_csv)
        result = score_against(synth_cents, "modern_rp", baseline_cents)
        composite = result["composite"]
        per_ph = result["per_phoneme"]
        rep_composites.append(composite)
        for ph in ALL_5_PHONEMES:
            if ph in per_ph:
                ph_scores[ph].append(per_ph[ph])

        posthoc_csv = rep_dir / "posthoc.csv"
        posthoc = score_posthoc_clips(manifest_path, ECAPA_REF, posthoc_csv)
        wers.append(posthoc.get("WER", float("nan")))
        ecapas.append(posthoc.get("ECAPA", float("nan")))
        dnsmos_list.append(posthoc.get("DNSMOS_OVR", float("nan")))

        _log(f"  composite={composite:.2f}  WER={posthoc.get('WER', 'nan'):.3f}  "
             f"ECAPA={posthoc.get('ECAPA', 'nan'):.3f}  DNSMOS={posthoc.get('DNSMOS_OVR', 'nan'):.2f}")

    elapsed = time.time() - t0
    per_ph_means = {ph: round(float(np.mean(vs)), 2) for ph, vs in ph_scores.items()}
    per_target_mean = float(np.mean(list(per_ph_means.values()))) if per_ph_means else float("nan")

    summary = {
        "cell_id": cell_id,
        "cand_idx": cand_idx,
        "emo_alpha": emo_alpha,
        "composite_modern_rp_mean": round(float(np.mean(rep_composites)), 2),
        "composite_modern_rp_std": round(float(np.std(rep_composites)), 2),
        "per_phoneme_target_mean": round(per_target_mean, 2),
        "per_phoneme_by_phoneme": per_ph_means,
        "wer_mean": round(float(np.nanmean(wers)), 4) if wers else float("nan"),
        "ecapa_mean": round(float(np.nanmean(ecapas)), 4) if ecapas else float("nan"),
        "dnsmos_ovr_mean": round(float(np.nanmean(dnsmos_list)), 4) if dnsmos_list else float("nan"),
        "elapsed_s": round(elapsed, 1),
    }

    out = cell_dir / "result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    _log(f"{cell_id} DONE: composite = {summary['composite_modern_rp_mean']:.2f}  elapsed={elapsed:.0f}s")
    return summary


def _load_baseline_result() -> dict | None:
    """Load cell_baseline_b result if available (for lift calculations)."""
    result_path = CELLS_DIR / "cell_baseline_b" / "result.json"
    if not result_path.exists():
        return None
    return json.loads(result_path.read_text())


def _write_lever_a_grid(results: list[dict], baseline: dict | None) -> None:
    out_csv = PHASE13_DIR / "lever_a_grid.csv"
    fieldnames = [
        "cell_id", "cand_idx", "emo_alpha",
        "composite_modern_rp_mean", "composite_modern_rp_std",
        "per_phoneme_target_mean", "composite_lift_vs_baseline",
        "per_target_lift_vs_baseline",
        "wer_mean", "ecapa_mean", "dnsmos_ovr_mean", "elapsed_s",
    ]
    baseline_composite = baseline.get("composite_modern_rp_mean", float("nan")) if baseline else float("nan")
    baseline_per_ph = baseline.get("per_phoneme_target_mean", float("nan")) if baseline else float("nan")

    rows = []
    for r in results:
        row = {k: r.get(k, "") for k in fieldnames}
        try:
            row["composite_lift_vs_baseline"] = round(r["composite_modern_rp_mean"] - baseline_composite, 2)
            row["per_target_lift_vs_baseline"] = round(r["per_phoneme_target_mean"] - baseline_per_ph, 2)
        except (TypeError, ValueError):
            row["composite_lift_vs_baseline"] = ""
            row["per_target_lift_vs_baseline"] = ""
        rows.append(row)

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    _log(f"Lever A grid written to {out_csv}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 0.13b Lever A driver")
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned cells without running")
    args = parser.parse_args()

    PHASE13_DIR.mkdir(parents=True, exist_ok=True)
    CELLS_DIR.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        index_path = EMO_DENSE_DIR / "index.json"
        if not index_path.exists():
            print("Lever A dry-run: emo_dense/index.json not yet generated.")
            print("  Run accent_coach_phase0_13_mine_emo.py first to mine candidates.")
            print(f"  Will generate: {len([None for _ in range(3) for _ in EMO_ALPHAS])} cells × {args.replicates} reps")
            return 0
        candidates = _load_candidates()
        print(f"Lever A plan: {len(candidates) * len(EMO_ALPHAS)} cells × {args.replicates} reps")
        for cand in candidates:
            for alpha in EMO_ALPHAS:
                cid = _cell_id(cand["candidate_idx"], alpha)
                print(f"  {cid}: cand_{cand['candidate_idx']} ({cand['speaker']}/{cand['clip_id']}) α={alpha}")
        return 0

    candidates = _load_candidates()
    baseline_cents = load_baseline_centroids()

    # Plan: cand × alpha cells
    cells_plan = []
    for cand in candidates:
        for alpha in EMO_ALPHAS:
            cells_plan.append((cand, alpha))


    results: list[dict] = []
    for cand, alpha in cells_plan:
        cand_idx = cand["candidate_idx"]
        emo_wav = EMO_DENSE_DIR / f"cand_{cand_idx}.wav"
        if not emo_wav.exists():
            _log(f"ERROR: {emo_wav} not found. Run accent_coach_phase0_13_mine_emo.py first.")
            return 1

        cell_id = _cell_id(cand_idx, alpha)
        _log(f"=== Running {cell_id} (N={args.replicates}) ===")
        r = run_lever_a_cell(
            cell_id=cell_id,
            cand_idx=cand_idx,
            emo_wav=emo_wav,
            emo_alpha=alpha,
            replicates=args.replicates,
            baseline_cents=baseline_cents,
        )
        results.append(r)

    baseline = _load_baseline_result()
    _write_lever_a_grid(results, baseline)

    # Print A3 table
    print("\n=== Lever A Per-Phoneme Summary (vs baseline) ===")
    print(f"{'Cell':<35} {'composite μ':>12} {'per-ph μ':>9} {'WER':>6} {'ECAPA':>7} {'DNSMOS':>7}")
    for r in results:
        print(f"{r['cell_id']:<35} "
              f"{r.get('composite_modern_rp_mean', '?'):>12.2f} "
              f"{r.get('per_phoneme_target_mean', float('nan')):>9.2f} "
              f"{r.get('wer_mean', float('nan')):>6.4f} "
              f"{r.get('ecapa_mean', float('nan')):>7.4f} "
              f"{r.get('dnsmos_ovr_mean', float('nan')):>7.3f}")

    # Apply §4.4 stop criteria
    if results:
        best = max(results, key=lambda x: x.get("per_phoneme_target_mean", -999))
        bph = best.get("per_phoneme_target_mean", float("nan"))
        bc = best.get("composite_modern_rp_mean", float("nan"))
        bwer = best.get("wer_mean", float("nan"))
        baseline_ph = baseline.get("per_phoneme_target_mean", float("nan")) if baseline else float("nan")
        lift = bph - baseline_ph if not (isinstance(bph, float) and bph != bph or isinstance(baseline_ph, float) and baseline_ph != baseline_ph) else float("nan")
        print(f"\nBest cell: {best['cell_id']}")
        print(f"  per-target phoneme mean: {bph:.2f} (lift vs baseline: {lift:.2f})")
        print(f"  composite_modern_rp:     {bc:.2f}")
        print(f"  WER:                     {bwer:.4f}")
        if lift >= 8 and bc >= 80 and bwer <= 0.05:
            print("\nVERDICT: GREEN — Lever A is a production path.")
        elif lift >= 3 and bc >= 77 and bwer <= 0.06:
            print("\nVERDICT: YELLOW — Partial win; emo conditioning helps but won't close full gap.")
        else:
            print("\nVERDICT: RED — Emo embedding cannot steer at phoneme granularity. Path forward: fine-tuning.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
