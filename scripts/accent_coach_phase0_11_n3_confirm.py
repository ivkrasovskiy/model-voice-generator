"""N=3 confirmation for Phase 0.11's winner candidate (baseline + Fry α=0.3).

Generates 2 additional reps for each cell (rep0 reused from Phase 0.11), then
reports mean ± std for fry / modern_rp / BC against BOTH original and cleaned
centroids — settles whether α=0.3 lift is real signal vs N=1 noise.

Cells: baseline (no emo), fry α=0.3. Each at N=3.
Total: 4 new generations × 19 phrases × ~50s/phrase ≈ 60 min.

Usage:
    .venv/bin/python scripts/accent_coach_phase0_11_n3_confirm.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from accent_coach.pipeline.experiment import (
    build_centroids_from_formants,
    extract_formants,
    generate_clips,
    load_baseline_centroids,
    score_against,
)

PHASE11_DIR  = PROJECT_ROOT / "tts_output/accent_coach/phase0_11"
CONFIRM_DIR  = PROJECT_ROOT / "tts_output/accent_coach/phase0_11_n3"
CAL_CSV      = PROJECT_ROOT / "tts_output/accent_coach/cal_25.csv"
SPK_REF      = PROJECT_ROOT / "tts_output/ref_interview.wav"
FRY_REF      = PROJECT_ROOT / "tts_output/ref_fry_emo.wav"
CLEANED_PATH = PROJECT_ROOT / "tts_output/accent_coach/cleaned_corpus/speaker_centroids_cleaned.json"

BASE_PARAMS = {
    "num_beams":       5,
    "temperature":     0.8,
    "top_p":           0.8,
    "cfg_rate":        0.7,
    "diffusion_steps": 25,
}

CELLS = [
    {"id": "00_baseline", "source_cell": "00_baseline",
     "emo_audio": None,           "emo_alpha": 1.0},
    {"id": "01_fry_a03", "source_cell": "01_fry_a03",
     "emo_audio": str(FRY_REF),   "emo_alpha": 0.3},
]


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def generate_and_score_rep(cell: dict, rep: int, baseline_cents: dict,
                            cleaned_cents: dict | None) -> dict:
    """Generate one rep for a cell, extract formants, build centroid, score against
    both original and cleaned targets. Idempotent: skip if result.json exists."""
    rep_dir = CONFIRM_DIR / "cells" / f"cell_{cell['id']}" / f"rep{rep}"
    rep_dir.mkdir(parents=True, exist_ok=True)
    result_path = rep_dir / "result.json"
    if result_path.exists():
        _log(f"  rep{rep}: cached")
        return json.loads(result_path.read_text())

    _log(f"  rep{rep}: generating (emo={cell['emo_audio'] is not None}, alpha={cell['emo_alpha']})")
    t0 = time.time()
    gen_params = {**BASE_PARAMS}
    if cell["emo_audio"]:
        gen_params["emo_audio"] = cell["emo_audio"]
        gen_params["emo_alpha"] = cell["emo_alpha"]

    manifest = generate_clips(phrases_csv=CAL_CSV, ref_audio=SPK_REF,
                              out_dir=rep_dir / "clips", gen_params=gen_params,
                              label="synth_BC")
    formants_csv = extract_formants(manifest=manifest, out_csv=rep_dir / "formants.csv",
                                     source_label="synth_BC")
    synth = build_centroids_from_formants(formants_csv)
    (rep_dir / "centroids.json").write_text(json.dumps(synth, indent=2))

    result = {"rep": rep, "n_phonemes": len(synth)}
    for label, base in [("orig", baseline_cents), ("clean", cleaned_cents)]:
        if base is None:
            continue
        for target in ["fry", "lindsey", "bbc_male", "modern_rp", "real_BC"]:
            try:
                s = score_against(synth, target, baseline_centroids=base)["composite"]
                result[f"{target}_{label}"] = round(s, 2)
            except Exception:
                result[f"{target}_{label}"] = None
    result["elapsed_s"] = round(time.time() - t0, 1)
    result_path.write_text(json.dumps(result, indent=2))
    _log(f"  rep{rep}: done in {result['elapsed_s']:.0f}s  "
         f"fry_orig={result.get('fry_orig')}  fry_clean={result.get('fry_clean')}")
    return result


def read_phase11_rep0(cell: dict, baseline_cents: dict,
                       cleaned_cents: dict | None) -> dict:
    """Use existing Phase 0.11 cell as rep0 (re-score against both target sets)."""
    src_dir = PHASE11_DIR / "cells" / f"cell_{cell['source_cell']}"
    cent_path = src_dir / "centroids.json"
    if not cent_path.exists():
        raise FileNotFoundError(f"Phase 0.11 source missing: {cent_path}")
    synth = json.loads(cent_path.read_text())
    result = {"rep": 0, "n_phonemes": len(synth), "source": str(src_dir.relative_to(PROJECT_ROOT))}
    for label, base in [("orig", baseline_cents), ("clean", cleaned_cents)]:
        if base is None:
            continue
        for target in ["fry", "lindsey", "bbc_male", "modern_rp", "real_BC"]:
            try:
                s = score_against(synth, target, baseline_centroids=base)["composite"]
                result[f"{target}_{label}"] = round(s, 2)
            except Exception:
                result[f"{target}_{label}"] = None
    return result


def main() -> int:
    CONFIRM_DIR.mkdir(parents=True, exist_ok=True)
    _log("=== Phase 0.11 N=3 confirmation ===")
    _log(f"  cells: {[c['id'] for c in CELLS]}")
    _log(f"  spk_ref: {SPK_REF.name}, fry_ref: {FRY_REF.name}")

    baseline_cents = load_baseline_centroids()
    cleaned_cents = None
    if CLEANED_PATH.exists():
        cleaned_cents = json.loads(CLEANED_PATH.read_text())
        _log(f"  cleaned overlay loaded ({len(cleaned_cents)} phonemes)")
    else:
        _log(f"  WARNING: cleaned overlay not found at {CLEANED_PATH} — only orig scoring")

    all_results: dict[str, list[dict]] = {}
    for cell in CELLS:
        _log(f"\n--- Cell {cell['id']} ---")
        reps = [read_phase11_rep0(cell, baseline_cents, cleaned_cents)]
        for rep in [1, 2]:
            reps.append(generate_and_score_rep(cell, rep, baseline_cents, cleaned_cents))
        all_results[cell["id"]] = reps

    # Aggregate: mean ± std per target per cell
    _log("\n=== N=3 aggregated results ===")
    target_set = [t + s for t in ["fry_", "modern_rp_", "real_BC_"]
                  for s in (["orig", "clean"] if cleaned_cents else ["orig"])]

    summary_rows = []
    for cell_id, reps in all_results.items():
        row = {"cell": cell_id}
        for target in target_set:
            vals = [r.get(target) for r in reps if r.get(target) is not None]
            if vals:
                row[f"{target}_mean"] = round(float(np.mean(vals)), 2)
                row[f"{target}_std"]  = round(float(np.std(vals)),  2)
                row[f"{target}_n"]    = len(vals)
        summary_rows.append(row)

    # Save and print
    out_csv = CONFIRM_DIR / "n3_summary.csv"
    fields = list(summary_rows[0].keys()) if summary_rows else []
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary_rows)
    _log(f"Wrote: {out_csv}")

    print()
    print(f"{'cell':<14}  {'fry_orig':>18}  {'fry_clean':>18}  "
          f"{'mrp_orig':>18}  {'mrp_clean':>18}  {'BC_orig':>18}")
    for r in summary_rows:
        def _fmt(prefix):
            m = r.get(f"{prefix}_mean")
            s = r.get(f"{prefix}_std")
            return f"{m:.2f} ± {s:.2f}" if m is not None else "N/A"
        print(f"{r['cell']:<14}  "
              f"{_fmt('fry_orig'):>18}  {_fmt('fry_clean'):>18}  "
              f"{_fmt('modern_rp_orig'):>18}  {_fmt('modern_rp_clean'):>18}  "
              f"{_fmt('real_BC_orig'):>18}")

    # Decision rule
    if len(summary_rows) == 2:
        base, alpha = summary_rows[0], summary_rows[1]
        for prefix in ["fry_clean", "modern_rp_clean"] if cleaned_cents else ["fry_orig", "modern_rp_orig"]:
            bm, bs = base.get(f"{prefix}_mean"), base.get(f"{prefix}_std")
            am, ast = alpha.get(f"{prefix}_mean"), alpha.get(f"{prefix}_std")
            if bm is None or am is None:
                continue
            lift = am - bm
            pooled_std = (bs**2 + ast**2) ** 0.5
            verdict = "REAL signal" if lift > pooled_std else "within noise"
            _log(f"\n{prefix}: α=0.3 lift = {lift:+.2f} (pooled σ = {pooled_std:.2f}) → {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
