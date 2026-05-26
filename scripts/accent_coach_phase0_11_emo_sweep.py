"""Phase 0.11 emo-conditioning sweep on cal_25 — simplified Fry-only design.

Holds inference knobs at Phase 0.9 baseline (beams=5, diff_steps=25, defaults
elsewhere). spk_audio_prompt = BC interview clip. Varies emo_audio_prompt:
  - cell 0:    baseline (no emo override; ≡ Phase 0.9 C_int_b5)
  - cells 1-4: emo_audio = Fry 14s clip, emo_alpha ∈ {0.3, 0.5, 0.7, 1.0}

Primary target: fry (Phase 0.9 baseline = 73.60 on cal_25). Also reports
lindsey, modern_rp, real_BC for context. Lindsey emo path dropped — Lindsey
source has multi-speaker contamination (demos). emo_text path dropped — too
speculative for the budget.

N=1 per cell. Resumable: cells with existing result.json are skipped.

Usage:
    .venv/bin/python scripts/accent_coach_phase0_11_emo_sweep.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from accent_coach.pipeline.experiment import (
    build_centroids_from_formants,
    extract_formants,
    generate_clips,
    load_baseline_centroids,
    score_against,
)

PHASE11_DIR  = PROJECT_ROOT / "tts_output/accent_coach/phase0_11"
CELLS_DIR    = PHASE11_DIR / "cells"
GRID_CSV     = PHASE11_DIR / "grid_results.csv"

CAL_CSV      = PROJECT_ROOT / "tts_output/accent_coach/cal_25.csv"
SPK_REF      = PROJECT_ROOT / "tts_output/ref_interview.wav"
FRY_REF      = PROJECT_ROOT / "tts_output/ref_fry_emo.wav"

# Fixed gen params (Phase 0.9 baseline — apples-to-apples)
BASE_PARAMS = {
    "num_beams":       5,
    "temperature":     0.8,
    "top_p":           0.8,
    "cfg_rate":        0.7,
    "diffusion_steps": 25,
}

# Decision threshold: primary target is fry (cleanest reference speaker)
PRIMARY_TARGET = "fry"
BC_FLOOR       = 65.0  # if BC drops below this, considered identity-breaking


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def define_cells() -> list[dict]:
    """5 cells: baseline + Fry alpha sweep."""
    cells: list[dict] = [
        {"id": "00_baseline", "emo_audio": None, "emo_alpha": 1.0,
         "use_emo_text": False, "emo_text": None,
         "note": "≡ Phase 0.9 C_int_b5 on cal_25"},
    ]
    for i, alpha in enumerate([0.3, 0.5, 0.7, 1.0], start=1):
        cells.append({
            "id": f"{i:02d}_fry_a{int(alpha*10):02d}",
            "emo_audio": str(FRY_REF), "emo_alpha": alpha,
            "use_emo_text": False, "emo_text": None,
            "note": f"Fry emo, alpha={alpha}",
        })
    return cells


def run_one_cell(cell: dict, baseline_centroids: dict) -> dict:
    """Generate + extract formants + build centroids + score for one cell."""
    cell_dir = CELLS_DIR / f"cell_{cell['id']}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    result_path = cell_dir / "result.json"

    if result_path.exists():
        _log(f"  cell {cell['id']}: result.json exists, loading")
        return json.loads(result_path.read_text())

    _log(f"=== Cell {cell['id']}: {cell['note']} ===")
    t_start = time.time()

    gen_params = {**BASE_PARAMS}
    if cell["emo_audio"]:
        gen_params["emo_audio"] = cell["emo_audio"]
    if cell["use_emo_text"]:
        gen_params["use_emo_text"] = True
        gen_params["emo_text"] = cell["emo_text"]
    if cell["emo_alpha"] != 1.0:
        gen_params["emo_alpha"] = cell["emo_alpha"]

    manifest = generate_clips(
        phrases_csv=CAL_CSV,
        ref_audio=SPK_REF,
        out_dir=cell_dir / "clips",
        gen_params=gen_params,
        label="synth_BC",
    )

    formants_csv = extract_formants(
        manifest=manifest,
        out_csv=cell_dir / "formants.csv",
        source_label="synth_BC",
    )

    synth_centroids = build_centroids_from_formants(formants_csv)
    n_phonemes = len(synth_centroids)

    bc = baseline_centroids
    scores = {}
    for target in ["fry", "lindsey", "bbc_male", "modern_rp", "real_BC"]:
        scores[target] = score_against(synth_centroids, target, baseline_centroids=bc)["composite"]

    elapsed = time.time() - t_start
    result = {
        "cell_id":      cell["id"],
        "note":         cell["note"],
        "params":       {k: v for k, v in gen_params.items()},
        "n_phonemes":   n_phonemes,
        "fry":          round(scores["fry"],        2),
        "lindsey":      round(scores["lindsey"],    2),
        "bbc_male":     round(scores["bbc_male"],   2),
        "modern_rp":    round(scores["modern_rp"],  2),
        "real_BC":      round(scores["real_BC"],    2),
        "elapsed_s":    round(elapsed,              1),
    }

    (cell_dir / "centroids.json").write_text(json.dumps(synth_centroids, indent=2))
    result_path.write_text(json.dumps(result, indent=2))
    _log(f"  cell {cell['id']} DONE: fry={scores['fry']:.2f}  "
         f"modern_rp={scores['modern_rp']:.2f}  BC={scores['real_BC']:.2f}  "
         f"elapsed={elapsed:.0f}s")
    return result


def append_grid_row(result: dict) -> None:
    """Append one result to grid_results.csv (create with header if missing)."""
    GRID_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = ["cell_id", "note", "fry", "lindsey", "bbc_male", "modern_rp",
              "real_BC", "n_phonemes", "elapsed_s", "params"]
    write_header = not GRID_CSV.exists()
    with GRID_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow({**{k: result.get(k) for k in fields if k != "params"},
                    "params": json.dumps(result.get("params", {}))})


def main() -> int:
    PHASE11_DIR.mkdir(parents=True, exist_ok=True)
    CELLS_DIR.mkdir(parents=True, exist_ok=True)

    for p in [SPK_REF, FRY_REF, CAL_CSV]:
        if not p.exists():
            _log(f"ERROR: required input missing: {p}")
            return 1

    _log("=== Phase 0.11 emo sweep (Fry-only, simplified) ===")
    _log(f"  spk_ref:     {SPK_REF.name}")
    _log(f"  fry_ref:     {FRY_REF.name}")
    _log(f"  phrases:     {CAL_CSV.name}")
    _log(f"  base params: {BASE_PARAMS}")
    _log(f"  primary target: {PRIMARY_TARGET}")

    baseline_centroids = load_baseline_centroids()
    results: list[dict] = []
    cells = define_cells()

    for cell in cells:
        r = run_one_cell(cell, baseline_centroids)
        results.append(r)
        marker = CELLS_DIR / f"cell_{cell['id']}" / "_appended"
        if not marker.exists():
            append_grid_row(r)
            marker.touch()

    _summarize(results)
    return 0


def _summarize(results: list[dict]) -> None:
    _log(f"\n=== Phase 0.11 summary ({len(results)} cells run) ===")
    print(f"{'cell':<22s}  {'fry':>6s}  {'modern_rp':>10s}  {'BC':>6s}  "
          f"{'lindsey':>8s}  {'bbc':>6s}  note")
    print("-" * 100)
    for r in sorted(results, key=lambda r: -r[PRIMARY_TARGET]):
        print(f"{r['cell_id']:<22s}  {r['fry']:>6.2f}  {r['modern_rp']:>10.2f}  "
              f"{r['real_BC']:>6.2f}  {r['lindsey']:>8.2f}  {r['bbc_male']:>6.2f}  {r['note']}")
    baseline = next((r for r in results if r["cell_id"] == "00_baseline"), None)
    if baseline:
        winners = [r for r in results if r["cell_id"] != "00_baseline"
                   and r[PRIMARY_TARGET] > baseline[PRIMARY_TARGET]
                   and r["real_BC"] >= BC_FLOOR]
        if winners:
            best = max(winners, key=lambda r: r[PRIMARY_TARGET])
            lift = best[PRIMARY_TARGET] - baseline[PRIMARY_TARGET]
            _log(f"\nBest above baseline ({PRIMARY_TARGET}): {best['cell_id']}  "
                 f"{PRIMARY_TARGET}={best[PRIMARY_TARGET]:.2f} (+{lift:.2f})  "
                 f"BC={best['real_BC']:.2f}")
        else:
            _log(f"\nNo cell exceeded baseline {PRIMARY_TARGET}={baseline[PRIMARY_TARGET]:.2f} "
                 f"with BC>={BC_FLOOR}. Negative result.")
    _log(f"Master table: {GRID_CSV}")


if __name__ == "__main__":
    sys.exit(main())
