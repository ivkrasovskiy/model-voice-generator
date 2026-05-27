"""Phase 0.13a — Lever B driver: formant-shift ceiling measurement.

Cells: cell_baseline_b, cell_strut, cell_foot, cell_thought, cell_mouth,
       cell_nurse, cell_all5.  Each at N=3 replicates.

Usage:
    .venv/bin/python scripts/accent_coach_phase0_13_lever_b.py \\
        --cell cell_baseline_b --replicates 3

    .venv/bin/python scripts/accent_coach_phase0_13_lever_b.py \\
        --cell all --replicates 3   # run every cell in order
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

from accent_coach.dsp.formant_shift import shift_vowels_to_centroid
from accent_coach.pipeline.experiment import (
    build_centroids_from_formants,
    extract_formants,
    generate_clips,
    load_baseline_centroids,
    score_against,
    score_posthoc_clips,
)

# ---------------------------------------------------------------------------
# Hard constraints (§1 of Phase 0.13 plan)
# ---------------------------------------------------------------------------

PHASE13_DIR  = PROJECT_ROOT / "tts_output/accent_coach/phase0_13"
CELLS_DIR    = PHASE13_DIR / "cells"
CAL_CSV      = PROJECT_ROOT / "tts_output/accent_coach/cal_25.csv"
SPK_REF      = PROJECT_ROOT / "tts_output/refs/production/ref_interview.wav"
ECAPA_REF    = PROJECT_ROOT / "tts_output/refs/indextts_baseline/ref_narrator.wav"

BASE_PARAMS = {
    "num_beams":       5,
    "temperature":     0.8,
    "top_p":           0.8,
    "cfg_rate":        0.7,
    "diffusion_steps": 25,
}

# IPA phoneme → cell name
SINGLE_CELLS: dict[str, str] = {
    "cell_strut":   "ʌ",
    "cell_foot":    "ʊ",
    "cell_thought": "ɔː",
    "cell_mouth":   "aʊ",
    "cell_nurse":   "ɜː",
}
ALL_5_PHONEMES = set(SINGLE_CELLS.values())

ALL_CELLS_ORDER = ["cell_baseline_b"] + list(SINGLE_CELLS.keys()) + ["cell_all5"]


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_target_centroid(baseline_cents: dict) -> dict:
    """Extract {phoneme: {f1, f2}} for the 5 failing phonemes from modern_rp."""
    out: dict[str, dict] = {}
    for ph in ALL_5_PHONEMES:
        entry = baseline_cents.get(ph, {}).get("modern_rp")
        if entry:
            out[ph] = {"f1": entry["f1"], "f2": entry["f2"]}
        else:
            _log(f"WARNING: modern_rp centroid missing for {ph}")
    return out


def _segments_by_clip(formants_csv: Path) -> dict[str, list[dict]]:
    """Load formants CSV → {clip_id: [segment_rows]}."""
    result: dict[str, list[dict]] = defaultdict(list)
    with formants_csv.open() as f:
        for row in csv.DictReader(f):
            cid = row.get("clip_id", "")
            # Skip rows with missing timing (old-format CSVs without start_s)
            if row.get("start_s", "") in ("", "nan") or row.get("end_s", "") in ("", "nan"):
                continue
            result[cid].append(row)
    return dict(result)


def _write_shifted_manifest(
    baseline_manifest: list[dict],
    shifted_dir: Path,
    phoneme_label: str,
) -> Path:
    """Write a new manifest.json pointing to shifted WAVs."""
    new_entries = []
    for entry in baseline_manifest:
        slug = entry.get("slug", entry.get("clip_id", "?"))
        orig_wav = Path(entry.get("wav_path", entry.get("path", "")))
        shifted_wav = shifted_dir / orig_wav.name
        new_entry = dict(entry)
        new_entry["wav_path"] = str(shifted_wav)
        if "path" in new_entry:
            new_entry["path"] = str(shifted_wav)
        new_entries.append(new_entry)
    out = shifted_dir / "manifest.json"
    out.write_text(json.dumps(new_entries, indent=2))
    return out


# ---------------------------------------------------------------------------
# Cell runners
# ---------------------------------------------------------------------------

def run_baseline(replicates: int, baseline_cents: dict) -> dict:
    """B1: generate N reps of cal_25 at Phase 0.9 params."""
    cell_dir = CELLS_DIR / "cell_baseline_b"
    rep_scores: list[dict] = []
    ph_scores: dict[str, list[float]] = defaultdict(list)

    for rep in range(replicates):
        rep_dir = cell_dir / f"rep_{rep}"
        _log(f"cell_baseline_b rep_{rep}: generating clips…")
        manifest_path = generate_clips(
            phrases_csv=CAL_CSV,
            ref_audio=SPK_REF,
            out_dir=rep_dir,
            gen_params=BASE_PARAMS,
            label="synth_BC",
        )

        formants_csv = rep_dir / "formants.csv"
        extract_formants(manifest_path, formants_csv, source_label="synth_BC")

        synth_cents = build_centroids_from_formants(formants_csv)
        result = score_against(synth_cents, "modern_rp", baseline_cents)
        composite = result["composite"]
        per_ph = result["per_phoneme"]

        _log(f"  composite_modern_rp = {composite:.2f}")
        rep_scores.append(composite)
        for ph in ALL_5_PHONEMES:
            if ph in per_ph:
                ph_scores[ph].append(per_ph[ph])

    summary = {
        "cell_id": "cell_baseline_b",
        "edited_phoneme": "none",
        "composite_modern_rp_mean": round(float(np.mean(rep_scores)), 2),
        "composite_modern_rp_std": round(float(np.std(rep_scores)), 2),
        "per_phoneme_target_mean": round(float(np.mean([v for vv in ph_scores.values() for v in vv])), 2),
        "per_phoneme_target_std": round(float(np.std([v for vv in ph_scores.values() for v in vv])), 2),
        "per_phoneme_by_phoneme": {ph: {"mean": round(float(np.mean(vs)), 2), "std": round(float(np.std(vs)), 2)}
                                   for ph, vs in ph_scores.items()},
        "rep_scores": rep_scores,
    }
    _save_result("cell_baseline_b", summary)
    _log(f"cell_baseline_b DONE: composite = {summary['composite_modern_rp_mean']:.2f} ± {summary['composite_modern_rp_std']:.2f}")

    # Gate check
    expected_lo, expected_hi = 72.0, 78.0
    if not (expected_lo <= summary["composite_modern_rp_mean"] <= expected_hi):
        _log(f"WARNING: baseline composite {summary['composite_modern_rp_mean']:.2f} outside expected [{expected_lo}, {expected_hi}]")
        _log("  Something may have drifted from Phase 0.11. Inspect before proceeding.")
    return summary


def run_shift_cell(
    cell_id: str,
    target_phonemes: set[str],
    replicates: int,
    baseline_cents: dict,
    target_centroid: dict,
) -> dict:
    """B4: shift target phonemes in baseline clips, re-extract, score."""
    cell_dir = CELLS_DIR / cell_id
    edited_phoneme = ", ".join(sorted(target_phonemes)) if len(target_phonemes) > 1 else next(iter(target_phonemes))

    rep_composites: list[float] = []
    ph_scores: dict[str, list[float]] = defaultdict(list)
    wers, ecapas, dnsmos_list = [], [], []
    t0_cell = time.time()

    for rep in range(replicates):
        baseline_rep_dir = CELLS_DIR / "cell_baseline_b" / f"rep_{rep}"
        baseline_manifest_path = baseline_rep_dir / "manifest.json"
        baseline_formants_csv = baseline_rep_dir / "formants.csv"

        if not baseline_manifest_path.exists():
            raise FileNotFoundError(
                f"Baseline rep_{rep} manifest missing. Run cell_baseline_b first.\n"
                f"  Expected: {baseline_manifest_path}"
            )

        # Ensure baseline formants exist (may need start_s/end_s — needs re-run if old)
        if not baseline_formants_csv.exists():
            extract_formants(baseline_manifest_path, baseline_formants_csv, source_label="synth_BC")

        baseline_manifest = json.loads(baseline_manifest_path.read_text())
        segs_by_clip = _segments_by_clip(baseline_formants_csv)

        rep_dir = cell_dir / f"rep_{rep}"
        rep_dir.mkdir(parents=True, exist_ok=True)

        _log(f"{cell_id} rep_{rep}: shifting phonemes {target_phonemes}…")
        edits_total = 0
        for entry in baseline_manifest:
            slug = entry.get("slug", entry.get("clip_id", "?"))
            orig_wav = Path(entry.get("wav_path", entry.get("path", "")))
            if not orig_wav.is_absolute():
                orig_wav = PROJECT_ROOT / orig_wav
            shifted_wav = rep_dir / orig_wav.name

            if shifted_wav.exists():
                edits_total += 1
                continue

            clip_segs = segs_by_clip.get(slug, segs_by_clip.get(str(orig_wav), []))
            r = shift_vowels_to_centroid(
                wav_in=orig_wav,
                wav_out=shifted_wav,
                segments=clip_segs,
                target_phonemes=target_phonemes,
                target_centroid=target_centroid,
            )
            edits_total += r["edits_applied"]

        shifted_manifest_path = _write_shifted_manifest(baseline_manifest, rep_dir, edited_phoneme)
        _log(f"  {edits_total} phoneme edits applied across {len(baseline_manifest)} clips")

        formants_csv = rep_dir / "formants.csv"
        extract_formants(shifted_manifest_path, formants_csv, source_label="synth_BC")

        synth_cents = build_centroids_from_formants(formants_csv)
        result = score_against(synth_cents, "modern_rp", baseline_cents)
        composite = result["composite"]
        per_ph = result["per_phoneme"]
        rep_composites.append(composite)
        for ph in ALL_5_PHONEMES:
            if ph in per_ph:
                ph_scores[ph].append(per_ph[ph])

        # Posthoc: WER / ECAPA / DNSMOS
        posthoc_csv = rep_dir / "posthoc.csv"
        posthoc = score_posthoc_clips(shifted_manifest_path, ECAPA_REF, posthoc_csv)
        wers.append(posthoc.get("WER", float("nan")))
        ecapas.append(posthoc.get("ECAPA", float("nan")))
        dnsmos_list.append(posthoc.get("DNSMOS_OVR", float("nan")))

        _log(f"  composite={composite:.2f}  WER={posthoc.get('WER', 'nan'):.3f}  "
             f"ECAPA={posthoc.get('ECAPA', 'nan'):.3f}  DNSMOS={posthoc.get('DNSMOS_OVR', 'nan'):.2f}")

    elapsed = time.time() - t0_cell
    summary = {
        "cell_id": cell_id,
        "edited_phoneme": edited_phoneme,
        "composite_modern_rp_mean": round(float(np.mean(rep_composites)), 2),
        "composite_modern_rp_std": round(float(np.std(rep_composites)), 2),
        "per_phoneme_target_mean": round(float(np.mean([v for vv in ph_scores.values() for v in vv])), 2) if ph_scores else float("nan"),
        "per_phoneme_target_std": round(float(np.std([v for vv in ph_scores.values() for v in vv])), 2) if ph_scores else float("nan"),
        "per_phoneme_by_phoneme": {ph: {"mean": round(float(np.mean(vs)), 2), "std": round(float(np.std(vs)), 2)}
                                   for ph, vs in ph_scores.items()},
        "wer_mean": round(float(np.nanmean(wers)), 4) if wers else float("nan"),
        "ecapa_mean": round(float(np.nanmean(ecapas)), 4) if ecapas else float("nan"),
        "dnsmos_ovr_mean": round(float(np.nanmean(dnsmos_list)), 4) if dnsmos_list else float("nan"),
        "elapsed_s": round(elapsed, 1),
    }
    _save_result(cell_id, summary)
    _log(f"{cell_id} DONE: composite = {summary['composite_modern_rp_mean']:.2f} ± "
         f"{summary['composite_modern_rp_std']:.2f}  elapsed={elapsed:.0f}s")
    return summary


def _save_result(cell_id: str, summary: dict) -> None:
    out = CELLS_DIR / cell_id / "result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))


def _write_lever_b_grid(results: list[dict]) -> None:
    out_csv = PHASE13_DIR / "lever_b_grid.csv"
    if not results:
        return
    fieldnames = ["cell_id", "edited_phoneme", "composite_modern_rp_mean",
                  "composite_modern_rp_std", "per_phoneme_target_mean",
                  "per_phoneme_target_std", "wer_mean", "ecapa_mean",
                  "dnsmos_ovr_mean", "elapsed_s"]
    rows = []
    for r in results:
        row = {k: r.get(k, "") for k in fieldnames}
        rows.append(row)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    _log(f"Lever B grid written to {out_csv}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 0.13a Lever B driver")
    parser.add_argument("--cell", required=True,
                        choices=ALL_CELLS_ORDER + ["all"],
                        help="Cell to run, or 'all' for all cells in order")
    parser.add_argument("--replicates", type=int, default=3)
    args = parser.parse_args()

    if args.replicates < 1:
        print("ERROR: replicates must be ≥ 1", file=sys.stderr)
        return 1

    PHASE13_DIR.mkdir(parents=True, exist_ok=True)
    CELLS_DIR.mkdir(parents=True, exist_ok=True)

    cells_to_run = ALL_CELLS_ORDER if args.cell == "all" else [args.cell]

    baseline_cents = load_baseline_centroids()
    target_centroid = _load_target_centroid(baseline_cents)

    results: list[dict] = []

    for cell_id in cells_to_run:
        _log(f"=== Running {cell_id} (N={args.replicates}) ===")

        if cell_id == "cell_baseline_b":
            r = run_baseline(args.replicates, baseline_cents)
        elif cell_id in SINGLE_CELLS:
            phoneme = SINGLE_CELLS[cell_id]
            r = run_shift_cell(
                cell_id=cell_id,
                target_phonemes={phoneme},
                replicates=args.replicates,
                baseline_cents=baseline_cents,
                target_centroid=target_centroid,
            )
        elif cell_id == "cell_all5":
            r = run_shift_cell(
                cell_id="cell_all5",
                target_phonemes=ALL_5_PHONEMES,
                replicates=args.replicates,
                baseline_cents=baseline_cents,
                target_centroid=target_centroid,
            )
        else:
            _log(f"Unknown cell: {cell_id}")
            return 1

        results.append(r)

    _write_lever_b_grid(results)

    # Print summary table
    print("\n=== Lever B Summary ===")
    print(f"{'Cell':<20} {'composite μ':>12} {'composite σ':>12} {'per-ph μ':>9} {'DNSMOS μ':>9}")
    for r in results:
        print(f"{r['cell_id']:<20} "
              f"{r.get('composite_modern_rp_mean', '?'):>12.2f} "
              f"{r.get('composite_modern_rp_std', '?'):>12.2f} "
              f"{r.get('per_phoneme_target_mean', float('nan')):>9.2f} "
              f"{r.get('dnsmos_ovr_mean', float('nan')):>9.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
