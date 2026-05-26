"""Single-cell runner for Phase 0.10 parameter sweep.

Given one set of generation params and a replicate index, runs the full
pipeline: generate clips → extract formants → build centroids → score
against lindsey and real_BC → compute loss.

Usage:
    .venv/bin/python scripts/accent_coach_phase0_10_run_cell.py \\
        --params-json '{"num_beams":5,"temperature":0.8,"top_p":0.8,"cfg_rate":0.7,"diffusion_steps":25}' \\
        --replicate 0 \\
        --out-dir tts_output/accent_coach/phase0_10/cell_smoke/
"""
from __future__ import annotations

import argparse
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

REF_AUDIO  = PROJECT_ROOT / "tts_output/ref_interview.wav"
CAL_25_CSV = PROJECT_ROOT / "tts_output/accent_coach/cal_25.csv"


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def run_cell(
    params: dict,
    replicate: int,
    out_dir: Path,
    cal_csv: Path | None = None,
    ref_audio: Path | None = None,
) -> dict:
    """Run one cell of the Phase 0.10 parameter sweep.

    Returns {'composite_lindsey', 'composite_BC', 'loss'}.
    """
    if cal_csv is None:
        cal_csv = CAL_25_CSV
    if ref_audio is None:
        ref_audio = REF_AUDIO

    out_dir = Path(out_dir)
    cell_dir = out_dir / f"rep{replicate}"
    cell_dir.mkdir(parents=True, exist_ok=True)

    _log(f"=== Cell  rep={replicate}  params={params} ===")
    t_start = time.time()

    # Step 1: generate clips
    _log(f"[rep{replicate}] Step 1/4: generating clips (cal={cal_csv.name}, ref={ref_audio.name})")
    manifest = generate_clips(
        phrases_csv=cal_csv,
        ref_audio=ref_audio,
        out_dir=cell_dir / "clips",
        gen_params=params,
        label="synth_BC",
    )

    # Step 2: extract formants
    _log(f"[rep{replicate}] Step 2/4: extracting formants")
    formants_csv = extract_formants(
        manifest=manifest,
        out_csv=cell_dir / "formants.csv",
        source_label="synth_BC",
    )

    # Step 3: build synth_BC centroids
    _log(f"[rep{replicate}] Step 3/4: building centroids")
    synth_centroids = build_centroids_from_formants(formants_csv)
    n_phonemes = len(synth_centroids)
    _log(f"  {n_phonemes} phonemes: {sorted(synth_centroids)}")
    if n_phonemes < 6:
        raise RuntimeError(
            f"Only {n_phonemes} phonemes from formant extraction — something went wrong"
        )

    # Step 4: score against lindsey + real_BC
    _log(f"[rep{replicate}] Step 4/4: scoring against lindsey and real_BC")
    baseline = load_baseline_centroids()

    result_lindsey = score_against(synth_centroids, "lindsey",  baseline_centroids=baseline)
    result_bc      = score_against(synth_centroids, "real_BC",  baseline_centroids=baseline)

    composite_lindsey = result_lindsey["composite"]
    composite_bc      = result_bc["composite"]
    loss = 0.5 * (100 - composite_lindsey) + 0.5 * (100 - composite_bc)

    elapsed = time.time() - t_start
    _log(f"[rep{replicate}] Done in {elapsed:.0f}s  "
         f"composite_lindsey={composite_lindsey:.2f}  "
         f"composite_BC={composite_bc:.2f}  "
         f"loss={loss:.4f}")

    result = {
        "replicate":          replicate,
        "params":             params,
        "composite_lindsey":  composite_lindsey,
        "composite_BC":       composite_bc,
        "loss":               loss,
        "elapsed_s":          round(elapsed, 1),
        "n_phonemes":         n_phonemes,
    }

    # Sanity check
    for name, val in [("composite_lindsey", composite_lindsey), ("composite_BC", composite_bc)]:
        if not (60 <= val <= 95):
            _log(f"WARNING: {name}={val:.2f} is outside expected range [60, 95] — check pipeline")

    # Persist result and centroids alongside the cell's outputs
    (cell_dir / "result.json").write_text(json.dumps(result, indent=2))
    (cell_dir / "centroids.json").write_text(json.dumps(synth_centroids, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 0.10 single-cell runner")
    parser.add_argument("--params-json", required=True,
                        help='JSON dict of generation params, e.g. \'{"num_beams":5,...}\'')
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cal", default=None,
                        help="Path to cal CSV (default: tts_output/accent_coach/cal_25.csv)")
    args = parser.parse_args()

    params = json.loads(args.params_json)
    cal_csv = Path(args.cal) if args.cal else None

    result = run_cell(
        params=params,
        replicate=args.replicate,
        out_dir=Path(args.out_dir),
        cal_csv=cal_csv,
    )

    print(f"\ncomposite_lindsey={result['composite_lindsey']:.2f}")
    print(f"composite_BC={result['composite_BC']:.2f}")
    print(f"loss={result['loss']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
