"""H3 robustness check — generate 15 sentences with 5 different BC reference clips.

Measures cross-ref variance to distinguish architecture-level distortion
from reference-clip sensitivity.

Usage:
    # Full run (generates ~75 clips, ~90 min CPU — confirm with owner first)
    .venv/bin/python scripts/accent_coach_refclip_robustness.py \\
        --ref-clips-json configs/accent_coach_phase0_5/ref_clips.json \\
        --sentences-csv  tts_output/cross_eval_50/eval_short.csv

    # Re-extract formants without regenerating audio
    .venv/bin/python scripts/accent_coach_refclip_robustness.py \\
        --ref-clips-json configs/accent_coach_phase0_5/ref_clips.json \\
        --sentences-csv  tts_output/cross_eval_50/eval_short.csv \\
        --skip-generation
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

DEFAULT_REFS_JSON = "configs/accent_coach_phase0_5/ref_clips.json"
DEFAULT_SENTENCES_CSV = "tts_output/cross_eval_50/eval_short.csv"
DEFAULT_OUT = "tts_output/refclip_robustness"
VENDOR_PYTHON = str(PROJECT_ROOT / "vendor/index-tts/.venv/bin/python")
INDEXTTS_GEN = str(PROJECT_ROOT / "scripts/indextts_gen.py")


def _build_manifest_for_ref(ref_out_dir: Path, sentences_csv: Path) -> Path:
    """Create a manifest.json from the generated clips + source CSV transcripts."""
    rows = []
    with sentences_csv.open() as f:
        src_rows = list(csv.DictReader(f))

    # First 15 sentences
    src_rows = src_rows[:15]

    for i, row in enumerate(src_rows):
        slug = row.get("slug", f"{i:04d}")
        transcript = row.get("prompt", row.get("text", ""))
        clip_path = ref_out_dir / f"{slug}.wav"
        if not clip_path.exists():
            # indextts_gen may use different naming
            candidates = sorted(ref_out_dir.glob("*.wav"))
            if i < len(candidates):
                clip_path = candidates[i]
            else:
                continue
        rows.append({
            "clip_id": slug,
            "path": str(clip_path.relative_to(PROJECT_ROOT)),
            "transcript": transcript,
        })

    manifest_path = ref_out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    return manifest_path


def run_robustness(
    refs_json: Path,
    sentences_csv: Path,
    out_dir: Path,
    skip_generation: bool,
) -> None:
    refs = json.loads(refs_json.read_text())

    if not skip_generation:
        print(f"\nGenerating {len(refs)} × 15 clips (~75 total). Estimated: 75–90 min CPU.", flush=True)
        print("Each reference clip will be used with the first 15 calibration sentences.", flush=True)

        for ref in refs:
            ref_path = ref["path"] if Path(ref["path"]).is_absolute() else PROJECT_ROOT / ref["path"]
            if not Path(ref_path).exists():
                print(f"  SKIP ref '{ref['name']}': file not found: {ref_path}", flush=True)
                continue

            ref_out = out_dir / ref["name"]
            ref_out.mkdir(parents=True, exist_ok=True)

            # Slice sentences CSV to first 15
            sentences_15 = out_dir / "sentences_15.csv"
            if not sentences_15.exists():
                with sentences_csv.open() as f:
                    src_rows = list(csv.DictReader(f))
                with sentences_15.open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(src_rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(src_rows[:15])

            print(f"\n  Generating ref='{ref['name']}' …", flush=True)
            subprocess.run([
                VENDOR_PYTHON, INDEXTTS_GEN,
                "--phrases-csv", str(sentences_15),
                "--ref-audio", str(ref_path),
                "--out-dir", str(ref_out),
            ], check=True)

    # Extract formants for each ref
    for ref in refs:
        ref_out = out_dir / ref["name"]
        formants_csv = ref_out / "formants.csv"

        if not ref_out.exists():
            print(f"  SKIP '{ref['name']}': output dir missing — run without --skip-generation first",
                  flush=True)
            continue

        manifest_path = ref_out / "manifest.json"
        if not manifest_path.exists():
            print(f"  Building manifest for '{ref['name']}' …", flush=True)
            manifest_path = _build_manifest_for_ref(ref_out, sentences_csv)

        print(f"\n  Extracting formants for ref='{ref['name']}' …", flush=True)
        subprocess.run([
            sys.executable,
            str(PROJECT_ROOT / "scripts/accent_coach_extract_formants.py"),
            "--manifest", str(manifest_path),
            "--out", str(formants_csv),
            "--source-label", f"synth_bc_ref_{ref['name']}",
        ], check=True)

    # Build summary.csv
    summary_rows = []
    for ref in refs:
        formants_csv = out_dir / ref["name"] / "formants.csv"
        if not formants_csv.exists():
            continue
        df = pd.read_csv(formants_csv)
        for ph, sub in df.groupby("phoneme"):
            summary_rows.append({
                "ref_name": ref["name"],
                "phoneme": str(ph),
                "F1_mean": round(float(sub.F1.mean()), 1),
                "F2_mean": round(float(sub.F2.mean()), 1),
                "n_tokens": len(sub),
            })

    summary_csv = out_dir / "summary.csv"
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
        print(f"\nRefclip robustness summary: {summary_csv}")
    else:
        print("\nNo formants extracted — summary.csv not written.", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="RefClip robustness — H3 cross-ref variance")
    parser.add_argument("--ref-clips-json", type=Path, default=DEFAULT_REFS_JSON)
    parser.add_argument("--sentences-csv", type=Path, default=DEFAULT_SENTENCES_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args()

    refs_json = args.ref_clips_json if args.ref_clips_json.is_absolute() else PROJECT_ROOT / args.ref_clips_json
    sentences_csv = args.sentences_csv if args.sentences_csv.is_absolute() else PROJECT_ROOT / args.sentences_csv
    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir

    if not refs_json.exists():
        print(f"ERROR: ref-clips JSON not found: {refs_json}", file=sys.stderr)
        return 1
    if not sentences_csv.exists():
        print(f"ERROR: sentences CSV not found: {sentences_csv}", file=sys.stderr)
        return 1

    if not args.skip_generation:
        print("\n>>> CHECKPOINT 3: About to run ~75 IndexTTS generations (~90 min CPU).")
        print("    Press Ctrl-C to abort, or wait 5 s to proceed …", flush=True)
        import time
        time.sleep(5)

    run_robustness(refs_json, sentences_csv, out_dir, args.skip_generation)
    return 0


if __name__ == "__main__":
    sys.exit(main())
