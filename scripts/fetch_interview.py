"""
Download a YouTube interview to raw_cumberbatch_data/ for further processing.

Requires yt-dlp on PATH: brew install yt-dlp
(Not a project dep — system install once, not training-pipeline Python.)

Usage:
  .venv/bin/python scripts/fetch_interview.py \
      --url "https://www.youtube.com/watch?v=XXXX" \
      --name graham_norton_2022

Then process with:
  .venv/bin/python scripts/process_audiobook.py \
      --source raw_cumberbatch_data/graham_norton_2022.wav \
      --name cumberbatch_graham_norton_2022 \
      --narrator-centroid data/cumberbatch_casanova/centroid.npy \
      --filter-threshold 0.65

Then combine into a training set:
  .venv/bin/python scripts/build_clean_dataset.py \
      --out-name combined_cas_sher_interview \
      --source cumberbatch_casanova_clean metadata \
      --source cumberbatch_sherlock_narrator metadata \
      --source cumberbatch_graham_norton_2022 metadata
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="YouTube URL")
    parser.add_argument("--name", required=True,
                        help="Output basename (no extension). Will be saved as "
                             "raw_cumberbatch_data/<name>.wav")
    args = parser.parse_args()

    if shutil.which("yt-dlp") is None:
        print("yt-dlp not found on PATH. Install with: brew install yt-dlp")
        sys.exit(1)

    out_dir = PROJECT_ROOT / "raw_cumberbatch_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.name}.wav"

    if out_path.exists():
        print(f"Already exists: {out_path}")
    else:
        cmd = [
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "--output", str(out_dir / f"{args.name}.%(ext)s"),
            args.url,
        ]
        print(f"Downloading: {args.url}")
        subprocess.run(cmd, check=True)
        print(f"Saved: {out_path}")

    print("\nNext — VAD + ECAPA filter + transcribe:")
    print("  .venv/bin/python scripts/process_audiobook.py \\")
    print(f"      --source {out_path} \\")
    print(f"      --name cumberbatch_{args.name} \\")
    print("      --narrator-centroid data/cumberbatch_casanova/centroid.npy \\")
    print("      --filter-threshold 0.65")


if __name__ == "__main__":
    main()
