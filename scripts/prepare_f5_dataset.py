"""
Convert build_manifest.py output into F5-TTS fine-tuning dataset format.

F5-TTS expects:
    data/<dataset_name>/
        wavs/                  # symlinked audio files
        metadata.csv           # name|text|duration columns (pipe-delimited)
        raw.arrow              # built by F5-TTS prepare_csv_wavs.py

Usage:
    uv run python scripts/prepare_f5_dataset.py --book casanova

Output:
    data/cumberbatch_casanova/
        wavs/seg_*.wav (symlinks)
        metadata.csv
"""

import argparse
import csv
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SEGMENTS_BASE = PROJECT_ROOT / "dataset/segments"
F5_DATA_BASE = PROJECT_ROOT / "data"


def prepare_dataset(book: str, dataset_name: str | None = None) -> Path:
    book_dir = SEGMENTS_BASE / book
    manifest_path = book_dir / "training_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No training_manifest.csv at {manifest_path}. "
            "Run build_manifest.py first."
        )

    name = dataset_name or f"cumberbatch_{book}"
    out_dir = F5_DATA_BASE / name
    wavs_dir = out_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open() as f:
        rows = list(csv.DictReader(f))

    print(f"Preparing F5-TTS dataset '{name}' from {len(rows)} clips...")

    metadata_path = out_dir / "metadata.csv"
    with metadata_path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(["audio_file", "text", "duration"])
        kept = 0
        for row in rows:
            audio_src = Path(row["audio_path"])
            if not audio_src.exists():
                print(f"  skip missing {audio_src}")
                continue
            audio_dst = wavs_dir / audio_src.name
            if not audio_dst.exists():
                audio_dst.symlink_to(audio_src.resolve())
            writer.writerow([audio_src.stem, row["text"], row["duration_sec"]])
            kept += 1

    print(f"  wrote {kept} entries to {metadata_path}")
    print(f"  wavs symlinked in {wavs_dir}")

    total_hrs = sum(float(r["duration_sec"]) for r in rows) / 3600
    print(f"  total duration: {total_hrs:.2f} hours")

    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--book", default="casanova")
    parser.add_argument("--name", default=None,
                        help="Dataset name (defaults to cumberbatch_<book>)")
    args = parser.parse_args()
    prepare_dataset(args.book, args.name)


if __name__ == "__main__":
    main()
