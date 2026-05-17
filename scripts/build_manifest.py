"""
Build the final training manifest from segmented + transcribed clips.

Combines:
    dataset/segments/<book>/manifest.csv     (segments + durations)
    dataset/segments/<book>/transcripts.csv  (whisper large-v3 text)
    dataset/style_analysis/<book>/narrator_chunks.txt  (optional narrator filter)

Outputs a training-ready manifest with quality filtering applied.

Usage:
    source .venv/bin/activate
    python scripts/build_manifest.py --book casanova

Output:
    dataset/segments/<book>/training_manifest.csv
        columns: audio_path, text, duration_sec, speaker
"""

import argparse
import csv
from pathlib import Path

SEGMENTS_BASE = Path("dataset/segments")


def build_manifest(book: str, min_logprob: float, max_no_speech: float,
                   min_dur: float, max_dur: float) -> None:
    book_dir = SEGMENTS_BASE / book
    transcripts_path = book_dir / "transcripts.csv"
    if not transcripts_path.exists():
        print(f"No transcripts at {transcripts_path}; run transcribe_v3.py first")
        return

    with transcripts_path.open() as f:
        rows = list(csv.DictReader(f))

    kept = []
    dropped = {"short": 0, "long": 0, "low_quality": 0, "no_speech": 0, "empty": 0}

    for row in rows:
        text = row["text"].strip()
        if not text or len(text) < 10:
            dropped["empty"] += 1
            continue
        dur = float(row.get("duration_sec") or 0)
        if dur < min_dur:
            dropped["short"] += 1
            continue
        if dur > max_dur:
            dropped["long"] += 1
            continue
        nsp = float(row.get("no_speech_prob") or 0)
        if nsp > max_no_speech:
            dropped["no_speech"] += 1
            continue
        lp = float(row.get("avg_logprob") or -10)
        if lp < min_logprob:
            dropped["low_quality"] += 1
            continue
        kept.append({
            "audio_path": str((book_dir / row["segment"]).resolve()),
            "text": text,
            "duration_sec": dur,
            "speaker": "benedict_cumberbatch",
        })

    out_path = book_dir / "training_manifest.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_path", "text", "duration_sec", "speaker"])
        writer.writeheader()
        writer.writerows(kept)

    total_hrs = sum(r["duration_sec"] for r in kept) / 3600
    print(f"\n=== {book} manifest ===")
    print(f"  kept:     {len(kept):5d} clips  ({total_hrs:.2f} hours)")
    for reason, n in dropped.items():
        print(f"  dropped {reason:12s} {n:5d}")
    print(f"  → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--book", default="casanova")
    parser.add_argument("--min-logprob", type=float, default=-1.0)
    parser.add_argument("--max-no-speech", type=float, default=0.3)
    parser.add_argument("--min-dur", type=float, default=2.0)
    parser.add_argument("--max-dur", type=float, default=15.0)
    args = parser.parse_args()

    build_manifest(args.book, args.min_logprob, args.max_no_speech,
                   args.min_dur, args.max_dur)


if __name__ == "__main__":
    main()
