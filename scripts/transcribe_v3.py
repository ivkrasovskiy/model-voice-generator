"""
Re-transcribe segmented clips with Whisper large-v3.

Better proper-noun handling than the original transcription (which botched
MEMOIRS → MEMOISE, San Samuele → Sam Simele, etc.). Runs on MPS.

Input:
    dataset/segments/<book>/seg_*.wav + manifest.csv

Output:
    dataset/segments/<book>/transcripts.csv
        columns: segment, text, no_speech_prob, avg_logprob, language

Usage:
    source .venv/bin/activate          # uses openai-whisper
    python scripts/transcribe_v3.py --book casanova
    # Or all:
    python scripts/transcribe_v3.py --book all
"""

import argparse
import csv
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import torch
from tqdm import tqdm

SEGMENTS_BASE = Path("dataset/segments")
BOOKS = ["casanova", "scales", "artists", "metamorphosis", "sherlock"]


def transcribe_book(book: str, model_name: str = "large-v3") -> None:
    book_dir = SEGMENTS_BASE / book
    manifest = book_dir / "manifest.csv"
    if not manifest.exists():
        print(f"No manifest at {manifest}, skipping {book}")
        return

    import whisper
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # whisper currently doesn't fully support MPS — falls back to CPU internally on some ops
    # but the load device matters for memory; safer to use CPU on Apple Silicon for large-v3
    load_device = "cpu"
    print(f"Loading whisper {model_name} on {load_device} (large-v3 is ~3 GB)...")
    model = whisper.load_model(model_name, device=load_device)

    with manifest.open() as f:
        rows = list(csv.DictReader(f))
    print(f"Transcribing {len(rows)} clips from {book}...")

    out_path = book_dir / "transcripts.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["segment", "duration_sec", "text", "no_speech_prob", "avg_logprob", "language"]
        )
        writer.writeheader()

        for row in tqdm(rows, desc=book):
            clip_path = book_dir / row["segment"]
            try:
                result = model.transcribe(
                    str(clip_path),
                    language="en",
                    fp16=False,
                    verbose=False,
                    condition_on_previous_text=False,  # avoid cross-clip hallucination
                    temperature=0.0,
                )
                # Per-clip aggregate stats
                segments = result.get("segments", [])
                no_speech_prob = (
                    sum(s.get("no_speech_prob", 0) for s in segments) / max(1, len(segments))
                ) if segments else 0.0
                avg_logprob = (
                    sum(s.get("avg_logprob", 0) for s in segments) / max(1, len(segments))
                ) if segments else 0.0

                writer.writerow({
                    "segment": row["segment"],
                    "duration_sec": row.get("duration_sec", ""),
                    "text": result["text"].strip(),
                    "no_speech_prob": round(no_speech_prob, 4),
                    "avg_logprob": round(avg_logprob, 4),
                    "language": result.get("language", "en"),
                })
            except Exception as e:
                print(f"  fail {row['segment']}: {e}")
                writer.writerow({
                    "segment": row["segment"],
                    "duration_sec": row.get("duration_sec", ""),
                    "text": "",
                    "no_speech_prob": 1.0,
                    "avg_logprob": -10,
                    "language": "",
                })

    print(f"Transcripts written → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--book", default="casanova", choices=BOOKS + ["all"])
    parser.add_argument("--model", default="large-v3",
                        choices=["large-v3", "large-v2", "medium", "small"])
    args = parser.parse_args()

    books = BOOKS if args.book == "all" else [args.book]
    for book in books:
        transcribe_book(book, args.model)


if __name__ == "__main__":
    main()
