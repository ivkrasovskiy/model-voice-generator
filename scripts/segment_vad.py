"""
Re-segment Casanova audiobook using silero-VAD at natural silence boundaries.

Memory-efficient: streams audio via ffmpeg instead of loading 3.3 GB into RAM.

1. ffmpeg extracts the audio at 16 kHz mono → tmp file (~150 MB for Casanova)
2. silero-VAD runs on this and produces speech timestamps
3. For each speech segment, ffmpeg slices the original WAV at the target SR
   directly into the output clip

Usage:
    uv run python scripts/segment_vad.py --book casanova
"""

import argparse
import csv
import subprocess
import tempfile
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import torch
from tqdm import tqdm

RAW_DIR = Path("raw_cumberbatch_data")
OUT_BASE = Path("dataset/segments")

BOOK_FILES = {
    "casanova": "Casanova",
    "scales": "Scales of Justice",
    "artists": "Artists in Crime",
    "metamorphosis": "Metamorphosis",
    "sherlock": "Sherlock Holmes",
}


def find_raw_file(book: str) -> Path:
    substr = BOOK_FILES[book]
    matches = list(RAW_DIR.glob(f"*{substr}*"))
    if not matches:
        raise FileNotFoundError(f"No raw audio file matching '{substr}' in {RAW_DIR}")
    return matches[0]


def load_silero_vad():
    """Load silero-VAD. Cached after first call."""
    print("Loading silero-VAD...")
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    return model, utils


def ffmpeg_extract_16k_mono(src: Path, dst: Path) -> None:
    """Use ffmpeg to extract audio as 16 kHz mono PCM16 — small and stream-friendly."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def ffmpeg_extract_segment(src: Path, start_sec: float, dur_sec: float,
                           dst: Path, target_sr: int) -> None:
    """Slice [start_sec, start_sec + dur_sec) at target_sr mono, write to dst."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start_sec:.3f}",
        "-i", str(src),
        "-t", f"{dur_sec:.3f}",
        "-ac", "1", "-ar", str(target_sr),
        "-c:a", "pcm_s16le",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def load_16k_audio(path: Path) -> torch.Tensor:
    """Load 16 kHz mono PCM WAV into a torch tensor."""
    import soundfile as sf
    audio, sr = sf.read(str(path), dtype="float32")
    assert sr == 16000, f"Expected 16k, got {sr}"
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return torch.from_numpy(audio)


def segment_audio(wav_path: Path, out_dir: Path, min_sec: float,
                  max_sec: float, target_sr: int) -> list[dict]:
    model, utils = load_silero_vad()
    get_speech_timestamps = utils[0]  # utils is a namedtuple/tuple

    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    print(f"Extracting 16k mono via ffmpeg → {tmp_path.name}")
    ffmpeg_extract_16k_mono(wav_path, tmp_path)
    print(f"  size: {tmp_path.stat().st_size / 1e6:.1f} MB")

    print("Loading 16k audio into memory...")
    audio16 = load_16k_audio(tmp_path)
    duration_hrs = len(audio16) / 16000 / 3600
    print(f"  duration: {duration_hrs:.2f} hours")

    print("Running silero-VAD (this takes a few minutes for 5+ hours of audio)...")
    speech_ts = get_speech_timestamps(
        audio16,
        model,
        sampling_rate=16000,
        threshold=0.5,
        min_speech_duration_ms=int(min_sec * 1000),
        max_speech_duration_s=max_sec,
        min_silence_duration_ms=300,
        speech_pad_ms=100,
    )
    print(f"  produced {len(speech_ts)} raw speech regions")

    # Refine: split overlong; merge short adjacent
    refined: list[tuple[float, float]] = []
    for ts in speech_ts:
        start = ts["start"] / 16000
        end = ts["end"] / 16000
        dur = end - start
        if dur > max_sec:
            n = int(np.ceil(dur / max_sec))
            step = dur / n
            for i in range(n):
                refined.append((start + i * step, start + (i + 1) * step))
        else:
            refined.append((start, end))

    merged: list[tuple[float, float]] = []
    for start, end in refined:
        if merged and (end - merged[-1][0]) <= max_sec and (start - merged[-1][1]) < 0.5:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    final = [(s, e) for s, e in merged if (e - s) >= min_sec]
    durs = [e - s for s, e in final]
    print(f"  after refinement: {len(final)} segments  "
          f"(min={min(durs):.2f}s  max={max(durs):.2f}s  avg={np.mean(durs):.2f}s)")

    # Free memory before clip extraction
    del audio16, model
    tmp_path.unlink(missing_ok=True)

    print(f"Extracting clips via ffmpeg → {out_dir}")
    records = []
    for i, (s, e) in enumerate(tqdm(final, desc="Clips"), 1):
        name = f"seg_{i:05d}.wav"
        ffmpeg_extract_segment(wav_path, s, e - s, out_dir / name, target_sr)
        records.append({
            "segment": name,
            "start_sec": round(s, 3),
            "end_sec": round(e, 3),
            "duration_sec": round(e - s, 3),
        })

    manifest = out_dir / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["segment", "start_sec", "end_sec", "duration_sec"])
        writer.writeheader()
        writer.writerows(records)

    total_hrs = sum(r["duration_sec"] for r in records) / 3600
    print(f"\n{wav_path.name}: {len(records)} clips, {total_hrs:.2f} hours")
    print(f"Manifest → {manifest}")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--book", default="casanova", choices=list(BOOK_FILES.keys()) + ["all"])
    parser.add_argument("--min-sec", type=float, default=3.0)
    parser.add_argument("--max-sec", type=float, default=12.0)
    parser.add_argument("--target-sr", type=int, default=24000)  # F5-TTS native sample rate
    args = parser.parse_args()

    books = list(BOOK_FILES.keys()) if args.book == "all" else [args.book]
    for book in books:
        raw = find_raw_file(book)
        out_dir = OUT_BASE / book
        print(f"\n=== {book}: {raw.name} ===")
        segment_audio(raw, out_dir, args.min_sec, args.max_sec, args.target_sr)


if __name__ == "__main__":
    main()
