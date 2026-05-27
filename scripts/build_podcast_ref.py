"""
Download + trim a YouTube interview clip to use as an IndexTTS-2 reference.

The current baseline `tts_output/refs/indextts_baseline/ref_narrator.wav` is audiobook material and
carries BC's narrator-creak. This script grabs a clip from an interview/podcast
(no creak, normal conversational voice) and produces a drop-in replacement.

IndexTTS-2 internally truncates speaker prompts to 15s
(see vendor/index-tts/indextts/infer_v2.py:435 `_load_and_cut_audio(..., 15)`),
so the first 15s of the trimmed file is what actually conditions the model.
Anything beyond that is unused by inference but kept on disk for future use.

Default config matches the locked baseline format: 22050 Hz, mono, s16le PCM.

Usage:
    .venv/bin/python scripts/build_podcast_ref.py
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_URL = "https://youtu.be/cHmkAStZBkc"
DEFAULT_START = "00:04:31"
DEFAULT_DURATION = "34"
DEFAULT_OUT = "tts_output/refs/production/ref_interview.wav"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--start", default=DEFAULT_START, help="ffmpeg -ss timestamp")
    parser.add_argument("--duration", default=DEFAULT_DURATION, help="ffmpeg -t seconds")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--force", action="store_true", help="overwrite if exists")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    out_path = project_root / args.out
    if out_path.exists() and not args.force:
        print(f"  {out_path} already exists, skipping (use --force to overwrite)")
        return 0

    for tool in ("yt-dlp", "ffmpeg"):
        if shutil.which(tool) is None:
            print(f"ERROR: {tool} not on PATH. brew install {tool}")
            return 1

    work_dir = project_root / "tts_output" / "_podcast_dl"
    work_dir.mkdir(parents=True, exist_ok=True)
    raw_path = work_dir / "raw.%(ext)s"

    print(f"yt-dlp → {raw_path}")
    yt_cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "-o", str(raw_path),
        "--force-overwrites",
        args.url,
    ]
    r = subprocess.run(yt_cmd)
    if r.returncode != 0:
        print(f"ERROR: yt-dlp failed (exit {r.returncode})")
        return 1

    downloaded = work_dir / "raw.wav"
    if not downloaded.exists():
        print(f"ERROR: expected {downloaded} not found")
        return 1

    print(f"ffmpeg trim {args.start} +{args.duration}s → {out_path} ({args.sr} Hz mono)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ff_cmd = [
        "ffmpeg", "-y",
        "-ss", args.start,
        "-t", args.duration,
        "-i", str(downloaded),
        "-ac", "1",
        "-ar", str(args.sr),
        "-c:a", "pcm_s16le",
        str(out_path),
    ]
    r = subprocess.run(ff_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"ERROR: ffmpeg failed:\n{r.stderr[-500:]}")
        return 1

    info = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(out_path)],
        capture_output=True, text=True,
    )
    dur = float(info.stdout.strip())
    print(f"✓ {out_path}  ({dur:.2f}s, {args.sr} Hz mono)")
    print("  Note: IndexTTS-2 will use only the first 15s of this clip.")
    print(f"  Cleanup work dir: rm -rf {work_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
