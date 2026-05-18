"""
Transcribe specific utterance files that don't yet have transcripts.

Used as a follow-up to clustering: extend the transcripts.json cache to cover
all clips in the narrator cluster (some of which fell below the initial ECAPA
threshold and were therefore skipped by process_audiobook.py).

Usage:
  .venv/bin/python scripts/transcribe_missing.py \
      --work-dir dataset/audiobook_work/sherlock \
      --filter narrator_cluster
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _dotenv_init import init_env_then_reexec, kill_stale_python
init_env_then_reexec(__file__)

import warnings
warnings.filterwarnings("ignore")

from lib.transcribe import load_whisper, transcribe_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--filter", choices=["narrator_cluster", "all"], default="narrator_cluster")
    args = parser.parse_args()

    work_dir = Path(args.work_dir).resolve()
    utts_dir = work_dir / "utts"
    transcripts_path = work_dir / "transcripts.json"
    cluster_csv = work_dir / "cluster_labels.csv"

    if not utts_dir.is_dir():
        raise FileNotFoundError(utts_dir)
    transcripts: dict[str, str] = {}
    if transcripts_path.exists():
        transcripts = json.loads(transcripts_path.read_text())
    print(f"Existing transcripts: {len(transcripts)}")

    # Determine which utterances to transcribe
    if args.filter == "narrator_cluster":
        if not cluster_csv.exists():
            raise FileNotFoundError(f"Need {cluster_csv} — run cluster_characters.py first")
        keep = set()
        with cluster_csv.open() as f:
            for row in csv.DictReader(f):
                if int(row["is_narrator_cluster"]):
                    keep.add(row["utt_file"])
        target = keep
        print(f"Target: narrator cluster ({len(target)} files)")
    else:
        target = {p.name for p in utts_dir.glob("*.wav")}
        print(f"Target: all utterances ({len(target)} files)")

    todo = [name for name in sorted(target)
            if name not in transcripts and (utts_dir / name).exists()]
    print(f"Already transcribed: {len(target) - len(todo)}")
    print(f"To transcribe: {len(todo)}")

    if not todo:
        print("Nothing to do.")
        return

    n_killed = kill_stale_python()
    if n_killed:
        print(f"Pre-launch: killed {n_killed} stale procs")

    print("Loading Whisper large-v3 on CPU...")
    model = load_whisper(device="cpu")

    print(f"Transcribing {len(todo)} clips...")
    start = time.time()
    for i, name in enumerate(todo):
        try:
            transcripts[name] = transcribe_file(utts_dir / name, model)
        except Exception as e:
            print(f"  failed {name}: {e}")
            transcripts[name] = ""

        if (i + 1) % 25 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (len(todo) - (i + 1)) / rate / 60
            print(f"  {i+1}/{len(todo)}  rate={rate:.1f}/s  eta={eta:.1f}m")
            transcripts_path.write_text(json.dumps(transcripts, indent=2))

    transcripts_path.write_text(json.dumps(transcripts, indent=2))
    print(f"\n✓ Updated {transcripts_path}: {len(transcripts)} total transcripts")


if __name__ == "__main__":
    main()
