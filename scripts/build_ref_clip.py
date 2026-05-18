"""
Build a combined reference clip for F5-TTS: concatenate a Casanova segment and a
Sherlock segment into a single file.

Two outputs:
  tts_output/ref_combined.wav  — full clip (20s + 10s = 30s).
                                  F5-TTS truncates this to ~12s at inference,
                                  so it effectively conditions on the Casanova part.
                                  Used for building a richer ECAPA embedding only.

  tts_output/ref_sherlock.wav  — standalone Sherlock 12s clip for when we want
                                  Sherlock-register conditioning at inference.

Usage:
  .venv/bin/python scripts/build_ref_clip.py
  (output paths printed to stdout)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import soundfile as sf
from lib.audio_io import read_wav_mono

PROJECT_ROOT = Path(__file__).parent.parent
TARGET_SR = 24000

# ---- Source clips ----
# Casanova: existing narrator ref (12s, already at ceiling)
CAS_REF = PROJECT_ROOT / "tts_output/ref_narrator.wav"

# Sherlock: pick a clean narrator clip from the val set (not in training)
# "Holmes took the hat and turned it around in his hands." — 3.5s, clean
SHER_CLIP = PROJECT_ROOT / "data/cumberbatch_sherlock_narrator/wavs/seg_00001.wav"


def best_sherlock_clip() -> Path:
    """Return the path to the best short Sherlock narrator clip not in training."""
    import csv
    import random

    train_texts: set[str] = set()
    for path in ["data/cumberbatch_sherlock_train_354/metadata.csv"]:
        with open(PROJECT_ROOT / path) as fh:
            for row in csv.DictReader(fh, delimiter="|"):
                train_texts.add(row["text"].strip())

    candidates = []
    meta = PROJECT_ROOT / "data/cumberbatch_sherlock_narrator/metadata.csv"
    base = meta.parent / "wavs"
    with open(meta) as fh:
        for row in csv.DictReader(fh, delimiter="|"):
            if row["text"].strip() not in train_texts and 3 <= float(row["duration"]) <= 8:
                candidates.append((row["audio_file"], row["text"], float(row["duration"])))

    if not candidates:
        raise RuntimeError("No clean Sherlock clips found")

    # Pick deterministically (seed 42), prefer mid-length
    rng = random.Random(42)
    pick = rng.choice(candidates)
    print(f"  Sherlock clip: {pick[0]}  {pick[2]:.1f}s  \"{pick[1][:60]}\"")
    return base / f"{pick[0]}.wav"


def main():
    sher_path = best_sherlock_clip()

    cas_wav, cas_sr = read_wav_mono(CAS_REF)
    sher_wav, sher_sr = read_wav_mono(sher_path)

    # Resample both to 24 kHz if needed
    from lib.audio_io import resample
    cas_wav = resample(cas_wav, cas_sr, TARGET_SR)
    sher_wav = resample(sher_wav, sher_sr, TARGET_SR)

    # Truncate Casanova to 20s, Sherlock to 10s
    cas_wav = cas_wav[: int(20 * TARGET_SR)]
    sher_wav = sher_wav[: int(10 * TARGET_SR)]

    # Combine with 0.5s silence separator
    silence = np.zeros(int(0.5 * TARGET_SR), dtype=np.float32)
    combined = np.concatenate([cas_wav, silence, sher_wav])

    out_combined = PROJECT_ROOT / "tts_output/ref_combined.wav"
    out_sherlock = PROJECT_ROOT / "tts_output/ref_sherlock.wav"

    sf.write(str(out_combined), combined, TARGET_SR)
    sf.write(str(out_sherlock), sher_wav, TARGET_SR)

    print(f"Combined ref: {out_combined}  ({len(combined)/TARGET_SR:.1f}s)")
    print("  → F5-TTS will use first ~12s (Casanova part) at inference")
    print(f"Sherlock ref: {out_sherlock}  ({len(sher_wav)/TARGET_SR:.1f}s)")
    print("  → Use for Sherlock-register inference conditioning")


if __name__ == "__main__":
    main()
