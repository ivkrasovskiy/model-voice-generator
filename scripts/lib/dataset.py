"""Dataset metadata.csv I/O + deterministic train/eval splits.

Conventions:
  - metadata.csv uses pipe-delimited rows: audio_file|text|duration
  - audio_file is the bare stem (no .wav suffix); WAVs live in <dataset_dir>/wavs/
"""

import csv
import random
import shutil
from pathlib import Path


def load_metadata(dataset_dir: str | Path, project_root: Path | None = None) -> list[dict]:
    """Load all valid clips from a dataset dir as {audio_file, text, duration, wav}.

    Skips rows whose WAV doesn't exist on disk.
    """
    dataset_dir = Path(dataset_dir)
    if project_root is not None and not dataset_dir.is_absolute():
        dataset_dir = project_root / "data" / str(dataset_dir)
    metadata = dataset_dir / "metadata.csv"
    if not metadata.exists():
        raise FileNotFoundError(metadata)
    entries = []
    with metadata.open() as f:
        for row in csv.DictReader(f, delimiter="|"):
            wav = dataset_dir / "wavs" / f"{row['audio_file']}.wav"
            if wav.exists():
                entries.append({
                    "audio_file": row["audio_file"],
                    "text": row["text"],
                    "duration": float(row["duration"]),
                    "wav": wav,
                })
    return entries


def write_metadata(dataset_dir: Path, entries: list[dict], copy_wavs: bool = True) -> Path:
    """Write entries as a fresh dataset directory.

    entries must have keys: audio_file (or wav), text, duration. If a 'wav' key
    exists and copy_wavs=True, the WAV is copied to <dataset_dir>/wavs/<new_id>.wav
    (renumbered seg_NNNNN).
    """
    dataset_dir = Path(dataset_dir)
    wavs_dir = dataset_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dataset_dir / "metadata.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="|")
        w.writerow(["audio_file", "text", "duration"])
        for i, e in enumerate(entries):
            new_id = f"seg_{i+1:05d}"
            dest = wavs_dir / f"{new_id}.wav"
            if copy_wavs and not dest.exists() and "wav" in e:
                shutil.copy(str(e["wav"]), str(dest))
            w.writerow([new_id, e["text"], f"{e['duration']:.3f}"])
    return csv_path


def deterministic_split(entries: list[dict], n_val: int, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Shuffle entries with the given seed, then split into (train, val) where
    val is the first n_val after shuffle.

    Reproducibility: same input + seed → identical split, across runs and platforms.
    """
    rng = random.Random(seed)
    shuffled = list(entries)
    rng.shuffle(shuffled)
    val = shuffled[:n_val]
    train = shuffled[n_val:]
    return train, val
