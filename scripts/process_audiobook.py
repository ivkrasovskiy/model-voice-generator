"""
Process a Cumberbatch audiobook WAV → narrator-only training dataset.

Pipeline (4 stages, resumable, each stage writes its own artifacts):

  1. CHUNK     — split source WAV into 60s chunks (fast, no model)
  2. VAD       — silero-VAD segment each chunk into 1.5-12s utterances
  3. TRANSCRIBE — Whisper large-v3 transcript per utterance (CPU, slow)
  4. FILTER    — ECAPA-embed each utterance, compare to a narrator centroid,
                 keep only clips with sim ≥ threshold (drops character voices)

Each stage is idempotent and skips work already on disk.

Usage:
  .venv/bin/python scripts/process_audiobook.py \
      --source raw_cumberbatch_data/"Sherlock Holmes Stories ｜ Read by Benedict Cumberbatch [84VVFgGmeSA].wav" \
      --name sherlock \
      --narrator-centroid data/cumberbatch_casanova/centroid.npy \
      --filter-threshold 0.65

Output: data/cumberbatch_<name>/wavs/*.wav + metadata.csv
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _dotenv_init import init_env_then_reexec

init_env_then_reexec(__file__)

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import torch
import torchaudio
from lib.audio_io import read_wav_mono
from lib.identity import cosine, embed_file, load_ecapa
from lib.transcribe import load_whisper, transcribe_file

PROJECT_ROOT = Path(__file__).parent.parent
TARGET_SR = 24000  # F5-TTS native rate
CHUNK_SEC = 60     # chunk size for staged processing
MIN_UTT = 1.5      # min utterance duration in seconds
MAX_UTT = 12.0     # max — anything above F5-TTS ref-cap of 12s won't help


def stage_chunk(source: Path, work_dir: Path) -> Path:
    """Split source WAV into 60s chunks. Skip if already present."""
    chunks_dir = work_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    sentinel = chunks_dir / ".done"
    if sentinel.exists():
        n = len(list(chunks_dir.glob("*.wav")))
        print(f"[CHUNK] cached: {n} chunks in {chunks_dir}")
        return chunks_dir

    print(f"[CHUNK] reading {source.name} (this can take a couple of minutes for multi-GB files)...")
    info = sf.info(str(source))
    print(f"  duration: {info.duration/60:.1f} min, sr: {info.samplerate}")

    # Stream in 60s windows to avoid loading the whole file at once
    block_frames = CHUNK_SEC * info.samplerate
    n_chunks = int(np.ceil(info.duration / CHUNK_SEC))
    print(f"  splitting into {n_chunks} × {CHUNK_SEC}s chunks at {TARGET_SR} Hz...")
    with sf.SoundFile(str(source), "r") as f:
        for i in range(n_chunks):
            data = f.read(block_frames, dtype="float32", always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=1)
            # Resample to 24 kHz
            if info.samplerate != TARGET_SR:
                t = torch.from_numpy(data).float().unsqueeze(0)
                data = torchaudio.functional.resample(t, info.samplerate, TARGET_SR).squeeze(0).numpy()
            out = chunks_dir / f"chunk_{i:05d}.wav"
            sf.write(str(out), data, TARGET_SR)
            if (i + 1) % 20 == 0:
                print(f"  chunk {i+1}/{n_chunks}")
    sentinel.write_text("done")
    print(f"[CHUNK] done → {chunks_dir}")
    return chunks_dir


def stage_vad(chunks_dir: Path, work_dir: Path) -> Path:
    """Silero-VAD each chunk into 1.5-12s utterances. Skip if already present."""
    utts_dir = work_dir / "utts"
    utts_dir.mkdir(parents=True, exist_ok=True)
    sentinel = utts_dir / ".done"
    if sentinel.exists():
        n = len(list(utts_dir.glob("*.wav")))
        print(f"[VAD] cached: {n} utterances in {utts_dir}")
        return utts_dir

    print("[VAD] loading silero-VAD model...")
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
    )
    get_speech_timestamps = utils[0]

    chunks = sorted(chunks_dir.glob("*.wav"))
    print(f"[VAD] processing {len(chunks)} chunks...")
    utt_idx = 0
    for ci, ch_path in enumerate(chunks):
        wav, sr = read_wav_mono(ch_path)
        # silero needs 16kHz
        wav16 = torchaudio.functional.resample(
            torch.from_numpy(wav).float().unsqueeze(0), sr, 16000
        ).squeeze(0)
        timestamps = get_speech_timestamps(
            wav16, model, sampling_rate=16000,
            min_speech_duration_ms=int(MIN_UTT * 1000),
            max_speech_duration_s=MAX_UTT,
            min_silence_duration_ms=300,
        )
        # Map 16k frames back to 24k samples
        scale = sr / 16000
        for t in timestamps:
            start = int(t["start"] * scale)
            end = int(t["end"] * scale)
            seg = wav[start:end]
            dur = (end - start) / sr
            if dur < MIN_UTT or dur > MAX_UTT:
                continue
            out = utts_dir / f"utt_{utt_idx:06d}.wav"
            sf.write(str(out), seg, sr)
            utt_idx += 1
        if (ci + 1) % 20 == 0:
            print(f"  chunk {ci+1}/{len(chunks)}: {utt_idx} utts so far")

    sentinel.write_text(f"{utt_idx}")
    print(f"[VAD] done: {utt_idx} utterances → {utts_dir}")
    return utts_dir


def stage_filter(utts_dir: Path, work_dir: Path, centroid_path: Path, threshold: float) -> list[dict]:
    """ECAPA-filter each utt against the narrator centroid. Returns kept entries."""
    filter_log = work_dir / "filter_log.csv"

    print(f"[FILTER] loading ECAPA + centroid from {centroid_path.name}...")
    ecapa = load_ecapa(device="cpu")
    centroid = np.load(str(centroid_path))
    print(f"  centroid dim={centroid.shape}, norm={np.linalg.norm(centroid):.3f}")

    utts = sorted(utts_dir.glob("*.wav"))
    print(f"[FILTER] embedding + scoring {len(utts)} utterances...")

    rows = []
    with filter_log.open("w", newline="") as logf:
        log_w = csv.writer(logf)
        log_w.writerow(["utt_file", "duration", "ecapa_sim", "kept"])
        for i, p in enumerate(utts):
            emb = embed_file(p, ecapa)
            if emb is None:
                print(f"  failed: {p.name}")
                log_w.writerow([p.name, "", "", 0])
                continue
            wav, sr = read_wav_mono(p)
            dur = len(wav) / sr
            sim = cosine(centroid, emb)
            kept = sim >= threshold
            log_w.writerow([p.name, f"{dur:.2f}", f"{sim:.4f}", int(kept)])
            if kept:
                rows.append({"utt_path": p, "duration": dur, "sim": sim})
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(utts)}  kept so far: {len(rows)}")

    print(f"[FILTER] kept {len(rows)}/{len(utts)} ({len(rows)/len(utts)*100:.1f}%)  threshold={threshold}")
    print(f"[FILTER] log → {filter_log}")
    return rows


def stage_transcribe(kept: list[dict], work_dir: Path) -> list[dict]:
    """Whisper-transcribe only the kept clips (saves CPU vs transcribing the rejects)."""
    transcript_json = work_dir / "transcripts.json"
    cache: dict[str, str] = {}
    if transcript_json.exists():
        cache = json.loads(transcript_json.read_text())
        print(f"[TRANSCRIBE] cached: {len(cache)} transcripts on disk")

    todo = [r for r in kept if r["utt_path"].name not in cache]
    if not todo:
        print(f"[TRANSCRIBE] all {len(kept)} clips already transcribed")
        return [{**r, "text": cache[r["utt_path"].name]} for r in kept]

    print("[TRANSCRIBE] loading Whisper large-v3 on CPU...")
    whisper_model = load_whisper(device="cpu")

    print(f"[TRANSCRIBE] transcribing {len(todo)} clips...")
    start = time.time()
    for i, r in enumerate(todo):
        try:
            cache[r["utt_path"].name] = transcribe_file(r["utt_path"], whisper_model)
            if (i + 1) % 50 == 0:
                rate = (i + 1) / (time.time() - start)
                eta = (len(todo) - (i + 1)) / rate / 60
                print(f"  {i+1}/{len(todo)}  rate={rate:.1f}/s  eta={eta:.1f}m")
                transcript_json.write_text(json.dumps(cache, indent=2))
        except Exception as e:
            print(f"  failed {r['utt_path'].name}: {e}")
            cache[r["utt_path"].name] = ""

    transcript_json.write_text(json.dumps(cache, indent=2))
    print(f"[TRANSCRIBE] done → {transcript_json}")
    return [{**r, "text": cache[r["utt_path"].name]} for r in kept]


def stage_export(rows: list[dict], out_dataset_dir: Path):
    """Copy kept WAVs into final dataset dir + write metadata.csv."""
    out_wavs = out_dataset_dir / "wavs"
    out_wavs.mkdir(parents=True, exist_ok=True)
    metadata_csv = out_dataset_dir / "metadata.csv"

    # Drop empty transcripts
    rows = [r for r in rows if r.get("text", "").strip()]
    print(f"[EXPORT] writing {len(rows)} clips to {out_dataset_dir}")
    with metadata_csv.open("w", newline="") as f:
        w = csv.writer(f, delimiter="|")
        w.writerow(["audio_file", "text", "duration"])
        for i, r in enumerate(rows):
            out_name = f"seg_{i+1:05d}"
            out_wav = out_wavs / f"{out_name}.wav"
            if not out_wav.exists():
                import shutil
                shutil.copy(str(r["utt_path"]), str(out_wav))
            w.writerow([out_name, r["text"], f"{r['duration']:.3f}"])
    print(f"[EXPORT] metadata → {metadata_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to raw audiobook WAV")
    parser.add_argument("--name", required=True, help="Short name (e.g. sherlock)")
    parser.add_argument("--narrator-centroid", required=True,
                        help="Path to centroid.npy from a known-narrator dataset")
    parser.add_argument("--filter-threshold", type=float, default=0.65,
                        help="ECAPA sim threshold to keep clip (lower = more permissive)")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    centroid_path = Path(args.narrator_centroid).resolve()
    if not centroid_path.exists():
        raise FileNotFoundError(f"Centroid missing: {centroid_path}. Run audit_dataset.py first.")

    work_dir = PROJECT_ROOT / "dataset" / "audiobook_work" / args.name
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dataset = PROJECT_ROOT / "data" / f"cumberbatch_{args.name}"
    print(f"Source:    {source}")
    print(f"Work dir:  {work_dir}")
    print(f"Output:    {out_dataset}")
    print(f"Threshold: {args.filter_threshold}")
    print()

    chunks_dir = stage_chunk(source, work_dir)
    utts_dir = stage_vad(chunks_dir, work_dir)
    kept = stage_filter(utts_dir, work_dir, centroid_path, args.filter_threshold)
    rows = stage_transcribe(kept, work_dir)
    stage_export(rows, out_dataset)

    total_sec = sum(r["duration"] for r in rows if r.get("text", "").strip())
    print(f"\n✓ Final dataset: {len(rows)} clips, {total_sec/60:.1f} min ({total_sec/3600:.2f} h)")
    print(f"  → {out_dataset}")


if __name__ == "__main__":
    main()
