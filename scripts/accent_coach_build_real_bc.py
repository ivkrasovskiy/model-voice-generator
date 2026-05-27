"""Build D1 — real BC speech corpus from YouTube interview.

Downloads audio, diarizes with whisperx + pyannote, identifies BC via ECAPA
cosine similarity to a reference clip, trims overlap, writes per-segment clips.

Usage:
    .venv/bin/python scripts/accent_coach_build_real_bc.py
    .venv/bin/python scripts/accent_coach_build_real_bc.py \\
        --primary-url https://youtu.be/cHmkAStZBkc \\
        --ref-wav tts_output/refs/production/ref_interview.wav \\
        --out-dir tts_output/real_bc_corpus
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import soundfile as sf

from scripts.lib.whisperx_helpers import transcribe_diarize

DEFAULT_PRIMARY = "https://youtu.be/cHmkAStZBkc"
DEFAULT_SECONDARY = "https://www.youtube.com/watch?v=UKfBtgDSCzw"
DEFAULT_REF = "tts_output/refs/production/ref_interview.wav"
DEFAULT_OUT = "tts_output/real_bc_corpus"
DEFAULT_MIN_NET = 300  # 5 minutes


class EscalateToOwner(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _run_cmd(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDERR: {result.stderr[-2000:]}")


def _audio_duration(path: Path) -> float:
    import librosa
    return librosa.get_duration(path=str(path))


def _load_mono_16k(path: Path) -> np.ndarray:
    from scripts.lib.audio_io import read_wav_at
    return read_wav_at(path, 16000)


def _resample_to_16k(src: Path, dst: Path) -> None:
    """Download/convert to 16 kHz mono WAV using ffmpeg."""
    _run_cmd([
        "ffmpeg", "-y", "-i", str(src),
        "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
        str(dst),
    ])


def _trim_audio(src: Path, start_s: float, dst: Path) -> None:
    _run_cmd([
        "ffmpeg", "-y", "-ss", str(start_s), "-i", str(src),
        "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
        str(dst),
    ])


def _write_clip(src_wav: np.ndarray, sr: int, start_s: float, end_s: float, dst: Path) -> None:
    start_i = int(start_s * sr)
    end_i = int(end_s * sr)
    clip = src_wav[start_i:end_i]
    sf.write(str(dst), clip, sr, subtype="PCM_16")


def _concat_spans(wav: np.ndarray, sr: int, spans: list[dict]) -> np.ndarray:
    """Concatenate audio from a list of {start, end} segments."""
    parts = []
    for s in spans:
        si = int(s["start"] * sr)
        ei = int(s["end"] * sr)
        parts.append(wav[si:ei])
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Overlap detection
# ---------------------------------------------------------------------------

def _has_overlap(word: dict, others: list[dict], ms: float = 50.0) -> bool:
    """True if word overlaps ≥ ms milliseconds with any segment in others."""
    th = ms / 1000.0
    for o in others:
        overlap = min(word["end"], o["end"]) - max(word["start"], o["start"])
        if overlap >= th:
            return True
    return False


# ---------------------------------------------------------------------------
# Silence-gap segmentation
# ---------------------------------------------------------------------------

def _group_by_gap(words: list[dict], gap_ms: float = 200.0) -> list[dict]:
    """Group word-level dicts into segments by silence gaps."""
    if not words:
        return []
    gap_s = gap_ms / 1000.0
    segments = []
    seg_words = [words[0]]
    for w in words[1:]:
        if w["start"] - seg_words[-1]["end"] > gap_s:
            text = " ".join(x["word"] for x in seg_words)
            segments.append({
                "start": seg_words[0]["start"],
                "end": seg_words[-1]["end"],
                "text": text,
            })
            seg_words = [w]
        else:
            seg_words.append(w)
    if seg_words:
        text = " ".join(x["word"] for x in seg_words)
        segments.append({
            "start": seg_words[0]["start"],
            "end": seg_words[-1]["end"],
            "text": text,
        })
    return segments


# ---------------------------------------------------------------------------
# ECAPA speaker identification
# ---------------------------------------------------------------------------

def _ecapa_embedding(wav: np.ndarray, sr: int) -> np.ndarray:
    from scripts.lib.identity import embed_wav, load_ecapa
    ecapa = load_ecapa()
    return embed_wav(wav, sr, ecapa)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten()
    b = b.flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_real_bc(
    primary_url: str,
    secondary_url: str,
    secondary_start: float,
    ref_wav_path: Path,
    out_dir: Path,
    min_net_seconds: int,
    hf_token: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    clips_dir = out_dir / "clips"
    verify_dir = out_dir / "samples_to_verify"
    raw_dir.mkdir(exist_ok=True)
    clips_dir.mkdir(exist_ok=True)
    verify_dir.mkdir(exist_ok=True)

    # Load reference embedding
    print(f"Loading reference clip: {ref_wav_path}", flush=True)
    ref_wav, ref_sr = sf.read(str(ref_wav_path))
    if ref_wav.ndim > 1:
        ref_wav = ref_wav.mean(axis=1)
    ref_emb = _ecapa_embedding(ref_wav.astype(np.float32), ref_sr)

    all_segments: list[tuple[str, str, dict]] = []
    debug_rows: list[dict] = []

    sources = [
        ("primary", primary_url, 0.0),
        ("secondary", secondary_url, secondary_start),
    ]

    for source_name, url, start_offset in sources:
        raw_tmp = raw_dir / f"{source_name}.tmp.wav"
        raw_wav = raw_dir / f"{source_name}.wav"

        if raw_wav.exists():
            print(f"\n=== {source_name}: using existing {raw_wav} ({raw_wav.stat().st_size//1024//1024}MB) ===",
                  flush=True)
        else:
            print(f"\n=== Downloading {source_name}: {url} ===", flush=True)
            _run_cmd([
                "yt-dlp", "-x", "--audio-format", "wav",
                "--audio-quality", "0",
                "-o", str(raw_tmp).replace(".wav", ""),
                url,
            ])

            # yt-dlp may write .tmp.wav or .tmp.wav.wav depending on source format
            tmp_candidates = list(raw_dir.glob(f"{source_name}.tmp*"))
            if not tmp_candidates:
                raise EscalateToOwner(f"yt-dlp produced no output for {source_name}")
            actual_tmp = tmp_candidates[0]

            if start_offset > 0:
                _run_cmd([
                    "ffmpeg", "-y", "-ss", str(start_offset),
                    "-i", str(actual_tmp),
                    "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
                    str(raw_wav),
                ])
            else:
                _resample_to_16k(actual_tmp, raw_wav)
            actual_tmp.unlink(missing_ok=True)

        print(f"Transcribing + diarizing {source_name} …", flush=True)
        words = transcribe_diarize(raw_wav, hf_token)
        if not words:
            print(f"  WARNING: no words found in {source_name}", flush=True)
            continue

        # Load full audio for ECAPA
        source_wav, source_sr = sf.read(str(raw_wav))
        if source_wav.ndim > 1:
            source_wav = source_wav.mean(axis=1)
        source_wav = source_wav.astype(np.float32)

        # Compute ECAPA similarity per speaker
        speaker_ids = sorted(set(w["speaker"] for w in words))
        source_debug: list[dict] = []
        for sid in speaker_ids:
            spk_words = [w for w in words if w["speaker"] == sid]
            spk_audio = _concat_spans(source_wav, source_sr, spk_words)
            if len(spk_audio) < source_sr:
                sim = 0.0
            else:
                emb = _ecapa_embedding(spk_audio, source_sr)
                sim = _cosine(emb, ref_emb)
            n_words = len(spk_words)
            total_dur = sum(w["end"] - w["start"] for w in spk_words)
            source_debug.append({
                "source": source_name, "speaker_id": sid,
                "ecapa_cos_to_ref": round(sim, 4),
                "n_words": n_words, "total_dur_s": round(total_dur, 2),
                "kept_bool": False,
            })
        debug_rows.extend(source_debug)

        best = max(source_debug, key=lambda r: r["ecapa_cos_to_ref"])
        print(f"  Best speaker: {best['speaker_id']} cos={best['ecapa_cos_to_ref']:.3f}", flush=True)
        if best["ecapa_cos_to_ref"] < 0.5:
            raise EscalateToOwner(
                f"No BC cluster in {source_name}: max cos={best['ecapa_cos_to_ref']:.2f}. "
                f"Check diarization_debug.csv"
            )

        bc_sid = best["speaker_id"]
        for row in debug_rows:
            if row["source"] == source_name and row["speaker_id"] == bc_sid:
                row["kept_bool"] = True

        # Keep BC words, drop overlap with other speakers (≥ 50 ms)
        bc_words = [w for w in words if w["speaker"] == bc_sid]
        other_words = [w for w in words if w["speaker"] != bc_sid]
        bc_words = [w for w in bc_words if not _has_overlap(w, other_words, ms=50.0)]

        # Group into segments by 200 ms silence gaps, filter < 1 s
        segments = _group_by_gap(bc_words, gap_ms=200.0)
        segments = [s for s in segments if s["end"] - s["start"] >= 1.0]
        print(f"  Kept {len(segments)} segments, {sum(s['end']-s['start'] for s in segments):.1f}s total", flush=True)

        all_segments.extend([(source_name, url, s) for s in segments])

        net = sum(s["end"] - s["start"] for _, _, s in all_segments)
        if source_name == "primary" and net >= min_net_seconds:
            print(f"  Primary sufficient ({net:.0f}s ≥ {min_net_seconds}s) — skipping secondary", flush=True)
            break

    # Write debug CSV
    debug_csv = out_dir / "diarization_debug.csv"
    with debug_csv.open("w", newline="") as f:
        if debug_rows:
            w = csv.DictWriter(f, fieldnames=list(debug_rows[0].keys()))
            w.writeheader()
            w.writerows(debug_rows)
    print(f"\nDiarization debug: {debug_csv}", flush=True)

    net = sum(s["end"] - s["start"] for _, _, s in all_segments)
    if net < min_net_seconds:
        raise EscalateToOwner(
            f"Insufficient BC audio: {net:.1f}s < {min_net_seconds}s. "
            f"Add more sources or lower --min-net-seconds."
        )
    print(f"\nTotal BC audio: {net:.1f}s across {len(all_segments)} segments", flush=True)

    # Load source wavs for clip writing
    source_wavs: dict[str, tuple[np.ndarray, int]] = {}
    for source_name in {s for s, _, _ in all_segments}:
        wav, sr = sf.read(str(raw_dir / f"{source_name}.wav"))
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        source_wavs[source_name] = (wav.astype(np.float32), sr)

    # Write clips + manifest
    manifest = []
    for i, (source_name, url, seg) in enumerate(all_segments):
        clip_path = clips_dir / f"{i:04d}.wav"
        wav, sr = source_wavs[source_name]
        _write_clip(wav, sr, seg["start"], seg["end"], clip_path)
        manifest.append({
            "clip_id": f"{i:04d}",
            "path": str(clip_path.relative_to(PROJECT_ROOT)),
            "source_url": url,
            "start_s": round(seg["start"], 3),
            "end_s": round(seg["end"], 3),
            "transcript": seg["text"],
        })

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Manifest: {manifest_path}  ({len(manifest)} clips)", flush=True)

    # Pick 5 random clips for owner listen-check
    sample_indices = random.sample(range(len(manifest)), min(5, len(manifest)))
    for j, idx in enumerate(sample_indices):
        src = PROJECT_ROOT / manifest[idx]["path"]
        dst = verify_dir / f"{j:02d}.wav"
        shutil.copy(src, dst)
    print(f"Samples to verify: {verify_dir}/", flush=True)
    print("\n>>> CHECKPOINT 1: listen to tts_output/real_bc_corpus/samples_to_verify/*.wav", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Build D1 real-BC corpus from YouTube")
    parser.add_argument("--primary-url", default=DEFAULT_PRIMARY)
    parser.add_argument("--secondary-url", default=DEFAULT_SECONDARY)
    parser.add_argument("--secondary-start", type=float, default=50.0,
                        help="Seconds to skip at start of secondary source")
    parser.add_argument("--ref-wav", default=DEFAULT_REF)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--min-net-seconds", type=int, default=DEFAULT_MIN_NET)
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("ERROR: set HF_TOKEN to use pyannote/speaker-diarization-3.1", file=sys.stderr)
        print("  export HF_TOKEN=hf_xxx", file=sys.stderr)
        return 1

    ref_wav_path = PROJECT_ROOT / args.ref_wav
    if not ref_wav_path.exists():
        print(f"ERROR: ref wav not found: {ref_wav_path}", file=sys.stderr)
        return 1

    out_dir = PROJECT_ROOT / args.out_dir

    try:
        build_real_bc(
            primary_url=args.primary_url,
            secondary_url=args.secondary_url,
            secondary_start=args.secondary_start,
            ref_wav_path=ref_wav_path,
            out_dir=out_dir,
            min_net_seconds=args.min_net_seconds,
            hf_token=hf_token,
        )
    except EscalateToOwner as e:
        print(f"\nEscalate to owner: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
