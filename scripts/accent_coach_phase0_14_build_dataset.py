"""WS-C C1 — Build LoRA training dataset from modern-RP corpus.

Produces train.jsonl and val.jsonl for GPT-LoRA fine-tuning.

Pipeline:
  1. Load kept_clips.json (fry + lindsey) — filter by sim >= 0.50 and
     0.8 s <= dur <= 10 s (under max_mel_tokens=1815).
  2. For bbc: duration filter only (no kept_clips / ECAPA sim available).
  3. Transcript quality gate: re-transcribe each clip with Whisper and drop
     clips where WER(stored_transcript, whisper_transcript) > 0.25.
     For bbc (no stored transcript), include all duration-passing clips and
     use the Whisper transcript directly.
  4. 90/10 train/val split by clip, fixed seed 0.
  5. Emit JSONL: {"wav": <abspath>, "text": <str>, "speaker": "fry|lindsey|bbc"}

Run: .venv/bin/python scripts/accent_coach_phase0_14_build_dataset.py

Acceptance: prints train/val counts; expect train ≈ 1400–1700.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import soundfile as sf

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

CORPUS_ROOT = PROJECT_ROOT / "tts_output/modern_rp_corpus"
CLEANED_ROOT = PROJECT_ROOT / "tts_output/accent_coach/cleaned_corpus"
OUT_ROOT = PROJECT_ROOT / "tts_output/accent_coach/phase0_14/lora/dataset"

SIM_THRESHOLD = 0.50
DUR_MIN = 0.8
DUR_MAX = 10.0
WER_THRESHOLD = 0.25
VAL_FRACTION = 0.10
RANDOM_SEED = 0


def _dur_ok(path: Path) -> float | None:
    """Return duration in seconds if in [DUR_MIN, DUR_MAX], else None."""
    try:
        info = sf.info(str(path))
        d = info.duration
        if DUR_MIN <= d <= DUR_MAX:
            return d
    except Exception:
        pass
    return None


def _wer(ref: str, hyp: str) -> float:
    from jiwer import wer as jiwer_wer
    return float(jiwer_wer(ref.lower(), hyp.lower()))


def _load_fry_lindsey(speaker: str, whisper_model) -> list[dict]:
    """Load and filter fry or lindsey clips."""
    clips_dir = CORPUS_ROOT / speaker / "clips"
    kept_path = CLEANED_ROOT / speaker / "kept_clips.json"
    trans_path = CLEANED_ROOT / speaker / "transcripts.json"

    if not kept_path.exists():
        print(f"  [{speaker}] kept_clips.json not found, skipping", file=sys.stderr)
        return []
    if not trans_path.exists():
        print(f"  [{speaker}] transcripts.json not found, skipping", file=sys.stderr)
        return []

    kept_clips: list[dict] = json.loads(kept_path.read_text())
    transcripts: dict[str, str] = json.loads(trans_path.read_text())

    kept_set = {
        e["clip"]: e for e in kept_clips
        if e.get("sim", 0.0) >= SIM_THRESHOLD
    }
    print(f"  [{speaker}] {len(kept_set)} clips pass sim>={SIM_THRESHOLD}", flush=True)

    results: list[dict] = []
    n_dur_fail = n_wer_fail = n_ok = 0
    for clip_name in kept_set:
        wav_path = clips_dir / clip_name
        if not wav_path.exists():
            continue
        dur = _dur_ok(wav_path)
        if dur is None:
            n_dur_fail += 1
            continue
        stored_text = transcripts.get(clip_name, "")
        if not stored_text:
            n_dur_fail += 1
            continue
        from lib.transcribe import transcribe_file
        hyp = transcribe_file(str(wav_path), model=whisper_model)
        w = _wer(stored_text, hyp)
        if w > WER_THRESHOLD:
            n_wer_fail += 1
            continue
        results.append({
            "wav": str(wav_path.resolve()),
            "text": hyp,
            "speaker": speaker,
        })
        n_ok += 1

    print(f"  [{speaker}] kept={n_ok}  dur_fail={n_dur_fail}  wer_fail={n_wer_fail}",
          flush=True)
    return results


def _load_bbc(whisper_model) -> list[dict]:
    """Load bbc clips — duration filter only (no kept_clips / ECAPA sim)."""
    clips_dir = CORPUS_ROOT / "bbc" / "clips"
    if not clips_dir.exists():
        print("  [bbc] clips dir not found", file=sys.stderr)
        return []

    results: list[dict] = []
    n_dur_fail = n_ok = 0
    for wav_path in sorted(clips_dir.glob("*.wav")):
        dur = _dur_ok(wav_path)
        if dur is None:
            n_dur_fail += 1
            continue
        from lib.transcribe import transcribe_file
        text = transcribe_file(str(wav_path), model=whisper_model)
        if not text.strip():
            n_dur_fail += 1
            continue
        results.append({
            "wav": str(wav_path.resolve()),
            "text": text,
            "speaker": "bbc",
        })
        n_ok += 1

    print(f"  [bbc] kept={n_ok}  dur_fail={n_dur_fail}", flush=True)
    return results


def _train_val_split(clips: list[dict], val_frac: float, seed: int
                     ) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = list(clips)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_frac))
    return shuffled[n_val:], shuffled[:n_val]


def _write_jsonl(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--whisper-model", default="large-v3",
                        help="Whisper model size (default: large-v3)")
    parser.add_argument("--speakers", nargs="*", default=["fry", "lindsey", "bbc"],
                        help="Speakers to include")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip Whisper transcription; use stored transcripts as-is")
    args = parser.parse_args()

    print(f"Loading Whisper {args.whisper_model}...", flush=True)
    if args.dry_run:
        whisper_model = None
        print("  dry-run: skipping Whisper load", flush=True)
    else:
        from lib.transcribe import load_whisper
        whisper_model = load_whisper(args.whisper_model)

    all_clips: list[dict] = []

    for speaker in args.speakers:
        print(f"\nProcessing {speaker}...", flush=True)
        if speaker == "bbc":
            if args.dry_run:
                print("  dry-run: skipping bbc (no stored transcripts)", flush=True)
            else:
                clips = _load_bbc(whisper_model)
                all_clips.extend(clips)
        else:
            if args.dry_run:
                kept_path = CLEANED_ROOT / speaker / "kept_clips.json"
                trans_path = CLEANED_ROOT / speaker / "transcripts.json"
                if kept_path.exists() and trans_path.exists():
                    kept = json.loads(kept_path.read_text())
                    trans = json.loads(trans_path.read_text())
                    clips_dir = CORPUS_ROOT / speaker / "clips"
                    for e in kept:
                        if e.get("sim", 0) < SIM_THRESHOLD:
                            continue
                        wav_path = clips_dir / e["clip"]
                        if not wav_path.exists():
                            continue
                        dur = _dur_ok(wav_path)
                        if dur is None:
                            continue
                        text = trans.get(e["clip"], "")
                        if text:
                            all_clips.append({
                                "wav": str(wav_path.resolve()),
                                "text": text,
                                "speaker": speaker,
                            })
                    print(f"  [{speaker}] dry-run: {len([c for c in all_clips if c['speaker']==speaker])} clips")
            else:
                clips = _load_fry_lindsey(speaker, whisper_model)
                all_clips.extend(clips)

    if not all_clips:
        print("ERROR: no clips collected.", file=sys.stderr)
        return 1

    train, val = _train_val_split(all_clips, VAL_FRACTION, RANDOM_SEED)

    train_path = OUT_ROOT / "train.jsonl"
    val_path = OUT_ROOT / "val.jsonl"
    _write_jsonl(train, train_path)
    _write_jsonl(val, val_path)

    print(f"\n{'='*50}")
    print(f"  Total clips: {len(all_clips)}")
    print(f"  Train: {len(train)}  Val: {len(val)}")
    print(f"  train.jsonl → {train_path}")
    print(f"  val.jsonl   → {val_path}")
    print(f"  By speaker: {dict.fromkeys(args.speakers, 0)}", end="")
    from collections import Counter
    sp_counts = Counter(c["speaker"] for c in all_clips)
    print(f"\r  By speaker: {dict(sp_counts)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
