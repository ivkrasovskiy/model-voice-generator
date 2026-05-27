"""Phase 0.13b — mine phoneme-dense emo clips from cleaned corpus (Task A1).

Selects 3 candidate clips (or 5-s windows within them) from the Fry/Lindsey
cleaned corpus that maximise density of the 5 failing phonemes:
    /ʌ/, /ʊ/, /ɔː/, /aʊ/, /ɜː/  (STRUT, FOOT, THOUGHT, MOUTH, NURSE)

Each candidate must contain ≥ 8 target-phoneme tokens.

Output:
    tts_output/accent_coach/phase0_13/emo_dense/cand_{0,1,2}.wav
    tts_output/accent_coach/phase0_13/emo_dense/index.json

Usage:
    .venv/bin/python scripts/accent_coach_phase0_13_mine_emo.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import soundfile as sf

TARGET_PHONEMES = {"ʌ", "ʊ", "ɔː", "aʊ", "ɜː"}
MIN_TARGET_TOKENS = 8       # §4.3 A1 gate
MAX_CLIP_DURATION_S = 5.0   # trim to 5 s window if clip is longer
N_CANDIDATES = 3

CLEANED_DIR   = PROJECT_ROOT / "tts_output/accent_coach/cleaned_corpus"
EMO_DENSE_DIR = PROJECT_ROOT / "tts_output/accent_coach/phase0_13/emo_dense"


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def load_formant_rows(speaker: str) -> list[dict]:
    """Load per-token rows from the cleaned corpus formants CSV."""
    csv_path = CLEANED_DIR / speaker / "formants.csv"
    if not csv_path.exists():
        _log(f"WARNING: {csv_path} not found, skipping {speaker}")
        return []
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def build_clip_stats(rows: list[dict]) -> dict[str, dict]:
    """Aggregate per-clip stats: total tokens, target tokens, raw file path."""
    clips: dict[str, dict] = {}
    for row in rows:
        clip_id = row.get("clip_id", "")
        if not clip_id:
            continue
        if clip_id not in clips:
            clips[clip_id] = {
                "clip_id": clip_id,
                "total_tokens": 0,
                "target_tokens": 0,
                "target_density": 0.0,
                "tokens": [],
            }
        clips[clip_id]["total_tokens"] += 1
        phoneme = row.get("phoneme", "")
        if phoneme in TARGET_PHONEMES:
            clips[clip_id]["target_tokens"] += 1
        clips[clip_id]["tokens"].append(row)
    for c in clips.values():
        denom = max(c["total_tokens"], 1)
        c["target_density"] = c["target_tokens"] / denom
    return clips


def _find_wav_path(clip_id: str, speaker: str) -> Path | None:
    """Resolve WAV path for a clip_id from the cleaned corpus manifest."""
    manifest_path = CLEANED_DIR / speaker / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        return None
    for entry in manifest:
        eid = entry.get("clip_id") or entry.get("slug", "")
        if eid == clip_id:
            p = Path(entry.get("path") or entry.get("wav_path", ""))
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            return p if p.exists() else None
    return None


def best_5s_window(tokens: list[dict]) -> tuple[float, float, int]:
    """Find the 5-s window maximising target-phoneme token count.

    Returns (start_s, end_s, n_target_tokens_in_window).
    """
    if not tokens:
        return 0.0, MAX_CLIP_DURATION_S, 0

    # Filter to tokens that have valid timing
    timed = []
    for t in tokens:
        try:
            s = float(t.get("start_s") or t.get("start_time") or 0)
            e = float(t.get("end_s") or t.get("end_time") or 0)
            timed.append((s, e, t.get("phoneme", "")))
        except (ValueError, TypeError):
            pass

    if not timed:
        return 0.0, MAX_CLIP_DURATION_S, 0

    # Try windows starting at each target-phoneme token
    target_times = [s for s, e, ph in timed if ph in TARGET_PHONEMES]
    if not target_times:
        return 0.0, MAX_CLIP_DURATION_S, 0

    best_start, best_count = 0.0, 0
    for ws in target_times:
        we = ws + MAX_CLIP_DURATION_S
        count = sum(1 for s, e, ph in timed if ph in TARGET_PHONEMES and s >= ws and e <= we)
        if count > best_count:
            best_count = count
            best_start = ws

    return best_start, best_start + MAX_CLIP_DURATION_S, best_count


def main() -> int:
    EMO_DENSE_DIR.mkdir(parents=True, exist_ok=True)

    # Load formant rows from Fry and Lindsey
    all_rows: list[tuple[str, dict]] = []  # (speaker, row)
    for speaker in ["fry", "lindsey"]:
        rows = load_formant_rows(speaker)
        for r in rows:
            all_rows.append((speaker, r))

    if not all_rows:
        print("ERROR: no formant rows found in cleaned corpus", file=sys.stderr)
        return 1

    # Build per-(speaker, clip_id) stats
    speaker_clips: dict[str, dict[str, dict]] = defaultdict(dict)
    for speaker, row in all_rows:
        cid = row.get("clip_id", "")
        if not cid:
            continue
        key = (speaker, cid)
        if key not in speaker_clips:
            speaker_clips[key] = {  # type: ignore[assignment]
                "speaker": speaker,
                "clip_id": cid,
                "total_tokens": 0,
                "target_tokens": 0,
                "tokens": [],
            }
        speaker_clips[key]["total_tokens"] += 1  # type: ignore[index]
        if row.get("phoneme", "") in TARGET_PHONEMES:
            speaker_clips[key]["target_tokens"] += 1  # type: ignore[index]
        speaker_clips[key]["tokens"].append(row)  # type: ignore[index]

    for v in speaker_clips.values():
        v["target_density"] = v["target_tokens"] / max(v["total_tokens"], 1)  # type: ignore[index]

    # Sort by target density descending
    ranked = sorted(speaker_clips.values(), key=lambda x: x["target_density"], reverse=True)  # type: ignore[arg-type]

    # Greedy: pick top-N distinct clips (different clip_ids)
    candidates: list[dict] = []
    seen_clips: set[str] = set()
    for clip_info in ranked:
        cid = clip_info["clip_id"]
        if cid in seen_clips:
            continue

        speaker = clip_info["speaker"]
        wav_path = _find_wav_path(cid, speaker)
        if wav_path is None:
            continue

        tokens = clip_info["tokens"]
        start_s, end_s, n_target_in_window = best_5s_window(tokens)

        # Use full clip if it's already short enough
        try:
            audio, sr = sf.read(str(wav_path))
            dur_s = len(audio) / sr
        except Exception as e:
            _log(f"WARNING: cannot read {wav_path}: {e}")
            continue

        if dur_s <= MAX_CLIP_DURATION_S:
            start_s, end_s = 0.0, dur_s
            # Count target tokens in full clip
            n_target_in_window = clip_info["target_tokens"]
        else:
            end_s = min(end_s, dur_s)

        if n_target_in_window < MIN_TARGET_TOKENS:
            continue  # not dense enough — keep looking

        candidates.append({
            "candidate_idx": len(candidates),
            "speaker": speaker,
            "clip_id": cid,
            "source_wav": str(wav_path),
            "start_s": round(start_s, 4),
            "end_s": round(end_s, 4),
            "clip_duration_s": round(dur_s, 3),
            "window_duration_s": round(end_s - start_s, 3),
            "target_density": round(clip_info["target_density"], 4),
            "target_tokens_in_window": n_target_in_window,
            "total_tokens": clip_info["total_tokens"],
        })
        seen_clips.add(cid)

        if len(candidates) >= N_CANDIDATES:
            break

    if len(candidates) < N_CANDIDATES:
        print(f"ERROR: only found {len(candidates)}/{N_CANDIDATES} dense candidates "
              f"with ≥{MIN_TARGET_TOKENS} target tokens. Corpus may be too small.", file=sys.stderr)
        if not candidates:
            return 1
        print("WARNING: proceeding with fewer candidates", file=sys.stderr)

    # Extract audio windows and write cand_{i}.wav
    for cand in candidates:
        i = cand["candidate_idx"]
        out_path = EMO_DENSE_DIR / f"cand_{i}.wav"
        cand["output_wav"] = str(out_path)

        if out_path.exists():
            _log(f"cand_{i}: already exists, skipping")
            continue

        audio, sr = sf.read(cand["source_wav"])
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        start_i = int(cand["start_s"] * sr)
        end_i = int(cand["end_s"] * sr)
        window = audio[start_i:end_i]

        sf.write(str(out_path), window, sr, subtype="FLOAT")
        _log(f"cand_{i}: wrote {out_path.name} "
             f"({cand['window_duration_s']:.2f}s, {cand['target_tokens_in_window']} target tokens, "
             f"density={cand['target_density']:.3f}) "
             f"from {cand['speaker']}/{cand['clip_id']}")

    # Write index.json
    index_path = EMO_DENSE_DIR / "index.json"
    index_path.write_text(json.dumps(candidates, indent=2))
    _log(f"Wrote {index_path}")

    # Verify gate: each candidate must have ≥ MIN_TARGET_TOKENS
    failed = [c for c in candidates if c["target_tokens_in_window"] < MIN_TARGET_TOKENS]
    if failed:
        print(f"ERROR: {len(failed)} candidates have < {MIN_TARGET_TOKENS} target-phoneme tokens:", file=sys.stderr)
        for c in failed:
            print(f"  cand_{c['candidate_idx']}: {c['target_tokens_in_window']} tokens", file=sys.stderr)
        return 1

    print(f"\nMined {len(candidates)} candidates. All have ≥ {MIN_TARGET_TOKENS} target-phoneme tokens.")
    for c in candidates:
        print(f"  cand_{c['candidate_idx']}: {c['target_tokens_in_window']} tokens "
              f"({c['speaker']}/{c['clip_id']}, density={c['target_density']:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
