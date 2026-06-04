"""Build ~14 s cloning reference clips per GA speaker (Phase 0.16, Track B).

Concatenates the highest-speaker-confidence corpus clips per speaker up to a
target duration, producing one clean reference WAV each — usable as IndexTTS-2
`spk_audio_prompt` (timbre/clone) and as `emo_audio_prompt` (accent/style donor
for the disentanglement test).

  .venv/bin/python scripts/accent_coach_build_genam_refs.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402


def build_ref(manifest: list[dict], speaker: str, target_s: float, out_path: Path) -> float:
    clips = [c for c in manifest if c["speaker"] == speaker]
    # Prefer high speaker-confidence, longer clips; fall back to order if no sim.
    clips.sort(key=lambda c: (c.get("spk_sim") or 0.0, c["end_s"] - c["start_s"]), reverse=True)
    pieces, total, sr = [], 0.0, 16000
    for c in clips:
        wav, sr = sf.read(str(PROJECT_ROOT / c["path"]))
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        pieces.append(wav.astype(np.float32))
        pieces.append(np.zeros(int(0.12 * sr), dtype=np.float32))
        total += len(wav) / sr
        if total >= target_s:
            break
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), np.concatenate(pieces[:-1]), sr, subtype="PCM_16")
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="tts_output/genam_lecture_corpus/manifest.json")
    ap.add_argument("--out-dir", default="tts_output/refs/genam")
    ap.add_argument("--target-s", type=float, default=14.0)
    args = ap.parse_args()

    manifest = json.loads((PROJECT_ROOT / args.manifest).read_text())
    out_dir = PROJECT_ROOT / args.out_dir
    speakers = sorted(set(c["speaker"] for c in manifest))
    for spk in speakers:
        out = out_dir / f"{spk}_ref.wav"
        dur = build_ref(manifest, spk, args.target_s, out)
        print(f"  {spk:12} → {out.relative_to(PROJECT_ROOT)}  ({dur:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
