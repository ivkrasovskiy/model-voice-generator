"""Extract per-token vowel formants from a corpus manifest.

Thin wrapper over accent_coach.pipeline.alignment + .formants. Does NOT
modify any pipeline code — uses Phase 0 parameters exactly as configured.

Usage:
    .venv/bin/python scripts/accent_coach_extract_formants.py \\
        --manifest tts_output/accent_coach/bc_cal_50/manifest.json \\
        --out tts_output/bc_cal_50/formants.csv \\
        --source-label synth_bc

Manifest field normalization:
    Standard format:   clip_id, path, transcript
    bc_cal_50 format:  slug -> clip_id, wav_path -> path, prompt -> transcript
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import soundfile as sf


def _normalize_entry(entry: dict) -> dict:
    """Unify different manifest field naming conventions."""
    out = dict(entry)
    if "path" not in out:
        out["path"] = out.get("wav_path", "")
    if "transcript" not in out:
        out["transcript"] = out.get("prompt", "")
    if "clip_id" not in out:
        out["clip_id"] = out.get("slug", out.get("path", "?"))
    return out


def _resolve_path(raw_path: str) -> Path:
    """Resolve path that may be relative to project root or absolute."""
    p = Path(raw_path)
    if p.is_absolute():
        return p
    full = PROJECT_ROOT / p
    if full.exists():
        return full
    # Last resort: try as-is from CWD
    return p


def extract_formants(manifest_path: Path, out_csv: Path, source_label: str) -> None:
    from accent_coach.pipeline.alignment import align_audio
    from accent_coach.pipeline.formants import extract_vowel_features

    manifest = json.loads(manifest_path.read_text())
    rows = []
    total = len(manifest)

    for idx, raw_entry in enumerate(manifest, 1):
        entry = _normalize_entry(raw_entry)
        clip_path = _resolve_path(entry["path"])
        transcript = entry["transcript"]
        clip_id = entry["clip_id"]

        if not clip_path.exists():
            print(f"  [{idx}/{total}] SKIP missing: {clip_path}", flush=True)
            continue

        print(f"  [{idx}/{total}] {clip_id}  {clip_path.name}", flush=True)

        try:
            # Alignment
            phonemes = align_audio(clip_path, transcript, sentence_id=0)

            # Load audio for formant extraction
            audio, sr = sf.read(str(clip_path))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)

            # Formant extraction (uses Phase 0 params: voiced_frac≥35%, F1≤900Hz, auto pitch ceil)
            vowel_features = extract_vowel_features(audio, sr, phonemes)

            for vf in vowel_features:
                rows.append({
                    "clip_id": clip_id,
                    "source_label": source_label,
                    "phoneme": vf.phoneme.phoneme,
                    "F1": round(vf.f1, 1),
                    "F2": round(vf.f2, 1),
                    "voiced_fraction": float("nan"),  # not exposed by pipeline; all survivors pass ≥35%
                    "duration_s": round(vf.duration_ms / 1000.0, 4),
                    "transcript": transcript,
                })

        except Exception as e:  # noqa: BLE001
            print(f"    ERROR: {e}", flush=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        fieldnames = ["clip_id", "source_label", "phoneme", "F1", "F2",
                      "voiced_fraction", "duration_s", "transcript"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    phoneme_counts: dict[str, int] = {}
    for r in rows:
        phoneme_counts[r["phoneme"]] = phoneme_counts.get(r["phoneme"], 0) + 1

    print(f"\nWrote {len(rows)} rows to {out_csv}")
    print(f"Phoneme token counts: { {k: phoneme_counts[k] for k in sorted(phoneme_counts)} }")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract vowel formants from a corpus manifest")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source-label", required=True)
    args = parser.parse_args()

    manifest_path = args.manifest if args.manifest.is_absolute() else PROJECT_ROOT / args.manifest
    out_csv = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out

    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    print(f"Extracting formants: {manifest_path.name} → {out_csv.name}  label={args.source_label}")
    extract_formants(manifest_path, out_csv, args.source_label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
