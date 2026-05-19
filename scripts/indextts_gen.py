"""
Generate eval phrases with IndexTTS-2 zero-shot voice cloning.

Architecture: audio-only reference (no ref_text). Eliminates F5-TTS-style leakage.

Pinned to IndexTTS-2 repo commit 830f6f8f and HF model revision 740dcaff
(see CLAUDE.md "IndexTTS-2 install pins"). Rebuild from those if vendor/ is lost.

Usage:
    vendor/index-tts/.venv/bin/python scripts/indextts_gen.py \\
        --phrases-csv tts_output/cross_eval_50/eval_short.csv \\
        --out-dir tts_output/eval_indextts_v2
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "0")
PROJECT_ROOT = Path(__file__).parent.parent
INDEXTTS_ROOT = PROJECT_ROOT / "vendor" / "index-tts"

sys.path.insert(0, str(INDEXTTS_ROOT))

import soundfile as sf
from indextts.infer_v2 import IndexTTS2

DEFAULT_REF = str(PROJECT_ROOT / "tts_output/ref_narrator.wav")
DEFAULT_PHRASES = str(PROJECT_ROOT / "tts_output/cross_eval_50/eval_short.csv")
DEFAULT_OUT = str(PROJECT_ROOT / "tts_output/eval_indextts_v2")

CFG_PATH = str(INDEXTTS_ROOT / "checkpoints/config.yaml")
MODEL_DIR = str(INDEXTTS_ROOT / "checkpoints")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phrases-csv", default=DEFAULT_PHRASES)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--ref-audio", default=DEFAULT_REF)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--label", default="indextts_v2")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading IndexTTS-2 on {args.device}...")
    tts = IndexTTS2(cfg_path=CFG_PATH, model_dir=MODEL_DIR, device=args.device)
    print(f"  Model loaded. Reference: {Path(args.ref_audio).name}")

    with open(args.phrases_csv) as f:
        phrases = list(csv.DictReader(f))
    print(f"  Generating {len(phrases)} phrases...")

    manifest = []
    for row in phrases:
        slug, prompt = row["slug"], row["prompt"]
        out_path = out_dir / f"indextts_{slug}.wav"
        print(f"\n--- {slug}: {prompt!r}")
        if out_path.exists():
            print("  already exists, skipping")
        else:
            tts.infer(spk_audio_prompt=args.ref_audio, text=prompt, output_path=str(out_path))
        info = sf.info(str(out_path))
        print(f"  → {out_path.name} ({info.duration:.1f}s, sr={info.samplerate})")
        manifest.append({
            "label": args.label,
            "step": 0,
            "slug": slug,
            "prompt": prompt,
            "wav_path": str(out_path),
            "sr": int(info.samplerate),
        })

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n✓ {len(manifest)} clips → {out_dir}")
    print(f"✓ manifest → {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
