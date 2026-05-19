"""
Generate eval phrases with IndexTTS-2 zero-shot voice cloning.

Architecture: audio-only reference (no ref_text). Eliminates F5-TTS-style leakage.

Pinned to IndexTTS-2 repo commit 830f6f8f and HF model revision 740dcaff
(see CLAUDE.md "IndexTTS-2 install pins"). Rebuild from those if vendor/ is lost.

Usage:
    vendor/index-tts/.venv/bin/python scripts/indextts_gen.py
"""

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

REF_AUDIO = str(PROJECT_ROOT / "tts_output/ref_narrator.wav")
PHRASES_CSV = PROJECT_ROOT / "tts_output/cross_eval_50/eval_short.csv"
OUT_DIR = PROJECT_ROOT / "tts_output" / "eval_indextts_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cpu"
CFG_PATH = str(INDEXTTS_ROOT / "checkpoints/config.yaml")
MODEL_DIR = str(INDEXTTS_ROOT / "checkpoints")

print(f"Loading IndexTTS-2 on {DEVICE}...")
tts = IndexTTS2(cfg_path=CFG_PATH, model_dir=MODEL_DIR, device=DEVICE)
print(f"  Model loaded. Reference: {Path(REF_AUDIO).name}")

phrases = []
with open(PHRASES_CSV) as f:
    for row in csv.DictReader(f):
        phrases.append(row)

print(f"  Generating {len(phrases)} phrases...")

manifest = []
for row in phrases:
    slug, prompt = row["slug"], row["prompt"]
    out_path = OUT_DIR / f"indextts_{slug}.wav"

    print(f"\n--- {slug}: {prompt!r}")
    if out_path.exists():
        print("  already exists, skipping")
    else:
        tts.infer(spk_audio_prompt=REF_AUDIO, text=prompt, output_path=str(out_path))

    info = sf.info(str(out_path))
    print(f"  → {out_path.name} ({info.duration:.1f}s, sr={info.samplerate})")
    manifest.append({
        "label": "indextts_v2",
        "step": 0,
        "slug": slug,
        "prompt": prompt,
        "wav_path": str(out_path),
        "sr": int(info.samplerate),
    })

(OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"\n✓ {len(manifest)} clips → {OUT_DIR}")
print(f"✓ manifest → {OUT_DIR / 'manifest.json'}")
print("\nScore with:")
print("  .venv/bin/python scripts/posthoc_eval.py \\")
print("      --score-only \\")
print("      --phrases-csv tts_output/cross_eval_50/eval_short.csv \\")
print("      --out-dir tts_output/eval_indextts_v2")
