"""
Generate eval phrases with XTTS-v2 zero-shot voice cloning.

Architecture (different from F5-TTS):
  - Speaker encoder extracts a fixed embedding from the reference clip — NO ref-text
    conditioning, so no reference-text leakage at generation time.
  - Autoregressive decoder generates mel conditioned on the speaker embedding + text.

This is the key architectural difference vs F5-TTS: XTTS-v2 trades some acoustic
fidelity for full text faithfulness (WER should be close to 0).

Usage (run in .venv_xtts, NOT .venv):
    .venv_xtts/bin/python scripts/xtts_gen.py
    # then score with the main .venv:
    .venv/bin/python scripts/posthoc_eval.py \\
        --score-only \\
        --phrases-csv tts_output/cross_eval_50/eval_short.csv \\
        --out-dir tts_output/eval_xtts_v2
"""

import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "0")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

import soundfile as sf
from TTS.api import TTS

REF_AUDIO = str(PROJECT_ROOT / "tts_output/ref_narrator.wav")
PHRASES_CSV = PROJECT_ROOT / "tts_output/cross_eval_50/eval_short.csv"
OUT_DIR = PROJECT_ROOT / "tts_output" / "eval_xtts_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# XTTS-v2 on MPS has known issues with the autoregressive decoder — use CPU.
# Slower (~8s/phrase) but correct. GPU on Mac is not worth fighting.
DEVICE = "cpu"

print(f"Loading XTTS-v2 on {DEVICE}...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)
print(f"  Model loaded. Reference: {Path(REF_AUDIO).name}")

# Load eval phrases
phrases = []
with open(PHRASES_CSV) as f:
    for row in csv.DictReader(f, delimiter=","):
        phrases.append(row)

print(f"  Generating {len(phrases)} phrases...")

manifest = []
for row in phrases:
    slug = row["slug"]
    prompt = row["prompt"]
    out_path = OUT_DIR / f"xtts_{slug}.wav"

    print(f"\n--- {slug}: {prompt!r}")
    if out_path.exists():
        print("  already exists, skipping")
    else:
        tts.tts_to_file(
            text=prompt,
            speaker_wav=REF_AUDIO,
            language="en",
            file_path=str(out_path),
        )

    info = sf.info(str(out_path))
    print(f"  → {out_path.name} ({info.duration:.1f}s, sr={info.samplerate})")
    manifest.append({
        "label": "xtts_v2",
        "step": 0,
        "slug": slug,
        "prompt": prompt,
        "wav_path": str(out_path),
        "sr": int(info.samplerate),
    })

manifest_path = OUT_DIR / "manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2))
print(f"\n✓ {len(manifest)} clips → {OUT_DIR}")
print(f"✓ manifest → {manifest_path}")
print("\nScore with:")
print("  .venv/bin/python scripts/posthoc_eval.py \\")
print("      --score-only \\")
print(f"      --phrases-csv {PHRASES_CSV} \\")
print(f"      --out-dir {OUT_DIR}")
