"""
Generate 6 eval phrases with OpenVoice V2 zero-shot cloning of Cumberbatch.

Architecture (different from F5-TTS):
  1. MeloTTS generates baseline speech in a default English voice
  2. ToneColorConverter swaps the timbre to match the target speaker
     (extracted from our 12s Cumberbatch ref clip)

Output: WAVs in tts_output/openvoice_eval/, plus manifest.json compatible with
posthoc_eval.py --score-only so we can score with the same harness.

Runs in the .venv_openvoice/ environment (NOT the F5-TTS .venv/).
"""

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
os.environ.setdefault("PYTHONHASHSEED", "0")

import torch
from melo.api import TTS
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

REF_AUDIO = PROJECT_ROOT / "tts_output/ref_narrator.wav"
CHECKPOINTS_DIR = PROJECT_ROOT / "openvoice_checkpoints" / "checkpoints_v2"
OUT_DIR = PROJECT_ROOT / "tts_output" / "openvoice_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_PHRASES = [
    ("stella_short", "Please call Stella; ask her to bring these things from the store."),
    ("rainbow",      "When the sunlight strikes raindrops in the air, they act as a prism."),
    ("casual",       "I haven't seen him since the last meeting, but I'll ask around tomorrow."),
    ("technical",    "The algorithm processes each frame independently before merging results."),
    ("deep_vowels",  "Whose woods these are I think I know; his house is in the village."),
    ("imperative",   "Stop. Don't move. There's something behind you."),
]

# OpenVoice on MPS hits known sampling bugs in MeloTTS — keep on CPU. Slow but correct.
DEVICE = "cpu"
SPEAKER_KEY = "en-newest"  # most recent EN base speaker; cleanest English

print(f"Loading ToneColorConverter on {DEVICE}...")
converter = ToneColorConverter(
    str(CHECKPOINTS_DIR / "converter" / "config.json"),
    device=DEVICE,
)
converter.load_ckpt(str(CHECKPOINTS_DIR / "converter" / "checkpoint.pth"))

print(f"Extracting target speaker embedding from {REF_AUDIO.name}...")
target_se, _ = se_extractor.get_se(
    str(REF_AUDIO), converter, vad=True,
)
print(f"  target embedding shape: {target_se.shape}")

print(f"Loading MeloTTS English ({SPEAKER_KEY})...")
tts_model = TTS(language="EN_NEWEST", device=DEVICE)
speaker_ids = tts_model.hps.data.spk2id
speaker_id = list(speaker_ids.values())[0]  # EN_NEWEST has one speaker
source_se = torch.load(
    str(CHECKPOINTS_DIR / "base_speakers" / "ses" / f"{SPEAKER_KEY}.pth"),
    map_location=DEVICE,
)

tmp_path = OUT_DIR / "_tmp_base.wav"
manifest = []

for slug, prompt in EVAL_PHRASES:
    print(f"\n--- {slug}: {prompt!r}")
    # Step 1: generate base speech with MeloTTS
    tts_model.tts_to_file(prompt, speaker_id, str(tmp_path), speed=1.0)

    # Step 2: convert tone colour to target (Cumberbatch)
    out_path = OUT_DIR / f"baseline_{slug}.wav"
    converter.convert(
        audio_src_path=str(tmp_path),
        src_se=source_se,
        tgt_se=target_se,
        output_path=str(out_path),
        message="openvoice_eval",
    )

    import soundfile as sf
    info = sf.info(str(out_path))
    print(f"  → {out_path.name} ({info.duration:.1f}s, sr={info.samplerate})")
    manifest.append({
        "label": "baseline", "step": 0, "slug": slug, "prompt": prompt,
        "wav_path": str(out_path), "sr": int(info.samplerate),
    })

if tmp_path.exists():
    tmp_path.unlink()

manifest_path = OUT_DIR / "manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2))
print(f"\n✓ {len(manifest)} clips generated → {OUT_DIR}")
print(f"✓ manifest → {manifest_path}")
print("\nNext: score with F5-TTS .venv:")
print("  .venv/bin/python scripts/posthoc_eval.py \\")
print("      --score-only --centroid-dir data/cumberbatch_casanova \\")
print("      --out-dir tts_output/openvoice_eval")
