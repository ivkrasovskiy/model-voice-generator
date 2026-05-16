"""
Zero-shot XTTS-v2 inference.
Runs on CPU (MPS conv1d channel limit prevents full MPS use).

Usage:
    uv run --python .venv/bin/python scripts/xtts_infer.py \
        --ref tts_output/ref_narrator.wav \
        --text "Your text here." \
        --out tts_output/xtts_out.wav

    # Or batch from a text file (one sentence per line):
    uv run --python .venv/bin/python scripts/xtts_infer.py \
        --ref tts_output/ref_narrator.wav \
        --texts-file prompts.txt \
        --out-dir tts_output/
"""

import argparse
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from TTS.api import TTS

DEFAULT_REF = "tts_output/ref_narrator.wav"
MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"


def load_model() -> TTS:
    # CPU only — XTTS speaker encoder exceeds MPS conv1d channel limit (65536)
    tts = TTS(MODEL, progress_bar=False)
    tts.to("cpu")
    return tts


def synthesize(tts: TTS, text: str, ref_wav: str, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tts.tts_to_file(text=text, file_path=out_path, speaker_wav=ref_wav, language="en")
    print(f"  saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default=DEFAULT_REF, help="Reference WAV (6–30s of target voice)")
    parser.add_argument("--text", help="Single text to synthesize")
    parser.add_argument("--texts-file", help="File with one sentence per line")
    parser.add_argument("--out", default="tts_output/xtts_out.wav", help="Output WAV path (single text)")
    parser.add_argument("--out-dir", default="tts_output/", help="Output directory (batch)")
    args = parser.parse_args()

    print("Loading XTTS-v2 (CPU)...")
    tts = load_model()

    if args.texts_file:
        texts = [l.strip() for l in Path(args.texts_file).read_text().splitlines() if l.strip()]
        for i, text in enumerate(texts):
            out = Path(args.out_dir) / f"xtts_{i+1:03d}.wav"
            print(f"[{i+1}/{len(texts)}] {text[:60]}...")
            synthesize(tts, text, args.ref, str(out))
    elif args.text:
        synthesize(tts, args.text, args.ref, args.out)
    else:
        # Demo
        demo_texts = [
            "The game is afoot, Watson. There is not a moment to lose if we are to catch our man before dawn.",
            "In the summer of seventeen fifteen, my father left the family home in Parma with nothing but a restless heart.",
            "She had the kind of beauty that makes a man forget his better judgment entirely.",
        ]
        for i, text in enumerate(demo_texts):
            out = f"tts_output/xtts_demo_{i+1}.wav"
            print(f"[{i+1}/{len(demo_texts)}] synthesizing...")
            synthesize(tts, text, args.ref, out)

    print("Done.")


if __name__ == "__main__":
    main()
