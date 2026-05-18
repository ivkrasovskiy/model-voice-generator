"""
Zero-shot F5-TTS inference (runs on MPS on Apple Silicon).

Requires .venv-f5:
    source .venv-f5/bin/activate
    python scripts/f5_infer.py --ref tts_output/ref_narrator.wav --text "..."

Or via CLI shortcut:
    f5-tts_infer-cli \
        --model F5TTS_v1_Base \
        --ref_audio tts_output/ref_narrator.wav \
        --ref_text "this an ideal opportunity for obtaining from her everything I wished." \
        --gen_text "The game is afoot Watson. There is not a moment to lose." \
        --output_dir tts_output/ \
        --output_file f5_demo_1.wav

Usage (Python API):
    source .venv-f5/bin/activate
    python scripts/f5_infer.py \
        --ref tts_output/ref_narrator.wav \
        --ref-text "this an ideal opportunity for obtaining from her everything I wished." \
        --text "Your generated text here." \
        --out tts_output/f5_out.wav
"""

import argparse
import sys
import warnings
from pathlib import Path

# Fix PYTHONHASHSEED from .env BEFORE F5-TTS imports (uv-venv quirk crashes subprocesses)
sys.path.insert(0, str(Path(__file__).parent))
from _dotenv_init import init_env_then_reexec

init_env_then_reexec(__file__)

warnings.filterwarnings("ignore")

import torch


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True, help="Reference WAV (clean narrator audio, 6–30s)")
    parser.add_argument("--ref-text", required=True,
                        help="Transcription of the reference WAV (must be accurate)")
    parser.add_argument("--text", help="Text to generate")
    parser.add_argument("--texts-file", help="File with one sentence per line")
    parser.add_argument("--out", default="tts_output/f5_out.wav")
    parser.add_argument("--out-dir", default="tts_output/")
    parser.add_argument("--model", default="F5TTS_v1_Base",
                        choices=["F5TTS_v1_Base", "E2TTS_Base"])
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    from f5_tts.api import F5TTS
    tts = F5TTS(model=args.model, device=device)

    if args.texts_file:
        texts = [ln.strip() for ln in Path(args.texts_file).read_text().splitlines() if ln.strip()]
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        for i, text in enumerate(texts):
            out = Path(args.out_dir) / f"f5_{i+1:03d}.wav"
            print(f"[{i+1}/{len(texts)}] {text[:60]}...")
            wav, sr, _ = tts.infer(
                ref_file=args.ref,
                ref_text=args.ref_text,
                gen_text=text,
            )
            import soundfile as sf
            sf.write(str(out), wav.squeeze().cpu().numpy(), sr)
            print(f"  saved → {out}")
    elif args.text:
        import soundfile as sf
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        wav, sr, _ = tts.infer(
            ref_file=args.ref,
            ref_text=args.ref_text,
            gen_text=args.text,
        )
        sf.write(args.out, wav.squeeze().cpu().numpy(), sr)
        print(f"Saved → {args.out}")
    else:
        print("Provide --text or --texts-file")


if __name__ == "__main__":
    main()
