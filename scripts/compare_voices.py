"""
A/B comparison: zero-shot F5-TTS vs fine-tuned F5-TTS.

Generates the same set of test sentences with:
  1. F5-TTS base (zero-shot, reference clip only)
  2. F5-TTS fine-tuned (loads partial checkpoint over base model)

Output files are named so the user can listen side-by-side:
    tts_output/compare/baseline_<n>.wav
    tts_output/compare/finetuned_<n>.wav

Usage:
    PYTHONHASHSEED=random uv run python scripts/compare_voices.py \
        --checkpoint runs/finetune_casanova/checkpoints/final.pt
"""

import argparse
import csv
import os
import sys
import warnings
from pathlib import Path

if os.environ.get("PYTHONHASHSEED", "missing") == "":
    os.environ["PYTHONHASHSEED"] = "random"
    os.execv(sys.executable, [sys.executable, __file__] + sys.argv[1:])

warnings.filterwarnings("ignore")

import torch
import soundfile as sf

PROJECT_ROOT = Path(__file__).parent.parent

REF_AUDIO = "tts_output/ref_narrator.wav"
REF_TEXT = "this an ideal opportunity for obtaining from her everything I wished. Briefly, I told her the object of my visit."
TEST_TEXTS = [
    "The game is afoot, Watson. There is not a moment to lose if we are to catch our man before dawn.",
    "In the summer of seventeen fifteen, my father left the family home in Parma with nothing but ambition and a restless heart.",
    "She had the kind of beauty that makes a man forget his better judgment, and I was no exception to that rule.",
    "Elementary, my dear fellow. The solution is quite obvious when you observe carefully.",
    "I had never seen such a woman before in my life, nor have I since.",
]


def make_tts():
    from f5_tts.api import F5TTS
    return F5TTS(model="F5TTS_v1_Base", device="mps")


def apply_finetune_checkpoint(tts, ckpt_path: Path):
    """Load partial state dict from fine-tune run over base model."""
    print(f"Loading fine-tune checkpoint: {ckpt_path}")
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = checkpoint["model"]
    model = tts.ema_model.transformer
    own = model.state_dict()
    n_loaded = 0
    for k, v in sd.items():
        # state_dict keys may be prefixed with "transformer." — strip it
        key = k.replace("transformer.", "", 1) if k.startswith("transformer.") else k
        if key in own:
            own[key] = v.to(own[key].device)
            n_loaded += 1
    model.load_state_dict(own, strict=False)
    print(f"  applied {n_loaded} parameter tensors from fine-tune")


def generate_set(tts, label: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, text in enumerate(TEST_TEXTS, 1):
        print(f"  [{label}] {i}/{len(TEST_TEXTS)}: {text[:60]}...")
        wav, sr, _ = tts.infer(ref_file=REF_AUDIO, ref_text=REF_TEXT, gen_text=text)
        out = out_dir / f"{label}_{i:02d}.wav"
        sf.write(str(out), wav.squeeze() if hasattr(wav, "squeeze") else wav, sr)
        print(f"     → {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",
                        default="runs/finetune_casanova/checkpoints/final.pt")
    parser.add_argument("--out-dir", default="tts_output/compare")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Manifest of texts so user knows what each file says
    with (out_dir / "texts.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "text"])
        for i, t in enumerate(TEST_TEXTS, 1):
            writer.writerow([i, t])

    print("=== Baseline (zero-shot F5-TTS) ===")
    tts = make_tts()
    generate_set(tts, "baseline", out_dir)

    print("\n=== Fine-tuned (F5-TTS + your weights) ===")
    apply_finetune_checkpoint(tts, Path(args.checkpoint))
    generate_set(tts, "finetuned", out_dir)

    print(f"\n✓ Done. Listen and compare:")
    print(f"  {out_dir}/baseline_01.wav  vs  {out_dir}/finetuned_01.wav")
    print(f"  (5 pairs total, same text)")


if __name__ == "__main__":
    main()
