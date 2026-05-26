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

import torch
import soundfile as sf
from indextts.infer_v2 import IndexTTS2

_DEFAULT_DEVICE = "cpu"  # MPS unsupported: bigvgan alias_free conv_transpose1d fails >65536 channels

DEFAULT_REF = str(PROJECT_ROOT / "tts_output/ref_interview.wav")
DEFAULT_PHRASES = str(PROJECT_ROOT / "tts_output/cross_eval_50/eval_short.csv")
DEFAULT_OUT = str(PROJECT_ROOT / "tts_output/eval_indextts_v2")

CFG_PATH = str(INDEXTTS_ROOT / "checkpoints/config.yaml")
MODEL_DIR = str(INDEXTTS_ROOT / "checkpoints")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phrases-csv", default=DEFAULT_PHRASES)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--ref-audio", default=DEFAULT_REF)
    parser.add_argument("--device", default=_DEFAULT_DEVICE)
    parser.add_argument("--label", default="indextts_v2")
    # Generation quality knobs (Track C sweep)
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="GPT sampling temperature (default 0.8; lower = more conservative)")
    parser.add_argument("--top-p", type=float, default=0.8,
                        help="Nucleus sampling top-p (default 0.8)")
    parser.add_argument("--top-k", type=int, default=30,
                        help="Top-k sampling (default 30)")
    parser.add_argument("--num-beams", type=int, default=3,
                        help="Beam search width (default 3; 1 = pure sampling)")
    # s2mel CFM diffusion knobs (Phase 0.10)
    parser.add_argument("--cfg-rate", type=float, default=0.7,
                        help="CFM inference cfg rate (default 0.7)")
    parser.add_argument("--diffusion-steps", type=int, default=25,
                        help="CFM diffusion steps (default 25)")
    # Emotion conditioning (Phase 0.11) — mutually exclusive: emo-audio XOR (use-emo-text + emo-text)
    parser.add_argument("--emo-audio", default=None,
                        help="Path to emo reference WAV (separate from spk reference); "
                             "if omitted, IndexTTS-2 uses spk_audio_prompt for emo too")
    parser.add_argument("--emo-alpha", type=float, default=1.0,
                        help="Emo conditioning strength (0..1, default 1.0)")
    parser.add_argument("--use-emo-text", action="store_true",
                        help="Use Qwen emotion classifier on --emo-text to derive emo vector")
    parser.add_argument("--emo-text", default=None,
                        help="Text prompt for emotion classifier (requires --use-emo-text)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "num_beams": args.num_beams,
        "inference_cfg_rate": args.cfg_rate,
        "diffusion_steps": args.diffusion_steps,
    }
    defaults = {"temperature": 0.8, "top_p": 0.8, "top_k": 30, "num_beams": 3,
                "inference_cfg_rate": 0.7, "diffusion_steps": 25}
    non_default = {k: v for k, v in gen_kwargs.items() if v != defaults[k]}

    infer_kwargs = {}
    if args.emo_audio:
        infer_kwargs["emo_audio_prompt"] = args.emo_audio
    if args.use_emo_text:
        if not args.emo_text:
            print("ERROR: --use-emo-text requires --emo-text", file=sys.stderr)
            return 1
        infer_kwargs["use_emo_text"] = True
        infer_kwargs["emo_text"] = args.emo_text
    if args.emo_alpha != 1.0:
        infer_kwargs["emo_alpha"] = args.emo_alpha

    print(f"Loading IndexTTS-2 on {args.device}...")
    tts = IndexTTS2(cfg_path=CFG_PATH, model_dir=MODEL_DIR, device=args.device)
    print(f"  Model loaded. Reference: {Path(args.ref_audio).name}")
    if non_default:
        print(f"  Non-default gen params: {non_default}")
    if infer_kwargs:
        print(f"  Emo kwargs: {infer_kwargs}")

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
            tts.infer(spk_audio_prompt=args.ref_audio, text=prompt,
                      output_path=str(out_path), **infer_kwargs, **gen_kwargs)
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
