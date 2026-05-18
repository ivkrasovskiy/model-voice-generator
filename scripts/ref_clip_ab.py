"""
A/B test the hypothesis: F5-TTS's gibberish on short prompts is reference-text leakage
specific to the chosen ref clip. Test 3 different ref clips on the `imperative` phrase
(worst-case leakage) and compare transcripts.

If transcripts contain phrases from each respective ref → confirmed leakage is ref-bound.
If transcripts contain the SAME garbage regardless of ref → it's an architectural issue.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _dotenv_init import init_env_then_reexec, kill_stale_python

init_env_then_reexec(__file__)

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import torch
import torchaudio

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data/cumberbatch_casanova"
OUT_DIR = PROJECT_ROOT / "tts_output/ref_clip_ab"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Three different ref clips, all ~11-12s. Look for very different lexical content.
REFS = [
    # Current ref — included as A control. "everything I wished" is the known leak.
    ("ref_A_current", PROJECT_ROOT / "tts_output/ref_narrator.wav",
     "this an ideal opportunity for obtaining from her everything I wished."),
    ("ref_B_dualists", DATA_DIR / "wavs/seg_00149.wav",
     "They taught me how to behave in the company of quarrelsome dualists. I avoided them with care."),
    ("ref_C_scrutinized", DATA_DIR / "wavs/seg_00170.wav",
     "She scrutinized me from head to foot, as if I was being offered for sale."),
]

# Test on the worst-leakage phrase (imperative) + one normal one (rainbow).
PROMPTS = [
    ("imperative", "Stop. Don't move. There's something behind you."),
    ("rainbow", "When the sunlight strikes raindrops in the air, they act as a prism."),
]

CFG_STRENGTH = 2.0
NFE_STEP = 32
SEED = 42
T_THRESHOLD = 0.08  # Selective CFG — best config


def _gen_segment(tts, ref_path, ref_text, segment, max_retries=5):
    for attempt in range(1, max_retries + 1):
        wav, sr, _ = tts.infer(
            ref_file=str(ref_path), ref_text=ref_text, gen_text=segment,
            cfg_strength=CFG_STRENGTH, nfe_step=NFE_STEP,
            speed=1.0,
            seed=SEED + (attempt - 1),
            selective_cfg_threshold=T_THRESHOLD,
        )
        w = np.asarray(wav.squeeze() if hasattr(wav, "squeeze") else wav, dtype=np.float32)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if np.isfinite(w).all() and float(np.abs(w).max()) > 0.01:
            return w, sr
        print(f"  attempt {attempt}: bad")
    return None, None


def main():
    n_killed = kill_stale_python()
    if n_killed:
        print(f"Pre-launch: killed {n_killed} stale procs")

    from f5_tts.api import F5TTS
    print("Loading F5TTS_v1_Base on mps...")
    tts = F5TTS(model="F5TTS_v1_Base", device="mps")

    print("\nLoading Whisper large-v3 on CPU for transcription...")
    import whisper
    whisper_model = whisper.load_model("large-v3", device="cpu")

    rows = []
    for ref_name, ref_path, ref_text in REFS:
        print(f"\n=== {ref_name}: {ref_path.name}")
        print(f"  ref text: {ref_text!r}")
        for prompt_slug, prompt in PROMPTS:
            from posthoc_eval import split_to_short_segments
            segments = split_to_short_segments(prompt, max_chars=50)
            pieces = []
            sr_out = None
            failed = False
            for seg in segments:
                wav_np, sr = _gen_segment(tts, ref_path, ref_text, seg)
                if wav_np is None:
                    failed = True
                    break
                pieces.append(wav_np)
                if sr_out is None:
                    sr_out = sr
                pieces.append(np.zeros(int(0.15 * sr), dtype=np.float32))
            if failed:
                print(f"  {prompt_slug}: FAILED")
                continue
            wav_full = np.concatenate(pieces[:-1])
            out_path = OUT_DIR / f"{ref_name}_{prompt_slug}.wav"
            sf.write(str(out_path), wav_full, sr_out)

            # Transcribe to detect leakage
            wav16 = torchaudio.functional.resample(
                torch.from_numpy(wav_full).float().unsqueeze(0), sr_out, 16000
            ).squeeze(0).numpy()
            result = whisper_model.transcribe(wav16, language="en", fp16=False, verbose=False)
            hyp = result["text"].strip()
            print(f"  {prompt_slug}: heard \"{hyp[:100]}\"")
            rows.append({
                "ref": ref_name, "ref_text_excerpt": ref_text[:50],
                "prompt": prompt_slug, "transcript": hyp,
                "out_path": str(out_path),
            })

    print("\n=== Summary ===")
    import json
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(rows, indent=2))
    print(f"Summary → {summary_path}")
    for r in rows:
        print(f"\n  {r['ref']:24s} | {r['prompt']:12s}")
        print(f"    ref text:   {r['ref_text_excerpt']}...")
        print(f"    transcript: {r['transcript'][:120]}")


if __name__ == "__main__":
    main()
