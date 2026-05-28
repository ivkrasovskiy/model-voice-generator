"""Pre-extract Whisper embeddings and mel-codes for all clips in the LoRA dataset.

Run once before training. Saves per-clip .pt files so the training loop
never calls the Whisper encoder (get_emb) or semantic_codec during training.

Usage:
  vendor/index-tts/.venv/bin/python scripts/accent_coach_phase0_14_cache_embeddings.py \
      --dataset-dir tts_output/accent_coach/phase0_14/lora/dataset \
      --cache-dir   tts_output/accent_coach/phase0_14/lora/emb_cache

Each .pt file contains:
  mel_codes : (1, T) int64  — quantized semantic codes (training label)
  spk_emb   : (1, T, 1024) float32  — Whisper encoder output (conditioning input)

Both tensors stored on CPU for portability.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
INDEXTTS_ROOT = PROJECT_ROOT / "vendor" / "index-tts"
sys.path.insert(0, str(INDEXTTS_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch

CFG_PATH = str(INDEXTTS_ROOT / "checkpoints/config.yaml")
MODEL_DIR = str(INDEXTTS_ROOT / "checkpoints")


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def _cache_path(cache_dir: Path, wav_path: str) -> Path:
    return cache_dir / (Path(wav_path).stem + ".pt")


def _compute_and_save(tts, wav_path: str, cache_path: Path) -> bool:
    """Compute mel_codes + spk_emb for one wav, save to cache_path. Returns True on success."""
    from lib.lora_targets import _emb_from_wav, _load_wav_16k

    try:
        wav_16k = _load_wav_16k(wav_path, "cpu")
        spk_emb = _emb_from_wav(tts, wav_16k)   # (1, T, 1024) — also the conditioning emb
        del wav_16k
        mel_codes, quant_emb = tts.semantic_codec.quantize(spk_emb)
        del quant_emb
        torch.save(
            {
                "mel_codes": mel_codes.cpu(),  # (1, T) int64
                "spk_emb":   spk_emb.cpu(),    # (1, T, 1024) float32
            },
            cache_path,
        )
        del spk_emb, mel_codes
        return True
    except Exception as e:
        _log(f"  SKIP {Path(wav_path).name}: {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir",
                        default="tts_output/accent_coach/phase0_14/lora/dataset")
    parser.add_argument("--cache-dir",
                        default="tts_output/accent_coach/phase0_14/lora/emb_cache")
    args = parser.parse_args()

    dataset_dir = PROJECT_ROOT / args.dataset_dir
    cache_dir   = PROJECT_ROOT / args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Collect all unique wav paths across train + val
    wavs: set[str] = set()
    for split in ("train.jsonl", "val.jsonl"):
        p = dataset_dir / split
        if not p.exists():
            continue
        with p.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    wavs.add(json.loads(line)["wav"])
    wavs_list = sorted(wavs)
    _log(f"Found {len(wavs_list)} unique clips to cache.")

    # Skip already-cached
    todo = [w for w in wavs_list if not _cache_path(cache_dir, w).exists()]
    _log(f"Already cached: {len(wavs_list) - len(todo)}  |  Remaining: {len(todo)}")
    if not todo:
        _log("All clips already cached. Nothing to do.")
        return 0

    # Load model
    try:
        if torch.backends.mps.is_available():
            device = "mps"
            _log("Device: mps")
        else:
            device = "cpu"
            _log("Device: cpu")
    except Exception:
        device = "cpu"
        _log("Device: cpu")

    _log("Loading IndexTTS2...")
    try:
        from indextts.infer_v2 import IndexTTS2
        tts = IndexTTS2(cfg_path=CFG_PATH, model_dir=MODEL_DIR, device=device)
    except RuntimeError as e:
        if "mps" in str(e).lower():
            _log(f"MPS error, falling back to cpu: {e}")
            device = "cpu"
            from indextts.infer_v2 import IndexTTS2
            tts = IndexTTS2(cfg_path=CFG_PATH, model_dir=MODEL_DIR, device=device)
        else:
            raise

    t0 = time.time()
    ok = 0
    for i, wav_path in enumerate(todo):
        cp = _cache_path(cache_dir, wav_path)
        success = _compute_and_save(tts, wav_path, cp)
        if success:
            ok += 1

        # Release MPS pool periodically
        if (i + 1) % 20 == 0:
            import gc
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()

        if (i + 1) % 50 == 0 or i == len(todo) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / (elapsed / 60)
            eta = (len(todo) - i - 1) / max(rate, 0.1)
            _log(f"  {i+1}/{len(todo)} done ({ok} OK)  {rate:.0f} clips/min  ETA {eta:.0f} min")

    elapsed = time.time() - t0
    _log(f"Done. {ok}/{len(todo)} cached in {elapsed/60:.1f} min → {cache_dir}")
    return 0 if ok == len(todo) else 1


if __name__ == "__main__":
    sys.exit(main())
