"""Inspect cached embeddings for the 3 known NaN-causing clips."""
import sys
from pathlib import Path

import torch

CACHE = Path("tts_output/accent_coach/phase0_14/lora/emb_cache")
CLIPS = ["0_309", "0_133", "1_133"]

for stem in CLIPS:
    p = CACHE / f"{stem}.pt"
    d = torch.load(p, map_location="cpu", weights_only=True)
    mc = d["mel_codes"]   # (1, T) int64
    se = d["spk_emb"]     # (1, T', 1024) float32

    mc_flat = mc.flatten()
    se_flat = se.flatten()

    print(f"\n=== {stem} ===")
    print(f"  mel_codes  shape={tuple(mc.shape)}  dtype={mc.dtype}")
    print(f"    min={mc_flat.min().item()}  max={mc_flat.max().item()}"
          f"  zeros={int((mc_flat == 0).sum())}  len={mc_flat.numel()}")
    print(f"  spk_emb    shape={tuple(se.shape)}  dtype={se.dtype}")
    print(f"    min={se_flat.min().item():.4f}  max={se_flat.max().item():.4f}"
          f"  nan={int(torch.isnan(se_flat).sum())}  inf={int(torch.isinf(se_flat).sum())}")
    print(f"    norm={se_flat.norm().item():.2f}")
