"""Diagnose MPS memory allocation per IndexTTS2 component and per training step.

Run with:
  vendor/index-tts/.venv/bin/python scripts/diagnose_mps_ram.py

Reports:
  - RAM after loading each sub-model (gpt, semantic_codec, s2mel, campplus, bigvgan)
  - mps_live = torch.mps.current_allocated_memory()  (live tensors only)
  - mps_pool = torch.mps.driver_allocated_memory()   (live + cached pool)
  - rss      = process resident set size
  - Per-step delta to catch leaks that survive empty_cache()
"""
from __future__ import annotations

import gc
import json
import resource
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
INDEXTTS_ROOT = PROJECT_ROOT / "vendor" / "index-tts"
sys.path.insert(0, str(INDEXTTS_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch

CFG_PATH    = str(INDEXTTS_ROOT / "checkpoints/config.yaml")
MODEL_DIR   = str(INDEXTTS_ROOT / "checkpoints")
CACHE_DIR   = PROJECT_ROOT / "tts_output/accent_coach/phase0_14/lora/emb_cache"
DATASET_DIR = PROJECT_ROOT / "tts_output/accent_coach/phase0_14/lora/dataset"


def _mem():
    live = torch.mps.current_allocated_memory() / 1e9
    pool = torch.mps.driver_allocated_memory() / 1e9
    ru   = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss  = ru / 1e9 if sys.platform == "darwin" else ru / 1e6
    return live, pool, rss


def _snap(label: str, prev=None):
    live, pool, rss = _mem()
    delta = ""
    if prev is not None:
        dl, dp, dr = live - prev[0], pool - prev[1], rss - prev[2]
        delta = f"  Δlive={dl:+.3f} Δpool={dp:+.3f} Δrss={dr:+.3f}"
    print(f"{label:45s}  live={live:.3f}GB  pool={pool:.3f}GB  rss={rss:.3f}GB{delta}",
          flush=True)
    return (live, pool, rss)


def main():
    device = "mps"
    print(f"\n=== MPS RAM Diagnosis — device={device} ===\n")

    baseline = _snap("baseline (before any import)")

    # Load model piece by piece using monkey-patching to intercept load order
    from indextts.infer_v2 import IndexTTS2
    snap_after_import = _snap("after IndexTTS2 import", baseline)

    tts = IndexTTS2(cfg_path=CFG_PATH, model_dir=MODEL_DIR, device=device)
    snap_after_load = _snap("after IndexTTS2.__init__() complete", snap_after_import)

    # Sub-model breakdown
    components = {
        "gpt":            tts.gpt,
        "semantic_codec": getattr(tts, "semantic_codec", None),
        "s2mel":          getattr(tts, "s2mel", None),
        "campplus":       getattr(tts, "campplus_model", None),
        "bigvgan":        getattr(tts, "bigvgan", None),
    }
    for name, mod in components.items():
        if mod is None:
            print(f"  {name}: not found")
            continue
        params = sum(p.numel() for p in mod.parameters())
        device_set = {str(p.device) for p in mod.parameters()}
        print(f"  {name}: {params/1e6:.1f}M params  devices={device_set}")

    print()
    snap_before_lora = _snap("before LoRA application", snap_after_load)

    from peft import LoraConfig, get_peft_model
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["c_attn", "c_proj", "c_fc"],
        lora_dropout=0.05, bias="none", task_type=None,
    )
    peft_model = get_peft_model(tts.gpt.gpt, lora_cfg)
    tts.gpt.gpt = peft_model
    for name, param in tts.gpt.named_parameters():
        if "gpt." not in name:
            param.requires_grad_(False)
    snap_after_lora = _snap("after LoRA application", snap_before_lora)

    # Optimizer
    trainable = [p for p in tts.gpt.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    snap_after_optim = _snap("after AdamW init", snap_after_lora)

    # Training step baseline
    from lib.lora_targets import extract_targets, gpt_ce_loss

    train_items = [json.loads(line) for line in (DATASET_DIR / "train.jsonl").read_text().splitlines() if line.strip()]
    items = train_items[:50]

    print("\n--- Per-step memory (first 50 clips) ---")
    print(f"{'step':>5}  {'loss':>8}  {'live':>8}  {'pool':>8}  {'rss':>8}  {'Δlive':>8}  {'Δpool':>8}")

    prev = snap_after_optim
    tts.gpt.train()

    for i, item in enumerate(items):
        # Pick ref from the same speaker
        spk = item.get("speaker")
        refs = [e["wav"] for e in train_items if e.get("speaker") == spk and e["wav"] != item["wav"]]
        if not refs:
            continue
        ref = refs[0]

        batch = extract_targets(tts, item["wav"], item["text"], ref, device, cache_dir=CACHE_DIR)
        loss = gpt_ce_loss(tts.gpt, batch)
        loss_val = loss.item()
        (loss / 16).backward()
        del batch, loss

        if (i + 1) % 16 == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad()

        gc.collect()
        torch.mps.synchronize()
        torch.mps.empty_cache()

        live, pool, rss = _mem()
        dl, dp = live - prev[0], pool - prev[1]
        print(f"  {i:3d}  {loss_val:8.4f}  {live:8.3f}  {pool:8.3f}  {rss:8.3f}  {dl:+8.4f}  {dp:+8.4f}")
        prev = (live, pool, rss)

    print("\n=== Summary ===")
    _snap("end of 50 steps", snap_after_optim)


if __name__ == "__main__":
    main()
