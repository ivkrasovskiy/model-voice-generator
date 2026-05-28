"""Identify which GPT layer produces exploding gradients on MPS.

Runs ONE forward+backward on a single clip WITHOUT enable_fallback_to_cpu,
so the MPS precision issue can be reproduced and the culprit layer pinpointed.

Usage (vendor venv, MPS device only):
  vendor/index-tts/.venv/bin/python scripts/diagnose_mps_grad_explosion.py

Output: per-layer grad norm table sorted descending.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

# Intentionally NOT setting PYTORCH_ENABLE_MPS_FALLBACK — this script
# is designed to reproduce the MPS precision issue without the fallback.

PROJECT_ROOT = Path(__file__).parent.parent
INDEXTTS_ROOT = PROJECT_ROOT / "vendor" / "index-tts"
sys.path.insert(0, str(INDEXTTS_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch  # noqa: E402
from lib.lora_targets import extract_targets, gpt_ce_loss  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402

CFG_PATH = str(INDEXTTS_ROOT / "checkpoints/config.yaml")
MODEL_DIR = str(INDEXTTS_ROOT / "checkpoints")
CACHE_DIR = PROJECT_ROOT / "tts_output/accent_coach/phase0_14/lora/emb_cache"
DATASET = PROJECT_ROOT / "tts_output/accent_coach/phase0_14/lora/dataset/train.jsonl"

# Deliberately do NOT enable fallback — we want to reproduce the explosion
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")
if device == "cpu":
    print("WARNING: running on CPU — will not reproduce MPS explosion, but shows layer norms")

print("Loading model...")
from indextts.infer_v2 import IndexTTS2  # noqa: E402

tts = IndexTTS2(cfg_path=CFG_PATH, model_dir=MODEL_DIR, device=device)

lora_cfg = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["c_attn", "c_proj", "c_fc"],
    lora_dropout=0.0, bias="none", task_type=None,
)
tts.gpt.gpt = get_peft_model(tts.gpt.gpt, lora_cfg)
tts.gpt.train()
for p in tts.gpt.parameters():
    p.requires_grad_(False)
for n, p in tts.gpt.named_parameters():
    if "lora_" in n:
        p.requires_grad_(True)

# Pick the first item in dataset (known-good, not in bad-clip list)
with open(DATASET) as fh:
    items = [json.loads(line) for line in fh]

BAD = {"0_309", "0_133", "1_133"}
item = next(e for e in items if Path(e["wav"]).stem not in BAD)
ref_wav = random.choice(
    [e["wav"] for e in items
     if e["speaker"] == item["speaker"] and e["wav"] != item["wav"]]
)

print(f"Clip: {Path(item['wav']).name}  ref: {Path(ref_wav).name}")

batch = extract_targets(tts, item["wav"], item["text"], ref_wav, device,
                        cache_dir=CACHE_DIR)

# Register per-layer backward hooks to capture grad norms
layer_grads: dict[str, float] = {}

def _make_hook(name: str):
    def hook(module, grad_input, grad_output):
        for i, g in enumerate(grad_output):
            if g is not None:
                layer_grads[f"{name}.out[{i}]"] = g.norm().item()
        for i, g in enumerate(grad_input):
            if g is not None:
                layer_grads[f"{name}.in[{i}]"] = g.norm().item()
    return hook

for name, module in tts.gpt.gpt.named_modules():
    if any(t in name for t in ("c_attn", "c_proj", "c_fc", "attn", "mlp")):
        module.register_full_backward_hook(_make_hook(name))

loss = gpt_ce_loss(tts.gpt, batch)
print(f"Forward loss: {loss.item():.4f}")
loss.backward()

print("\nPer-layer gradient norms (sorted descending, top 30):")
print(f"{'Layer':<60} {'Norm':>12}")
print("-" * 74)
for name, norm in sorted(layer_grads.items(), key=lambda x: -x[1])[:30]:
    flag = " *** EXPLOSION" if norm > 100 else ""
    print(f"{name:<60} {norm:>12.1f}{flag}")

print("\nTrainable param grad norms:")
for name, p in tts.gpt.named_parameters():
    if p.requires_grad and p.grad is not None:
        print(f"  {name:<60} {p.grad.norm().item():>12.4f}")
