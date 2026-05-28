"""WS-C C3/C4 — LoRA fine-tuning of IndexTTS-2 GPT (UnifiedVoice backbone).

Run in vendor venv:
  vendor/index-tts/.venv/bin/python scripts/accent_coach_phase0_14_lora_train.py \
      --dataset-dir tts_output/accent_coach/phase0_14/lora/dataset \
      --dry-run       # 32 clips / 200 steps to confirm loss decreases

LoRA targets: c_attn, c_proj, c_fc (HF GPT2 Conv1D layers in all 24 layers).
LoRA config: r=16, lora_alpha=32, lora_dropout=0.05, bias=none.
Frozen: conditioning_encoder, perceiver_encoder, emo_*, mel_head (first run).
Device: mps (falls back to cpu on unsupported-op error).
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

# Must be set before torch is imported; enables CPU fallback for MPS ops that
# lack float32 precision (e.g. SDPA backward) — prevents forward NaN on bad clips.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

PROJECT_ROOT = Path(__file__).parent.parent
INDEXTTS_ROOT = PROJECT_ROOT / "vendor" / "index-tts"
sys.path.insert(0, str(INDEXTTS_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch

CFG_PATH = str(INDEXTTS_ROOT / "checkpoints/config.yaml")
MODEL_DIR = str(INDEXTTS_ROOT / "checkpoints")
CKPT_ROOT = PROJECT_ROOT / "tts_output/accent_coach/phase0_14/lora/ckpt"


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def _load_dataset(jsonl_path: Path) -> list[dict]:
    items = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _pick_ref(items: list[dict], exclude_wav: str) -> str | None:
    """Pick a different clip from the same speaker as exclude_wav."""
    exclude_speaker = next(
        (e["speaker"] for e in items if e["wav"] == exclude_wav), None
    )
    same_spk = [e["wav"] for e in items
                if e["speaker"] == exclude_speaker and e["wav"] != exclude_wav]
    if not same_spk:
        return None
    import random
    return random.choice(same_spk)


def _build_lora_model(tts, device: str, train_mel_head: bool = False, lora_r: int = 16):
    """Apply LoRA to tts.gpt.gpt (HF GPT2). Returns peft_model."""
    from peft import LoraConfig, get_peft_model

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,
        target_modules=["c_attn", "c_proj", "c_fc"],
        lora_dropout=0.05,
        bias="none",
        task_type=None,
    )
    gpt2 = tts.gpt.gpt  # the HF GPT2Model backbone
    peft_model = get_peft_model(gpt2, lora_cfg)
    tts.gpt.gpt = peft_model

    # Freeze everything outside GPT2 backbone
    for name, param in tts.gpt.named_parameters():
        if "gpt." not in name:  # conditioning_encoder, perceiver_encoder, emo_*, head layers
            if name == "mel_head" and train_mel_head:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

    trainable = sum(p.numel() for p in tts.gpt.parameters() if p.requires_grad)
    total = sum(p.numel() for p in tts.gpt.parameters())
    _log(f"LoRA trainable: {trainable:,} / {total:,} = {100*trainable/total:.2f}%")
    return peft_model


def _train_step(tts, batch: dict, optimizer, grad_accum: int = 1, scaler=None) -> float | None:
    from lib.lora_targets import gpt_ce_loss
    loss = gpt_ce_loss(tts.gpt, batch)
    loss_val = loss.item()
    if not math.isfinite(loss_val):
        return None  # skip backward — don't corrupt accumulated gradients
    scaled = loss / grad_accum
    if scaler is not None:
        scaler.scale(scaled).backward()
    else:
        scaled.backward()
    # Per-step inner clip: caps each sample's gradient contribution to norm ≤ 1.0.
    # Without this, a single outlier clip (observed grad_norms up to 46K) biases AdamW
    # m2 estimates for all subsequent steps → eventual NaN from 0/0 in the Adam denominator.
    torch.nn.utils.clip_grad_norm_(
        [p for p in tts.gpt.parameters() if p.requires_grad and p.grad is not None], 1.0
    )
    return loss_val


def _eval_loss(tts, val_items: list[dict], device: str,
               n_eval: int = 50, cache_dir=None) -> float:
    from lib.lora_targets import extract_targets, gpt_ce_loss
    tts.gpt.eval()
    losses: list[float] = []
    eval_items = val_items[:n_eval]
    with torch.no_grad():
        for item in eval_items:
            ref = _pick_ref(val_items + val_items, item["wav"])
            if ref is None:
                continue
            try:
                batch = extract_targets(tts, item["wav"], item["text"], ref, device,
                                        cache_dir=cache_dir)
                loss = gpt_ce_loss(tts.gpt, batch)
                losses.append(loss.item())
                del batch
            except Exception as e:
                _log(f"  eval skip {Path(item['wav']).name}: {e}")
    tts.gpt.train()
    return float(sum(losses) / len(losses)) if losses else float("nan")


def _run_training(
    tts,
    train_items: list[dict],
    val_items: list[dict],
    device: str,
    args: argparse.Namespace,
) -> None:
    from lib.lora_targets import extract_targets

    optimizer = torch.optim.AdamW(
        [p for p in tts.gpt.parameters() if p.requires_grad],
        lr=args.lr,
    )
    warmup_steps = 20 if args.dry_run else 50
    def lr_lambda(step: int) -> float:
        return min(1.0, step / max(warmup_steps, 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # dry-run: small accum so we get updates within 32 clips; full: 16 for stability
    grad_accum = 4 if args.dry_run else 16
    step = 0
    best_eval_loss = float("inf")
    best_step = 0
    steps_since_best = 0
    stop_reason: str | None = None
    t0_run = time.time()
    items_cycle = list(train_items)

    tts.gpt.train()
    optimizer.zero_grad()

    epoch = 0
    # dry-run: loop many epochs until the 200-step check fires (32 clips × 7 epochs ≈ 224 steps)
    max_epochs = 50 if args.dry_run else args.epochs
    dry_run_items = items_cycle[:32] if args.dry_run else items_cycle

    while epoch < max_epochs and stop_reason is None:
        epoch += 1
        _log(f"=== Epoch {epoch}/{max_epochs} ({len(dry_run_items)} items) ===")
        import random
        random.shuffle(dry_run_items)

        for item in dry_run_items:
            if stop_reason is not None:
                break
            ref = _pick_ref(train_items, item["wav"])
            if ref is None:
                continue
            try:
                batch = extract_targets(
                    tts, item["wav"], item["text"], ref, device,
                    cache_dir=args.cache_dir,
                )
                loss_val = _train_step(tts, batch, optimizer, grad_accum=grad_accum)
                del batch
            except Exception as e:
                _log(f"  skip {Path(item['wav']).name}: {e}")
                continue

            if loss_val is None:
                # NaN/Inf forward — clear any partial grads and skip this sample
                _log(f"  NaN loss skipped: {Path(item['wav']).name}")
                optimizer.zero_grad()
                step += 1
                continue

            # Cheap per-step check: did this backward introduce NaN into any gradient?
            # Check one representative parameter (lora_B of layer 0, typically first updated).
            _first_p = next((p for p in tts.gpt.parameters() if p.requires_grad), None)
            if _first_p is not None and _first_p.grad is not None and not torch.isfinite(_first_p.grad).all():
                _log(f"  backward NaN: {Path(item['wav']).name} step={step} — zeroing grads")
                optimizer.zero_grad()
                step += 1
                continue

            if (step + 1) % grad_accum == 0:
                # Full NaN/Inf check on ALL accumulated grads before stepping
                _corrupt = next((n for n, p in tts.gpt.named_parameters()
                                 if p.requires_grad and p.grad is not None
                                 and not torch.isfinite(p.grad).all()), None)
                if _corrupt is not None:
                    _log(f"  corrupt grads ({_corrupt}) at step {step} — skipping optimizer step")
                    optimizer.zero_grad()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in tts.gpt.parameters() if p.requires_grad], 1.0
                    )
                    _log(f"  opt_step grad_norm={grad_norm:.3f} (step {step})")
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    # Detect NaN parameters produced by this optimizer step
                    nan_p = next((n for n, p in tts.gpt.named_parameters()
                                  if p.requires_grad and torch.isnan(p).any()), None)
                    if nan_p is not None:
                        _log(f"  FATAL: param {nan_p} is NaN after optimizer step — stopping")
                        return

            step += 1

            # MPS holds freed tensors in a pool until empty_cache().
            # synchronize() first drains the command queue so pending-op refs are released;
            # without it, empty_cache() can't reclaim those pages and the pool keeps growing.
            if step % 20 == 0:
                gc.collect()
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    torch.mps.synchronize()
                    torch.mps.empty_cache()

            log_every = 5 if args.dry_run else 20
            if step % log_every == 0:
                elapsed = time.time() - t0_run
                mem_info = ""
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    with contextlib.suppress(Exception):
                        live = torch.mps.current_allocated_memory() / 1e9
                        pool = torch.mps.driver_allocated_memory() / 1e9
                        mem_info = f" mps_live={live:.2f}GB mps_pool={pool:.2f}GB"
                with contextlib.suppress(Exception):
                    # ru_maxrss: bytes on macOS, kilobytes on Linux
                    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    rss_gb = ru / 1e9 if sys.platform == "darwin" else ru / 1e6
                    mem_info += f" rss={rss_gb:.2f}GB"
                loss_str = f"{loss_val:.4f}" if loss_val is not None else "nan(skipped)"
                _log(f"  step={step} loss={loss_str} lr={scheduler.get_last_lr()[0]:.2e} "
                     f"({elapsed/60:.1f} min, {step/(elapsed/60):.1f} steps/min{mem_info})")

            if args.dry_run and step >= 200:
                _log("Dry-run: 200 steps reached. Inspect loss trend above.")
                return

            # Hard step cap
            if not args.dry_run and args.max_steps and step >= args.max_steps:
                stop_reason = f"max_steps={args.max_steps} reached"
                break

            # Eval + early stopping (every eval_every steps)
            if not args.dry_run and step % args.eval_every == 0:
                eval_l = _eval_loss(tts, val_items, device, cache_dir=args.cache_dir)
                steps_since_best = step - best_step
                improved = eval_l < best_eval_loss
                _log(f"  eval_loss={eval_l:.4f} (best={best_eval_loss:.4f} "
                     f"@ step {best_step}, stale {steps_since_best} steps)")
                if improved:
                    best_eval_loss = eval_l
                    best_step = step
                    steps_since_best = 0
                    best_dir = CKPT_ROOT / "best"
                    best_dir.mkdir(parents=True, exist_ok=True)
                    tts.gpt.gpt.save_pretrained(str(best_dir))
                    _log(f"  ✓ new best → {best_dir}")
                elif steps_since_best >= args.patience:
                    stop_reason = (f"early stop: no improvement for {steps_since_best} steps "
                                   f"(patience={args.patience})")
                    break

            # Periodic checkpoint (independent of eval frequency)
            if not args.dry_run and step % 300 == 0:
                ckpt_dir = CKPT_ROOT / f"step_{step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                tts.gpt.gpt.save_pretrained(str(ckpt_dir))
                _log(f"  checkpoint → {ckpt_dir}")

    # Final checkpoint
    if not args.dry_run:
        if stop_reason:
            _log(f"Stopped: {stop_reason}")
        _log(f"Best eval_loss={best_eval_loss:.4f} at step {best_step} (saved to ckpt/best/)")
        ckpt_dir = CKPT_ROOT / f"step_{step}_final"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        tts.gpt.gpt.save_pretrained(str(ckpt_dir))
        _log(f"Final checkpoint → {ckpt_dir}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir",
                        default="tts_output/accent_coach/phase0_14/lora/dataset")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank (default 16; try 8 if loss unstable)")
    parser.add_argument("--train-mel-head", action="store_true",
                        help="Also fine-tune mel_head (default OFF for run 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run 32 clips for 200 steps; confirm loss decreases")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Dir of pre-extracted .pt embeddings (from cache_embeddings.py); "
                             "skips Whisper encoder during training — faster + less RAM")
    parser.add_argument("--device", default=None,
                        help="Force device: 'cpu' or 'mps'. Default: auto (mps if available)")
    parser.add_argument("--max-steps", type=int, default=900,
                        help="Hard cap on training steps (default 900; 0 = unlimited)")
    parser.add_argument("--eval-every", type=int, default=100,
                        help="Evaluate val loss every N steps (default 100)")
    parser.add_argument("--patience", type=int, default=200,
                        help="Early stop if best eval_loss unchanged for N steps (default 200)")
    args = parser.parse_args()

    dataset_dir = PROJECT_ROOT / args.dataset_dir
    train_path = dataset_dir / "train.jsonl"
    val_path = dataset_dir / "val.jsonl"
    if not train_path.exists():
        print(f"ERROR: {train_path} not found. Run build_dataset.py first.",
              file=sys.stderr)
        return 1

    # Device selection
    if args.device:
        device = args.device
        _log(f"Device: {device} (forced via --device)")
    else:
        try:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            _log(f"Device: {device}")
        except Exception:
            device = "cpu"
            _log("Device: cpu (mps probe failed)")

    _log("Loading IndexTTS2...")
    try:
        from indextts.infer_v2 import IndexTTS2
        tts = IndexTTS2(cfg_path=CFG_PATH, model_dir=MODEL_DIR, device=device)
    except RuntimeError as e:
        if "mps" in str(e).lower():
            _log(f"MPS error ({e}), retrying on cpu")
            device = "cpu"
            from indextts.infer_v2 import IndexTTS2
            tts = IndexTTS2(cfg_path=CFG_PATH, model_dir=MODEL_DIR, device=device)
        else:
            raise

    _log("Applying LoRA...")
    _build_lora_model(tts, device, train_mel_head=args.train_mel_head, lora_r=args.lora_r)

    train_items = _load_dataset(train_path)
    val_items = _load_dataset(val_path) if val_path.exists() else []
    _log(f"Dataset: {len(train_items)} train / {len(val_items)} val")

    if args.dry_run:
        _log("=== DRY RUN: 32 clips / 200 steps ===")
        _log("Expected: train loss strictly decreases. If not, stop and report.")
    else:
        _log(f"=== FULL RUN: {args.epochs} epochs ===")

    _run_training(tts, train_items, val_items, device, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
