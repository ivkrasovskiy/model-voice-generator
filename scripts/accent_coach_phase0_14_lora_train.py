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


def _build_lora_model(tts, device: str, train_mel_head: bool = False):
    """Apply LoRA to tts.gpt.gpt (HF GPT2). Returns peft_model."""
    from peft import LoraConfig, get_peft_model

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
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


def _train_step(tts, batch: dict, optimizer, scaler=None) -> float:
    from lib.lora_targets import gpt_ce_loss
    loss = gpt_ce_loss(tts.gpt, batch)
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()
    return loss.item()


def _eval_loss(tts, val_items: list[dict], device: str, n_eval: int = 50) -> float:
    from lib.lora_targets import extract_targets
    tts.gpt.eval()
    losses: list[float] = []
    eval_items = val_items[:n_eval]
    with torch.no_grad():
        for item in eval_items:
            ref = _pick_ref(val_items + val_items, item["wav"])
            if ref is None:
                continue
            try:
                batch = extract_targets(tts, item["wav"], item["text"], ref, device)
                from lib.lora_targets import gpt_ce_loss
                loss = gpt_ce_loss(tts.gpt, batch)
                losses.append(loss.item())
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
    warmup_steps = 50
    def lr_lambda(step: int) -> float:
        return min(1.0, step / max(warmup_steps, 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    grad_accum = 16
    step = 0
    best_eval_loss = float("inf")
    t0_run = time.time()
    items_cycle = list(train_items)

    tts.gpt.train()
    optimizer.zero_grad()

    epoch = 0
    max_epochs = 1 if args.dry_run else args.epochs
    dry_run_items = items_cycle[:32] if args.dry_run else items_cycle

    while epoch < max_epochs:
        epoch += 1
        _log(f"=== Epoch {epoch}/{max_epochs} ({len(dry_run_items)} items) ===")
        import random
        random.shuffle(dry_run_items)

        for item in dry_run_items:
            ref = _pick_ref(train_items, item["wav"])
            if ref is None:
                continue
            try:
                batch = extract_targets(tts, item["wav"], item["text"], ref, device)
                loss_val = _train_step(tts, batch, optimizer)
            except Exception as e:
                _log(f"  skip {Path(item['wav']).name}: {e}")
                continue

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in tts.gpt.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            step += 1

            if step % 20 == 0:
                elapsed = time.time() - t0_run
                _log(f"  step={step} loss={loss_val:.4f} lr={scheduler.get_last_lr()[0]:.2e} "
                     f"({elapsed/60:.1f} min, {step/(elapsed/60):.1f} steps/min)")

            if args.dry_run and step >= 200:
                _log("Dry-run: 200 steps reached. Inspect loss trend above.")
                return

            if not args.dry_run and step % 300 == 0:
                ckpt_dir = CKPT_ROOT / f"step_{step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                tts.gpt.gpt.save_pretrained(str(ckpt_dir))
                _log(f"  checkpoint → {ckpt_dir}")
                eval_l = _eval_loss(tts, val_items, device)
                _log(f"  eval_loss={eval_l:.4f} (best={best_eval_loss:.4f})")
                if eval_l < best_eval_loss:
                    best_eval_loss = eval_l
                else:
                    _log("  WARNING: eval loss not improving — check for overfit")

    # Final checkpoint
    if not args.dry_run:
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
    parser.add_argument("--train-mel-head", action="store_true",
                        help="Also fine-tune mel_head (default OFF for run 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run 32 clips for 200 steps; confirm loss decreases")
    args = parser.parse_args()

    dataset_dir = PROJECT_ROOT / args.dataset_dir
    train_path = dataset_dir / "train.jsonl"
    val_path = dataset_dir / "val.jsonl"
    if not train_path.exists():
        print(f"ERROR: {train_path} not found. Run build_dataset.py first.",
              file=sys.stderr)
        return 1

    # Device selection: try mps, fall back to cpu
    try:
        if torch.backends.mps.is_available():
            device = "mps"
            _log("Device: mps")
        else:
            device = "cpu"
            _log("Device: cpu (mps not available)")
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
    _build_lora_model(tts, device, train_mel_head=args.train_mel_head)

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
