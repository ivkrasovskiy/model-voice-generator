"""
Fine-tune F5-TTS v1 Base on Cumberbatch Casanova narrator voice.

Memory-efficient setup for M3 Pro 18 GB unified memory:
  - Freezes early DiT transformer blocks; trains only the last N blocks + output
  - Adafactor optimizer (smaller state than Adam)
  - Per-clip forward/backward, gradient accumulation = effective batch size

Purpose is to verify the fine-tuning pipeline works and loss decreases.
For production-quality fine-tuning, rent a 24 GB+ GPU and unfreeze the whole model.

Usage:
    PYTHONHASHSEED=random uv run python scripts/finetune_f5.py \
        --dataset cumberbatch_casanova \
        --max-steps 200 \
        --train-last-n 4

Output:
    runs/finetune_casanova/
        loss_log.csv
        checkpoints/step_*.pt
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

if os.environ.get("PYTHONHASHSEED", "missing") == "":
    os.environ["PYTHONHASHSEED"] = "random"
    os.execv(sys.executable, [sys.executable, __file__] + sys.argv[1:])

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).parent.parent
DATA_BASE = PROJECT_ROOT / "data"
RUNS_BASE = PROJECT_ROOT / "runs"


def load_pretrained_f5tts(device: str):
    from f5_tts.model import DiT, CFM
    from f5_tts.model.utils import get_tokenizer
    from cached_path import cached_path
    from safetensors.torch import load_file
    import f5_tts

    print(f"Loading F5TTS_v1_Base on {device}...")

    model_cfg = dict(
        dim=1024, depth=22, heads=16, ff_mult=2,
        text_dim=512, text_mask_padding=False,
        qk_norm=None, conv_layers=4,
        pe_attn_head=1,
    )
    n_mel_channels = 100
    target_sample_rate = 24000

    # Find vocab.txt via importlib.resources (works with namespace packages)
    from importlib.resources import files
    tokenizer_path = files("f5_tts.infer.examples").joinpath("vocab.txt")

    vocab_char_map, vocab_size = get_tokenizer(str(tokenizer_path), "custom")
    transformer = DiT(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels)

    cfm = CFM(
        transformer=transformer,
        mel_spec_kwargs=dict(
            n_fft=1024, hop_length=256, win_length=1024,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
        ),
        odeint_kwargs=dict(method="euler"),
        vocab_char_map=vocab_char_map,
    )

    ckpt_path = cached_path("hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors")
    print(f"Loading checkpoint: {ckpt_path}")
    state_dict = load_file(str(ckpt_path))

    # Strip ema_model. prefix if present
    new_sd = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("ema_model."):
            nk = nk[len("ema_model."):]
        new_sd[nk] = v
    missing, unexpected = cfm.load_state_dict(new_sd, strict=False)
    print(f"Loaded weights — missing: {len(missing)}, unexpected: {len(unexpected)}")
    if len(missing) > 50:
        print(f"  First 5 missing: {missing[:5]}")
    if len(unexpected) > 50:
        print(f"  First 5 unexpected: {unexpected[:5]}")

    cfm = cfm.to(device)
    return cfm, target_sample_rate, vocab_char_map


def freeze_model_except_last_n(cfm, train_last_n: int):
    """Freeze every parameter except the last N transformer blocks + output proj."""
    transformer = cfm.transformer

    # Freeze everything first
    for p in cfm.parameters():
        p.requires_grad = False

    # Unfreeze last N transformer blocks
    n_blocks = len(transformer.transformer_blocks)
    unfreeze_from = max(0, n_blocks - train_last_n)
    for i, block in enumerate(transformer.transformer_blocks):
        if i >= unfreeze_from:
            for p in block.parameters():
                p.requires_grad = True

    # Unfreeze final norm + output projection
    for name in ["norm_out", "proj_out"]:
        if hasattr(transformer, name):
            for p in getattr(transformer, name).parameters():
                p.requires_grad = True

    trainable = sum(p.numel() for p in cfm.parameters() if p.requires_grad)
    total = sum(p.numel() for p in cfm.parameters())
    print(f"Trainable params: {trainable/1e6:.1f}M / {total/1e6:.1f}M  "
          f"({trainable/total*100:.1f}%)")
    print(f"Trainable: last {train_last_n}/{n_blocks} transformer blocks + output")


def load_dataset_entries(dataset_name: str) -> list[dict]:
    data_dir = DATA_BASE / dataset_name
    metadata_csv = data_dir / "metadata.csv"
    if not metadata_csv.exists():
        raise FileNotFoundError(f"No metadata.csv at {metadata_csv}")

    entries = []
    with metadata_csv.open() as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            wav_path = data_dir / "wavs" / f"{row['audio_file']}.wav"
            if wav_path.exists():
                entries.append({
                    "audio_path": str(wav_path),
                    "text": row["text"],
                    "duration": float(row["duration"]),
                })
    print(f"Dataset {dataset_name}: {len(entries)} valid entries")
    return entries


def train(args):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}, torch {torch.__version__}")

    cfm, sample_rate, vocab_char_map = load_pretrained_f5tts(device)
    freeze_model_except_last_n(cfm, args.train_last_n)

    entries = load_dataset_entries(args.dataset)
    if len(entries) < 10:
        raise RuntimeError(f"Too few entries: {len(entries)}")

    import torchaudio
    from f5_tts.model.modules import MelSpec

    mel_spec = MelSpec(
        n_fft=1024, hop_length=256, win_length=1024,
        n_mel_channels=100, target_sample_rate=sample_rate,
    ).to(device)

    trainable_params = [p for p in cfm.parameters() if p.requires_grad]

    # Adafactor — much smaller state than Adam (sqrt of 1 axis vs full matrix)
    try:
        from transformers.optimization import Adafactor
        optimizer = Adafactor(
            trainable_params, lr=args.lr, scale_parameter=False,
            relative_step=False, warmup_init=False,
        )
        print(f"Optimizer: Adafactor, lr={args.lr}")
    except ImportError:
        # Fallback to SGD (no state)
        optimizer = torch.optim.SGD(trainable_params, lr=args.lr * 100, momentum=0.0)
        print(f"Optimizer: SGD (Adafactor unavailable), lr={args.lr * 100}")

    run_dir = RUNS_BASE / args.run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "loss_log.csv"
    log_file = log_path.open("w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["step", "loss", "elapsed_sec"])

    cfm.train()
    rng = torch.Generator()
    rng.manual_seed(42)

    print(f"\n=== Training ===")
    print(f"  max_steps={args.max_steps}  lr={args.lr}  accum={args.grad_accum}")
    print(f"  device={device}  entries={len(entries)}")

    start = time.time()
    step = 0
    losses = []
    optimizer.zero_grad()

    while step < args.max_steps:
        idx = torch.randint(0, len(entries), (1,), generator=rng).item()
        entry = entries[idx]

        try:
            wav, sr = torchaudio.load(entry["audio_path"])
            if sr != sample_rate:
                wav = torchaudio.functional.resample(wav, sr, sample_rate)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            wav = wav.to(device)

            with torch.no_grad():
                mel = mel_spec(wav)  # (1, n_mel, T)
            mel = mel.transpose(1, 2)  # (1, T, n_mel)

            # F5-TTS CFM accepts text as list[str] and tokenizes internally
            mel_lengths = torch.tensor([mel.shape[1]], device=device)
            outputs = cfm(mel, text=[entry["text"]], lens=mel_lengths)

            # CFM returns (loss, cond, pred) or similar — pull the loss tensor
            if isinstance(outputs, tuple):
                loss = outputs[0]
            else:
                loss = outputs

            if not loss.requires_grad:
                print(f"  step {step}: loss tensor has no grad — check freezing")
                continue

            (loss / args.grad_accum).backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            losses.append(loss.item())
            step += 1

            if step % 5 == 0 or step == 1:
                recent = sum(losses[-10:]) / min(10, len(losses))
                elapsed = time.time() - start
                print(f"  step {step:4d}/{args.max_steps}  loss={loss.item():.4f}  "
                      f"avg_last10={recent:.4f}  elapsed={elapsed:.1f}s  "
                      f"(~{elapsed/step:.2f}s/step)")
                log_writer.writerow([step, f"{loss.item():.6f}", f"{elapsed:.2f}"])
                log_file.flush()

            if step % args.save_every == 0:
                ckpt_path = run_dir / "checkpoints" / f"step_{step:06d}.pt"
                torch.save({
                    "model": {k: v for k, v in cfm.state_dict().items() if "transformer_blocks" in k or "norm_out" in k or "proj_out" in k},
                    "step": step,
                }, ckpt_path)
                print(f"  saved partial checkpoint → {ckpt_path}")
        except Exception as e:
            print(f"  step {step} failed: {e}")
            import traceback; traceback.print_exc()
            if step == 0:
                raise
            continue

    log_file.close()
    final = run_dir / "checkpoints" / "final.pt"
    torch.save({
        "model": {k: v for k, v in cfm.state_dict().items() if "transformer_blocks" in k or "norm_out" in k or "proj_out" in k},
        "step": step,
    }, final)
    print(f"\nFinal partial checkpoint → {final}")
    print(f"Loss log → {log_path}")

    if len(losses) >= 20:
        first = sum(losses[:10]) / 10
        last = sum(losses[-10:]) / 10
        delta = first - last
        verdict = "DECREASING ✓" if last < first else "NOT DECREASING ✗"
        print(f"\nLoss summary:")
        print(f"  first 10 steps avg: {first:.4f}")
        print(f"  last 10 steps avg:  {last:.4f}")
        print(f"  delta: {delta:+.4f}  → {verdict}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cumberbatch_casanova")
    parser.add_argument("--run-name", default="finetune_casanova")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--train-last-n", type=int, default=4,
                        help="Train only the last N DiT transformer blocks (out of 22)")
    parser.add_argument("--save-every", type=int, default=100)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
