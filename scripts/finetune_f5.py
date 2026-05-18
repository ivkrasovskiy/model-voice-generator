"""
Fine-tune F5-TTS v1 Base on Cumberbatch Casanova narrator voice.

Memory-efficient setup for M3 Pro 18 GB unified memory:
  - Freezes early DiT transformer blocks; trains only the last N blocks + output
  - Adafactor optimizer (smaller state than Adam)
  - Per-clip forward/backward, gradient accumulation = effective batch size

Observability:
  - TensorBoard scalars (loss, grad_norm, lr, samples/sec, ETA)
  - Periodic audio samples (the Speech Accent Archive "Stella" phrase) logged
    as TB audio and saved to runs/<name>/samples/step_NNNNNN.wav
  - Periodic Resemblyzer speaker-similarity score (cosine to Casanova ref)
  - Early-stop if rolling-100 loss fails to improve > 1% over last 500 steps

Health checks (warnings, non-fatal):
  - Loss spike > 3x rolling mean
  - Grad-norm clipped > 50% of recent steps
  - NaN/Inf loss

Usage:
    PYTHONHASHSEED=random uv run python scripts/finetune_f5.py \
        --dataset cumberbatch_casanova \
        --max-steps 3000 \
        --train-last-n 4 \
        --eval-every 250

    # Watch in another terminal:
    tensorboard --logdir runs/finetune_casanova/tb

Output:
    runs/finetune_casanova/
        loss_log.csv
        events.jsonl            # one record per step
        checkpoints/step_*.pt
        samples/step_NNNNNN.wav # eval samples
        tb/                     # TensorBoard event files
"""

import argparse
import csv
import json
import sys
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _dotenv_init import init_env_then_reexec, kill_stale_python

init_env_then_reexec(__file__)

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
DATA_BASE = PROJECT_ROOT / "data"
RUNS_BASE = PROJECT_ROOT / "runs"

# Listen sample (logged as TB audio every eval step): Speech Accent Archive
# standard, truncated to F5-TTS single-batch limit (< 80 chars).
EVAL_PHRASE_LISTEN = "Please call Stella; ask her to bring these things from the store."

# Six scoring prompts (≤ 80 chars each). Kept in sync with EVAL_PHRASES in
# posthoc_eval.py — see that file for per-phrase rationale.
EVAL_PHRASES_SCORE = [
    "Please call Stella; ask her to bring these things from the store.",
    "When the sunlight strikes raindrops in the air, they act as a prism.",
    "I haven't seen him since the last meeting, but I'll ask around tomorrow.",
    "The algorithm processes each frame independently before merging results.",
    "Whose woods these are I think I know; his house is in the village.",
    "Stop. Don't move. There's something behind you.",
]

REF_AUDIO_PATH = PROJECT_ROOT / "tts_output/ref_narrator.wav"
REF_TEXT = "this an ideal opportunity for obtaining from her everything I wished."

# Two short phrases (≤ 50 chars) used for the in-training ECAPA eval hook.
# Chosen from the cross-eval set: short enough for single-batch F5-TTS,
# high ECAPA ceiling on baseline (0.86, 0.87), different registers.
ECAPA_EVAL_PHRASES = [
    "I found all the guests around him.",         # 34 chars, Casanova
    "That would indeed have been ingenious.",      # 38 chars, Sherlock
]


def load_pretrained_f5tts(device: str):
    from cached_path import cached_path
    from f5_tts.model import CFM, DiT
    from f5_tts.model.utils import get_tokenizer
    from safetensors.torch import load_file

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


def load_teacher(device: str):
    """Load a frozen copy of the pretrained F5TTS to use as KD teacher.

    The teacher is loaded onto the same device but never updated. Its velocity
    predictions anchor the student and prevent identity drift during fine-tuning.
    """
    teacher, _, _ = load_pretrained_f5tts(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    n = sum(p.numel() for p in teacher.parameters()) / 1e6
    print(f"Teacher loaded and frozen: {n:.1f}M params")
    return teacher


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
    from lib.dataset import load_metadata

    entries = load_metadata(dataset_name, project_root=PROJECT_ROOT)
    # finetune loop expects audio_path key (str), not wav (Path)
    for e in entries:
        e["audio_path"] = str(e.pop("wav"))
    print(f"Dataset {dataset_name}: {len(entries)} valid entries")
    return entries


def split_train_eval(
    entries: list[dict], n_eval: int, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    from lib.dataset import deterministic_split

    train_set, eval_set = deterministic_split(entries, n_val=n_eval, seed=seed)
    return train_set, eval_set


@torch.no_grad()
def compute_eval_loss(cfm, mel_spec, eval_entries: list[dict], device: str, sample_rate: int) -> float:
    """Forward-pass-only loss on the held-out eval set. No backward, no audio gen.

    Returns mean loss across all eval clips. Skips NaN/Inf to avoid pollution.
    """
    import torchaudio
    was_training = cfm.training
    cfm.eval()
    losses = []
    try:
        for entry in eval_entries:
            try:
                wav, sr = torchaudio.load(entry["audio_path"])
                if sr != sample_rate:
                    wav = torchaudio.functional.resample(wav, sr, sample_rate)
                if wav.shape[0] > 1:
                    wav = wav.mean(0, keepdim=True)
                wav = wav.to(device)
                mel = mel_spec(wav).transpose(1, 2)
                mel_lengths = torch.tensor([mel.shape[1]], device=device)
                outputs = cfm(mel, text=[entry["text"]], lens=mel_lengths)
                loss_val = float(outputs[0] if isinstance(outputs, tuple) else outputs)
                if np.isfinite(loss_val):
                    losses.append(loss_val)
            except Exception:
                continue
    finally:
        if was_training:
            cfm.train()
    return float(np.mean(losses)) if losses else float("nan")


def load_eval_machinery(device: str, load_resemblyzer: bool = True):
    """Pre-load vocoder and optionally Resemblyzer voice encoder + reference embedding."""
    from f5_tts.infer.utils_infer import load_vocoder

    print("Loading vocoder (vocos) for in-training eval...")
    vocoder = load_vocoder(vocoder_name="vocos", device=device)

    voice_encoder, ref_emb = None, None
    if load_resemblyzer:
        print("Loading Resemblyzer voice encoder (CPU)...")
        from resemblyzer import VoiceEncoder, preprocess_wav
        voice_encoder = VoiceEncoder(device="cpu", verbose=False)
        ref_wav = preprocess_wav(str(REF_AUDIO_PATH))
        ref_emb = voice_encoder.embed_utterance(ref_wav)
        print(f"  ref embedding: shape={ref_emb.shape}, norm={np.linalg.norm(ref_emb):.3f}")

    return vocoder, voice_encoder, ref_emb


def load_ecapa_machinery():
    """Load ECAPA-TDNN and embed the reference narrator clip (CPU)."""
    from lib.identity import embed_file, load_ecapa

    print("Loading ECAPA-TDNN (CPU) for in-training identity eval...")
    ecapa = load_ecapa(device="cpu")
    ref_emb = embed_file(REF_AUDIO_PATH, ecapa=ecapa)
    if ref_emb is None:
        raise RuntimeError(f"Could not embed ref audio: {REF_AUDIO_PATH}")
    print(f"  ECAPA ref embedding: shape={ref_emb.shape}, norm={np.linalg.norm(ref_emb):.3f}")
    return ecapa, ref_emb


@contextmanager
def _ema_applied(cfm: torch.nn.Module, ema: "EMATracker"):
    """Temporarily swap EMA shadow weights into cfm in-place, restore on exit.

    Used for inference-under-EMA without saving/loading checkpoints.
    """
    live_weights = {
        n: p.data.clone()
        for n, p in cfm.named_parameters()
        if n in ema.shadow
    }
    for n, p in cfm.named_parameters():
        if n in ema.shadow:
            p.data.copy_(ema.shadow[n].to(p.device))
    try:
        yield cfm
    finally:
        for n, p in cfm.named_parameters():
            if n in live_weights:
                p.data.copy_(live_weights[n])


@torch.no_grad()
def run_ecapa_eval(cfm, ema, ecapa_model, ref_ecapa_emb, vocoder, device, step, writer):
    """Generate ECAPA_EVAL_PHRASES with EMA weights, log cosine vs real ref clip.

    Uses EMA weights (not live) — avoids noisy mid-training identity scores.
    No Whisper — just ECAPA cosine similarity.
    """
    from f5_tts.infer.utils_infer import infer_process
    from lib.identity import cosine, embed_wav

    was_training = cfm.training
    sims = []

    with _ema_applied(cfm, ema):
        cfm.eval()
        for phrase in ECAPA_EVAL_PHRASES:
            wav, sr, _ = infer_process(
                ref_audio=str(REF_AUDIO_PATH),
                ref_text=REF_TEXT,
                gen_text=phrase,
                model_obj=cfm,
                vocoder=vocoder,
                device=device,
                show_info=lambda *a, **k: None,
            )
            if wav is None:
                continue
            w = wav.squeeze() if hasattr(wav, "squeeze") else wav
            w = np.asarray(w, dtype=np.float32)
            if not np.isfinite(w).all() or float(np.abs(w).max()) < 0.01:
                continue
            gen_emb = embed_wav(w, int(sr), ecapa_model)
            sims.append(cosine(ref_ecapa_emb, gen_emb))

    if was_training:
        cfm.train()

    if sims:
        mean_sim = float(np.mean(sims))
        writer.add_scalar("eval/ecapa_ema_mean", mean_sim, step)
        print(f"  [ecapa step {step}] ecapa_ema={mean_sim:.4f}  (n={len(sims)})")
        return mean_sim
    print(f"  [ecapa step {step}] ECAPA eval skipped — all phrases failed inference")
    return None


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    from lib.identity import cosine

    return cosine(a, b)


def run_eval(
    cfm,
    vocoder,
    voice_encoder,
    ref_emb: np.ndarray,
    device: str,
    out_dir: Path,
    step: int,
    writer,  # tensorboard SummaryWriter
    sample_rate: int,
):
    """Generate eval audio, score with Resemblyzer, log to TB.

    - 1 listen sample (EVAL_PHRASE_LISTEN) → saved + add_audio to TB
    - 3 score samples (EVAL_PHRASES_SCORE) → mean speaker similarity to ref
    Returns mean cosine similarity (or None on failure).
    """
    import soundfile as sf
    from f5_tts.infer.utils_infer import infer_process
    from resemblyzer import preprocess_wav

    was_training = cfm.training
    cfm.eval()

    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    sims = []
    listen_path = None

    def _infer_with_retry(prompt: str, max_retries: int = 5):
        for attempt in range(1, max_retries + 1):
            wav, sr, _ = infer_process(
                ref_audio=str(REF_AUDIO_PATH),
                ref_text=REF_TEXT,
                gen_text=prompt,
                model_obj=cfm,
                vocoder=vocoder,
                device=device,
                show_info=lambda *_a, **_k: None,
            )
            if wav is None:
                continue
            w = wav.squeeze() if hasattr(wav, "squeeze") else wav
            w = np.asarray(w, dtype=np.float32)
            finite = np.isfinite(w).all()
            peak = float(np.abs(w[np.isfinite(w)]).max()) if finite else 0.0
            if finite and peak > 0.01:
                return w, sr
            print(f"    attempt {attempt}: {'NaN/Inf' if not finite else f'SILENT (peak={peak:.4f})'} — retrying")
        return None, None

    try:
        for i, prompt in enumerate(EVAL_PHRASES_SCORE):
            wav_np, sr = _infer_with_retry(prompt)
            if wav_np is None:
                print(f"  [eval step {step}] prompt {i+1} failed after retries — skipped")
                continue

            # Listen sample = first one (Stella phrase). Save to disk + TB.
            if i == 0:
                listen_path = samples_dir / f"step_{step:06d}.wav"
                sf.write(str(listen_path), wav_np, sr)
                writer.add_audio(
                    "eval/sample_audio",
                    torch.from_numpy(wav_np).unsqueeze(0).float(),
                    global_step=step,
                    sample_rate=sr,
                )

            # Resemblyzer scoring
            gen_wav = preprocess_wav(wav_np, source_sr=sr)
            gen_emb = voice_encoder.embed_utterance(gen_wav)
            sim = cosine_sim(ref_emb, gen_emb)
            sims.append(sim)

        if sims:
            mean_sim = float(np.mean(sims))
            std_sim = float(np.std(sims))
            writer.add_scalar("eval/speaker_sim_mean", mean_sim, global_step=step)
            writer.add_scalar("eval/speaker_sim_std", std_sim, global_step=step)
            print(f"  [eval step {step}] speaker_sim={mean_sim:.4f}±{std_sim:.4f}  "
                  f"listen={listen_path.name if listen_path else 'n/a'}")
            return mean_sim
    except Exception as e:
        print(f"  [eval step {step}] FAILED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if was_training:
            cfm.train()

    return None


def save_partial_checkpoint(
    cfm, run_dir: Path, step: int, name: str | None = None, ema_sd: dict | None = None
):
    """Save only the trainable (requires_grad) weights. ~60% smaller than saving all blocks.

    If ema_sd is provided, also saves ema_<name>.pt alongside — EMA weights produce
    NaN-free audio at inference (same format, loaded by posthoc_eval.apply_checkpoint).
    """
    trainable_names = {n for n, p in cfm.named_parameters() if p.requires_grad}
    sd = {k: v for k, v in cfm.state_dict().items() if k in trainable_names}
    fname = name if name else f"step_{step:06d}.pt"
    path = run_dir / "checkpoints" / fname
    torch.save({"model": sd, "step": step}, path)
    if ema_sd is not None:
        ema_path = run_dir / "checkpoints" / f"ema_{fname}"
        torch.save({"model": ema_sd, "step": step}, ema_path)
    return path


class EMATracker:
    """Exponential moving average over a model's trainable parameters.

    Kept separate from the live model so we can checkpoint raw and EMA
    independently. EMA weights produce stable audio at inference; raw weights
    are what the optimizer updates.

    decay=0.9999 is the standard F5-TTS value; use lower (0.999) for small
    datasets where the EMA should track faster.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        # Shadow dict: {param_name: ema_tensor}. Only trainable params.
        self.shadow: dict[str, torch.Tensor] = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict:
        """Return {trainable_key: ema_tensor} in the checkpoint format expected by posthoc_eval."""
        # posthoc_eval.apply_checkpoint reads key as "transformer.<block>.<param>"
        # model.named_parameters already includes the full qualified name; map to
        # the same format the live checkpoint uses (strip the top-level cfm. prefix
        # if present — named_parameters on the CFM object already excludes it).
        return {k: v.cpu() for k, v in self.shadow.items()}


def train(args):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}, torch {torch.__version__}")

    cfm, sample_rate, vocab_char_map = load_pretrained_f5tts(device)
    freeze_model_except_last_n(cfm, args.train_last_n)

    teacher = None
    if args.kd_lambda > 0:
        print(f"\nKD mode: λ={args.kd_lambda} — loading frozen teacher...")
        teacher = load_teacher(device)
        from f5_tts.model.utils import list_str_to_idx

    all_entries = load_dataset_entries(args.dataset)
    if len(all_entries) < 10:
        raise RuntimeError(f"Too few entries: {len(all_entries)}")

    entries, eval_entries = split_train_eval(all_entries, n_eval=args.n_eval_clips)
    print(f"Split: {len(entries)} train / {len(eval_entries)} eval (held out, seed=42)")

    import torchaudio
    from f5_tts.model.modules import MelSpec
    from torch.utils.tensorboard import SummaryWriter

    mel_spec = MelSpec(
        n_fft=1024, hop_length=256, win_length=1024,
        n_mel_channels=100, target_sample_rate=sample_rate,
    ).to(device)

    trainable_params = [p for p in cfm.parameters() if p.requires_grad]

    ema = EMATracker(cfm, decay=args.ema_decay)
    print(f"EMA tracker initialised (decay={args.ema_decay}, {len(ema.shadow)} params)")

    try:
        from transformers.optimization import Adafactor
        optimizer = Adafactor(
            trainable_params, lr=args.lr, scale_parameter=False,
            relative_step=False, warmup_init=False,
        )
        print(f"Optimizer: Adafactor, lr={args.lr}")
    except ImportError:
        optimizer = torch.optim.SGD(trainable_params, lr=args.lr * 100, momentum=0.0)
        print(f"Optimizer: SGD (Adafactor unavailable), lr={args.lr * 100}")

    run_dir = RUNS_BASE / args.run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "loss_log.csv"
    log_file = log_path.open("w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["step", "loss", "grad_norm", "elapsed_sec"])

    events_path = run_dir / "events.jsonl"
    events_file = events_path.open("w")

    tb_dir = run_dir / "tb"
    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"TensorBoard: tensorboard --logdir {tb_dir}")

    vocoder, voice_encoder, ref_emb = (None, None, None)
    ecapa_model, ref_ecapa_emb = (None, None)
    need_vocoder = args.eval_every > 0 or args.ecapa_every > 0
    if need_vocoder:
        vocoder, voice_encoder, ref_emb = load_eval_machinery(
            device, load_resemblyzer=(args.eval_every > 0)
        )
    if args.ecapa_every > 0:
        ecapa_model, ref_ecapa_emb = load_ecapa_machinery()

    cfm.train()
    rng = torch.Generator()
    rng.manual_seed(42)

    print("\n=== Training ===")
    print(f"  max_steps={args.max_steps}  lr={args.lr}  accum={args.grad_accum}")
    print(f"  eval_every={args.eval_every}  ecapa_every={args.ecapa_every}  save_every={args.save_every}")
    print(f"  early_stop: stop if best eval_loss not beaten in {args.early_stop_window} steps")
    print(f"  kd_lambda={args.kd_lambda}  device={device}  entries={len(entries)}")

    start = time.time()
    step = 0
    losses: list[float] = []
    rolling100: deque[float] = deque(maxlen=100)
    grad_norms: deque[float] = deque(maxlen=100)
    spike_count = 0
    nan_count = 0
    best_eval_loss = float("inf")
    best_eval_step = 0
    stop_reason = "max_steps"
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
                mel = mel_spec(wav)
            mel = mel.transpose(1, 2)

            mel_lengths = torch.tensor([mel.shape[1]], device=device)
            outputs = cfm(mel, text=[entry["text"]], lens=mel_lengths)
            cfm_loss = outputs[0] if isinstance(outputs, tuple) else outputs

            kd_loss_val = 0.0
            if teacher is not None:
                # KD: fresh (x0, time, phi) — anchor student velocity to frozen teacher.
                # Uses full audio conditioning (cond=x1) to directly regularise the
                # conditioned predictions that drive speaker identity at inference.
                with torch.no_grad():
                    x1 = mel  # [1, T, 100] — clean mel, no grad
                    x0 = torch.randn_like(x1)
                    kd_time = torch.rand((1,), dtype=x1.dtype, device=device)
                    t_e = kd_time.unsqueeze(-1).unsqueeze(-1)
                    phi = (1 - t_e) * x0 + t_e * x1
                    kd_mask = torch.ones(1, mel.shape[1], dtype=torch.bool, device=device)
                    kd_text = list_str_to_idx([entry["text"]], cfm.vocab_char_map).to(device)
                    pred_teacher = teacher.transformer(
                        x=phi, cond=x1, text=kd_text, time=kd_time, mask=kd_mask,
                    )

                pred_student = cfm.transformer(
                    x=phi, cond=x1, text=kd_text, time=kd_time, mask=kd_mask,
                )
                kd_loss = F.mse_loss(pred_student, pred_teacher.detach())
                kd_loss_val = kd_loss.item()
                loss = cfm_loss + args.kd_lambda * kd_loss
            else:
                loss = cfm_loss

            if not loss.requires_grad:
                print(f"  step {step}: loss has no grad — freeze misconfigured")
                continue

            loss_val = loss.item()
            if not np.isfinite(loss_val):
                nan_count += 1
                print(f"  step {step}: NaN/Inf loss — skipping ({nan_count} total)")
                optimizer.zero_grad()
                continue

            (loss / args.grad_accum).backward()

            grad_norm = 0.0
            if (step + 1) % args.grad_accum == 0:
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                )
                optimizer.step()
                optimizer.zero_grad()
                ema.update(cfm)
                grad_norms.append(grad_norm)

            losses.append(loss_val)
            rolling100.append(loss_val)
            step += 1

            elapsed = time.time() - start
            sec_per_step = elapsed / step
            eta_sec = sec_per_step * (args.max_steps - step)
            rolling_mean = float(np.mean(rolling100))

            # Health: loss spike > 3x rolling mean (only meaningful once we have history)
            if len(rolling100) >= 20 and loss_val > 3 * rolling_mean:
                spike_count += 1

            # TB scalars every step (cheap)
            writer.add_scalar("loss/step", loss_val, step)
            writer.add_scalar("loss/rolling100", rolling_mean, step)
            if teacher is not None:
                writer.add_scalar("loss/cfm", float(cfm_loss.item()), step)
                writer.add_scalar("loss/kd", kd_loss_val, step)
            if grad_norm > 0:
                writer.add_scalar("grad/norm_pre_clip", grad_norm, step)
            writer.add_scalar("progress/eta_min", eta_sec / 60, step)
            writer.add_scalar("progress/sec_per_step", sec_per_step, step)

            # JSONL event
            events_file.write(json.dumps({
                "step": step, "loss": loss_val, "rolling100": rolling_mean,
                "kd_loss": kd_loss_val, "grad_norm": grad_norm,
                "elapsed_sec": elapsed, "eta_min": eta_sec / 60,
            }) + "\n")
            events_file.flush()

            if step % 5 == 0 or step == 1:
                kd_str = f"  kd={kd_loss_val:.4f}" if teacher is not None else ""
                print(f"  step {step:5d}/{args.max_steps}  loss={loss_val:.4f}  "
                      f"r100={rolling_mean:.4f}  gn={grad_norm:.3f}{kd_str}  "
                      f"elapsed={elapsed:.0f}s  eta={eta_sec/60:.1f}m")
                log_writer.writerow([step, f"{loss_val:.6f}", f"{grad_norm:.4f}", f"{elapsed:.2f}"])
                log_file.flush()

            # Release MPS allocator buffers every N steps so the high-water mark
            # doesn't drift into swap. KD doubles per-step activation memory and
            # this allocator does not auto-release between iterations.
            if step % args.mps_cache_every == 0 and device == "mps":
                import gc
                gc.collect()
                torch.mps.empty_cache()

            # Held-out eval loss (cheap, no audio gen, no NaN issue)
            if args.eval_loss_every > 0 and step % args.eval_loss_every == 0:
                eval_loss = compute_eval_loss(cfm, mel_spec, eval_entries, device, sample_rate)
                writer.add_scalar("loss/eval_mean", eval_loss, step)
                events_file.write(json.dumps({
                    "step": step, "event": "eval_loss", "eval_loss": eval_loss,
                    "train_rolling100": rolling_mean,
                }) + "\n")
                events_file.flush()
                gap = eval_loss - rolling_mean
                gap_flag = " ⚠️  OVERFIT?" if gap > 0.15 else ""
                print(f"  [eval_loss step {step}] eval={eval_loss:.4f}  train_r100={rolling_mean:.4f}  "
                      f"gap={gap:+.4f}{gap_flag}")

                # Auto-save the best eval-loss checkpoint — otherwise the global minimum
                # falls between save_every saves and is lost (happened on the 8blk run:
                # best eval @ step ~1600, but saves were at 1500 / 2000).
                if np.isfinite(eval_loss) and eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    best_eval_step = step
                    best_path = save_partial_checkpoint(
                        cfm, run_dir, step, name="best.pt", ema_sd=ema.state_dict()
                    )
                    writer.add_scalar("loss/best_eval", best_eval_loss, step)
                    events_file.write(json.dumps({
                        "step": step, "event": "best_eval",
                        "eval_loss": eval_loss, "saved": str(best_path.name),
                    }) + "\n")
                    events_file.flush()
                    print(f"    → new best eval={best_eval_loss:.4f}; saved {best_path.name}")

                # Early stop: if global best eval-loss hasn't been beaten in
                # `early_stop_window` steps, the model has plateaued/diverged.
                # Uses *global* best (not previous-step delta), so a one-off
                # bad eval doesn't trip the stop and a sustained drift does.
                if best_eval_step > 0 and (step - best_eval_step) >= args.early_stop_window:
                    print(f"\n  EARLY STOP: eval_loss has not beaten global best "
                          f"({best_eval_loss:.4f} @ step {best_eval_step}) for "
                          f"{step - best_eval_step} steps (window={args.early_stop_window})")
                    stop_reason = "early_stop_eval_plateau"
                    break

            # ECAPA identity eval using EMA weights (no Whisper, 2 short phrases)
            if args.ecapa_every > 0 and step % args.ecapa_every == 0:
                run_ecapa_eval(cfm, ema, ecapa_model, ref_ecapa_emb,
                               vocoder, device, step, writer)
                if device == "mps":
                    import gc
                    gc.collect()
                    torch.mps.empty_cache()

            # Audio eval hook (separate, broken on raw weights — kept off by default)
            if args.eval_every > 0 and step % args.eval_every == 0:
                run_eval(cfm, vocoder, voice_encoder, ref_emb, device,
                         run_dir, step, writer, sample_rate)

            # Checkpoint
            if step % args.save_every == 0:
                ckpt = save_partial_checkpoint(cfm, run_dir, step)
                print(f"  saved checkpoint → {ckpt.name}")

            # Early stop is handled inside the eval_loss block above —
            # we stop when global-best eval_loss hasn't been beaten in
            # `early_stop_window` steps. Tracking training-loss rolling100
            # was misleading: it's noisy and doesn't catch divergence.

            # Health warnings every 250 steps
            if step % 250 == 0:
                spike_rate = spike_count / max(step, 1)
                clip_rate = sum(1 for g in grad_norms if g >= 1.0) / max(len(grad_norms), 1)
                if spike_rate > 0.05:
                    print(f"  HEALTH: loss spikes {spike_count}/{step} steps ({spike_rate*100:.1f}%) — consider lr=lr/2")
                if clip_rate > 0.5:
                    print(f"  HEALTH: grad-norm clipped {clip_rate*100:.0f}% of recent steps — lr likely too high")
                if nan_count > 0:
                    print(f"  HEALTH: {nan_count} NaN/Inf losses skipped so far")

        except Exception as e:
            print(f"  step {step} failed: {e}")
            import traceback
            traceback.print_exc()
            if step == 0:
                raise
            continue

    log_file.close()
    events_file.close()
    final = save_partial_checkpoint(cfm, run_dir, step, name="final.pt")
    print(f"\nFinal partial checkpoint → {final}")
    print(f"Loss log → {log_path}")
    print(f"Events JSONL → {events_path}")
    print(f"Stop reason: {stop_reason}")

    if len(losses) >= 20:
        first = sum(losses[:10]) / 10
        last = sum(losses[-10:]) / 10
        delta = first - last
        verdict = "DECREASING ✓" if last < first else "NOT DECREASING ✗"
        print("\nLoss summary:")
        print(f"  first 10 steps avg: {first:.4f}")
        print(f"  last 10 steps avg:  {last:.4f}")
        print(f"  delta: {delta:+.4f}  → {verdict}")
        print(f"  best eval_loss:     {best_eval_loss:.4f} @ step {best_eval_step}  "
              f"({'saved as best.pt' if best_eval_step > 0 else 'no eval taken'})")
        print(f"  spikes: {spike_count}  NaN: {nan_count}")

    writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cumberbatch_casanova")
    parser.add_argument("--run-name", default="finetune_casanova")
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--train-last-n", type=int, default=4,
                        help="Train only the last N DiT transformer blocks (out of 22)")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=0,
                        help="Generate audio + speaker-sim every N steps (0 = disabled; broken without EMA)")
    parser.add_argument("--eval-loss-every", type=int, default=100,
                        help="Compute held-out eval loss every N steps (0 = disabled)")
    parser.add_argument("--n-eval-clips", type=int, default=50,
                        help="How many clips to hold out for eval-loss computation")
    parser.add_argument("--early-stop-window", type=int, default=200,
                        help="Stop if global-best eval_loss has not been beaten "
                             "in this many steps. Checked at each eval_loss_every tick.")
    parser.add_argument("--kd-lambda", type=float, default=0.0,
                        help="Knowledge distillation weight: λ × L2(student_vel, teacher_vel) "
                             "added to CFM loss. 0 = disabled. Try 0.2, 0.5, 1.0.")
    parser.add_argument("--mps-cache-every", type=int, default=25,
                        help="Flush MPS allocator every N steps (KD doubles activations and "
                             "MPS doesn't auto-release). Lower = lower memory, slightly slower.")
    parser.add_argument("--ecapa-every", type=int, default=100,
                        help="Compute ECAPA cosine vs ref clip every N steps using EMA weights "
                             "(0 = disabled). Loads ECAPA-TDNN on CPU; costs ~15s per eval.")
    parser.add_argument("--ema-decay", type=float, default=0.9999,
                        help="EMA decay for trainable parameters (0.9999 = standard F5-TTS value; "
                             "use 0.999 for small datasets to track faster). EMA weights are saved "
                             "as ema_best.pt alongside best.pt and produce NaN-free audio at eval.")
    args = parser.parse_args()

    # Free MPS memory by killing any stale F5-TTS / eval procs from prior runs.
    # CLAUDE.md notes that 3 stale F5-TTS procs ≈ 21 GB → OOM on 18 GB M3 Pro.
    n_killed = kill_stale_python()
    if n_killed:
        print(f"Pre-launch: killed {n_killed} stale F5-TTS/eval process(es)")

    train(args)


if __name__ == "__main__":
    main()
