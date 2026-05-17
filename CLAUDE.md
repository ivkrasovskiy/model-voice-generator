# CLAUDE.md

Guidance for Claude Code when working in this repo.

## Goal

Mimic Benedict Cumberbatch's **voice identity** — timbre, low-frequency depth, RP accent — using the Casanova audiobook as training data. **Audiobook narration cadence is a non-goal**; we want his voice in *any* content, not his reading style. Priority order: **ECAPA / CENT speaker-similarity > WER (intelligibility) > DNSMOS (perceived quality)**.

## Hardware

- macOS, **Apple M3 Pro, 18 GB unified memory**
- Use **MPS** (`torch.device("mps")`) for inference and training
- F5-TTS fine-tune with 8 trainable DiT blocks + Adafactor fits on 18 GB (~3 GB peak)

## Environment

```bash
uv sync                       # install/update deps from pyproject.toml
.venv/bin/python scripts/<script>  # PYTHONHASHSEED auto-fixed via _dotenv_init.py
```

## Active scripts

| Script | Purpose |
|---|---|
| `scripts/finetune_f5.py` | F5-TTS partial fine-tune — 8 DiT blocks, Adafactor, held-out eval loss, **auto-saves `best.pt`** on eval-loss improvement, TensorBoard |
| `scripts/posthoc_eval.py` | 2-phase eval: F5-TTS gen → Whisper WER + ECAPA + CENT (centroid) + DNSMOS. **Resume-safe** (incremental manifest); flags: `--cfg-strength --nfe-step --seed --first-n --centroid-dir --baseline-only` |
| `scripts/eval_compare.py` | Side-by-side Markdown report across N eval dirs (per-phrase + aggregate + Δ vs baseline) |
| `scripts/f5_infer.py` | One-off zero-shot inference |
| `scripts/_dotenv_init.py` | `init_env_then_reexec()` + `kill_stale_python()` — both called from main() of training/eval scripts so memory cleanup is automatic |
| `scripts/status.sh` | Pipeline snapshot (clip counts, last loss, running procs) |

## Dataset

- **Training data**: `data/cumberbatch_casanova/` — 1914 clips, avg 6.4s, 3.4h total. Silero-VAD segmented, Whisper large-v3 transcribed.
- **Reference clip**: `tts_output/ref_narrator.wav` — 12s from Casanova chunk 067. **At the F5-TTS auto-clip ceiling** (longer refs get silently truncated to ~12s — see [docs/literature_notes.md](docs/literature_notes.md)).

## Training runs to date

| Run dir | Blocks | Steps | Best eval loss | Note |
|---|---|---|---|---|
| `runs/finetune_casanova/` | 4 | 1000 | ~1.00 train | Killed at plateau |
| `runs/finetune_casanova_8blk/` | 8 | 2000 | 0.843 @ step 1000 | True min 0.793 @ ~1600 was uncaptured (fixed by `best.pt` auto-save) |

## Eval results (latest — seeded, 6 phrases, with centroid)

`tts_output/sweep_v2_cfg{20,30,40}/` — compared in [`tts_output/cfg_sweep_report.md`](tts_output/cfg_sweep_report.md).

| Config | WER ↓ | **ECAPA** ↑ | **CENT** ↑ | DNSMOS ↑ |
|---|---|---|---|---|
| **baseline cfg=2.0** | 1.25 | **0.837** | **0.743** | **3.94** |
| baseline cfg=3.0 | 1.13 | 0.808 | 0.722 | 3.90 |
| baseline cfg=4.0 | **0.92** | 0.755 | 0.680 | 3.66 |

Findings:
- **Monotonic tradeoff**: higher cfg trades identity (ECAPA/CENT/DNSMOS) for intelligibility (WER). cfg=2.0 (default) currently wins on identity.
- **Short-prompt failure**: `imperative` ("Stop. Don't move…") shows the trade most starkly — WER 2.57 → 1.00 but ECAPA 0.84 → 0.68. Documented F5-TTS fix exists (see literature notes).
- **CENT (centroid) is ~0.10 lower than single-ref ECAPA** — gen is closer to the specific 12s ref than to the average Cumberbatch sound. CENT is the more honest identity measure.
- **Prior fine-tunes hurt identity** (8blk@1000 ECAPA 0.71 vs baseline 0.84). Training pulls the model toward the narration register and away from raw voice.

## Next experiments

See [docs/literature_notes.md](docs/literature_notes.md) for the literature review behind this ordering (Selective CFG paper, short-text fix, F5R-TTS, etc.).

**Cheap inference fixes — do first**:
1. **`local_speed=0.3` for short phrases** (`len(segment) < 10 bytes`). Documented F5-TTS workaround for the short-text hallucination. Likely fixes the `imperative` WER without identity cost. ~10 min.
2. **CFG sweep below 2.0** — test cfg ∈ {1.0, 1.5, 2.0}. Literature suggests SIM may peak in this range; our prior sweep only covered 2.0-4.0. ~25 min using existing harness.
3. **BigVGAN vs Vocos vocoder A/B**. ~15 min. May recover DNSMOS lost at higher cfg.

**Medium effort**:
4. **Selective CFG patch** — apply standard CFG for first ~9 timesteps, text-only-conditioned CFG thereafter (Sept-2025 paper). ~20-line patch + ~30 min test.

**Training (only if inference fixes are insufficient — identity-preserving order)**:
5. **Knowledge distillation** — frozen F5 teacher, `λ × L2(student_vel, teacher_vel)` added to CFM loss. Counters ECAPA drift by anchoring student to pretrained prior. Sweep `λ ∈ {0.2, 0.5, 1.0}`. Validate teacher+student fits in 18 GB first. ~50 min.
6. **EMA + save-every 200** — unblocks in-training audio eval (NaN without EMA). `best.pt` auto-save reduces the urgency of save-every gap, but EMA is still needed for audio sampling.
7. **DNSMOS top-75% filter on training clips** — drop the noisy clips behind late-run gradient spikes.

**Big investigation**: [F5R-TTS (arXiv 2504.02407)](https://arxiv.org/abs/2504.02407) — RL successor with WER+SIM as joint reward. Reports +4.6% SIM / -29.5% WER vs F5-TTS. Check weight/code availability before committing.

## Eval harness — recommended workflow

```bash
# Quick subset check first (kills bad configs cheaply)
.venv/bin/python scripts/posthoc_eval.py \
    --baseline-only --cfg-strength X --first-n 2 --seed 42 \
    --centroid-dir data/cumberbatch_casanova \
    --out-dir tts_output/expN_subset

# If promising, full 6 phrases — manifest is incremental, mid-run crash recoverable
.venv/bin/python scripts/posthoc_eval.py \
    --baseline-only --cfg-strength X --seed 42 \
    --centroid-dir data/cumberbatch_casanova \
    --out-dir tts_output/expN_full

# Side-by-side Markdown report
.venv/bin/python scripts/eval_compare.py \
    tts_output/expA tts_output/expB tts_output/expC \
    --baseline tts_output/expA --out tts_output/expA_vs_BC.md
```

All inference is reproducible with `--seed 42`. Same args twice → identical WAVs.

## Memory / MPS — hard-won lessons (mostly automated now)

1. **Stale Python procs**: `kill_stale_python()` is called automatically from `main()` of `finetune_f5.py` and `posthoc_eval.py`. No manual `pkill` needed.
2. **PYTHONHASHSEED**: auto-fixed via `_dotenv_init.init_env_then_reexec(__file__)`. Just import at top.
3. **F5-TTS on MPS produces NaN audio (~30-50% per batch)**: `posthoc_eval.split_to_short_segments()` pre-splits phrases to ≤50 chars (single-batch), retries up to 5× per segment, calls `torch.mps.empty_cache()` after each gen.
4. **2-phase eval** (F5 → drop → Whisper/ECAPA/DNSMOS) — load order matters. Already enforced.
5. **In-training audio eval needs EMA** — direct inference on training weights produces NaN. The F5TTS API uses EMA internally (that's why posthoc works). Fix planned in Experiment 6.
6. **Whisper large-v3 needs ~3 GB RAM** — don't run concurrently with active training.

## Conventions

- Code runs **locally** on the user's Mac (M3 Pro).
- Do not commit `.env`, `dataset/`, `raw_cumberbatch_data/`, `tts_output/`, `.venv/`, `runs/`, `logs/`, `data/` — all gitignored.
- Always use `--run-name` (training) or `--out-dir` (eval) to separate experiments; never overwrite existing artifacts.
- `tensorboard --logdir runs/ --port 6006` shows all training runs together.
