# CLAUDE.md

Guidance for Claude Code when working in this repo.

## Goal

Mimic Benedict Cumberbatch's **voice identity** — timbre, low-frequency depth, RP accent. **Audiobook narration cadence is a non-goal**; we want his voice in *any* content. Priority: **ECAPA(gen, real_clip) > WER > DNSMOS**.

## Hardware

- macOS, **Apple M3 Pro, 18 GB unified memory**
- Use **MPS** for inference and training. F5-TTS fine-tune with 8 trainable DiT blocks + Adafactor fits on 18 GB (~3–5 GB peak with KD).
- `torch.mps.empty_cache()` every 25 training steps (`--mps-cache-every`).

## Environment

```bash
uv sync                       # install/update deps (ruff in dev group)
uv sync --group dev           # if ruff not installed yet
.venv/bin/python scripts/<script>   # PYTHONHASHSEED auto-fixed via _dotenv_init.py
uv run ruff check scripts/    # lint — must be zero errors before commits
```

## Repository structure

```
scripts/
  lib/                    # Shared helpers (no duplication across scripts)
    audio_io.py           # read_wav_mono, resample, read_wav_at
    identity.py           # load_ecapa, embed_wav/embed_file, cosine, robust_centroid
    dataset.py            # load_metadata, write_metadata, deterministic_split
    transcribe.py         # load_whisper, transcribe_wav, transcribe_file
    inference.py          # split_to_short_segments, gen_segment, gen_phrase, mps_reset
    metrics.py            # compute_wer, compute_dnsmos, load_dnsmos
  posthoc_eval.py         # 2-phase eval; uses lib throughout
  finetune_f5.py          # Training; EMA, KD, early-stop on eval-loss
  eval_compare.py         # Side-by-side Markdown report
  build_ref_clip.py       # Build tts_output/ref_combined.wav + ref_sherlock.wav
  build_clean_dataset.py  # Merge datasets with audit/cluster filter
  process_audiobook.py    # VAD+ECAPA-filter a raw audiobook WAV → dataset
  cluster_characters.py   # k-means on utterance ECAPA → narrator cluster
  audit_dataset.py        # ECAPA distribution audit + centroid.npy
  make_experiment_v1.py   # Set up cross-eval holdout sets and training subsets
  cross_eval_compare.py   # Compare model outputs broken down by source (cas/sher)
  transcribe_missing.py   # Transcribe narrator-cluster clips lacking transcripts
  sample_matched_subset.py # Stratified random subset (duration-matched)
```

## Active scripts — key flags

| Script | Key flags |
|---|---|
| `posthoc_eval.py` | `--selective-cfg --t-threshold 0.08` (best inference), `--phrases-csv` (per-clip ECAPA), `--listen-checkpoints` |
| `finetune_f5.py` | `--kd-lambda 5` (KD), `--ema-decay 0.9999`, `--early-stop-window 200`, `--mps-cache-every 25` |
| `posthoc_eval.py` | No `--centroid-dir` needed when `--phrases-csv` has `ref_audio_path` — real clips used directly |

## Datasets

| Directory | Source | Clips | Duration | Note |
|---|---|---|---|---|
| `data/cumberbatch_casanova/` | Raw Casanova | 1914 | 3.4h | Original, has character voices |
| `data/cumberbatch_casanova_clean/` | Casanova filtered | 1735 | 3.2h | ECAPA sim ≥ 0.65 |
| `data/cumberbatch_sherlock_narrator/` | Sherlock narrator cluster | 379 | 30 min | k=5 cluster, sim 0.83 to Casanova |
| `data/cumberbatch_casanova_train_354/` | Casanova subset | 354 | 37.8 min | Duration-matched to Sherlock |
| `data/cumberbatch_sherlock_train_354/` | Sherlock subset | 354 | 28 min | Val set (25 clips) excluded |

- **Centroid**: `data/cumberbatch_casanova/centroid.npy` — robust centroid of Casanova, used to filter Sherlock
- **Eval set**: `tts_output/cross_eval_50/eval_short.csv` — 10 Casanova + 5 Sherlock short phrases (<60 chars), verified not in any training set, each with `ref_audio_path` to real clip
- **Reference clips**: `tts_output/ref_narrator.wav` (12s Casanova, F5-TTS inference), `tts_output/ref_combined.wav` (18s Casanova+Sherlock), `tts_output/ref_sherlock.wav` (6s Sherlock)

## Training runs

| Run dir | Blocks | λ | Steps | Best eval | Note |
|---|---|---|---|---|---|
| `runs/finetune_casanova/` | 4 | 0 | 1000 | ~1.00 | Killed at plateau |
| `runs/finetune_casanova_8blk/` | 8 | 0 | 2000 | 0.843 | Identity collapse (ECAPA 0.71) |
| `runs/finetune_kd_8blk_lam05/` | 8 | 0.5 | 1000 | 0.916 | λ too small (2.5% of loss) |
| `runs/finetune_kd_sher_lam10_aborted/` | 8 | 10 | 400 | 1.059 | Diverged; λ=10 too aggressive |
| `runs/finetune_kd_sher_lam5/` | 8 | 5 | 300 (ES) | **1.038 @ step 100** | Sherlock data, proper early stop |
| `runs/finetune_kd_cas_lam5/` | 8 | 5 | 600 (max) | **0.918 @ step 450** | Casanova data, proper early stop |

Early stop now works correctly: stops when global-best eval_loss not beaten in `--early-stop-window` steps.

## Eval results

**Best inference config**: `--selective-cfg --t-threshold 0.08 --cfg-strength 2.0`

Prior eval used centroid ECAPA (less honest). **New standard**: `ECAPA(generated, real_clip)` per phrase, using `--phrases-csv eval_short.csv` which has `ref_audio_path` per row.

Old 6-phrase results (centroid-based, for reference):

| Config | WER ↓ | ECAPA ↑ | DNSMOS ↑ |
|---|---|---|---|
| F5-TTS sel-CFG (best) | 1.10 | 0.845 | 3.90 |
| F5-TTS baseline cfg=2.0 | 1.25 | 0.837 | 3.94 |
| F5-TTS KD λ=0.5 | 1.17 | 0.828 | 3.95 |
| OpenVoice V2 | **0.02** | 0.400 | 3.28 |

Cross-eval on 15 short phrases (per-clip ECAPA): `tts_output/cross_eval_v1/` — **currently running**.

**Key architectural finding**: F5-TTS (mel-conditioned, high identity / gibberish leakage) vs OpenVoice V2 (speaker-embedding, perfect text / generic voice). Need hybrid.

## Key findings

- **Selective CFG** (arXiv 2509.19668): standard CFG for t≤0.08, text-conditioned CFG thereafter. Patches in `.venv`: `cfm.py`, `utils_infer.py`, `api.py`.
- **Reference text leakage**: F5-TTS injects ref text mid-output. Leakage is ref-text-bound (different ref = different leaked phrases). Not fixable without architecture change.
- **Fine-tuning hurt identity** every time. λ=5 KD with proper early stop is the best we've managed (eval_loss 0.918 Casanova, 1.038 Sherlock). In-training ECAPA pending.
- **Narration register bias**: Casanova training data pulls model toward narrator-speak. Need diverse register data (interviews, non-narration audiobooks).
- **EMA now implemented**: `EMATracker` in `finetune_f5.py`. `ema_best.pt` saved alongside `best.pt`. Checkpoints now 456 MB (was 1.2 GB — bug fixed).
- **6-phrase eval was too small**: differences of ±0.01 ECAPA are inside noise. New eval set is 15 verified phrases with real-clip ECAPA targets.

## Next steps (priority order)

1. **Listen to cross-eval results** (`tts_output/cross_eval_v1/`) — running now (~40 min)
2. **In-training ECAPA hook**: every 100 steps, generate 2 short phrases using EMA weights, log ECAPA vs real clip. No Whisper during training. Unblocked by EMA implementation.
3. **Data: add Sherlock + interviews** — ECAPA-filter Sherlock for narrator-only; download 2-3 Cumberbatch interviews + diarize. Primary lever for identity improvement.
4. **Re-train with combined data + λ=5 KD** — compare Casanova-only vs Casanova+Sherlock on cross-eval set.
5. **XTTS-v2 comparison** — hybrid mel+embedding, was used before, worth re-testing.
6. **F5R-TTS** (arXiv 2504.02407) — RL fine-tune with WER+SIM reward, most principled fix for the gibberish/identity trade-off.

## Memory / MPS

1. `kill_stale_python()` runs automatically at script start.
2. `--mps-cache-every 25` in training (default) keeps allocator from growing to 21 GB.
3. KD doubles per-step activation footprint (~3–5 GB total vs ~2 GB without KD).
4. EMA tracking adds ~480 MB (shadow copy of trainable params) — acceptable.
5. Never run Whisper + active training simultaneously.

## Conventions

- Code runs locally on user's Mac (M3 Pro).
- Do not commit: `.env`, `dataset/`, `raw_cumberbatch_data/`, `tts_output/`, `.venv/`, `runs/`, `logs/`, `data/`
- Always use `--run-name` / `--out-dir` to separate experiments.
- `tensorboard --logdir runs/ --port 6006` shows all training runs.
- **Check progress**: `ls tts_output/cross_eval_v1/*.wav | wc -l` (0→45 for current cross-eval)
