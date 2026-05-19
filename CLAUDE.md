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
  posthoc_eval.py         # 2-phase eval; uses lib throughout. --score-only consumes
                          # a manifest.json from any *_gen.py script.
  finetune_f5.py          # Training; EMA, KD, early-stop on eval-loss
  xtts_gen.py             # XTTS-v2 zero-shot gen (.venv_xtts/); writes manifest.json
  openvoice_gen.py        # OpenVoice V2 zero-shot gen (.venv_openvoice/); writes manifest.json
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

## Environments (multiple venvs, one per model family)

| venv | Created by | For | Reason |
|---|---|---|---|
| `.venv/` | `uv sync` (main) | F5-TTS, fine-tune, posthoc eval | Project's primary deps |
| `.venv_xtts/` | `uv venv` + `uv pip install TTS "torch<2.6" "transformers<4.44"` | XTTS-v2 generation | Coqui TTS needs old torch + transformers |
| `.venv_openvoice/` | (existing) | OpenVoice V2 generation | Conflicts with MeloTTS pins |
| `.venv_indextts/` | per [indextts_plan.md](docs/indextts_plan.md) | IndexTTS-2 (planned) | CUDA-targeted repo; isolate |

All scoring goes back through the main `.venv/` via `posthoc_eval.py --score-only` — that path is venv-agnostic (reads `manifest.json` + WAVs).

## Active scripts — key flags

| Script | Key flags |
|---|---|
| `posthoc_eval.py` | `--selective-cfg --t-threshold 0.08` (best inference), `--phrases-csv` (per-clip ECAPA), `--listen-checkpoints`, `--score-only` (skip gen, score existing manifest.json) |
| `finetune_f5.py` | `--kd-lambda 5` (KD), `--ema-decay 0.9999`, `--early-stop-window 200`, `--mps-cache-every 25` |
| `xtts_gen.py` | Run with `.venv_xtts/bin/python`; writes manifest compatible with `posthoc_eval.py --score-only` |
| `openvoice_gen.py` | Run with `.venv_openvoice/bin/python`; same manifest contract |

## Datasets

| Directory | Source | Clips | Duration | Note |
|---|---|---|---|---|
| `data/cumberbatch_casanova/` | Raw Casanova | 1914 | 3.4h | Original, has character voices |
| `data/cumberbatch_casanova_clean/` | Casanova filtered | 1735 | 3.2h | ECAPA sim ≥ 0.65 |
| `data/cumberbatch_sherlock_narrator/` | Sherlock narrator cluster | 379 | 30 min | k=5 cluster, sim 0.83 to Casanova |
| `data/cumberbatch_casanova_train_354/` | Casanova subset | 354 | 37.8 min | Duration-matched to Sherlock |
| `data/cumberbatch_sherlock_train_354/` | Sherlock subset | 354 | 28 min | Val set (25 clips) excluded |
| `data/cumberbatch_combined_cas_sher/` | Casanova_clean + Sherlock_narrator | 2114 | 3.5h | Used for `finetune_kd_combined_lam5_v2` |

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
| `runs/finetune_kd_combined_lam5_v2/` | 8 | 5 | 450 (ES) | **0.986 @ step 250** | Combined data (2114 clips); ECAPA flat vs baseline |

Early stop now works correctly: stops when global-best eval_loss not beaten in `--early-stop-window` steps.

## Eval results

**Best inference config**: `--selective-cfg --t-threshold 0.08 --cfg-strength 2.0`

Per-clip ECAPA standard: `ECAPA(generated, real_clip)` per phrase, using `--phrases-csv eval_short.csv` with `ref_audio_path` per row. The old centroid-ECAPA results are deprecated — they masked the WER/identity trade-off.

15-phrase eval (current, real-clip ECAPA):

| Model | WER ↓ | ECAPA ↑ | DNSMOS OVR ↑ | Note |
|---|---|---|---|---|
| F5-TTS baseline (sel-CFG) | 1.388 | **0.827** | 3.86 | Ref-text leakage caps WER |
| F5-TTS `finetune_kd_combined_lam5_v2/best.pt` | 1.404 | 0.828 | 3.91 | No identity gain over baseline |
| F5-TTS `finetune_kd_combined_lam5_v2/ema_best.pt` | 1.369 | 0.826 | 3.87 | Slightly better WER, same identity |
| XTTS-v2 zero-shot (ref_narrator.wav) | **0.108** | 0.689 | 2.58 | No leakage — but lower identity + quality |

**The architectural trade-off is confirmed by data**: F5-TTS owns identity (ECAPA 0.83) via mel-conditioning, but ref-text leakage cripples WER. XTTS-v2 owns intelligibility (WER 0.11) via speaker-embedding conditioning, but identity drops to 0.69. **No single off-the-shelf model on M3 Pro currently does both.**

## Key findings

- **Selective CFG** (arXiv 2509.19668): standard CFG for t≤0.08, text-conditioned CFG thereafter. Patches in `.venv`: `cfm.py`, `utils_infer.py`, `api.py`.
- **Reference text leakage**: F5-TTS injects ref text mid-output. Architectural — not fixable by fine-tuning. Confirmed by XTTS-v2 (no leakage → near-zero WER).
- **Fine-tuning F5-TTS is exhausted**: 7 runs, all land at ECAPA ~0.82-0.83. Per-clip ECAPA differences (±0.01) are inside noise. Further F5-TTS fine-tunes are not worth the time on this dataset.
- **F5R-TTS (arXiv 2504.02407) is NOT doable on M3 Pro**: their gain requires re-pretraining F5-TTS from scratch on 7,226h with a modified output head, then 8× A100 40GB for RL phase. Custom checkpoint not released. See [docs/indextts_plan.md](docs/indextts_plan.md) for the evidence and what replaces it.
- **Narration register bias**: Casanova training data pulls model toward narrator-speak. Diverse register data (interviews, non-narration) is still a lever but only if we move to a model that can actually use it.
- **EMA implemented**: `EMATracker` in `finetune_f5.py`. `ema_best.pt` saved alongside `best.pt`. Checkpoints 456 MB.
- **Eval methodology stabilized**: 15-phrase set with real-clip ECAPA targets. Old 6-phrase / centroid-based numbers are deprecated.

## Next steps (priority order)

1. **IndexTTS-2 (Sept 2025) trial** — see [docs/indextts_plan.md](docs/indextts_plan.md). Self-contained plan with feasibility gate, install gotchas, and fallback ladder (CosyVoice 2 → Fish Speech v1.5). This is the only path forward that hasn't been exhausted.
2. **If IndexTTS-2 fails on M3 Pro**: rent a 24 GB cloud GPU and run IndexTTS-2 or CosyVoice 2 properly. See [docs/finetune_layers.md](docs/finetune_layers.md) §"When to migrate off F5-TTS" for cost.
3. **If all modern models stall**: hybrid post-processing — XTTS-v2 for content + voice-conversion step using a fine-tuned BC encoder. Architecturally hard, last resort.

Dropped from previous plan:
- ~~F5R-TTS RL fine-tune~~ — not feasible on this hardware. See [literature_notes.md §4](docs/literature_notes.md).
- ~~In-training ECAPA hook~~ — further F5-TTS training won't move the number.
- ~~More Sherlock/interview data~~ — only worth it once we have a model architecturally capable of using it without leakage.

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
