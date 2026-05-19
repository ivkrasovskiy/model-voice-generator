# CLAUDE.md

Guidance for Claude Code when working in this repo.

## Goal

Mimic Benedict Cumberbatch's **voice identity** — timbre, low-frequency depth, RP accent. **Audiobook narration cadence is a non-goal**; we want his voice in *any* content. Priority: **ECAPA(gen, real_clip) > WER > DNSMOS**.

## Current state (2026-05-19)

**IndexTTS-2 is the current winner.** Zero-shot, no fine-tune: **ECAPA 0.784, WER 0.037, DNSMOS 2.90**. Voice is recognizable, intelligible, no gibberish. Verified by ear on 15 short phrases. Trade-offs vs F5-TTS confirmed: F5-TTS has higher ECAPA on paper (0.827) but its WER 1.4 means many outputs are unusable. IndexTTS-2 is the practical winner.

**F5-TTS is retired** from active development on this project. Seven fine-tune runs all landed at ECAPA 0.82-0.83 with WER 1.4 — architectural ceiling reached. Reference-text leakage cannot be removed without rewriting the architecture (see [docs/literature_notes.md §6](docs/literature_notes.md)).

**Next**: push ECAPA from 0.784 → 0.80+ via two experiments planned in [docs/indextts_experiments.md](docs/indextts_experiments.md):
- **Experiment A** (low risk, half-day): speaker embedding centroid — average ref-audio embeddings from 20 high-quality BC clips
- **Experiment B** (high effort, ~1 week, only if A doesn't reach 0.82): LoRA fine-tune of the IndexTTS-2 GPT submodule

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
| `vendor/index-tts/.venv/` | `cd vendor/index-tts && uv sync --no-dev` | **IndexTTS-2 (current winner)** | Mac torch + repo's own deps |

All scoring goes back through the main `.venv/` via `posthoc_eval.py --score-only` — that path is venv-agnostic (reads `manifest.json` + WAVs).

## IndexTTS-2 install pins (DO NOT TOUCH without verifying smoke test)

These are the exact versions that produced the locked baseline (ECAPA 0.784, WER 0.037 on 15-phrase eval).

| Component | Pin | Where |
|---|---|---|
| IndexTTS-2 repo SHA | `830f6f8f94a51fea23ab1d639027a86200075a4e` | `vendor/index-tts/` (git-ignored). Date: 2026-03-16. |
| HuggingFace weights revision | `740dcaff396282ffb241903d150ac011cd4b1ede` | `vendor/index-tts/checkpoints/` (~8.3 GB) |
| Python | 3.10.20 (via uv) | `vendor/index-tts/.venv/` |
| torch | 2.8.0 | per repo's pyproject.toml |
| transformers | 4.52.1 | per repo's pyproject.toml |

**Rebuild from scratch** (if `vendor/index-tts/` is lost):
```bash
git clone https://github.com/index-tts/index-tts.git vendor/index-tts
cd vendor/index-tts && git checkout 830f6f8f && uv sync --no-dev
.venv/bin/huggingface-cli download IndexTeam/IndexTTS-2 \
    --revision 740dcaff396282ffb241903d150ac011cd4b1ede \
    --local-dir checkpoints
cd .. && .. && vendor/index-tts/.venv/bin/python scripts/indextts_smoke_test.py
```

**Regression protection**:
- `tts_output/eval_indextts_v2/scores.regression_baseline.csv` — locked baseline scores; never overwrite
- `scripts/indextts_smoke_test.py` — runs in ~2 min; asserts `cas_01` and `sher_03` WER ≤ 0.10 and ECAPA ≥ 0.74. Run after any experiment that touches IndexTTS-2 inference paths.
- Future experiment outputs go in **new** dirs (`tts_output/eval_indextts_centroid_*`, `tts_output/eval_indextts_lora_*`) — never reuse `eval_indextts_v2/`.

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
- **Eval sets** (both in `tts_output/cross_eval_50/`, both with leakage protection: clips disjoint from train354 subsets):
  - `eval_short.csv` — 15 short phrases (3-5s, 10 Casanova + 5 Sherlock). Primary metric.
  - `eval_long.csv` — 8 long phrases (9-12s, 5 Casanova + 3 Sherlock). Tests register/breath stability.
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
| IndexTTS-v2 zero-shot (ref_narrator.wav, audiobook) | **0.037** | 0.784 | 2.90 | Locked baseline. No leakage, marginal ECAPA. |
| IndexTTS-v2 centroid-20 (Exp A) | 0.042 | 0.700 | 2.95 | **Regression −0.084 ECAPA** vs baseline. Mean-averaging `spk_cond_emb` across clips loses speaker info: frame-t features carry phoneme content, not just speaker identity, so the average is mush. |
| IndexTTS-v2 interview ref (4:47-5:01 of [cHmkAStZBkc](https://youtu.be/cHmkAStZBkc)) | 0.051 | 0.310 | 2.73 | ECAPA crash — but ECAPA target clips are all audiobook BC. Measures distribution shift, not necessarily identity loss. **Listen-check required.** |
| XTTS-v2 zero-shot (ref_narrator.wav) | 0.108 | 0.689 | 2.58 | No leakage — but lower identity + quality |

8-phrase long eval (new, 9-12s phrases — tests register/breath stability):

| Model | WER ↓ | ECAPA ↑ | DNSMOS OVR ↑ | Note |
|---|---|---|---|---|
| IndexTTS-v2 zero-shot (ref_narrator.wav, audiobook) | **0.013** | **0.846** | 3.48 | Long context lets ECAPA lock onto speaker more confidently than short eval (0.846 vs 0.784) |
| IndexTTS-v2 centroid-20 (Exp A) | 0.017 | 0.738 | 3.21 | Same regression pattern as short — centroid hurts identity |
| IndexTTS-v2 interview ref | 0.028 | 0.322 | 3.13 | Same crash pattern as short. Both eval targets are audiobook BC, so this is a target-mismatch artifact, not necessarily identity loss. |

**The architectural trade-off is confirmed by data**: F5-TTS owns identity (ECAPA 0.83) via mel-conditioning, but ref-text leakage cripples WER. XTTS-v2 owns intelligibility (WER 0.11) via speaker-embedding conditioning, but identity drops to 0.69. **No single off-the-shelf model on M3 Pro currently does both.**

## Key findings

- **Selective CFG** (arXiv 2509.19668): standard CFG for t≤0.08, text-conditioned CFG thereafter. Patches in `.venv`: `cfm.py`, `utils_infer.py`, `api.py`.
- **Reference text leakage**: F5-TTS injects ref text mid-output. Architectural — not fixable by fine-tuning. Confirmed by XTTS-v2 (no leakage → near-zero WER).
- **Fine-tuning F5-TTS is exhausted**: 7 runs, all land at ECAPA ~0.82-0.83. Per-clip ECAPA differences (±0.01) are inside noise. Further F5-TTS fine-tunes are not worth the time on this dataset.
- **F5R-TTS (arXiv 2504.02407) is NOT doable on M3 Pro**: their gain requires re-pretraining F5-TTS from scratch on 7,226h with a modified output head, then 8× A100 40GB for RL phase. Custom checkpoint not released. See [docs/indextts_plan.md](docs/indextts_plan.md) for the evidence and what replaces it.
- **Narration register bias**: Casanova training data pulls model toward narrator-speak. Diverse register data (interviews, non-narration) is still a lever but only if we move to a model that can actually use it.
- **EMA implemented**: `EMATracker` in `finetune_f5.py`. `ema_best.pt` saved alongside `best.pt`. Checkpoints 456 MB.
- **Eval methodology stabilized**: 15-phrase set with real-clip ECAPA targets. Old 6-phrase / centroid-based numbers are deprecated.

## Next steps

**Experiment A (centroid) result: regression.** Mean-averaging `spk_cond_emb` across 20 refs drops ECAPA by 0.08 on short eval, 0.11 on long eval. Hypothesis: W2V-BERT features at frame `t` carry phoneme content as well as speaker info — averaging across temporally-unaligned clips destroys both. The CAMPPlus `style` vector (192-d global) is the only naturally averageable component, but on its own it can't carry enough identity. Centroid lever is exhausted as designed.

**LoRA (Experiment B) is postponed.** New focus is on the *reference-clip* side of the pipeline, where the cheap wins live:

1. **Creak filter / podcast refs** — current `ref_narrator.wav` is 12s of Casanova audiobook, which carries BC's narrator-creak. Test single-clip refs from podcast/interview material. Expected impact: cleaner voice quality without ECAPA loss. ~1h.
2. **Emotion-vector knob** — IndexTTS-2 accepts an 8-d `emo_vector` (calm, sad, …). Push "calm" up to see if it attenuates creak in generation. ~1h.
3. **Allosaurus/Charsiu phoneme analysis pipeline** (independent of TTS) — extract IPA tier from BC's audiobook + interviews + user's recordings for accent comparison. Creak doesn't affect phoneme identity, so the existing data is fine. ~half day.

Dropped from previous plan:
- ~~F5R-TTS RL fine-tune~~ — not feasible on this hardware. See [literature_notes.md §4](docs/literature_notes.md).
- ~~In-training ECAPA hook~~ — further F5-TTS training won't move the number.
- ~~More Sherlock/interview data~~ — only worth it once we have a model architecturally capable of using it without leakage.
- ~~Experiment A (centroid)~~ — completed 2026-05-19, regression. Artifacts in `tts_output/eval_indextts_centroid_{short,long}/`.
- ~~Experiment B (LoRA)~~ — postponed; the cheap reference-side experiments above must come first.

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
