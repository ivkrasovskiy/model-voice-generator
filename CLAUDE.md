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
| `scripts/posthoc_eval.py` | 2-phase eval: F5-TTS gen → Whisper WER + ECAPA + CENT (centroid) + DNSMOS. **Resume-safe** (incremental manifest); flags: `--cfg-strength --nfe-step --seed --first-n --centroid-dir --baseline-only --speed-fix --selective-cfg --t-threshold` |
| `scripts/finetune_f5.py` | F5-TTS partial fine-tune — 8 DiT blocks, Adafactor, held-out eval loss, **auto-saves `best.pt`**, TensorBoard; **`--kd-lambda`** adds KD loss (frozen teacher velocity anchor) |
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

Full inference sweep compared in [`tts_output/all_configs_report.md`](tts_output/all_configs_report.md). **⚠️ 6 phrases is too few for statistical significance — expanding planned.**

| Config | WER ↓ | **ECAPA** ↑ | **CENT** ↑ | DNSMOS ↑ |
|---|---|---|---|---|
| F5-TTS NFE=64 + sel-CFG | 1.32 | **0.849** | **0.752** | 3.91 |
| **F5-TTS sel-CFG t=0.08, cfg=2.0** | 1.10 | 0.845 | 0.741 | **3.90** |
| F5-TTS baseline cfg=2.0 | 1.25 | 0.837 | 0.743 | 3.94 |
| F5-TTS KD λ=0.5 + sel-CFG | 1.17 | 0.828 | 0.735 | 3.95 |
| OpenVoice V2 | **0.02** | 0.400 | 0.392 | 3.28 |

Findings:
- **F5-TTS vs OpenVoice exposes architectural trade-off**: F5-TTS conditions on full mel (high identity, lots of gibberish from ref-text leakage). OpenVoice distils to a 256-d speaker embedding (perfect text, generic voice). Neither solves the goal alone — we need a hybrid that keeps mel detail with a clean text path.
- **Gibberish is real, not Whisper noise** — user confirmed audible nonsense in F5-TTS output (e.g. "everything i wished" injected mid-phrase). WER 1.10 means ~1 in 10 words is wrong; in practice the broken segments are concentrated in short prompts.
- **NFE=64 is a marginal identity win, big intelligibility loss** — more denoising steps = more time for the model to "settle into" reference text leakage. `imperative` WER goes 2.71 → 3.29.
- **Selective CFG (arXiv 2509.19668)**: still the best pure-F5-TTS config. `stella_short` WER 1.83 → 0.42.
- **KD λ=0.5 fine-tune is essentially identical to no fine-tune** (effective KD contribution was ~2.5% of loss). Need λ ≥ 5–10 to actually test.
- **Below-2.0 cfg disproved**: cfg=1.5 (ECAPA 0.832) and cfg=1.0 (0.822) both worse than cfg=2.0 (0.837).
- **speed-fix was a dead end**: built-in to F5-TTS already, and leakage is the real cause of `imperative` failure anyway.
- **CENT (centroid) is more honest than single-ref ECAPA**, particularly important for OpenVoice (the embedding-based approach scores ~0.40 on both, confirming it really is a different speaker, not a metric artefact).
- **Prior fine-tunes hurt identity** (8blk@1000 ECAPA 0.71 vs baseline 0.84). KD guard implemented but not properly tested yet.

## Next experiments

See [docs/literature_notes.md](docs/literature_notes.md) for the literature review behind this ordering.

**Inference fixes — DONE / exhausted**:
- ~~`local_speed=0.3` for short phrases~~ — **done, no effect**. `imperative` WER stuck at 2.5 regardless; root cause is ref-text leakage (Issue #85), not duration formula.
- ~~CFG sweep below 2.0~~ — **done, disproved**. cfg=1.5 and cfg=1.0 are both worse than cfg=2.0 on ECAPA/CENT.
- **BigVGAN vocoder A/B** — skipped; BigVGAN is tied to `F5TTS_Base` (older weights), not v1_Base. Not a clean A/B.

**Inference fixes — DONE**:
- ~~Selective CFG~~ — **done, best config**. Patches threaded through cfm.py → utils_infer.py → api.py. Use `--selective-cfg --t-threshold 0.08` on all future eval runs.
- ~~CFG sweep below 2.0~~ — **done, disproved**.
- ~~speed-fix~~ — **done, no effect** (leakage, not duration).
- BigVGAN A/B — skipped (tied to older model weights, not a clean A/B).

**Training runs to date (updated)**:

| Run dir | Blocks | Steps | Best eval loss | ECAPA (sel-CFG) | Note |
|---|---|---|---|---|---|
| `runs/finetune_casanova/` | 4 | 1000 | ~1.00 train | n/a | Killed at plateau |
| `runs/finetune_casanova_8blk/` | 8 | 2000 | 0.843 @ step 1000 | 0.71 (std cfg, 1 phrase) | Identity collapse |
| `runs/finetune_kd_8blk_lam05/` | 8 | 1000 (ES) | **0.916 @ step 800** | **0.828** | KD λ=0.5 prevents collapse but drift persists |

KD finding: λ=0.5 prevented catastrophic collapse (0.71 → 0.828) but still below zero-shot (0.845). Root cause: training data is narration-register Casanova — fine-tuning shifts the model toward narration speech regardless of KD strength. Report: [`tts_output/kd_vs_baseline_report.md`](tts_output/kd_vs_baseline_report.md).

**Next experiments — priority order updated after OpenVoice finding**:

1. **Expand eval set to 30+ phrases.** Current 6-phrase set has std-dev ≥ 0.01 ECAPA, so anything we've called a "win" of < 0.02 is noise. Use Harvard Sentences or LibriSpeech test-clean transcripts. ~30 min compute per config.
2. **Try XTTS-v2** — hybrid mel + speaker-embedding architecture, the third architectural choice we haven't tested. CLAUDE.md history notes XTTS was abandoned; reopen and check if it sits between F5-TTS and OpenVoice on the trade-off.
3. **Try F5R-TTS if weights available** ([arXiv 2504.02407](https://arxiv.org/abs/2504.02407)) — RL successor with WER+SIM as joint reward. Reports +4.6% SIM / -29.5% WER. Specifically designed to fix the gibberish-vs-identity trade-off we see in F5-TTS.
4. **Data work**: ECAPA-filter Sherlock Holmes audiobook to extract narrator-only segments; add to training set. Skip Kafka & Scales of Justice (poor quality). Then refit KD with λ ≥ 10. ~2h work.
5. **Interviews + diarisation** — yt-dlp → pyannote → ECAPA-cluster filter → keep Cumberbatch only. Highest-leverage data work but largest effort.

**Killed / done**:
- Selective CFG patch — done, modest win, implementation in `cfm.py`/`utils_infer.py`/`api.py`/`posthoc_eval.py`
- BigVGAN A/B — tied to older base model, not a clean comparison
- cfg sweep below 2.0 — disproved
- speed-fix — already in F5-TTS
- KD at λ=0.5 — effectively negligible, needs much higher λ to test properly

**Current best config**: `--selective-cfg --t-threshold 0.08 --cfg-strength 2.0` (and NFE=64 if compute allows, but only if expanded eval confirms the small identity gain is real).

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
