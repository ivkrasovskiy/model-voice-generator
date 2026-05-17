# CLAUDE.md

Guidance for Claude Code when working in this repo.

## Goal

Mimic Benedict Cumberbatch's **voice identity** — timbre, low-frequency depth, RP accent — using the Casanova audiobook as training data. **Audiobook narration cadence is a non-goal**; we want his voice in *any* content, not his reading style. Priority order: **ECAPA speaker-similarity > WER (intelligibility) > DNSMOS (perceived quality)**.

## Hardware

- macOS, **Apple M3 Pro, 18 GB unified memory**
- Use **MPS** (`torch.device("mps")`) for inference and training
- F5-TTS fine-tune with 8 trainable DiT blocks + Adafactor fits on 18 GB (~3 GB peak)

## Environment

```bash
export PYTHONHASHSEED=random   # ALWAYS set before launching any F5-TTS script
uv sync                        # install/update deps from pyproject.toml
.venv/bin/python scripts/<script>
```

`.env` contains `PYTHONHASHSEED=random`. Scripts import `_dotenv_init` at the top.

## Active scripts

| Script | Purpose |
|---|---|
| `scripts/finetune_f5.py` | F5-TTS partial fine-tune — 8 DiT blocks, Adafactor, held-out eval loss, TensorBoard |
| `scripts/posthoc_eval.py` | 2-phase eval: generate WAVs (F5-TTS only), then score with Whisper WER + ECAPA + DNSMOS |
| `scripts/_dotenv_init.py` | PYTHONHASHSEED fix — import at top of every F5-TTS script |

Retired (ran, results in dataset/runs dirs): `segment_vad.py`, `transcribe_v3.py`, `build_manifest.py`, `prepare_f5_dataset.py`, `compare_voices.py`, `analyze_*.py`

## Dataset

- **Training data**: `data/cumberbatch_casanova/` — 1914 clips, avg 6.4s, 3.4h total
  - Re-segmented on silence boundaries (silero-VAD), NOT the old 30s fixed chunks
  - Re-transcribed with Whisper large-v3 (proper nouns correct)
  - Casanova is narrator-uniform — no character voice filtering needed
- **Reference clip**: `tts_output/ref_narrator.wav` — 12s from Casanova chunk 067

## Training runs (2026-05-17)

| Run dir | Blocks | Steps | Best eval loss | Best checkpoint |
|---|---|---|---|---|
| `runs/finetune_casanova/` | 4 | 1000 (killed at plateau) | ~1.00 (train) | `step_000500.pt` |
| `runs/finetune_casanova_8blk/` | 8 | 2000 (early-stop) | **0.843 @ step 1000** | `step_001000.pt` |

## Eval results — Stella phrase A/B (same prompt, all configs)

Listen: `tts_output/posthoc_stella/`, `tts_output/posthoc_8blk/`, `tts_output/posthoc_8blk_1500/`

| Config | WER ↓ | ECAPA ↑ | DNSMOS OVR ↑ |
|---|---|---|---|
| Baseline (zero-shot) | 1.333 | **0.837** | 3.84 |
| 4-block step_500 | 1.833 | 0.778 | **4.08** |
| **8-block step_1000** | **0.833** | 0.709 | 4.01 |
| 8-block step_1500 | 1.083 | 0.780 | 4.06 |

**Under the timbre-first goal, baseline zero-shot (ECAPA 0.84) is currently best**; fine-tuning has actively hurt speaker identity (8blk@1000 dropped to 0.71). Fine-tune wins WER + DNSMOS but loses on the metric that matters most — training pulls the model toward the *narration register* and away from raw voice identity. Also: eval/train gap turned positive at step 1500 (overfitting); true best eval (0.793) was @ step ~1600 but uncaptured.

## Next experiments

Run in order; each with its own `--run-name`. Stop early if goal is met. **Primary metric: ECAPA. Floor: WER ≤ 1.5.** Baseline ECAPA (0.84) is the bar to beat.

### Pre-flight, no training (~45 min total)

Do these first — fine-tuning may turn out unnecessary if we can recover ECAPA on the baseline with better inference. Single-prompt eval also makes current rankings statistically unreliable.

**0a. Expand the eval prompt set to 6 phrases** (replace the existing list in [scripts/posthoc_eval.py](scripts/posthoc_eval.py) and [scripts/finetune_f5.py](scripts/finetune_f5.py) `EVAL_PHRASES_SCORE`). Each phrase ≤ 80 chars (F5-TTS single-batch limit). Cover varied domains and phonetic patterns relevant to Cumberbatch's voice:

| # | Phrase | Why this one |
|---|---|---|
| 1 | `Please call Stella; ask her to bring these things from the store.` | Keep — backward-compat with existing scores; phonetically balanced (Speech Accent Archive control). |
| 2 | `When the sunlight strikes raindrops in the air, they act as a prism.` | Rainbow Passage opener — classic speech-pathology eval; dense fricatives + diphthongs. |
| 3 | `I haven't seen him since the last meeting, but I'll ask around tomorrow.` | Casual conversational register — tests voice *outside* narration mode. |
| 4 | `The algorithm processes each frame independently before merging results.` | Modern/technical domain — far OOD from a 1700s Casanova audiobook; stresses generalization. |
| 5 | `Whose woods these are I think I know; his house is in the village.` | Deep open vowels (oh/ou/aw) — exercises chest resonance, where Cumberbatch's signature low frequencies live. |
| 6 | `Stop. Don't move. There's something behind you.` | Short imperatives — dynamic range without narrator cadence; tests crisp RP plosives (t, d, p, b). |

After updating, re-score baseline / 8blk@1000 / 8blk@1500 against all 6 phrases. **Report per-phrase scores** (not just mean) — variance across phrases tells us which phonetic features the fine-tune is preserving vs breaking. N=6 also drops the std from undefined (N=1) to interpretable.

**0b. Inference sweep on baseline (zero-shot) first, then on 8blk@1000.** Pure inference, no training. May recover ECAPA without any retrain:
- `cfg_strength` ∈ {1.0, 2.0 (default), 3.0}
- `nfe_step` ∈ {32 (default), 64}
- Longer reference clip: cut a 20–30s ref from a different Casanova chunk (current ref is 12s from chunk 067) and A/B
- Optional: try `BigVGAN` vocoder vs `Vocos` (F5-TTS supports both)

Score the full 3×2×2 = 12-cell grid (or a smart subset) on the 6 new phrases. **If baseline+tuned-inference beats fine-tune on ECAPA *and* clears the WER ≤ 1.5 floor, stop — fine-tuning is the wrong direction.**

### Training (only if pre-flight insufficient)

Reordered so the identity-preserving approach goes first.

1. **Knowledge distillation** — frozen F5 teacher, add `λ × L2(student_vel, teacher_vel)` to the CFM loss. Directly counters ECAPA drift by anchoring student to the pretrained prior. Start `λ = 0.5`, sweep {0.2, 0.5, 1.0}. **First validate teacher+student forward fits in 18 GB** before committing to full run. ~50 min if it fits.
2. **EMA + save-every 200** — capture true eval minimum (was @ step ~1600, missed by save-every=500). Also unblocks in-training audio eval. ~30 min. Combine with #1.
3. **DNSMOS top-75% filter** (not 50% — too aggressive on 3.4 h). Removes noisy clips behind gradient spikes at steps 1520/1540. ~20 min score + 30 min train.
4. **LR cool-down for last 500 steps** (5e-6 vs 1e-5) — cheap fix for late spikes (loss 3.4 @ 1925, 2.6 @ 1935). Can be combined with any of the above. ~30 min.

## Memory constraints — hard-won lessons

1. **Kill stale Python processes BEFORE every new launch** — mandatory pre-launch:
   ```bash
   pkill -f "scripts/finetune_f5\.py"; pkill -f "scripts/posthoc_eval\.py"
   ps -A | grep -E "python.*(finetune|posthoc|regen|whisper)" | grep -v grep  # must be empty
   ```
   Three stale F5-TTS procs ≈ 21 GB → OOM. Even "successfully completed" runs leave orphans.

2. **PYTHONHASHSEED must be set** before any F5-TTS launch. Empty string crashes subprocesses.

3. **F5-TTS on MPS produces NaN audio** (~30-50% per batch). Long phrases auto-chunk into multiple batches — NaN probability compounds. Fix: `split_to_short_segments()` in `posthoc_eval.py` pre-splits to ≤50 chars, retries each segment independently, concatenates. Call `torch.mps.empty_cache()` after each generation to prevent fragmentation.

4. **Load Whisper/ECAPA/DNSMOS AFTER F5-TTS generation is complete** (2-phase approach in posthoc_eval.py). Loading all models simultaneously pushes MPS into instability.

5. **In-training audio eval requires EMA weights** — direct inference on training weights produces NaN. The F5TTS API uses EMA internally (that's why posthoc works). Fix planned in next experiment.

6. **Whisper large-v3 needs ~3 GB RAM**. Don't run concurrently with active training.

## Conventions

- Code runs **locally** on the user's Mac (M3 Pro).
- Do not commit `.env`, `dataset/`, `raw_cumberbatch_data/`, `tts_output/`, `.venv/` — gitignored.
- Always use `--run-name` to separate experiments; never overwrite existing checkpoints.
- `tensorboard --logdir runs/ --port 6006` shows all runs together.
