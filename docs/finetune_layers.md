# F5-TTS fine-tune: layer freezing, what to unfreeze, and where this sits vs SOTA

This doc explains the architectural choices in [scripts/finetune_f5.py](../scripts/finetune_f5.py) and how the picture changes if you move off-laptop.

## F5-TTS architecture in 60 seconds

F5-TTS is a **conditional flow-matching (CFM)** TTS model. Three pieces:

| Component | Role | Params | Trainable here? |
|---|---|---|---|
| **Text tokenizer + embedding** | char → token IDs → 512-dim text embeddings | tiny | frozen |
| **DiT transformer** (22 blocks, dim=1024, 16 heads) | predicts mel-spectrogram velocities; cross-attends text + audio condition | ~333 M | **partial** (4/22) |
| **Vocos vocoder** (separate model) | mel → 24 kHz waveform | ~14 M | always frozen, not part of CFM |

The CFM model is trained to **predict the velocity field** of a flow that transforms Gaussian noise into a mel-spectrogram conditioned on (ref-audio, ref-text, target-text). At inference, an ODE solver integrates that velocity field for N steps (default 32) to produce the target mel, which Vocos decodes to audio.

**Loss**: L2 between predicted velocity and the (cond → noise) interpolant. Values 0.8–1.2 are normal during fine-tune; below ~0.5 on a small dataset usually means memorization.

## What's frozen right now

In `freeze_model_except_last_n(cfm, train_last_n=4)`:

```
Frozen (full pretrained, no updates):
  ├── text_embed                  ← linguistic features
  ├── audio_cond_embed           ← reference-audio conditioning
  ├── transformer_blocks[0..17]   ← 18 of 22 DiT blocks
  └── (input norm, time conditioning, etc.)

Trainable (~61 M / 333 M = 18%):
  ├── transformer_blocks[18..21]  ← last 4 DiT blocks
  ├── norm_out                    ← final layer norm
  └── proj_out                    ← 1024 → 100-dim mel projection
```

Why this split: the canonical "speaker adaptation" recipe for transformer-based TTS keeps the language/prosody machinery and rewrites only the **acoustic projection** layers — the parts that decide *how* the mel is rendered, not *what* phonemes appear when. The last 4 blocks specifically tend to encode timbre + voicing detail; earlier blocks encode language/duration structure that you don't want to disturb (especially with only 3.4 hours of data).

## What unfreezing each layer would buy you

Ordered by usefulness for **speaker timbre cloning** (the goal here):

| Unfreeze | What it changes | Likely benefit on this dataset | RAM cost (rough) |
|---|---|---|---|
| **norm_out + proj_out** (already on) | Final acoustic projection | Required — without it the model can't shift its output distribution at all | +0 GB (tiny) |
| **last 2–4 blocks** (currently 4) | Timbre, voicing detail, breath patterns | Most of the speaker-similarity gain happens here | +2–3 GB |
| **blocks 14–17** (next 4) | Prosody, rhythm, pace | Useful for matching Cumberbatch's deliberate audiobook cadence | +2–3 GB |
| **blocks 6–13** (middle) | Mid-level phoneme-to-acoustic mapping | Diminishing returns; risk of catastrophic forgetting on a small dataset | +4–6 GB |
| **blocks 0–5** (early) | Low-level text → linguistic features | Almost never worth touching for a single-speaker clone | +4–6 GB |
| **text_embed** | Tokenizer-side embeddings | Only useful if you're adding new tokens / new language | +small |

**On a 24 GB GPU** (your rental target), unfreeze everything except `text_embed` and use Adam instead of Adafactor — that's the F5-TTS authors' default recipe and what their published checkpoints used. Expect a ~2× speaker-similarity gain over the current frozen setup.

**On the M3 Pro 18 GB**, you're already near the ceiling at 4 blocks unfrozen + Adafactor. Pushing to 8 blocks unfrozen on this machine will OOM at the first long clip (>10 s).

## Loss patterns: when to stop, when it's broken

These thresholds are baked into the early-stop + health-check logic in `finetune_f5.py`:

| Signal | Threshold | What it means | Action |
|---|---|---|---|
| Rolling-100 loss flat for 500 steps (improvement < 1 %) | early-stop | Model is at its capacity ceiling for this freeze config | Stop; sample-test the checkpoint |
| Individual step loss > 3× rolling mean | "spike" | Bad clip (clipping noise / mis-transcription) OR LR too hot | If > 5 % of steps, halve LR |
| Grad-norm clipped at 1.0 in > 50 % of recent steps | clipping-saturated | LR is too high; gradients are saturating the clip | Halve LR; restart |
| NaN/Inf loss | hard fail | Numerical blow-up (clip with silence / corrupt audio) | Logged + skipped; check dataset if > 5 NaN |
| Rolling-100 loss drops below 0.5 on this dataset | overfitting | Model is memorizing the 3.4 h training set | Add data, increase dropout, or accept |
| Rolling-100 loss never drops below first-100 mean | not learning | Frozen too aggressively, LR too low, or text/audio mismatch | Unfreeze more / raise LR / inspect data |

**Industry convention from the HuggingFace TTS community + F5-TTS authors' guidance**: for small-dataset speaker adaptation, **3–10 K steps** is the typical sweet spot. Beyond that, the model starts to lose the pretrained prior (catastrophic forgetting) and zero-shot quality on out-of-domain text degrades.

## How this model compares to 2025 SOTA TTS

F5-TTS (Oct 2024) is solid but no longer cutting edge. The landscape as of early 2026:

| Model | License | Released | Strength | Why we picked F5-TTS instead |
|---|---|---|---|---|
| **F5-TTS** | MIT | 2024-10 | Fast inference, MPS-compatible, permissive license, simple to fine-tune | — this is what we use |
| **CosyVoice 2** (Alibaba) | Apache-2.0 | 2024-12 | Currently SOTA for English voice cloning quality; instruction-following | LLM-style autoregressive — much harder to fine-tune on 18 GB |
| **GPT-SoVITS v3** | MIT | 2025-Q1 | Excellent English cloning, great for audiobooks specifically | Heavier; requires CUDA, no good MPS path |
| **Spark-TTS** (SparkAudio) | Apache-2.0 | 2025-01 | Strong cross-lingual zero-shot; small model | Newer/less battle-tested; tooling is rough |
| **Voicebox / Audiobox** (Meta) | research-only | 2024 | Best zero-shot quality in published benchmarks | Not open weights — can't use |
| **Bark** (Suno) | MIT | 2023 | Expressive non-speech (laughs, sighs) | Outdated; Bark Small was deprecated 2024-Q4 |
| **XTTS-v2** (Coqui) | CPML (non-commercial) | 2023 | Multilingual, good for non-English | Unmaintained since 2023; doesn't work on MPS (`Output channels > 65536`) |
| **StyleTTS 2** | MIT | 2023 | Highest naturalness for trained voices | No usable zero-shot path; needs full retrain per voice |

**Why F5-TTS for this project specifically**:
1. **Runs on Apple MPS** at acceptable speed (~70 s per ~6 s clip on M3 Pro) — most others are CUDA-only
2. **Permissive MIT license** — you can ship a fine-tuned voice without legal concerns
3. **Single-speaker fine-tune fits in 18 GB** with the freezing strategy above
4. **Stable inference API** via the `f5_tts.api.F5TTS` wrapper

**When to migrate off F5-TTS**:
- If you rent a 24 GB+ GPU anyway, **CosyVoice 2** or **GPT-SoVITS v3** will give meaningfully better cloning quality on this dataset. CosyVoice 2 is the current best-in-class for English clone from a short reference.
- If you need expressive non-speech (Cumberbatch's pauses, breath, laughs), **Bark** does this better but at lower base quality.
- If you go multilingual, **Spark-TTS** has the best zero-shot cross-lingual transfer.

For *this* project — single-voice audiobook narrator clone, on-laptop, with permissive license — F5-TTS is the right tool. Migrating to CosyVoice 2 would mean +1 week of pipeline work (different dataset format, different inference path) for an estimated 10–20 % improvement in subjective speaker similarity. Worth it only after you've maxed out F5-TTS with a 24 GB GPU fine-tune.

## Practical playbook

**If results from this fine-tune are good enough**: keep using F5-TTS, this script. No upgrade needed.

**If they're underwhelming** (mostly likely outcome given 18 GB + 4 blocks unfrozen):
1. Rent a 24 GB GPU (RunPod A5000 ≈ $0.35/hr; ~$2 for an 8 K-step run)
2. Set `--train-last-n 22` and switch Adafactor → Adam in the same script
3. Run 8 K steps with eval every 1 K
4. Pick the checkpoint with the best Resemblyzer score (not necessarily the last one)

**If F5-TTS still underwhelms after that**: migrate to CosyVoice 2.
