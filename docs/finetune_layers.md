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

F5-TTS (Oct 2024) is solid but no longer cutting edge. The landscape as of May 2026:

| Model | License | Released | Strength | Status here |
|---|---|---|---|---|
| **F5-TTS** | MIT | 2024-10 | MPS-compatible, simple to fine-tune | **Current best identity (ECAPA 0.83), capped WER 1.4 by ref-text leakage** |
| **XTTS-v2** (Coqui) | CPML (non-commercial) | 2023 | Multilingual, speaker-encoder conditioning | **Tested: WER 0.11 but ECAPA 0.69. CPU works in `.venv_xtts/` after pinning `torch<2.6`, `transformers<4.44`** |
| **IndexTTS-2** | open | 2025-09 | Audio-only ref, AR with duration control; reportedly beats F5 and CosyVoice 2 | **Proposed next trial — see [indextts_plan.md](./indextts_plan.md)** |
| **CosyVoice 2** (Alibaba) | Apache-2.0 | 2024-12 | Supervised semantic tokens; strong English clone | Fallback if IndexTTS-2 fails |
| **Fish Speech v1.5+** | open | 2024 | Most mature pip-install path | Fallback B |
| **GPT-SoVITS v3** | MIT | 2025-Q1 | Excellent English cloning | CUDA-only, no good MPS path |
| **Spark-TTS** (SparkAudio) | Apache-2.0 | 2025-01 | Cross-lingual zero-shot | Less battle-tested; tooling rough |
| **Voicebox / Audiobox** (Meta) | research-only | 2024 | Best published benchmarks | Not open weights |
| **Bark** (Suno) | MIT | 2023 | Expressive non-speech | Outdated |
| **StyleTTS 2** | MIT | 2023 | Highest naturalness for trained voices | No zero-shot path; full retrain per voice |
| **F5R-TTS** (arXiv 2504.02407) | MIT | 2025-04 | RL fine-tune of F5-TTS with WER+SIM reward | **Not feasible**: requires re-pretrain from scratch + 8× A100 — see [literature_notes.md §4](./literature_notes.md) |

**Why F5-TTS still anchors this project**:
1. Runs on Apple MPS at acceptable speed (~70 s per ~6 s clip on M3 Pro)
2. Permissive MIT license
3. Single-speaker fine-tune fits in 18 GB with the freezing strategy above
4. Stable inference API via `f5_tts.api.F5TTS`

**Why F5-TTS is no longer enough**:
- Reference-text leakage caps WER at ~1.4 architecturally. Fine-tuning won't move it.
- 7 fine-tune runs, all land at ECAPA 0.82-0.83. Diminishing returns confirmed empirically.
- XTTS-v2 proved the trade-off is architectural: no-ref-text → no leakage → near-zero WER.

**When to migrate off F5-TTS** (current decision tree):
1. **First**: Try IndexTTS-2 on M3 Pro CPU per [indextts_plan.md](./indextts_plan.md). It's the only model that *might* beat F5 on identity AND XTTS on WER on this hardware.
2. **If that fails on M3 Pro**: rent a 24 GB GPU (RunPod A5000 ≈ $0.35/hr) for IndexTTS-2 / CosyVoice 2 inference. Cheap enough to be a one-day experiment.
3. **If no modern model can match F5-TTS on identity**: keep F5-TTS for production, accept the WER trade-off, and either (a) post-process to splice in XTTS-v2 outputs for high-stakes phrases, or (b) accept the leakage and ship.

## Practical playbook

**Current state**: F5-TTS fine-tuning is exhausted. Don't run more F5 fine-tune experiments without a new lever (e.g. a different architecture or a 24 GB GPU + Adam + all 22 blocks unfrozen).

**To move forward**: read and execute [indextts_plan.md](./indextts_plan.md).

**Legacy fine-tune playbook (kept for reference, low priority)**:
1. Rent a 24 GB GPU
2. Set `--train-last-n 22`, switch Adafactor → Adam
3. Run 8 K steps with eval every 1 K
4. Pick checkpoint with best per-clip ECAPA (not Resemblyzer; that metric is deprecated)
5. If F5-TTS *still* underwhelms (likely, given the architectural ceiling), move to IndexTTS-2 / CosyVoice 2.
