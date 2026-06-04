# Voice cloning Benedict Cumberbatch — project history

Single-file archive of the experiments, data work, and dead-ends that led to the
current working setup. The active state lives in [../CLAUDE.md](../CLAUDE.md);
**this file is reference material, not instructions**.

Hardware throughout: Apple M3 Pro, 18 GB unified memory (MPS where it worked,
CPU otherwise). All training/inference local.

---

## Project goal

Mimic Benedict Cumberbatch's **voice identity** — timbre, low-frequency depth,
RP accent. Audiobook narration cadence is a *non-goal*; we want his voice in
any content. Priority metric: ECAPA-TDNN cosine similarity against real BC
recordings, then WER (intelligibility), then DNSMOS (audio quality).

Constraint: 18 GB unified memory rules out anything that needs a 24 GB CUDA GPU.

---

## Data

### Sources

| Source | Provenance | Used for |
|---|---|---|
| Casanova audiobook (BC narrating) | ~3.5 h raw audio | Primary speaker dataset; ref clips |
| Sherlock audiobook (BC narrating) | ~30 min after narrator-only filtering | Secondary speaker dataset |
| Interview ([youtu.be/cHmkAStZBkc](https://youtu.be/cHmkAStZBkc), 4:31-5:05) | 34 s extracted via `yt-dlp` | Current production reference clip |

### Dataset construction pipeline (now retired)

These scripts produced the F5-TTS training data. Kept in git history; deleted
from working tree because the IndexTTS-2 pipeline is zero-shot and needs none of
this machinery.

1. **`process_audiobook.py`** — raw audiobook WAV → VAD-split segments → ECAPA
   pre-filter against a coarse speaker model. Produced
   `data/cumberbatch_casanova/` (1914 clips).
2. **`cluster_characters.py`** — k-means (k=5) on per-utterance ECAPA embeddings
   to separate narrator from character voices. Yielded
   `data/cumberbatch_sherlock_narrator/` (379 clips, ~30 min).
3. **`audit_dataset.py`** — per-clip ECAPA-against-centroid scoring; produced
   `audit.csv` and `centroid.npy` per dataset. Output drives all later filtering.
4. **`build_clean_dataset.py`** — merges + audit-thresholds → cleaned Casanova
   (1735 clips, ECAPA-sim ≥ 0.65) and a combined cas+sher set (2114 clips).
5. **`sample_matched_subset.py`** — duration-matched stratified subsets
   (`cumberbatch_casanova_train_354/`, `cumberbatch_sherlock_train_354/`) for
   fair fine-tune comparison.
6. **`make_experiment_v1.py`** — built cross-eval holdout: every clip in
   `eval_short.csv` / `eval_long.csv` is excluded from any train354 subset, so
   training-time leakage cannot contaminate evaluation.
7. **`transcribe_missing.py`** — Whisper transcription for clips lacking text.

### Eval sets

Constructed once in `tts_output/cross_eval_50/` and used unchanged ever since:

| File | Phrases | Duration | Purpose |
|---|---|---|---|
| `eval_short.csv` | 15 (10 cas + 5 sher) | 3–5 s each | Primary metric, comparable to all prior runs |
| `eval_long.csv` | 8 (5 cas + 3 sher) | 9–12 s each | Tests register/breath stability and long-context identity |

Each row carries `ref_audio_path` pointing to the real BC recording for that
phrase, enabling per-clip `ECAPA(generated, real_clip)` rather than
ECAPA-vs-centroid (a softer metric that masked the WER/identity trade-off).

Methodology note: the old 6-phrase / centroid-ECAPA results in `posthoc_6phrase/`
and similar dirs are deprecated. Anything else here predates that decision.

---

## Approaches tried (chronological)

### 1. F5-TTS (Oct 2024)

**[Paper — arXiv 2410.06885](https://arxiv.org/abs/2410.06885)** ·
**[Repo — SWivid/F5-TTS](https://github.com/SWivid/F5-TTS)**

Conditional flow-matching TTS. 22-block DiT, predicts mel velocities; Vocos
vocoder. Reference is `(ref_audio, ref_text, target_text)`.

**What we did**:
- Baseline zero-shot evaluation: ECAPA **0.827**, but **WER 1.388** —
  near-unusable. F5-TTS leaks reference *text* into generation
  ([Issue #85](https://github.com/SWivid/F5-TTS/issues/85)).
- Selective CFG patch ([arXiv 2509.19668](https://arxiv.org/html/2509.19668v1)):
  standard CFG for `t ≤ 0.08`, text-conditioned CFG after. Best inference
  config we found; held the same identity at slightly better WER.
- Seven fine-tune runs on the Casanova + Sherlock data, freezing all but the
  last 4 DiT blocks + `norm_out` + `proj_out` (~18 % trainable, the standard
  "speaker adaptation" recipe):

  | Run | Blocks | KD λ | Steps | Best eval | Note |
  |---|---|---|---|---|---|
  | `finetune_casanova` | 4 | 0 | 1000 | ~1.00 | Killed at plateau |
  | `finetune_casanova_8blk` | 8 | 0 | 2000 | 0.843 | Identity collapse (ECAPA 0.71) |
  | `finetune_kd_8blk_lam05` | 8 | 0.5 | 1000 | 0.916 | λ too small (2.5 % of loss) |
  | `finetune_kd_sher_lam10_aborted` | 8 | 10 | 400 | 1.059 | Diverged; λ=10 too aggressive |
  | `finetune_kd_sher_lam5` | 8 | 5 | 300 (ES) | 1.038 @ step 100 | Sherlock data |
  | `finetune_kd_cas_lam5` | 8 | 5 | 600 | 0.918 @ step 450 | Casanova data |
  | `finetune_kd_combined_lam5_v2` | 8 | 5 | 450 (ES) | 0.986 @ step 250 | Combined 2114 clips |

- KD = frozen-teacher knowledge distillation against the pretrained F5-TTS, to
  preserve identity while specializing on BC.
- EMA tracking added (`EMATracker` in `finetune_f5.py`); `ema_best.pt` saved
  alongside `best.pt`. EMA helps later, hurts early
  ([Discussion #57](https://github.com/SWivid/F5-TTS/discussions/57)).

**Why retired**: every fine-tune landed at **ECAPA 0.82–0.83**, regardless of
data or λ. Per-clip ECAPA differences (±0.01) are inside noise. The ceiling
is architectural — `ref_text` injected into the prompt causes mid-output
leakage that no amount of fine-tuning fixes. Confirmed by XTTS-v2 below.

**Useful pieces salvaged**:
- Selective CFG idea (works across architectures in principle).
- Per-clip ECAPA evaluation methodology.
- The cross-eval set with strict train/eval disjoint guarantees.

### 2. XTTS-v2 (Coqui, 2023) — architectural confirmation

**[Coqui-AI/TTS](https://github.com/coqui-ai/TTS)**

Speaker-encoder-based (audio-only conditioning, no `ref_text`). Ran 2026-05-19
in a dedicated `.venv_xtts/` after pinning `torch<2.6` and `transformers<4.44`.

Result: **WER 0.108, ECAPA 0.689, DNSMOS 2.58.**

**This was the smoking gun.** XTTS-v2 has *no* `ref_text` input, and its WER
dropped 13× compared to F5-TTS. So leakage was an F5-TTS architectural
property, not a data or training problem. But XTTS-v2's speaker encoder is
less faithful than F5-TTS's mel conditioning, giving up ECAPA 0.83 → 0.69.

**Conclusion**: no single off-the-shelf model on M3 Pro hit both axes. Either
take F5-TTS's identity and live with garbled words, or take XTTS-v2's clean
words and lose identity. The architecturally-correct answer is to find a model
with audio-only conditioning AND a better speaker representation than XTTS's
encoder — see IndexTTS-2 below.

### 3. OpenVoice V2 — quick check, dropped

**[myshell-ai/OpenVoice](https://github.com/myshell-ai/OpenVoice)**

Two-stage: base TTS + tone-color converter. Tested in `.venv_openvoice/`.
Results worse than XTTS-v2 across the board; no obvious lever for improvement.
Dropped without a full sweep. Useful only as a potential post-processing voice
converter (see "Future work" below).

### 4. F5R-TTS (April 2025) — literature scan, not implemented

**[arXiv 2504.02407](https://arxiv.org/abs/2504.02407)** ·
**[Repo — shaw0fr/F5R-TTS](https://github.com/shaw0fr/F5R-TTS) (2 stars)**

Reformulates F5-TTS's deterministic flow-matching output as Gaussian
distributions, enabling GRPO with WER + SIM as joint rewards. Reported gains:
**+4.6 % SIM, −29.5 % WER** vs vanilla F5-TTS.

**Verdict: not feasible on M3 Pro 18 GB.** Hard blockers:
1. Authors pretrain from scratch on WenetSpeech4TTS (7,226 h) with a modified
   final linear layer. No probabilistic checkpoint is released.
2. RL phase needs 8× A100 40 GB, batch 6,400, 1,100 updates, 100 h of speech.
   Rollouts (multiple gens per step + Whisper + ECAPA in the reward loop) blow
   up memory.
3. Our 18 GB MPS budget accommodates neither.
4. The +4.6 % is *relative to their baseline*; absolute gain on top of our
   already-0.83 ECAPA is unclear.

Replaced in the plan by IndexTTS-2 (different architecture, open weights,
audio-only ref).

### 5. IndexTTS-2 (September 2025) — current winner

**[Paper — arXiv 2502.05512](https://arxiv.org/abs/2502.05512)** ·
**[Repo — index-tts/index-tts](https://github.com/index-tts/index-tts)** ·
**[Weights — IndexTeam/IndexTTS-2](https://huggingface.co/IndexTeam/IndexTTS-2)**

Audio-only reference, autoregressive (GPT-style) over discrete mel tokens, with
explicit duration control. Speaker conditioning is dual:
- `spk_cond_emb` from W2V-BERT v2 (semantic features, time-varying)
- `style` from CAMPPlus (192-d global timbre vector)

These feed a GPT + s2mel decoder + BigVGAN vocoder pipeline.

**Pinned versions** (locked baseline, never bump without smoke test):

| Component | Pin |
|---|---|
| IndexTTS-2 repo SHA | `830f6f8f94a51fea23ab1d639027a86200075a4e` |
| HuggingFace weights revision | `740dcaff396282ffb241903d150ac011cd4b1ede` |
| Python | 3.10.20 |
| torch | 2.8.0 |
| transformers | 4.52.1 |

**Zero-shot baseline (2026-05-19, `ref_narrator.wav`, 12 s audiobook clip)**:

| Eval set | WER | ECAPA | DNSMOS OVR |
|---|---|---|---|
| 15-phrase short | 0.037 | 0.784 | 2.90 |
| 8-phrase long | 0.013 | 0.846 | 3.48 |

First model on this project to clear the architectural trade-off: no leakage,
clean words, AND recognizable BC by ear.

#### Experiment A: speaker centroid (failed)

Hypothesis: averaging speaker-conditioning tensors from N=20 ref clips would
reduce single-clip noise and improve identity. Implemented as a subclass that
mean-averages `spk_cond_emb` (truncated to min T) and `style` across clips,
keeps single-clip `S_ref`/`ref_mel`/`prompt_condition` from a representative.

Result: ECAPA dropped **0.784 → 0.700** on short eval, **0.846 → 0.738** on
long. Worse on both axes.

Root cause: `spk_cond_emb` from W2V-BERT is `(B, T, C)` where each time step
carries phoneme content as well as speaker info. Mean-averaging across
temporally-unaligned clips averages out the phonetic detail and produces
mush. Only `style` (CAMPPlus, time-collapsed 192-d global) is naturally
averageable — and on its own, not enough information to carry identity.

Centroid lever is exhausted as designed. Multi-ref averaging on this
architecture is dead unless one rebuilds the speaker-conditioning path
end-to-end.

#### Experiment B: LoRA fine-tune (designed, not run)

**Postponed indefinitely.** Plan was:
- LoRA adapters on GPT attention layers (q/k/v/o projections), rank 16
- Loss: cross-entropy on the discrete mel tokens, conditioned on
  `(text, spk_cond_emb)`
- Dataset: the 354-clip Casanova subset, 25 held out for eval
- Training: ~500-2000 steps, AdamW, MPS-cache-flush every 25 steps

The plan was sized at ~1 week with uncertain outcome, because IndexTTS-2 ships
**no training script** — the loss formulation has to be reverse-engineered from
inference code. Cheaper reference-side experiments (below) came first.

#### Reference-clip experiments

Once Exp A failed, focus shifted to the *reference* side of the pipeline.
IndexTTS-2 truncates the speaker prompt to the first 15 s
(`_load_and_cut_audio` in `infer_v2.py`), so the choice of those 15 s matters
more than averaging.

**Audiobook → interview ref**. Replaced the 12 s audiobook clip with 14 s of
BC interview (no narrator-creak, conversational register). Results:

| Eval | WER | ECAPA | DNSMOS OVR |
|---|---|---|---|
| Short, audiobook ref | 0.037 | 0.784 | 2.90 |
| Short, interview ref | 0.051 | 0.310 | 2.73 |
| Long, audiobook ref | 0.013 | 0.846 | 3.48 |
| Long, interview ref | 0.028 | 0.322 | 3.13 |

**ECAPA crashed**, but this is a methodology artifact, not necessarily identity
loss: `ref_audio_path` in the eval CSVs points to audiobook clips, so the
metric measures distance from audiobook-BC. Switching the input to interview-BC
deliberately shifts the output distribution. WER stayed clean (<0.06) and
DNSMOS dropped only modestly (likely Opus codec artifacts from YouTube).

Listen-test was the gate. Verdict: user accepted the interview-ref output as
the new working configuration. The audiobook creak that motivated the swap is
absent in generation.

---

## Findings worth remembering

1. **Leakage is architectural.** F5-TTS, F5R-TTS, anything with `ref_text`
   inputs will leak. Audio-only conditioning is the only fix.
2. **ECAPA cosine similarity has a moving target.** Comparing
   ECAPA(generated, reference) is meaningful only if "reference" represents
   the *desired* voice. Audiobook BC and interview BC are far apart in
   ECAPA-space even though they're the same person. Always pair ECAPA with
   listen-tests.
3. **Mean-averaging time-varying speaker embeddings destroys information.**
   Centroid-style speaker representations only work on time-collapsed vectors
   (CAMPPlus, x-vector, ECAPA-emb). W2V-BERT-style frame-varying features
   carry phoneme content and lose meaning when averaged across clips.
4. **Reference clip selection is the cheapest knob.** A clean 14 s of the
   right register beats 20 clips averaged together, and beats fine-tuning.
   IndexTTS-2 caps prompts at 15 s — more audio is wasted.
5. **DNSMOS doesn't track perceptual quality reliably across codecs.** Lossy
   reference audio drops DNSMOS BAK by 0.5+ even when output sounds fine to
   the ear. Use it as a regression flag, not a quality target.

---

## Open questions / future work

In the order I'd try them if returning to this project:

1. **Build an interview-targeted eval set.** Current eval references are all
   audiobook clips; ECAPA against them is unfair to interview-conditioned
   generation. Need 10-20 short BC interview phrases with clean per-clip
   reference audio.
2. **Allosaurus / Charsiu IPA pipeline** for accent comparison (BC vs. user's
   own recordings). Independent of TTS — runs on raw audio + transcripts.
3. **Emotion-vector probe**. IndexTTS-2 accepts an 8-d `emo_vector`. We never
   swept it. Cheap experiment with potentially large effect on register.
4. **LoRA fine-tune of IndexTTS-2 GPT** (the postponed Exp B). Worth running
   only if reference-side knobs are exhausted and a clear identity-vs-eval
   target gap remains.
5. **Move IndexTTS-2 to MPS.** Currently CPU-only (~45-90 s per clip). A
   working MPS path would 5-10× iteration speed for any future experiment.
6. **24 GB cloud GPU fallback.** $0.35/hr (RunPod A5000) buys full-parameter
   fine-tuning of IndexTTS-2 or CosyVoice 2 — both architecturally suited and
   blocked here only by RAM.

---

## References & links

### Papers
- [F5-TTS (arXiv 2410.06885)](https://arxiv.org/abs/2410.06885) — base
  flow-matching architecture
- [Selective CFG (arXiv 2509.19668)](https://arxiv.org/html/2509.19668v1) —
  per-timestep CFG schedule, ported to F5-TTS as a `.venv` patch
- [F5R-TTS (arXiv 2504.02407)](https://arxiv.org/abs/2504.02407) — GRPO RL on
  F5-TTS, blocked by pretraining requirement
- [IndexTTS-2 (arXiv 2502.05512)](https://arxiv.org/abs/2502.05512) — current
  production model
- [F5-TTS cross-lingual (arXiv 2509.14579)](https://arxiv.org/html/2509.14579v4)
  — language-agnostic extension, not relevant here

### F5-TTS issues / discussions referenced
- [#85 reference-text leakage](https://github.com/SWivid/F5-TTS/issues/85)
- [#1155 short-text duration fix](https://github.com/SWivid/F5-TTS/issues/1155)
- [#965 best reference audio](https://github.com/SWivid/F5-TTS/issues/965)
- [Discussion #57 EMA caveat for early fine-tunes](https://github.com/SWivid/F5-TTS/discussions/57)
- [Discussion #769 small-dataset fine-tuning](https://github.com/SWivid/F5-TTS/discussions/769)
- [HF Space discussion #48 vocoder choice](https://huggingface.co/spaces/mrfakename/E2-F5-TTS/discussions/48)

### Repos referenced
- [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS)
- [coqui-ai/TTS](https://github.com/coqui-ai/TTS) (XTTS-v2)
- [myshell-ai/OpenVoice](https://github.com/myshell-ai/OpenVoice)
- [shaw0fr/F5R-TTS](https://github.com/shaw0fr/F5R-TTS)
- [index-tts/index-tts](https://github.com/index-tts/index-tts)
- [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) (potential fallback)
- [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech) (potential fallback)

### Tools used
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — interview clip download
- [ffmpeg](https://ffmpeg.org/) — audio trim / resample
- [Whisper](https://github.com/openai/whisper) — transcription for WER
- [ECAPA-TDNN via SpeechBrain](https://github.com/speechbrain/speechbrain) —
  speaker similarity metric
- [DNSMOS](https://github.com/microsoft/DNS-Challenge) — non-intrusive MOS
- [uv](https://github.com/astral-sh/uv) — Python package management

---

*Accent coach development (per-phoneme scoring, rhythm diagnostics, RP/GenAm norms) is tracked separately in [accent_coach_history.md](accent_coach_history.md).*
