# F5-TTS literature notes

External findings that inform the experiment plan in [../CLAUDE.md](../CLAUDE.md). Last updated: 2026-05-19, after XTTS-v2 comparison and F5R-TTS feasibility review.

## Our experimental finding (recap)

Seeded cfg sweep on F5-TTS baseline (no fine-tune), 6 phrases, ECAPA + centroid-ECAPA + DNSMOS:

| cfg | WER ↓ | ECAPA ↑ | CENT ↑ | DNSMOS ↑ |
|---|---|---|---|---|
| 2.0 (default) | 1.25 | **0.837** | **0.743** | **3.94** |
| 3.0 | 1.13 | 0.808 | 0.722 | 3.90 |
| 4.0 | **0.92** | 0.755 | 0.680 | 3.66 |

Monotonic: higher cfg trades identity for intelligibility across all metrics. Imperative phrase shows it sharpest (WER 2.57 → 1.00 but ECAPA 0.84 → 0.68).

---

## 1. Selective CFG paper — directly studies the cfg/identity tradeoff

**[Pataranutaporn et al., "Selective Classifier-free Guidance for Zero-shot TTS" — arXiv 2509.19668 (Sept 2025)](https://arxiv.org/html/2509.19668v1)**

Paper's reported tradeoff for standard CFG: **higher cfg → better speaker similarity, worse WER**. This is the *opposite direction on SIM* compared to our finding (we measured worse identity at higher cfg).

Possible reasons for the contradiction:
- They likely use a different SIM metric (paper doesn't name it; many use Resemblyzer or WeSpeaker rather than ECAPA-TDNN).
- Non-monotonic curve: their sweet spot may sit at cfg < 2.0. Our test covered 2.0-4.0 only.
- Language- and model-dependent. Paper itself notes the method "improves F5-TTS on English but fails for Mandarin." Implementation specifics matter.

**Their proposed method (selective CFG)**: standard CFG for the first ~9 timesteps (when words form), then switch to text-only-conditioned CFG (i.e. amplify the speaker-vs-text condition rather than the conditioned-vs-unconditioned). Empirically gives most of CFG's WER benefit with less identity loss. Implementable as a ~20-line patch to F5-TTS's CFG function.

Recommended threshold: `t_threshold = 0.08` (between timestep 9-10 for F5-TTS).

## 2. Short-text gibberish — known issue with documented workaround

**[F5-TTS Issue #1155 — "regarding speed adjustment for small length text"](https://github.com/SWivid/F5-TTS/issues/1155)**

F5-TTS's duration formula `duration = ref_audio_len + (ref_audio_len/ref_text_len * gen_text_len / local_speed)` collapses on very short text. Documented fix:

```python
if len(gen_text.encode("utf-8")) < 10:
    local_speed = 0.3
```

Slowing generation gives the model better text-audio alignment on short prompts. This directly explains our `imperative` phrase failure (WER 2.57 at cfg=2.0) — segments like "Stop." and "Don't move." are well under 10 bytes.

## 3. Reference clip — 12s is the ceiling, not a starting point

**[F5-TTS infer README](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/infer/README.md)**

> Use reference audio <12s and leave proper silence space (e.g. 1s) at the end.

Long reference audio is **auto-clipped to ~12s**. Our 12s `ref_narrator.wav` is right at the cap. Going longer at inference won't help on F5-TTS — the clip is silently truncated.

The 30s recommendation seen on some F5-TTS landing pages refers to *generation duration cap* (ref + output combined), not reference length.

## 4. F5R-TTS — investigated 2026-05-19, NOT feasible on M3 Pro

**[F5R-TTS — arXiv 2504.02407 (April 2025)](https://arxiv.org/abs/2504.02407)** · **[official code (shaw0fr/F5R-TTS, 2 stars)](https://github.com/shaw0fr/F5R-TTS)**

Builds on F5-TTS by reformulating deterministic flow-matching outputs as probabilistic Gaussian distributions, which enables **GRPO** with **WER + SIM as joint rewards**.

Reported gains: **+4.6% SIM, -29.5% WER** vs vanilla F5-TTS.

**Verdict: cannot reproduce on this hardware.** Hard blockers from the full paper:

1. **The win is from pretraining, not RL alone**: they modify the F5-TTS final linear layer to predict Gaussian μ(x) and σ(x), then **pretrain from scratch on WenetSpeech4TTS (7,226 h)**. The released code targets that path, not RL-from-vanilla-F5-TTS. No probabilistic checkpoint is released.
2. **RL phase compute**: 8× A100 40 GB, batch size 6,400, 1,100 updates, 100 h of speech data. Rollouts (multiple generations per step + Whisper + ECAPA in the loop) blow up memory.
3. **18 GB MPS budget** cannot accommodate any of this — neither the from-scratch pretraining nor the RL rollouts.
4. The +4.6% relative SIM gain on *their* baseline doesn't even guarantee a meaningful absolute improvement on ours (already at ECAPA 0.83).

Replaced in the experiment plan by **IndexTTS-2** (see §10 below), which addresses the same WER/identity trade-off via a different architecture (audio-only reference, no leakage by construction) and ships with open weights.

## 5. Fine-tune EMA gotcha

## 5. Fine-tune EMA gotcha

**[F5-TTS Finetune practice — Discussion #57](https://github.com/SWivid/F5-TTS/discussions/57)**

> For early-stage finetuned checkpoints, consider disabling the `use_ema` parameter.

This explains why our in-training audio eval produced NaN (noted in CLAUDE.md memory-lessons). The F5TTS API uses EMA by default, but EMA isn't meaningful in the first few hundred steps. When we add EMA tracking, evaluate BOTH ema and non-ema variants.

## 6. Reference text leakage — known unresolved

**[F5-TTS Issue #85 — "Sometimes generates output with a phrase from reference audio"](https://github.com/SWivid/F5-TTS/issues/85)**

A user reports phrases from the reference text appearing in generations. Discussion has no resolution. We're seeing exactly this — many of our generations contain fragments of "everything I wished" (our ref_text) inserted mid-output. Likely the same root cause as the short-text problem (#2 above): when the model can't find a clean alignment, it pads with reference-derived content.

The short-text fix (`local_speed=0.3`) may also reduce leakage on short prompts. Untested.

## 7. Vocoder choice — Vocos vs BigVGAN

**[F5-TTS HF Space — discussion #48](https://huggingface.co/spaces/mrfakename/E2-F5-TTS/discussions/48)**

F5-TTS supports both `vocos` (default, what we use) and `bigvgan` vocoders. Multiple reports of quality differences on the same generation. Easy A/B — vocoder choice is a parameter at F5TTS init.

## 8. Other potentially relevant

- **[F5-TTS Best reference audio suggestion — Issue #965](https://github.com/SWivid/F5-TTS/issues/965)** — community discussion on ref-clip selection: clean, neutral prosody, sentence-aligned.
- **[F5-TTS Discussion #769 — small-dataset fine-tuning on 12GB VRAM](https://github.com/SWivid/F5-TTS/discussions/769)** — anecdotes on fine-tuning with 10-60 min of audio; relevant to our 3.4h dataset.
- **[F5-TTS Cross-Lingual — arXiv 2509.14579](https://arxiv.org/html/2509.14579v4)** — adds language-agnostic capability; unlikely to be useful for our single-speaker single-language task.

## 9. XTTS-v2 (Coqui) — empirical confirmation of the architectural trade-off

Ran 2026-05-19 (15-phrase eval, CPU, in `.venv_xtts/`). Results: **WER 0.108, ECAPA 0.689, DNSMOS OVR 2.58**.

This is the smoking gun for what's wrong with F5-TTS:
- XTTS-v2 uses a **speaker encoder** (audio-only conditioning), so no reference-text leakage.
- Result: WER drops from 1.4 → 0.11 (∼13× improvement), without any fine-tuning.
- But identity drops too: ECAPA 0.83 → 0.69, DNSMOS 3.9 → 2.6. The speaker encoder produces a less faithful imitation of BC than F5-TTS's mel conditioning.

So the leakage problem is **architectural**, not a data or training problem. F5-TTS will never beat its own ceiling without an architecture change. Either an explicitly leakage-free architecture (IndexTTS-2, CosyVoice 2) or RL training that punishes leaked tokens (F5R-TTS, not feasible here — see §4).

Install gotchas to remember:
- XTTS-v2 needs `torch < 2.6` (PyTorch 2.6 changed `weights_only` default).
- XTTS-v2 needs `transformers < 4.44` (newer drops `BeamSearchScorer`).
- MPS broken; CPU works at ~2 s/phrase. Acceptable for 15-phrase eval.
- All install via `uv pip install --python .venv_xtts/bin/python …`.

## 10. IndexTTS-2 — the proposed next move

**[IndexTTS-2 paper — arXiv 2502.05512 (Feb 2025; v2 released Sept 2025)](https://arxiv.org/abs/2502.05512)** · **[code (index-tts/index-tts)](https://github.com/index-tts/index-tts)** · **[weights (IndexTeam/IndexTTS-2)](https://huggingface.co/IndexTeam/IndexTTS-2)**

Why it should beat both F5-TTS and XTTS-v2:
- **Audio-only reference** (`spk_audio_prompt`, no `ref_text`) → no architectural leakage, like XTTS but newer.
- **Autoregressive with explicit duration control** — duration is a controllable input, not emergent. Reduces the failure modes that produce gibberish.
- Paper reports it **beats F5-TTS and CosyVoice 2 on WER, SIM, and emotional fidelity** in zero-shot benchmarks.
- Open weights, MIT-style license.

Risks:
- Docs assume CUDA 12.8. No documented MPS or CPU path. On M3 Pro this means CPU inference (likely working but slow) or rent a cloud GPU.
- Newer repo — install pins are likely fragile. Same playbook as XTTS-v2: dedicated `.venv_indextts/`, pin `torch<2.6` and `transformers<4.44` if needed.

Full implementation plan: **[indextts_plan.md](./indextts_plan.md)** — feasibility gate first, then `scripts/indextts_gen.py`, then score via `posthoc_eval.py --score-only`.

## Prioritized recommendations for next experiments

The F5-TTS exploration is exhausted. Items below are ordered by current ROI.

**Now:**

1. **IndexTTS-2 trial** ([indextts_plan.md](./indextts_plan.md)). Step 0 (feasibility gate) is 30 min; full eval ~3 h wall-clock. Highest expected value of any remaining experiment.

**If IndexTTS-2 fails or underwhelms:**

2. **CosyVoice 2** (Apache-2.0, HF `FunAudioLLM/CosyVoice2-0.5B`). Same plan structure, different model. Slightly older (Dec 2024) but more mature codebase.
3. **Fish Speech v1.5+** (HF `fishaudio/s2-pro`). Most mature pip-install path of the three.

**If on-laptop M3 Pro can't run any of them:**

4. Rent a 24 GB GPU (RunPod A5000 ~$0.35/hr) and run IndexTTS-2 / CosyVoice 2 with proper CUDA.
5. Or **hybrid post-processing**: XTTS-v2 for words → fine-tuned voice-conversion stage for timbre. Architecturally hard, last resort.

**Done (no longer recommended):**

- ~~Knowledge distillation training~~ — λ=5 KD attempted on Casanova-only, Sherlock-only, and combined. All land at ECAPA ~0.82-0.83. F5-TTS fine-tuning is exhausted on this dataset.
- ~~F5R-TTS investigation~~ — checked 2026-05-19, not doable on M3 Pro (see §4).
- ~~`local_speed=0.3` short-text fix~~, ~~cfg < 2.0 sweep~~, ~~BigVGAN A/B~~ — all already absorbed into the selective-CFG baseline; no remaining ROI on F5-TTS inference knobs.

## Sources

- [Selective CFG for Zero-shot TTS (arXiv 2509.19668)](https://arxiv.org/html/2509.19668v1)
- [F5R-TTS — Flow-matching + GRPO (arXiv 2504.02407)](https://arxiv.org/abs/2504.02407)
- [IndexTTS-2 (arXiv 2502.05512)](https://arxiv.org/abs/2502.05512)
- [F5-TTS official infer README](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/infer/README.md)
- [F5-TTS Issue #85 — reference text leakage](https://github.com/SWivid/F5-TTS/issues/85)
- [F5-TTS Issue #1155 — short-text duration fix](https://github.com/SWivid/F5-TTS/issues/1155)
- [F5-TTS Discussion #57 — fine-tune practice (EMA note)](https://github.com/SWivid/F5-TTS/discussions/57)
- [F5-TTS HF Space discussion #48 — gibberish + vocoder workarounds](https://huggingface.co/spaces/mrfakename/E2-F5-TTS/discussions/48)
- [F5-TTS Issue #965 — best reference audio](https://github.com/SWivid/F5-TTS/issues/965)
- [F5-TTS Discussion #769 — small-dataset fine-tuning](https://github.com/SWivid/F5-TTS/discussions/769)
- [F5-TTS original paper (arXiv 2410.06885)](https://arxiv.org/abs/2410.06885)
