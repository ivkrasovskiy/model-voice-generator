# F5-TTS literature notes

External findings that inform the experiment plan in [../CLAUDE.md](../CLAUDE.md). Last updated: 2026-05-17, after the cfg-sweep on baseline.

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

## 4. F5R-TTS — the architectural alternative

**[F5R-TTS — arXiv 2504.02407 (April 2025)](https://arxiv.org/abs/2504.02407)**

Builds on F5-TTS by reformulating deterministic flow-matching outputs as probabilistic Gaussian distributions, which enables **GRPO (Group Relative Policy Optimization)** with **WER + SIM as joint rewards**.

Reported gains: **+4.6% SIM, -29.5% WER** vs vanilla F5-TTS.

This is the most principled solution to the cfg tradeoff — instead of tuning at inference time, it bakes the WER/SIM optimization into training via RL. Tradeoff: significant architectural change, weight/code availability unconfirmed from the abstract.

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

## Prioritized recommendations for next experiments

Cheap to test, high information:

1. **`local_speed=0.3` on phrases with any segment < 10 bytes.** Likely fixes our `imperative` WER without identity cost. ~10 min, single-line code change.
2. **Sweep cfg ∈ {1.0, 1.5, 2.0}.** Probe the below-default range where the Selective CFG paper claims identity improves. ~25 min using the existing harness.
3. **BigVGAN A/B vs Vocos.** ~15 min. May recover DNSMOS lost at high cfg.

Medium effort:

4. **Selective CFG patch** (item 1 above). ~1 hour to implement + 30 min to test.

Big effort:

5. **Knowledge distillation training** (in CLAUDE.md plan). Identity-preserving fine-tune approach.
6. **F5R-TTS investigation** — confirm weight/code availability; if open, this could be a step-change.

## Sources

- [Selective CFG for Zero-shot TTS (arXiv 2509.19668)](https://arxiv.org/html/2509.19668v1)
- [F5R-TTS — Flow-matching + GRPO (arXiv 2504.02407)](https://arxiv.org/abs/2504.02407)
- [F5-TTS official infer README](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/infer/README.md)
- [F5-TTS Issue #85 — reference text leakage](https://github.com/SWivid/F5-TTS/issues/85)
- [F5-TTS Issue #1155 — short-text duration fix](https://github.com/SWivid/F5-TTS/issues/1155)
- [F5-TTS Discussion #57 — fine-tune practice (EMA note)](https://github.com/SWivid/F5-TTS/discussions/57)
- [F5-TTS HF Space discussion #48 — gibberish + vocoder workarounds](https://huggingface.co/spaces/mrfakename/E2-F5-TTS/discussions/48)
- [F5-TTS Issue #965 — best reference audio](https://github.com/SWivid/F5-TTS/issues/965)
- [F5-TTS Discussion #769 — small-dataset fine-tuning](https://github.com/SWivid/F5-TTS/discussions/769)
- [F5-TTS original paper (arXiv 2410.06885)](https://arxiv.org/abs/2410.06885)
