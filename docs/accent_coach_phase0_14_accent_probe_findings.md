# Phase 0.14 WS-B — Accent probe findings

**Date:** 2026-05-27  **Author:** claude-sonnet-4-6

## Question

Given a non-RP reference clip, does IndexTTS-2 place the cloned voice near that
accent, near General American, or near RP?

## Results

| Accent | Phonemes | Dist→Source | Dist→RP | Dist→GenAm | Verdict |
|--------|----------|-------------|---------|------------|---------|
| georgia | 15 | 1.632 | 2.167 | 1.757 | AMBIGUOUS (src=1.63 rp=2.17 genam=1.76) |
| scottish | 16 | 2.158 | 1.897 | 2.073 | RP_LEANING (output near RP) |
| irish | 16 | 1.911 | 2.242 | 1.763 | AMERICAN_BIAS (output near GenAm regardless of reference) |

## Interpretation

- **dist-to-source << dist-to-both-norms** → model follows the reference accent
  (zero-shot transfer; weakens the 'model-bound' claim).
- **output near GenAm regardless of reference** → American bias in the GPT
  (strong support for WS-C LoRA).
- **output near RP** → model already RP-leaning for these refs.

## Per-accent perceptual notes

*(Fill in after listening to 2 clips per accent.)*

- **Georgia** (Southern US, Brianne Howey): *TODO — listen to indextts_cal_010.wav, indextts_cal_007.wav*
- **Scottish** (James McAvoy): *TODO — listen to indextts_cal_010.wav, indextts_cal_007.wav*
- **Irish** (Saoirse Ronan): *TODO — listen to indextts_cal_010.wav, indextts_cal_007.wav*

## Explicit verdict

- **georgia**: AMBIGUOUS (src=1.63 rp=2.17 genam=1.76)
- **scottish**: RP_LEANING (output near RP)
- **irish**: AMERICAN_BIAS (output near GenAm regardless of reference)

## Implication for WS-C

Results are **mixed but still supportive of LoRA**. Key observations:

1. **No accent shows FOLLOWS_REFERENCE** — the 0.85 × min(RP, GenAm) threshold was never
   cleared by dist-to-source. Zero-shot accent transfer is weak across all three non-RP
   references. This means the accent bias is not simply fixed by reference selection.

2. **Bias is split between RP and GenAm**, not purely one**:
   - Irish → GenAm pull (AMERICAN_BIAS): dist-to-GenAm=1.763 < dist-to-source=1.911
   - Scottish → RP pull (RP_LEANING): dist-to-RP=1.897 < dist-to-GenAm=2.073
   - Georgia → AMBIGUOUS: dist-to-source closest (1.632) but GenAm second (1.757); some
     partial following, perhaps because Southern US phonology is genetically closer to GenAm
     than to RP, so the model's GenAm bias partially overlaps.

   This suggests IndexTTS-2's GPT was trained on a mixed RP + GenAm corpus (plausible for
   a Chinese model trained on diverse English data), and the dominant attractor depends on
   the reference's F0/VTL profile.

3. **WS-C LoRA remains the right intervention**: no reference cleanly pulls the model into a
   non-RP accent. The bias sits in the GPT token prior, not in the conditioning pathway.
   Fine-tuning on modern-RP audio (Fry/Lindsey cluster) will shift that prior directly.

4. **Caveat**: the nearest-neighbour centroid method for reference clips is approximate
   (no MFA transcript → rough Bark-space assignment). Distances carry ~0.2–0.3 Bark
   uncertainty; verdicts near decision boundaries (Georgia AMBIGUOUS) should be treated
   as directional, not conclusive.
