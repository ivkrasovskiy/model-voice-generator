# Accent Coach — Phase 0.10 Findings

> **Verdict: RED**
> Best `composite_lindsey_mean = 62.10` (needs ≥ 82). Best `composite_BC_mean = 70.80`
> (needs ≥ 75). Neither threshold met. The GREEN gate cannot be evaluated directly
> against the Phase 0.9 result because the scoring target changed from `modern_rp`
> (aggregate) to `lindsey` (single speaker) — see Caveat 1.

---

## Scores table — top 10 trials

| Rank | Trial | loss | lindsey ± std | BC ± std | beams | temp | top_p | cfg_rate | diff_steps |
|---|---|---|---|---|---|---|---|---|---|
| 1 | #0 ¹ | 33.55 | 62.10 ± 0.00 | 70.80 ± 0.00 | 5 | 0.70 | 0.74 | 0.49 | 25 |
| 2 | #25 | 36.87 | 59.10 ± 1.16 | 67.17 ± 1.86 | 5 | 0.63 | 0.75 | 0.89 | 60 |
| 3 | #15 | 37.15 | 59.90 ± 2.52 | 65.80 ± 2.20 | 5 | 0.64 | 0.71 | 1.03 | 60 |
| 4 | #13 | 37.75 | 57.87 ± 3.30 | 66.63 ± 3.44 | 5 | 0.66 | 0.79 | 0.65 | 60 |
| 5 | #21 | 38.03 | 56.57 ± 5.76 | 67.37 ± 3.08 | 5 | 0.65 | 0.76 | 0.68 | 60 |
| 6 | #10 | 38.83 | 57.27 ± 0.71 | 65.07 ± 1.41 | 5 | 0.77 | 0.85 | 0.37 | 15 |
| 7 | #14 | 39.25 | 58.17 ± 2.15 | 63.33 ± 1.51 | 7 | 0.65 | 0.79 | 0.69 | 60 |
| 8 | #31 | 39.53 | 56.87 ± 1.92 | 64.07 ± 2.75 | 5 | 0.68 | 0.75 | 0.60 | 60 |
| 9 | #7  | 40.55 | 57.83 ± 3.50 | 61.07 ± 2.03 | 3 | 0.58 | 0.81 | 1.24 | 40 |
| 10 | #2  | 41.65 | 58.10 ± 0.00 | 58.60 ± 0.00 | 3 | 0.58 | 0.81 | 1.24 | 40 |

¹ Trial #0 is the N=1 pilot — std=0.00 is not a real measurement. Best reliable N=3 result is trial #25.

Study: 35 total trials, 14 COMPLETE, 21 PRUNED (60% pruning rate).

---

## Stop criteria evaluation

| Criterion | Threshold | Best result | Pass? |
|---|---|---|---|
| `composite_lindsey_mean` ≥ 82 | GREEN gate | 62.10 (trial #0, N=1) | ✗ |
| `lindsey_mean − std` ≥ 80 | GREEN gate | 62.10 (same) | ✗ |
| `composite_BC_mean` ≥ 75 | GREEN gate | 70.80 (trial #0, N=1) | ✗ |
| Best reliable N=3 lindsey | — | 59.90 ± 2.52 (trial #15) | ✗ |
| Best reliable N=3 BC | — | 67.17 ± 1.86 (trial #25) | ✗ |

**Verdict: RED.** However, the thresholds were written for `modern_rp` scoring.
Against `lindsey` alone, the frozen Phase 0.7 synth_BC already scores only 58.80
(see Caveat 1). The GREEN gate of 82 is not achievable with zero-shot TTS against
a single strict RP speaker — it would require fine-tuning or a different target
framing.

---

## Finding 1 — CFM diffusion steps are the dominant parameter

fANOVA importance scores across 14 complete trials:

| Parameter | Importance |
|---|---|
| `diffusion_steps` | **0.518** |
| `temperature` | 0.214 |
| `cfg_rate` | 0.120 |
| `top_p` | 0.109 |
| `num_beams` | 0.040 |

`diffusion_steps` accounts for over half the variance in loss. This was invisible
in Phase 0.9, which kept it hard-coded at 25. Higher steps (60) consistently
appears in top N=3 trials (#25, #15, #13, #21, #14, #31) — 6 of the top 8.

`num_beams` (the Phase 0.9 lever) has near-zero importance here. This does NOT
contradict Phase 0.9 — the Phase 0.9 sweep only varied beams. When the CFM knobs
are also free, they dominate. The `beams=5` preference holds: all top-6 trials
use beams=5.

**Recommendation**: lock `beams=5` (confirmed), set `diffusion_steps=60` for
best vowel quality (at ~2× generation cost vs. 25 steps).

---

## Finding 2 — lindsey and real_BC scores are weakly correlated

Spearman ρ = **0.491**, p = 0.075 (across 14 complete trials).

Positive but not significant at α=0.05. The two objectives are partially
independent — a configuration can push lindsey-score up without a proportional
gain in BC-score and vice versa. In particular, BC scores cluster tightly
(61–71) while lindsey scores spread more (54–62), suggesting lindsey is the
harder and more discriminating target.

The equal weighting (0.5/0.5) in the loss is reasonable: if they were fully
correlated, one target would be redundant; if fully independent, neither alone
would be sufficient.

---

## Finding 3 — Posthoc WER/ECAPA/DNSMOS vs Phase 0.9 baseline

Best trial (#0) scored on `eval_short.csv` (15 phrases), N=3 replicates:

| Metric | Phase 0.10 best (trial #0) | Phase 0.9 `C_int_b5` | Δ |
|---|---|---|---|
| WER ↓ | 0.042 ± 0.000 | 0.037 | +0.005 (worse) |
| ECAPA ↑ | 0.293 ± 0.003 | 0.276 | +0.017 ✓ |
| DNSMOS OVR ↑ | 2.70 ± 0.09 | 2.67 | +0.03 ✓ |

**WER** is slightly higher than Phase 0.9 (0.042 vs 0.037) — both are essentially
zero-error rates; the difference is 1–2 words across 15 phrases.

**ECAPA** improved by +0.017 — the best trial uses `cfg_rate=0.49` (below the
default 0.70), which appears to keep the voice closer to the reference speaker
by reducing the CFM guidance push away from the reference embedding.

**DNSMOS** marginally improved (+0.03). Generation quality is not degraded by
the parameter changes.

Overall: Phase 0.10 best config maintains Phase 0.9 quality on intelligibility
and modestly improves on identity and naturalness.

---

## Cross-speaker validation matrix (Phase 0.12+)

After corpus cleanup and N=3 confirmation runs, the following modern_rp composite
scores were measured across all sources — including the owner (L2 Russian accent,
non-native English speaker):

| Source | modern_rp | real_BC | fry | bbc_male | lindsey |
|---|---|---|---|---|---|
| fry | 98.1 | 92.4 | — | 91.4 | 79.7 |
| bbc_male | 97.9 | 81.2 | 91.4 | — | 75.6 |
| lindsey | 94.5 | 72.8 | 79.7 | 75.6 | — |
| real_BC | **88.9** | — | 92.4 | 81.2 | 72.8 |
| synth_BC | **75.4** | 77.6 | 77.8 | 74.0 | 63.5 |
| **owner** | **59.3** | 49.9 | 58.7 | 60.9 | 51.5 |

**What this confirms:**

1. **Scoring system is correctly calibrated.** Native RP speakers cluster 94–98,
   well above the BC target (88.9). The ordering — native RP > real BC > synth BC >
   L2 non-native — is exactly what a valid phonetic distance metric should produce.

2. **Synth BC is substantially better than L2 baseline.** The gap between synth_BC
   (75.4) and owner (59.3) is **+16.1 points** — the TTS clone sits meaningfully
   closer to RP than an L2 Russian-accented speaker. This validates that the TTS
   pipeline is doing useful accent transfer work.

3. **Remaining challenge: 13.5-point gap from real BC.** real_BC scores 88.9 vs
   synth_BC's 75.4. This gap is the open problem — it cannot be closed by parameter
   tuning alone (Phase 0.10/0.11 showed ~2–3 point ceiling from sweep/emo). Closing
   it would require either a different reference conditioning strategy or fine-tuning.

4. **Per-phoneme failure pattern (synth-specific).** The gap between real_BC and
   synth_BC concentrates on 5 phonemes where real BC scores 86–100 but synth scores
   30–59: /ʌ/ (STRUT), /ʊ/ (FOOT), /ɔː/ (THOUGHT), /aʊ/ (MOUTH), /ɜː/ (NURSE).
   These are TTS artifacts, not BC accent features — the model is not correctly
   capturing BC's vowel placement for these sounds relative to the RP centroid.

---

## Remaining TTS accent levers

Two approaches remain unexplored that are feasible without reference clip curation
(which requires user-facing clip selection and is not viable in production).

---

### Lever A — Targeted emo clips for the 5 failing phonemes

**Idea.** Phase 0.11 used full-sentence Fry emo clips, which carry broadband
phonetic information. A more surgical variant: construct short emo clips (~3–5 s)
that are dense in the 5 failing vowels (/ʌ/, /ʊ/, /ɔː/, /aʊ/, /ɜː/) from a clean
RP speaker. IndexTTS-2's CFM stage pulls the mel-spectrogram distribution toward
the emo embedding — if the emo clip is phonetically loaded with exactly the sounds
that are wrong, the guidance vector should correct those specific regions more than
a generic sentence does.

**How to test.** Collect or synthesize 2–3 short RP sentences rich in the failing
phonemes (e.g. "The nurse thought about the foot of the strut"). Extract a 5 s
window. Run with `emo_alpha ∈ {0.3, 0.5}`. Score per-phoneme on /ʌ ʊ ɔː aʊ ɜː/
specifically, not just composite. Success criterion: those 5 phonemes move from
30–59 toward 70+ without composite dropping below baseline 75.4.

**Caveat.** This is architecturally plausible but unproven at phoneme granularity.
The emo embedding is a sentence-level vector — it's unclear how much it can steer
individual phoneme production vs. overall prosody/timbre.

---

### Lever B — Post-processing formant shift via vocoder

**Idea.** After generation, use pyworld (WORLD vocoder) to extract the F0 +
spectral envelope + aperiodicity from the generated WAV, shift F1/F2 for the
failing phonemes toward the RP centroid, then resynthesize. This is deterministic:
it always moves the acoustic measurements (and therefore the scores) in the right
direction because it directly edits the signal.

**How to test.** Implement a `shift_vowels_to_rp(wav_path, synth_centroids,
target_centroids)` function:
1. Run forced alignment (whisperx) to get per-phoneme time segments.
2. For each segment of a failing phoneme, extract WORLD envelope, estimate current
   F1/F2 via parselmouth, compute the (Δf1, Δf2) needed to hit the RP centroid.
3. Apply a smooth frequency warp to the spectral envelope for that segment only.
4. Resynthesize. Write the patched WAV.
Score both pre- and post-patched WAVs. The pre/post delta gives an empirical
ceiling — the maximum improvement achievable if the formant shift were perfect and
artifact-free.

**Caveat.** Vocoder resynthesis introduces audible artifacts (buzziness, formant
smearing) even with WORLD. DNSMOS OVR will drop. This is best used as a
measurement of theoretical headroom (how much of the 13.5-point gap is recoverable
from signal-level edits) rather than as a production path. If the ceiling is <5
points, the gap is fundamentally in the model and only fine-tuning closes it. If
the ceiling is 10+ points, the signal is recoverable and a cleaner vocoder
(differentiable, neural) could be worth building.

---



**Caveat 1 — Target change invalidates direct GREEN gate comparison.**
Phase 0.9 scored synth_BC against `modern_rp` (aggregate of Fry + Lindsey + BBC).
Phase 0.10 scores against `lindsey` alone. The same frozen Phase 0.7 synth_BC
centroids score 74.60 vs `modern_rp` but only 58.80 vs `lindsey`. The Phase 0.9
GREEN threshold of 82 was calibrated for modern_rp scoring and is not directly
applicable here. A fair Phase 0.10 GREEN gate would be approximately lindsey ≥ 70
(the lindsey-equivalent of the modern_rp 82 threshold, extrapolated from the
~16-point systematic offset observed at baseline).

**Caveat 2 — Best trial is N=1 (pilot).**
Trial #0 was the only pilot trial (N=1, no replicates). Its loss=33.55 and
lindsey=62.10 cannot be trusted as stable estimates — there is no std. The best
fully replicated (N=3) result is trial #25 at loss=36.87, lindsey=59.10±1.16.

**Caveat 3 — High pruning rate (60%) limits the search.**
21 of 35 trials were pruned, mostly after 1–2 reps. The MedianPruner was
aggressive relative to the variance between reps — per-trial variance is high
enough (std up to 5.76 for lindsey) that pruning after rep 1 discards potentially
good configurations. A wider warmup window would help in Phase 0.11.

**Caveat 4 — cal_25 has only 19 phrases (not 25).**
The greedy selection stopped early at 19 because all available phonemes hit ≥2
hits. Sonorants (l, r, m, n, ŋ) and /e/, /ɒ/ are absent from the formant data —
the aligner does not produce these labels for the BC calibration clips. Scoring
covers 11 of 17 spec phonemes.

**Caveat 5 — MPS unavailable; all runs on CPU.**
BigVGAN anti-aliased convolution exceeds the MPS 65536 output channel limit.
All generation ran on CPU at ~50–85 s/phrase depending on diffusion_steps.
