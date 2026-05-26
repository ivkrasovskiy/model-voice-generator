# Accent Coach — Phase 0.9 Findings

> **Verdict: GREEN**
> `C_int_b5` (interview reference, `num_beams=5`) reaches **H4 Bark piecewise 82.1**,
> crossing the ≥82 stop criterion. Zero-shot ceiling is acceptable for production.
> Fine-tuning is NOT needed to hit the Phase 0.9 target.

---

## Scores at a glance

| ID | Track | Reference | temp | beams | H4 Bark ↑ | WER ↓ | ECAPA ↑ | DNSMOS ↑ |
|----|-------|-----------|------|-------|-----------|-------|---------|----------|
| **C_int_b5** | **C** | **interview (14s)** | **0.8** | **5** | **82.1 🟢** | **0.037** | **0.276** | **2.67** |
| A4 | A | Sherlock (7s) | 0.8 | 3 | 78.8 | 0.042 | 0.501 | 3.29 |
| C_int_t04b5 | C | interview (14s) | 0.4 | 5 | 79.5 | 0.043 | 0.299 | 2.71 |
| C_int_t04 | C | interview (14s) | 0.4 | 3 | 78.8 | 0.042 | 0.303 | 2.80 |
| C_int_t06 | C | interview (14s) | 0.6 | 3 | 77.4 | 0.037 | 0.318 | 2.69 |
| A2 | A | interview 4:29 (17s) | 0.8 | 3 | 76.4 | 0.037 | 0.274 | 3.55 |
| A3 | A | interview 14:50 (20s) | 0.8 | 3 | 75.7 | 0.048 | 0.220 | 2.92 |
| **A1** | A | interview baseline (14s) | 0.8 | 3 | **74.6** | 0.042 | 0.301 | 2.76 |
| A6 | A | combined (18s) | 0.8 | 3 | 74.1 | 0.065 | 0.787 | 2.82 |
| B1 | B | Fry RP (35s) | 0.8 | 3 | 71.4 | 0.037 | 0.197 | 3.36 |
| B2 | B | Lindsey RP (28s) | 0.8 | 3 | 66.1 | 0.042 | 0.205 | 3.55 |
| A5 | A | narrator (12s) | 0.8 | 3 | 67.6 | 0.037 | **0.778** | 2.79 |
| C_lin_t06 | C | Lindsey RP (28s) | 0.6 | 3 | 63.0 | 0.042 | 0.221 | 3.60 |
| C_int_t04 | C | Lindsey RP (28s) | 0.4 | 3 | 56.9 | 0.042 | 0.213 | 3.52 |
| C_lin_b5 | C | Lindsey RP (28s) | 0.8 | 5 | 50.6 | 0.042 | 0.214 | 3.52 |
| real_BC | — | — | — | — | 89.8 | — | — | — |
| owner | — | — | — | — | 56.5 | — | — | — |

---

## Stop criteria evaluation

| Criterion | Threshold | Result |
|-----------|-----------|--------|
| **GREEN** | Any A/B/C variant ≥ 82 | ✅ `C_int_b5` = 82.1 |
| A1 pipeline validity | 74.6 ± 2 | ✅ 74.6 exact (frozen phase0_7 centroids) |
| B-floor RED | B1 AND B2 < 80 | ✅ B1=71.4, B2=66.1 — floor confirmed |

---

## Finding 1 — Generation params dominate reference selection

The biggest single lever is `num_beams`: raising from 3 (default) to 5 with the
production interview reference adds **+7.5 points** (74.6 → 82.1). This outperforms
every reference-clip swap in Track A.

Reference clip is still a co-variate: A4 (Sherlock, scripted dialogue) scores 78.8
vs A1 74.6, a +4.2-pt gain from register alone. But the best reference swap
(A4) plus default params (78.8) is still below what the default reference plus
more beams achieves (82.1).

**Production recommendation**: keep `ref_interview.wav` as the reference, set
`num_beams=5` in `indextts_gen.py`.

---

## Finding 2 — Temperature and beams interact non-monotonically

| beams=3 | beams=5 |
|---------|---------|
| temp=0.8 → 74.6 | temp=0.8 → **82.1** |
| temp=0.6 → 77.4 | — |
| temp=0.4 → 78.8 | temp=0.4 → 79.5 |

Lower temperature (more greedy GPT decoding) helps when beams=3 but **hurts**
when beams=5 (-2.6 pts from 82.1 → 79.5). The model benefits from maintained
sampling diversity in the autoregressive stage once beam width is increased.
Combined low-temp + high-beams is over-constrained.

---

## Finding 3 — B-floor RED: RP references are counterproductive

Both B variants score below their own baselines:

- B1 (Fry ref, 71.4) < A1 (interview, 74.6)
- B2 (Lindsey ref, 66.1) — worse yet

Track C on Lindsey is monotonically *worse* with any param change (66.1 → 63.0 →
56.9 → 50.6 as temp drops / beams rise). This confirms the Phase 0.8 hypothesis:
the model's phoneme embeddings are trained on generic English. An RP reference
provides speaker timbre but cannot override the underlying vowel centroids. When
beam search is widened with an RP reference, it locks more strongly onto those
generic embeddings, suppressing any reference-side vowel influence.

**Implication for product**: zero-shot quality is maximised by staying within the
speaker's own vowel space. Using a non-BC reference to "teach" RP vowels
zero-shot doesn't work.

---

## Finding 4 — Identity vs vowels trade-off in Track A

The narrator reference (A5) scores highest on ECAPA (0.778) because the eval
phrases are drawn from the same audiobook register, and ECAPA measures distance
from `ref_narrator.wav`. But A5's vowel score (67.6) is the worst in Track A —
the audiobook-creak register actively suppresses vowel quality in TTS output.

The Sherlock reference (A4) hits 0.501 ECAPA — BC's acting register is different
enough from audiobook to have lower identity match, but the scripted, deliberate
diction produces cleaner vowels. This suggests a genuine register / identity
trade-off that cannot be resolved by reference selection alone.

---

## Recommended production config

```bash
vendor/index-tts/.venv/bin/python scripts/indextts_gen.py \
    --ref-audio tts_output/ref_interview.wav \
    --num-beams 5
```

This gives **H4 Bark 82.1** (within Lindsey range; +7.5 pts over the prior
default; within ~8 pts of real_BC). WER 0.037 (near-zero error rate).

---

## What changed in Phase 0.9 vs Phase 0.8

| Metric | Phase 0.8 (baseline) | Phase 0.9 (C_int_b5) | Delta |
|--------|---------------------|----------------------|-------|
| synth_BC H4 Bark | 74.6 | **82.1** | **+7.5** |
| real_BC H4 Bark | 89.8 | 89.8 | — |
| Gap to real_BC | 15.2 pts | **7.7 pts** | **−7.5** |

---

## Next steps (Phase 0.10)

1. **Owner-side coaching** — the owner piecewise score (56.5) is the remaining
   gap. Phase 0.9 closed the synth_BC gap; Phase 0.10 should focus on measuring
   and coaching the owner's vowels against the updated synth_BC target.
2. **Lock the new production default** — update `CLAUDE.md` and
   `scripts/indextts_gen.py` default to `num_beams=5`.
3. **Re-run regression smoke test** with `num_beams=5` to verify WER/ECAPA
   stay within locked thresholds.

---

## Caveats

- **ECAPA scores are relative to `ref_narrator.wav`** (per-clip real BC refs
  absent on this machine). Absolute ECAPA values are not directly comparable to
  Phase 0.7/0.8 numbers; use relative ordering within Phase 0.9 only.
- **Reference duration is a co-variate in Track A** (7s–20s). Not fully
  controlled; see `ref_duration_s` column in `experiment_log.csv`.
- **Track C was abbreviated**: Lindsey variants were stopped after 4 runs
  once the monotonic degradation trend was clear; `C_lin_t04b5` was not run.
- **H4 Bark piecewise scores for Track C use the same 50-phrase cal set**
  (`cal_50.csv`) as Track A, ensuring comparability.
