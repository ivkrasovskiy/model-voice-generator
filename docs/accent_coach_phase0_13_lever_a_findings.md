# Phase 0.13b — Lever A Findings: Phoneme-Dense Emo Clip Sweep

**Date**: 2026-05-27  
**Status**: RED — Emo embedding cannot steer at phoneme granularity.

---

## 1. Verdict

**RED.** Best cell (cand1/α=0.7) achieves per-target phoneme mean lift of +2.88 pts
(48.63 → 51.51), below the YELLOW threshold of ≥ 3 pts. Composite score for that
cell is 63.83 — 3.7 pts *below* baseline (67.53), violating the composite ≥ 77
threshold. Emo conditioning at any alpha does not selectively improve the 5 failing
phonemes; it primarily changes prosody and DNSMOS without phoneme-level steering.

**Path forward**: fine-tuning. See §5 of combined findings doc.

---

## 2. Cell Results

Baseline: composite = 67.53, per-phoneme target mean = 48.63 (all from `cell_baseline_b`).

| Cell | cand | α | composite μ | composite σ | per-ph μ | composite Δ | per-ph Δ | WER | ECAPA | DNSMOS |
|---|---|---|---|---|---|---|---|---|---|---|
| cell_a_cand0_alpha03 | lindsey/4_138 | 0.3 | 68.03 | 3.56 | 42.74 | +0.50 | −5.89 | 0.008 | 0.290 | 2.726 |
| cell_a_cand0_alpha05 | lindsey/4_138 | 0.5 | 65.87 | 1.68 | 41.35 | −1.66 | −7.28 | 0.010 | 0.261 | 3.036 |
| cell_a_cand0_alpha07 | lindsey/4_138 | 0.7 | 64.17 | 1.69 | 35.33 | −3.36 | −13.30 | 0.018 | 0.209 | 3.213 |
| cell_a_cand1_alpha03 | fry/1_066 | 0.3 | 66.10 | 0.28 | 42.15 | −1.43 | −6.48 | 0.006 | 0.289 | 2.937 |
| cell_a_cand1_alpha05 | fry/1_066 | 0.5 | 67.53 | 2.24 | 46.92 | 0.00 | −1.71 | 0.030 | 0.274 | 3.218 |
| **cell_a_cand1_alpha07** | **fry/1_066** | **0.7** | **63.83** | **5.33** | **51.51** | **−3.70** | **+2.88** | **0.002** | **0.260** | **3.432** |
| cell_a_cand2_alpha03 | fry/1_004 | 0.3 | 63.23 | 3.52 | 44.71 | −4.30 | −3.92 | 0.004 | 0.299 | 2.929 |
| cell_a_cand2_alpha05 | fry/1_004 | 0.5 | 64.87 | 5.07 | 44.57 | −2.66 | −4.06 | 0.002 | 0.283 | 3.169 |
| cell_a_cand2_alpha07 | fry/1_004 | 0.7 | 65.17 | 1.47 | 49.89 | −2.36 | +1.26 | 0.006 | 0.275 | 3.360 |

---

## 3. §4.4 Stop Criteria Application

| Criterion | GREEN | YELLOW | RED | Measured (best cell) | Status |
|---|---|---|---|---|---|
| Per-target mean lift | ≥ 8 pts | [3, 8) | < 3 | **+2.88 pts** | **RED** |
| Composite modern_rp | ≥ 80 | ≥ 77 | < 77 | **63.83** | **RED** |
| WER | ≤ 0.05 | ≤ 0.06 | > 0.06 | 0.002 | GREEN |

**VERDICT: RED.** Two hard criteria fail independently. Even the per-ph lift criterion
is RED by 0.12 pts; the composite criterion fails by 13 pts.

---

## 4. Observations

### 4.1 Per-phoneme target mean vs composite trade-off

The cell with the highest per-phoneme target mean (cand1/α=0.7, per-ph=51.51) has
the second-worst composite (63.83). High α shifts the emo embedding strongly toward
the Fry reference, which improves the 5 target phonemes slightly but degrades the
other 12 phonemes enough to pull the composite down. This decoupling confirms that
the emo vector operates globally on phoneme statistics, not selectively on the
5 failing phonemes.

### 4.2 α monotonically increases DNSMOS

DNSMOS rises with α across all three candidates (e.g., lindsey: 2.73 → 3.04 → 3.21;
fry_066: 2.94 → 3.22 → 3.43). Higher emo alpha produces more expressive, higher-
quality-sounding audio — but this does not translate to phoneme-level accuracy.

### 4.3 Composite degradation with α

For lindsey and fry_004, composite decreases monotonically as α increases. For
fry_066 there is a non-monotonic bump at α=0.5 (composite=67.53 = baseline), but
α=0.7 falls to 63.83. In all cases the emo conditioning does not achieve composite
improvement above baseline at any sustainable α.

### 4.4 Per-phoneme target mean is disconnected from composite

The metric used for per-target mean (mean score over the 5 failing phonemes) can
rise while composite falls: emo conditioning narrows the gap on 5 phonemes at the
cost of introducing errors on others. A production Lever A would need to improve
both simultaneously — which this sweep shows is not achievable via emo conditioning
alone with the current model.
