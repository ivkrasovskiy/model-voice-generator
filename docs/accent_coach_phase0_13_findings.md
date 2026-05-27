# Phase 0.13 — Combined Findings

**Date**: 2026-05-27  
**Levers tested**: B (formant shift), A (phoneme-dense emo clips)  
**Diagnostic**: Phase 0.13c (F3 normalization bench)

---

## 1. Executive Summary

| Component | Verdict | Key number |
|---|---|---|
| Lever B (formant shift) | **YELLOW** | cell_all5 composite lift +5.47 pts (67.53 → 73.00) |
| Lever A (emo conditioning) | **RED** | Best per-target lift +2.88 pts; composite drops −3.7 pts |
| F3 normalization bench | **RED** | No method satisfies YELLOW; Bark stays as production norm |

**Overall conclusion**: The 13.5-pt synth_BC vs modern_rp gap is model-bound. Signal-level
post-processing (WORLD formant shift) recovers ~5.5 pts at the cost of vocoder artifact.
Emo conditioning cannot steer phoneme-level formants. F3 normalization cannot tighten
the native RP cluster without destroying L2 discrimination.

**Path forward**: LoRA / adapter fine-tuning on a BC-leaning RP corpus targeting the
5 failing phonemes (/ʌ/, /ʊ/, /ɔː/, /aʊ/, /ɜː/).

---

## 2. Lever B — Formant Shift Ceiling

Full details: [accent_coach_phase0_13_lever_b_findings.md](accent_coach_phase0_13_lever_b_findings.md)

| Cell | composite μ | per-ph μ | DNSMOS μ |
|---|---|---|---|
| cell_baseline_b (no edit) | 67.53 | 48.63 | 2.578 |
| cell_strut (/ʌ/ only) | 71.20 | 54.09 | 2.235 |
| cell_foot (/ʊ/ only) | 71.23 | 51.57 | 2.222 |
| cell_thought (/ɔː/ only) | 70.07 | 46.95 | 2.215 |
| cell_mouth (/aʊ/ only) | 68.60 | 45.65 | 2.232 |
| cell_nurse (/ɜː/ only) | 71.40 | 53.75 | 2.211 |
| **cell_all5 (all 5)** | **73.00** | **54.97** | **2.211** |

**Key observations**:
- Composite lift of +5.47 pts is real but bounded; ceiling is partly an artefact of
  WORLD vocoder re-synthesis changing formant measurements on non-shifted phonemes.
- WORLD re-synthesis costs −0.37 DNSMOS (2.578 → 2.211) as a fixed overhead independent
  of how many phonemes are shifted.
- /ʌ/ responds strongly (+31.6 pts per-phoneme); /aʊ/ collapses (−33.3 pts) due to
  diphthong mid-point targeting — the WORLD shift cannot represent a formant trajectory.
- A neural vocoder would reduce DNSMOS degradation but would not fix the /aʊ/ diphthong
  failure or the overall model-level gap.

---

## 3. Lever A — Phoneme-Dense Emo Clip Sweep

Full details: [accent_coach_phase0_13_lever_a_findings.md](accent_coach_phase0_13_lever_a_findings.md)

3 candidates × 3 emo_alpha values × N=3 reps = 9 cells.

| Cell | composite μ | per-ph μ | per-ph Δ | composite Δ | DNSMOS |
|---|---|---|---|---|---|
| cand0/α=0.3 (lindsey) | 68.03 | 42.74 | −5.89 | +0.50 | 2.726 |
| cand0/α=0.5 | 65.87 | 41.35 | −7.28 | −1.66 | 3.036 |
| cand0/α=0.7 | 64.17 | 35.33 | −13.30 | −3.36 | 3.213 |
| cand1/α=0.3 (fry/1_066) | 66.10 | 42.15 | −6.48 | −1.43 | 2.937 |
| cand1/α=0.5 | 67.53 | 46.92 | −1.71 | 0.00 | 3.218 |
| **cand1/α=0.7** | **63.83** | **51.51** | **+2.88** | **−3.70** | **3.432** |
| cand2/α=0.3 (fry/1_004) | 63.23 | 44.71 | −3.92 | −4.30 | 2.929 |
| cand2/α=0.5 | 64.87 | 44.57 | −4.06 | −2.66 | 3.169 |
| cand2/α=0.7 | 65.17 | 49.89 | +1.26 | −2.36 | 3.360 |

**Key observations**:
- Best per-target phoneme lift is +2.88 pts (< YELLOW threshold of 3 pts) — technically RED.
- The cell achieving the best per-target lift (cand1/α=0.7) simultaneously drops the
  composite by −3.7 pts: emo conditioning improves 5 phonemes at the cost of regressing
  the other 12. No cell achieves both.
- Higher α raises DNSMOS monotonically (prosodic expressiveness increases) but does not
  improve phoneme accuracy.
- Phoneme-dense emo clip selection does not outperform Phase 0.11's broadband Fry
  sentence sweep — the emo vector is a global acoustic modifier, not a phoneme-targeted
  one.

---

## 4. Phase 0.13c — F3 Normalization Bench

Full details: [accent_coach_phase0_13c_norm_findings.md](accent_coach_phase0_13c_norm_findings.md)

| Method | native_tightness | l2_discrimination | Verdict |
|---|---|---|---|
| bark_control | 0.540 | 0.898 | baseline |
| syrdal_gopal | 0.564 | 1.045 | ✗ tightness worse |
| nearey_intrinsic | 0.081 | 0.152 | ✗ discrimination destroyed |
| f_ratios | 0.032 | 0.070 | ✗ discrimination destroyed |

Bark is confirmed as the best speaker-blind normalization for this use case.
Intrinsic F3-based normalization reproduces the Lobanov failure mode (erases
L2-compressed vowel space). Syrdal-Gopal preserves discrimination but does not
tighten the native cluster — no method satisfies even the YELLOW threshold.

---

## 5. Root Cause Analysis

The 13.5-pt gap between synth_BC (67.53) and modern_rp ceiling (real_BC = 88.9) is
confirmed to be **model-bound**:

1. **Signal editing (Lever B) recovers ~5.5 pts** — but at vocoder cost and with a
   diphthong failure mode. This establishes the ceiling for post-hoc spectral editing.
2. **Conditioning (Lever A) provides no phoneme-level steering** — emo vectors operate
   globally, not on individual phoneme trajectories in formant space.
3. **Scoring normalization (F3 bench) cannot resolve the native cluster boundary**
   without collapsing L2 discrimination.
4. **The gap is concentrated in diphthongs and tense vowels** where IndexTTS-2 places
   synth_BC in American-influenced positions (/ʊ/ at 30 pts, /ʌ/ at 45.6 pts, /ɔː/ at
   47.6 pts, /ɜː/ at 51.1 pts, /aʊ/ at 68.8 pts).

The model has no signal path through which per-phoneme formant targets can be injected
without modifying weights.

---

## 6. Recommended Next Step: Fine-Tuning (Phase 0.14)

Per §8 of the Phase 0.13 plan, the RED-both outcome points to:

> (a) LoRA fine-tune on a BC-leaning RP corpus; (b) switch base model; (c) accept the
> 13.5-pt gap and ship as-is.

**Recommended path: (a) LoRA fine-tuning.**

Inputs available:
- **Lever B emo-dense clips** (`phase0_13/emo_dense/cand_0-2.wav`): phoneme-dense RP
  candidates with high density of the 5 failing phonemes. Directly usable as fine-tuning
  targets.
- **Phase 0.12 cleaned corpus** (`cleaned_corpus/{fry,lindsey,bbc_male}/`): ECAPA-filtered
  RP corpus (sim ≥ 0.50) with formant ground truth. Can anchor the fine-tuning distribution.
- **Per-phoneme formant targets** from `speaker_centroids_cleaned.json`: explicit F1/F2
  targets for each of the 5 phonemes under modern_rp.
- **cal_25 phrase set**: reuse for A/B evaluation continuity.

The WORLD-shifted clips from Lever B (`cell_all5/rep_{0,1,2}/`) represent the target
vowel space for the 5 failing phonemes and could serve as weak supervision signal for
the fine-tune, though audio quality is limited by WORLD artifacts.

Do **not** open Phase 0.14 for normalization switching (F3 bench is RED — Bark stays).
