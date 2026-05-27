# Phase 0.13a — Lever B Findings: Formant-Shift Ceiling

**Date**: 2026-05-26  
**Status**: YELLOW — Pursue Lever A.

---

## 1. Verdict

**YELLOW.** `cell_all5` composite lift = **+5.47 pts** (67.53 → 73.00), falling in the [5, 10)
YELLOW band of §3.6. DNSMOS drop = **0.37 pts** (2.578 → 2.211), within the GREEN
sub-threshold of ≤ 0.5. Only 1 of 5 target phonemes degraded (/aʊ/ = −33.3 pts),
not the ≥ 3 required for a RED per-phoneme trigger.

**Action**: Proceed to Phase 0.13b (Lever A). Note that Lever B's ceiling is bounded
by both the WORLD vocoder quality cost and the diphthong-shift anomaly on /aʊ/. A
neural vocoder follow-up for the formant-shift path is not yet justified — Lever A
should run first.

---

## 2. Cell Results

| Cell | Edited | composite μ | composite σ | per-ph μ | DNSMOS μ | WER μ |
|---|---|---|---|---|---|---|
| cell_baseline_b | none | 67.53 | 1.38 | 48.63 | 2.578 | 0.014 |
| cell_strut | /ʌ/ | 71.20 | 3.93 | 54.09 | 2.235 | 0.012 |
| cell_foot | /ʊ/ | 71.23 | 4.07 | 51.57 | 2.222 | 0.012 |
| cell_thought | /ɔː/ | 70.07 | 2.60 | 46.95 | 2.215 | 0.019 |
| cell_mouth | /aʊ/ | 68.60 | 4.04 | 45.65 | 2.232 | 0.012 |
| cell_nurse | /ɜː/ | 71.40 | 4.38 | 53.75 | 2.211 | 0.014 |
| **cell_all5** | all 5 | **73.00** | **2.48** | **54.97** | **2.211** | **0.019** |

Baseline DNSMOS scored post-hoc via `score_posthoc_clips` on existing rep_0/1/2 clips.

---

## 3. Per-Phoneme Breakdown (cell_all5 vs baseline)

| Phoneme | Baseline | cell_all5 | Lift | Notes |
|---|---|---|---|---|
| /ʌ/ (STRUT) | 45.60 | 77.20 | **+31.60** | Largest gain; monophthong, well-defined target |
| /ʊ/ (FOOT) | 30.00 | 42.13 | **+12.13** | Good gain; was the floor phoneme |
| /ɜː/ (NURSE) | 51.13 | 62.87 | **+11.74** | Good gain |
| /ɔː/ (THOUGHT) | 47.63 | 57.17 | **+9.54** | Modest gain |
| /aʊ/ (MOUTH) | 68.77 | 35.47 | **−33.30** | Severe degradation — diphthong anomaly |

---

## 4. §3.6 Stop Criteria Application

| Criterion | Threshold | Measured | Status |
|---|---|---|---|
| Composite lift | ≥ 10 pts (GREEN); ≥ 5 pts (YELLOW) | +5.47 pts | YELLOW trigger |
| DNSMOS drop | ≤ 0.5 (GREEN); ≤ 1.0 (YELLOW) | 0.37 pts | GREEN sub-threshold |
| ≥ 3 per-target phonemes lift < 5 | RED if triggered | 1/5 (/aʊ/ only) | Not triggered |

**VERDICT: YELLOW.** Composite lift falls in [5, 10), DNSMOS drop is within GREEN range,
and the per-phoneme RED trigger is not activated. Per §3.6: pursue Lever A.

---

## 5. Observations and Caveats

### 5.1 WORLD vocoder re-synthesis cost

All shift cells have DNSMOS ≈ 2.21, regardless of how many phonemes are shifted.
The baseline (TTS output, no WORLD processing) scores DNSMOS = 2.578. The entire
0.37-pt DNSMOS drop is attributable to WORLD `wav2world / synthesize` round-trip
quality loss, not to the formant shifting itself. This is a fixed-cost ceiling of
the WORLD-based approach.

### 5.2 /aʊ/ diphthong anomaly

Shifting /aʊ/ toward its modern_rp centroid midpoint drops the score from 68.77 to
35.47 (−33.3 pts). The effect appears in `cell_mouth` (shift /aʊ/ only) and carries
over into `cell_all5`. Three possible explanations:

1. **Midpoint mismatch**: /aʊ/ is a diphthong; targeting a single F1/F2 midpoint
   doesn't represent the trajectory. The scoring function measures the full phoneme
   centroid, so a midpoint-only warp distorts the score.
2. **Overshoot**: The target centroid for /aʊ/ in modern_rp differs significantly
   from the synth_BC midpoint, causing the warp to overshoot into a low-scoring
   region of Bark space.
3. **WORLD spectral artefact**: The WORLD spectral envelope warp may be less stable
   for wide-bandwidth formant regions characteristic of /aʊ/.

A neural vocoder (e.g. HiFi-GAN) would not fix explanations 1 or 2. Fixing /aʊ/
likely requires either (a) trajectory-aware shifting (warp start and endpoint
separately) or (b) addressing the gap at the model level (Lever A / fine-tuning).

### 5.3 WORLD re-synthesis changes non-shifted phonemes

Unshifted phonemes differ from their baseline values after WORLD round-trip. Example:
/ɔː/ in cell_strut (only /ʌ/ shifted) scores 70.53 vs baseline 47.63 (+22.9 pts).
This suggests WORLD synthesis changes spectral shape in ways that affect Praat formant
extraction unpredictably. The composite scores are thus not a clean isolation of
the shift effect — WORLD re-synthesis introduces its own formant-measurement
confound. This makes the per-cell comparisons harder to interpret at the
phoneme level; the composite and overall per-phoneme mean are more reliable
summary statistics.

---

## 6. Next Step

Run **Phase 0.13b (Lever A)** per §4 of the plan:

```bash
# Mine emo candidates (already done — index.json exists)
# Run lever A: 3 candidates × 3 alphas × N=3 reps
.venv/bin/python scripts/accent_coach_phase0_13_lever_a.py --replicates 3
```

Estimated wall time: ~8–13 hours CPU (9 cells × 3 reps × 19 clips × ~60–90 s/clip).

---

## 7. Phase 0.14 WS-A — Segment-Splice Validation (2026-05-27)

**Question**: Does running WORLD only on a ±30 ms context window around each target
segment (instead of the full clip) recover the DNSMOS quality loss observed in §1?

**Method**: `cell_all5 --replicates 1 --splice` via Phase 0.14 `formant_shift.py` splice
path. Baseline WAVs are the same (re-generated Phase 0.13 `cell_baseline_b` clips).
Outputs written to `tts_output/accent_coach/phase0_14/cells/cell_all5/rep_0/`.

**Results** (1 replicate):

| Method | Composite | DNSMOS OVR | WER |
|---|---|---|---|
| Phase 0.13 full-resynth (3 reps) | 73.00 ± 2.48 | 2.211 | 0.019 |
| Phase 0.14 splice (1 rep) | **74.70** | **2.633** | 0.018 |
| Phase 0.13 baseline (no shift) | 67.53 | 2.578 | 0.014 |

**Verdict**: **GREEN** — acceptance criterion met.
- DNSMOS splice (2.633) ≥ full-resynth (2.211): **+0.422 pts recovered**
- DNSMOS splice (2.633) is also *above* baseline (2.578), likely within 1-rep noise but
  confirms the splice does not degrade naturalness relative to untouched audio.
- Composite lift vs baseline: **+7.17 pts** (67.53 → 74.70), slightly better than
  full-resynth's +5.47 pts — consistent with DNSMOS improvement feeding back into scoring.

**Conclusion**: The global WORLD round-trip was the source of the DNSMOS penalty in
Phase 0.13. Windowed splice (±30 ms context, 20 ms equal-power crossfade) eliminates
that penalty while preserving the formant-shift composite gain. The splice path is the
recommended default for Lever B going forward.
