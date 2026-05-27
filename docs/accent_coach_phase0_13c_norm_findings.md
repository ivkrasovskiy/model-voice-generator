# Phase 0.13c — F3 Normalization Bench Findings

**Date**: 2026-05-26  
**Status**: RED — Bark remains the production normalization.

---

## 1. Verdict

**RED.** No method satisfies even the YELLOW threshold (§5.6).

Nearey-intrinsic and F-ratios tighten the native RP cluster dramatically (85–94% reduction in native stddev) but destroy L2 discrimination (83–92% loss) — the same failure mode as Lobanov per-speaker z-scoring. Syrdal-Gopal preserves and slightly improves discrimination but makes native tightness *worse* (4.4% increase). No method satisfies: T(M) ≤ 0.90·T_bark **AND** D(M) ≥ 0.70·D_bark.

**Conclusion**: Bark is confirmed as the best speaker-blind normalization available with the current per-token F3 estimates. The per-token F3 approach cannot disentangle speaker identity from phonetic category — exactly as Lobanov showed at the utterance level.

---

## 2. Bench Table

| Method | native_tight | l2_disc | bc_gap | nt_rel | l2_rel | bc_rel |
|---|---|---|---|---|---|---|
| bark_control | 0.5399 | 0.8976 | 0.4904 | 1.000 | 1.000 | 1.000 |
| syrdal_gopal | 0.5635 | 1.0445 | 0.4606 | 1.044 | 1.164 | 0.939 |
| nearey_intrinsic | 0.0808 | 0.1521 | 0.0517 | 0.150 | 0.169 | 0.105 |
| f_ratios | 0.0324 | 0.0697 | 0.0296 | 0.060 | 0.078 | 0.060 |

Metrics definition:
- **native_tightness**: mean over 17 phonemes of (σ_dim1 + σ_dim2) across the 3 native RP speakers {fry, lindsey, bbc_male}. Lower = tighter cluster.
- **l2_discrimination**: mean over 17 phonemes of ‖owner_centroid − modern_rp_centroid‖. Higher = better L2 learner is distinguishable.
- **bc_gap**: mean over 17 phonemes of ‖real_BC_centroid − modern_rp_centroid‖. Higher = real BC clips are more distinct from native RP mean.
- `*_rel_bark`: ratio to the bark_control row.

---

## 3. Stop Criteria Application (§5.6)

Let T_b = 0.5399, D_b = 0.8976, G_b = 0.4904.

**GREEN** threshold: T(M) ≤ 0.4319 AND D(M) ≥ 0.7630 AND G(M) ≥ 0.3678  
**YELLOW** threshold: T(M) ≤ 0.4859 AND D(M) ≥ 0.6283 (but not GREEN)

| Method | T ≤ 0.4319? | T ≤ 0.4859? | D ≥ 0.7630? | D ≥ 0.6283? | GREEN? | YELLOW? |
|---|---|---|---|---|---|---|
| syrdal_gopal | ✗ (0.564) | ✗ (0.564) | ✓ (1.045) | ✓ | ✗ | ✗ |
| nearey_intrinsic | ✓ (0.081) | ✓ | ✗ (0.152) | ✗ | ✗ | ✗ |
| f_ratios | ✓ (0.032) | ✓ | ✗ (0.070) | ✗ | ✗ | ✗ |

**Verdict: RED.** No method satisfies YELLOW.

---

## 4. Per-Phoneme Breakdown

### Syrdal-Gopal (best discrimination, worst tightness)

Syrdal-Gopal maps F1 and F2 to their Bark-difference from F3. Because F3 varies strongly by speaker VTL in predictable ways, the difference (Z3−Z1) and (Z3−Z2) partially corrects for speaker size but not reliably enough: the native spread in this space is 4.4% *worse* than Bark. The method does increase L2 discrimination (1.164×) and BC gap is near-Bark (0.939×), suggesting BC's vowels are phonetically distinctive in Syrdal-Gopal space — but the native cluster is no tighter.

### Nearey-intrinsic / F-ratios (high tightness, zero discrimination)

Both achieve tightness reductions of 85–94%, but l2_discrimination drops to 0.169×–0.078× and bc_gap drops to 0.105×–0.060×. The normalization collapses all speakers into nearly the same vowel space — including the L2 learner and real BC, making them indistinguishable from the RP cluster. This confirms the Lobanov-style failure mode: per-token intrinsic normalization over-normalises, erasing the between-speaker and L2-compression signal that coaching depends on.

---

## 5. Recommendation

**Stay on Bark.** RED outcome across all three F3 methods closes the door on per-token F3-based normalization for this use case.

The underlying reason (see memory: [Phase 0.6 Lobanov limitation](../.claude/projects/-Users-ivkrasovskii-model-voice-generator/memory/project_phase0_6_lobanov_limit.md)):
> Lobanov / per-speaker z-scoring erases L2 vowel-space compression. The same mechanism applies to per-token intrinsic normalization: by dividing by F3 (a proxy for VTL), you remove exactly the signal that separates L2-compressed vowel spaces from native RP.

Syrdal-Gopal is the only method that preserves discrimination, but it does not tighten the native cluster. If tightening native variance is important in a future phase (e.g. to reduce σ_rp and sharpen the scoring function), a hierarchical approach — e.g. Syrdal-Gopal for the native cluster boundary, Bark for coaching — could be explored in Phase 0.14. This is a speculative path, not recommended now.

**Do NOT open Phase 0.14** for normalization switching. The RED verdict makes this a closed question unless new evidence emerges from a different corpus or formant estimator.
