# Accent Coach — Phase 0.6 Plan

> **Status**: ready to execute. Measurement-only — no scoring code modified.
> **Builds on**: [accent_coach_phase0_5_findings.md](accent_coach_phase0_5_findings.md),
> [accent_coach_phase0_5_table.csv](accent_coach_phase0_5_table.csv)
> **Triggered by**: owner's pushback that the Phase 0.5 conclusion ("owner closer to
> modern RP than Deterding 1997 is to modern RP") is implausible given a Slavic-L1 accent.

---

## Hypothesis

The Phase 0.5 raw F1/F2 distance comparison gave the wrong answer because of two defects:

1. **Compressed vowel space.** Owner's F1 spans only ~120 Hz across 9 vowels (438–560
   in [accent_coach_phase0_5_table.csv:56-64](accent_coach_phase0_5_table.csv#L56-L64)),
   vs ~270 Hz for native speakers. All owner vowels cluster near the centre of F1×F2 space.
   Raw 2D Euclidean distance to a specific modern-RP target is small not because the vowel
   is well-produced, but because a centralised blob accidentally overlaps the target. This
   is the canonical Slavic-L1 signature.
2. **Possible BBC contamination of modern_rp pool** (deprioritised, see Phase E). Even if
   real, the multi-baseline check below should make the result BBC-independent.

**Predictions if hypothesis is correct:**
- Owner's vowel-space convex-hull area is 30–50% of native speakers'.
- With Lobanov normalisation (per-speaker z-scoring strips out vowel-space scale), the
  owner sits in a distinct cluster from {Fry, Lindsey, real_bc, synth_bc}; Deterding sits
  in its own location (old RP).
- In Lobanov space, **d(owner, native_baseline) > d(Deterding, native_baseline)** — the
  opposite of the Phase 0.5 raw-Hz finding.

---

## Multi-baseline robustness

To make the conclusion BBC-independent, every distance comparison runs against three
baselines. If the hypothesis holds across all three, the BBC question is moot.

| Baseline ID         | Composition                            | Rationale                                  |
|---------------------|----------------------------------------|--------------------------------------------|
| `modern_rp_full`    | BBC + Fry + Lindsey                    | Reproduces Phase 0.5 baseline             |
| `modern_rp_no_bbc`  | Fry + Lindsey                          | Removes any female-voice contamination     |
| `native_wide`       | Fry + Lindsey + real_bc + synth_bc     | Broadest confirmed-male native pool        |

Decision rule:
- **V3 holds for all 3 baselines** → BBC question is irrelevant; skip Phase E.
- **V3 holds for ≥ 2 of 3** → likely robust; Phase E optional.
- **V3 fails for `modern_rp_no_bbc` or `native_wide`** → hypothesis may be wrong; report
  honestly, do not force Phase E.
- **V3 holds only for `modern_rp_full`** → run Phase E to confirm BBC contamination.

---

## Phase A — Vowel-space geometry metrics

**Script**: `scripts/accent_coach_phase0_6_geometry.py`
**Input**: [docs/accent_coach_phase0_5_table.csv](accent_coach_phase0_5_table.csv)
**Output**: `docs/accent_coach_phase0_6_geometry.csv`

For each source in
`{deterding_rp, modern_rp_bbc, modern_rp_fry, modern_rp_lindsey, real_bc, synth_bc, owner,
modern_rp_full, modern_rp_no_bbc, native_wide}`, compute:

| Column              | Definition                                                                 |
|---------------------|----------------------------------------------------------------------------|
| `n_vowels`          | Number of phonemes available (should be 9)                                 |
| `f1_range_hz`       | `max(F1_mean) − min(F1_mean)` across 9 vowels                              |
| `f2_range_hz`       | `max(F2_mean) − min(F2_mean)` across 9 vowels                              |
| `area_hz2`          | Convex hull area in F1×F2 plane (`scipy.spatial.ConvexHull`)               |
| `articulation_idx`  | `area_hz2 / median(area_hz2 over {fry, lindsey, real_bc, synth_bc})`       |

For the aggregated baselines (`modern_rp_full`, `modern_rp_no_bbc`, `native_wide`),
compute per-phoneme F1/F2 as the **simple mean of constituent speakers' means** (equal
weight per speaker — this is what we want for cross-speaker comparison, not token-weighted).

**Acceptance**: file written; row count = 10; `articulation_idx(owner) < 1.0`.

---

## Phase B — Lobanov normalisation

**New module**: `accent_coach/diagnostics/lobanov.py`

```python
def lobanov_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Per-source z-scoring of F1 and F2 across phonemes.

    For each source, compute mu/sigma over its 9 vowels independently, then
    z = (x - mu) / sigma. Strips out vowel-space scale; preserves shape.

    Args:
        df: long-format with columns [source, phoneme, F1_mean, F2_mean]
    Returns:
        Same df with added [F1_lobanov, F2_lobanov]
    """
```

**Script**: `scripts/accent_coach_phase0_6_lobanov.py`
**Input**: [docs/accent_coach_phase0_5_table.csv](accent_coach_phase0_5_table.csv) +
the 3 aggregated baseline rows computed in Phase A (re-derive here, do not depend on
the geometry CSV).
**Outputs**:
- `docs/accent_coach_phase0_6_lobanov_table.csv` — long-format with `F1_lobanov`, `F2_lobanov`.
- `docs/accent_coach_phase0_6_lobanov_distances.csv` — full pairwise Euclidean distance
  matrix between sources in 18-D Lobanov space (9 vowels × 2 formants, flattened).
- `docs/img/accent_coach_phase0_6_dendrogram.png` — hierarchical clustering (Ward).
- `docs/img/accent_coach_phase0_6_lobanov_vowel_space.png` — all speakers overlaid in
  Lobanov F1×F2 space, one panel per phoneme or all-in-one with vowel labels.

**Algorithm**:
1. Load Phase 0.5 table; filter to the 9 target phonemes.
2. Append the 3 aggregated baseline rows (`modern_rp_full`, `modern_rp_no_bbc`,
   `native_wide`) computed exactly as in Phase A.
3. Apply `lobanov_normalize`.
4. Reshape so each source becomes a vector in 18-D (`{ph}_F1z`, `{ph}_F2z`, ordered).
5. Pairwise Euclidean distances → distance matrix (write CSV).
6. `scipy.cluster.hierarchy.linkage(matrix, method="ward")`; plot dendrogram.
7. Plot Lobanov vowel positions of all sources on one figure for visual check.

**Acceptance**: outputs written; dendrogram visibly groups {fry, lindsey, real_bc,
synth_bc} together; owner is visibly elsewhere (or the verdict explains why not).

---

## Phase C — Verdicts

**Script**: `scripts/accent_coach_phase0_6_compute_verdicts.py`
**Inputs**: Phase A + Phase B outputs.
**Outputs**: `docs/accent_coach_phase0_6_verdicts.{json,md}`.

| ID | Verdict label                              | Test                                                                                                                     | Pass condition         |
|----|--------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|------------------------|
| V1 | Owner vowel space is compressed            | `articulation_idx(owner)` from Phase A                                                                                   | ≤ 0.7 → CONFIRMED      |
| V2 | Owner clusters separately from natives     | In Ward dendrogram, owner's nearest neighbour distance > median pairwise distance among {fry, lindsey, real_bc, synth_bc} | binary                 |
| V3 | Distance reversal (per baseline)           | `d_lobanov(owner, baseline) > d_lobanov(deterding, baseline)` for each baseline in {full, no_bbc, native_wide}            | binary per baseline    |
| V4 | Cross-baseline robustness                  | V3 holds for ≥ 2 of 3 baselines                                                                                          | binary                 |

Verdict JSON shape:
```json
{
  "V1": {"value": 0.42, "pass": true, "threshold": 0.7},
  "V2": {"owner_nn_distance": ..., "native_median_distance": ..., "pass": true},
  "V3": {"modern_rp_full": {"d_owner": ..., "d_deterding": ..., "pass": true}, ...},
  "V4": {"baselines_passing": 3, "pass": true, "phase_e_required": false}
}
```

---

## Phase D — Findings doc

**File**: `docs/accent_coach_phase0_6_findings.md` (new; supersedes Phase 0.5 interpretation).

Sections (mandatory):
1. **TL;DR** — one paragraph, what changed and why.
2. **Vowel space plots** — embed `accent_coach_phase0_6_lobanov_vowel_space.png` and
   `accent_coach_phase0_6_dendrogram.png`.
3. **Geometry table** — Phase A output, formatted.
4. **Lobanov distances** — owner-vs-each-baseline and Deterding-vs-each-baseline.
5. **Verdicts** — Phase C JSON rendered as a table.
6. **Corrected interpretation** — explain the compressed-vowel-space mechanism in
   plain language, contrast with Phase 0.5's misleading raw-Hz reading.
7. **Phase 1 implications** — only state what changes for `rp_norms.py` if Phase E
   has run or is not needed; otherwise mark "pending BBC resolution".

Also add a top banner to [accent_coach_phase0_5_findings.md](accent_coach_phase0_5_findings.md):

```
> **SUPERSEDED** by Phase 0.6 (accent_coach_phase0_6_findings.md). The raw-Hz
> distance interpretation here is misleading for L2 speakers because it ignores
> vowel-space compression.
```

---

## Phase E — BBC contamination audit (conditional)

**Run only if** V4 = FALSE (cross-baseline robustness failed) or specifically requested.

- **E1**: Manual gender label on 10 random clips from
  `tts_output/modern_rp_corpus/bbc/clips/*.wav`. Output:
  `tts_output/modern_rp_corpus/bbc/speaker_audit.csv` with columns
  `[clip_id, gender, notes]`. Owner does this; Sonnet scaffolds the CSV.
- **E2**: Within-BBC F0 distribution (parselmouth) — flag clip_id ranges that look
  female (F0 > 165 Hz, F1 systematically higher).
- **E3**: Re-run Phase A/B with BBC split into `bbc_male` and `bbc_female`; update
  Phase D findings.

---

## Out of scope (DO NOT touch)

- `accent_coach/reference/rp_norms.py` — no edits; Phase 1 decision waits.
- `accent_coach/scoring/` or any per-utterance scoring code.
- Re-running Phase 0 bench.
- Re-extracting formants from raw audio (use existing CSVs only).

---

## Execution order

```
A → B → C → D → (E only if V4 = FALSE)
```

Each phase reads the previous phase's outputs. If any acceptance criterion fails,
stop and report; do not improvise alternative metrics. If a verdict comes out the
opposite of the prediction, report honestly — the goal is to find out what's true,
not to confirm the hypothesis.
