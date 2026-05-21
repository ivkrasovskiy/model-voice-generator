# Accent Coach — Phase 0.7 Plan

> **Status**: ready to execute. Closes out the open Phase 0.6 question (Phase E
> BBC audit) and then **pivots back to the product** — update `rp_norms.py` and
> re-run the Phase 0 bench against modern RP.
> **Builds on**:
> [`docs/accent_coach_phase0_6_findings.md`](accent_coach_phase0_6_findings.md),
> [`docs/accent_coach_phase0_5_findings.md`](accent_coach_phase0_5_findings.md).

---

## Where we are

Phase 0.6 ended mixed. Summary of verdicts:

| ID | Verdict                                | Result |
|----|----------------------------------------|--------|
| V1 | Owner vowel space compressed (≤ 0.7)   | **FAIL** (1.13). F1 compressed, F2 normal → hull area unchanged. |
| V2 | Owner clusters separately from natives | PASS |
| V3 | Distance reversal vs `modern_rp_full`   | FAIL (Δ = 0.046) |
| V3 | Distance reversal vs `modern_rp_no_bbc` | FAIL (Δ = 0.158) |
| V3 | Distance reversal vs `native_wide`      | PASS (Δ = 0.171) |
| V4 | Cross-baseline robustness (≥ 2/3)       | **FAIL** (1/3) |

**Critical observation not in the findings doc**: `modern_rp_no_bbc` already
excludes BBC and V3 still fails. So Phase E (BBC gender audit) cannot rescue
V3 — it can only explain whether the BBC-inclusive failure is a contamination
artefact. The deeper mechanism is that **Lobanov's per-speaker z-scoring strips
out vowel-space compression**, the very signal that's diagnostic of Slavic-L1
transfer. The owner's centralised vowels get stretched to native-sized "shape"
by Lobanov, accidentally landing near modern-RP positions for mid-vowels.

**Implication for the product**: the V3/V4 summary-distance question is mostly
academic. The accent coach scores **per phoneme**, not in aggregated Lobanov
space. Per-phoneme raw-Hz distance to modern RP — which Phase 0.5 already
showed — is the metric the product actually uses.

---

## Goals of Phase 0.7

1. **Close Phase E** quickly so the BBC-contamination question is no longer
   open. Downscope: skip manual labelling, use automatic F0 distribution.
2. **Update `rp_norms.py`** to modern RP (Fry + Lindsey, plus BBC-male if E
   says BBC is clean). This is the gating action that was deferred at the end
   of Phase 0.5.
3. **Re-run Phase 0 bench** against the updated norms. Verify per-phoneme
   diagnostics correctly flag the owner's /iː/, /æ/, /ʌ/ (the predicted
   Slavic-L1 problem phonemes).
4. **Write Phase 0.7 findings** that resolves the Phase 0 acceptance gate
   that was marked NOT MET in
   [`docs/accent_coach_plan.md`](accent_coach_plan.md).

What this plan does **not** do:
- Re-extract formants from raw audio (existing CSVs are authoritative).
- Build a new scoring methodology — the existing per-phoneme exponential
  decay in [`accent_coach/comparison/vowels.py`](../accent_coach/comparison/vowels.py)
  is fine. Only the reference values change.
- Re-engineer Lobanov for L2 (recognise it as a clustering tool, not a
  coaching metric, and move on).

---

## Phase E — BBC speaker audit (downscoped)

**Script**: `scripts/accent_coach_phase0_7_bbc_audit.py`
**Input**: `tts_output/modern_rp_corpus/bbc/clips/*.wav` (451 clips, all from
[`https://www.youtube.com/watch?v=vxxXQN76jDE`](https://www.youtube.com/watch?v=vxxXQN76jDE)).
**Outputs**:
- `tts_output/modern_rp_corpus/bbc/speaker_audit.csv` with columns
  `[clip_id, median_f0_hz, mean_f0_hz, voiced_fraction, gender_guess, transcript_excerpt]`.
- `docs/img/accent_coach_phase0_7_bbc_f0_hist.png` — histogram of median F0
  across the 451 clips, with 165 Hz divider line.
- Stdout summary: `n_male` / `n_female` / `n_ambiguous`.

**Algorithm**:
1. For each clip, use `parselmouth.Sound.to_pitch(time_step=0.01,
   pitch_floor=70, pitch_ceiling=400)`; compute median over voiced frames.
2. `gender_guess`:
   - `female` if `median_f0 > 175 Hz` AND `voiced_fraction > 0.3`
   - `male` if `median_f0 < 155 Hz`
   - `ambiguous` otherwise
3. Hand-check by listening to 5 clips at random from each `gender_guess`
   bucket (owner — Sonnet scaffolds a notebook cell that loads + plays
   them via `IPython.display.Audio`).

**Decision rule**:
- `n_female / n_total < 0.10` → BBC is essentially male. Keep it in
  modern_rp pool for Phase F.
- `0.10 ≤ n_female / n_total < 0.30` → BBC is mostly male but contaminated.
  Create `modern_rp_bbc_male` baseline (filter to `gender_guess == "male"`)
  and use that.
- `n_female / n_total ≥ 0.30` → BBC is not usable for male-target norms.
  Drop BBC entirely from modern_rp pool in Phase F.

**Acceptance**: CSV written, histogram plot saved, decision rule applied to
arrive at a definitive choice. Update
[`docs/accent_coach_phase0_6_findings.md`](accent_coach_phase0_6_findings.md)
with a one-paragraph "Phase E resolution" appendix citing the outcome.

---

## Phase F — Update `rp_norms.py` to modern RP

**File**: [`accent_coach/reference/rp_norms.py`](../accent_coach/reference/rp_norms.py).

**Source values**: pooled mean across {fry, lindsey} or {fry, lindsey,
bbc_male} depending on Phase E outcome. Values come from the same Phase 0.5
formants CSV that drove
[`docs/accent_coach_phase0_5_table.csv`](accent_coach_phase0_5_table.csv).
Re-derive here from `tts_output/modern_rp_corpus/formants.csv` to keep the
provenance explicit; do not depend on the Phase 0.5 table file.

**Changes**:
1. Add new constants `RP_VOWEL_F1_F2_MALE_MODERN` (and `_FEMALE_MODERN` if
   female data is available — Phase E may not give us female references; if
   not, mark `# TODO(cite)` and keep using Deterding-female until a future
   pass).
2. Re-point `get_rp_norms` to the `_MODERN` tables.
3. Keep the original `RP_VOWEL_F1_F2_MALE` (Deterding) **defined but unused**
   under a `_LEGACY` alias, with a comment pointing at
   `accent_coach_phase0_5_findings.md` for the supersession rationale.
4. Every numeric constant carries a citation comment:
   `# Mean over modern_rp_fry + modern_rp_lindsey; n_tokens=...; see
   accent_coach_phase0_7_findings.md`.

**Script** (small helper, not a long-lived utility):
`scripts/accent_coach_phase0_7_rebuild_norms.py` — reads
`tts_output/modern_rp_corpus/formants.csv`, applies a voiced-fraction filter
(≥ 0.6) and duration filter (≥ 50 ms), computes per-phoneme mean per source,
then pools across the chosen sources. Prints the new dict (formatted) that
the owner pastes into `rp_norms.py`. No automatic file edits to the norms
module — Sonnet writes the dict by hand so the citation comments are
deliberate.

**Acceptance**:
- `uv run ruff check accent_coach/ scripts/accent_coach_phase0_7_*.py` is
  clean.
- `uv run python -c "from accent_coach.reference.rp_norms import
  RP_VOWEL_F1_F2_MALE_MODERN; print(RP_VOWEL_F1_F2_MALE_MODERN['iː'])"`
  prints something close to (350, 1950) — within 30 Hz of the Phase 0.5
  `modern_rp` row.
- Existing unit tests for `rp_norms` still pass.

---

## Phase G — Re-run Phase 0 bench

**Goal**: prove that with modern-RP norms in place, the original Phase 0
acceptance gate from [`docs/accent_coach_plan.md:21-43`](accent_coach_plan.md#L21)
moves in the right direction. The gate is:

- Experiment A (synth BC vs RP norms): composite ≥ 85
- Experiment B (real BC vs RP norms): composite ≥ 80
- Experiment C (owner vs RP norms): composite ≤ 65
- Experiment D (owner vs synth BC): within ±5 of C

**Run**:
```bash
.venv/bin/python scripts/accent_coach_bench.py \
    --manifest tts_output/accent_coach/bench/phase0_7_manifest.csv \
    --run-id phase0_7
```

Reuse existing audio: `tts_output/accent_coach/bc_cal_50/`,
`tts_output/accent_coach/users/owner/`, real BC reference clips.
Manifest writer (small one-off): `scripts/accent_coach_phase0_7_make_manifest.py`
emits the bench CSV from those directories.

**Expected**:
- Owner composite rises from ~53 (Deterding) to **~60–68** against modern RP.
  Reason: he was being penalised for not speaking 1990s RP on /æ/, /ɛ/,
  /ʌ/. Modern RP targets are closer to where he actually lands on those
  vowels.
- The /iː/, /ʊ/, /ɔː/ gap from
  [`docs/accent_coach_plan.md:65-69`](accent_coach_plan.md#L65) should
  persist or widen — these are the genuine Slavic-L1 markers.
- BC composites should stay > 80 (the modern RP targets are *closer* to BC's
  own vowels than Deterding was, per Phase 0.5).

**Decision rule for Phase 1 readiness**:
- A ≥ 80 AND B ≥ 80 AND (B − C) ≥ 15 → Phase 0 gate cleared. Phase 1 (UI)
  unblocked.
- Otherwise → analyse which skill is dragging the composite. If it's
  rhythm/stress (likely — the TTS-timing artefact is unfixed), down-weight
  those to 5% each and re-score. Document the weight change.

**Acceptance**: `tts_output/accent_coach/bench/phase0_7/report.md` exists
and contains the four experiment scores plus per-skill breakdowns.

---

## Phase H — Findings doc + Phase 0 sign-off

**File**: `docs/accent_coach_phase0_7_findings.md`.

Sections:
1. **TL;DR** — one paragraph: BBC outcome, what changed in `rp_norms.py`,
   bench result, whether Phase 0 gate is met.
2. **Phase E resolution** — BBC audit numbers + decision.
3. **Modern RP table** — the new F1/F2 dict, with the Deterding column next
   to it for contrast.
4. **Bench results** — A/B/C/D scores, per-skill breakdown, comparison to
   the Phase 0 numbers in
   [`docs/accent_coach_plan.md:21-43`](accent_coach_plan.md#L21).
5. **Per-phoneme diagnostic dump for the owner** — top 5 phonemes by
   F1/F2 deviation, with the advice string from
   [`accent_coach/diagnostics/advice.py`](../accent_coach/diagnostics/advice.py).
   Eyeball-check that it reads like phonetician advice.
6. **Phase 1 entry criteria** — explicit go/no-go with the bench scores.

Also:
- Update [`docs/accent_coach_plan.md`](accent_coach_plan.md) section
  "Current status" to point at the new findings doc.
- Add `SUPERSEDED` banner to
  [`docs/accent_coach_phase0_findings.md`](accent_coach_phase0_findings.md)
  if Phase 0.7 changes its conclusions.

---

## Out of scope (do NOT touch)

- `vendor/` (IndexTTS install — sacred per
  [`CLAUDE.md`](../CLAUDE.md)).
- `scripts/indextts_*.py`, `scripts/posthoc_eval.py`,
  `scripts/build_podcast_ref.py`.
- Lobanov-related code in `accent_coach/diagnostics/lobanov.py` — keep it,
  it's useful for the clustering plot. Just stop treating Lobanov shape
  distance as a coaching metric.
- New normalisation schemes. Per-phoneme raw-Hz Z-against-RP is what the
  product uses; that's adequate.
- Re-extracting formants from audio. CSVs are authoritative.
- FastAPI / React / SQLite (Phase 1).

---

## Execution order

```
E → F → G → H
```

E and F are sequential (F depends on E's BBC inclusion decision). G depends
on F (it loads the updated norms). H summarises all of them.

If Phase E's outcome is `n_female / n_total ≥ 0.30` (BBC unusable), Phase F
proceeds with Fry+Lindsey only — there is no rerun of E. If Phase G's bench
fails the Phase 0 gate, Phase H is still written, but it documents the
failure and the next investigation rather than declaring victory.

---

## Estimated effort

| Phase | Task | Time |
|-------|------|------|
| E | parselmouth F0 over 451 clips + plot + listening check | ~45 min |
| F | rebuild script + paste new dict + verify imports | ~30 min |
| G | manifest + bench run + read report | ~30 min (bench is ~20 min wall time) |
| H | findings doc | ~45 min |
| Total | | ~2.5 hours |
