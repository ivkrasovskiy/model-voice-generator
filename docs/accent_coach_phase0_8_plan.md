# Accent Coach — Phase 0.8 Plan

> **Status**: ready to execute. Reframes Phase 0 around the one signal the
> pipeline measures reliably (formants) and drops the rest until they can be
> repaired.
>
> **Builds on**:
> [`docs/accent_coach_phase0_7_findings.md`](accent_coach_phase0_7_findings.md),
> [`docs/accent_coach_phase0_6_findings.md`](accent_coach_phase0_6_findings.md).

---

## Where we are

Phase 0.7 ended with three problems:

1. **Aspiration, rhythm, stress, intonation scorers are not trustworthy** —
   real BC scores 13.4 on aspiration, 17.5 on rhythm. These are scorer bugs,
   not speaker gaps. Until they are diagnosed the composite is noise.
2. **Vowel composite score collapses the meaningful range** — `exp(-d/sigma)`
   gives owner 76.9 and BC 79.5 despite owner having 150-250 Hz deviations on
   six phonemes vs BC's 20-80 Hz. The score hides the gap that the raw Hz
   table makes obvious.
3. **F0 / vocal-tract-length bias confounds raw Hz comparisons** — higher-F0
   speakers produce systematically higher formants independent of vowel
   quality. Lobanov removes this *and* erases L2 vowel-space compression
   (Phase 0.6). We need something between raw Hz and Lobanov.

The unresolved question from Phase 0.7's open section: **which normalisation
scheme preserves the owner-vs-modern-RP gap while removing the F0 confound,
and which one also keeps modern-RP speakers (Fry, Lindsey, BBC-male, real BC)
clustered together?**

Phase 0.8 answers that question and nothing else.

---

## Goals

1. **Pick a vowel-space distance metric** that satisfies the four-cluster
   criterion below. Implement it in `accent_coach/diagnostics/`.
2. **Replace the exponential-decay composite** with a representation that
   exposes the per-phoneme gap rather than hiding it (raw Hz table +
   piecewise-linear score thresholded against RP within-cluster variance).
3. **Document Phase 0.8 findings** with the chosen metric and the bench
   numbers it produces.

Explicitly **out of scope**:
- Aspiration, rhythm, stress, intonation scorers. Frozen for Phase 0.8.
  Re-opened in Phase 0.9 only if Phase 0.8 produces a clean vowel metric.
- New audio capture or formant re-extraction. The `formants.csv` from
  Phase 0.7 is authoritative.
- UI work, FastAPI, SQLite (still Phase 1).
- Lobanov as a coaching metric (keep the code for clustering plots; do not
  feed it into scoring).

---

## Inputs (frozen)

- `tts_output/accent_coach/bench/phase0_7/speaker_centroids.json` — F1/F2
  centroids per (speaker_group, phoneme) for the seven groups: real BC,
  synth BC, Fry, Lindsey, BBC-male, modern RP avg, owner.
- `tts_output/modern_rp_corpus/formants.csv` — raw per-token formant data
  (needed for per-speaker F0 estimation).
- `tts_output/accent_coach/users/owner/formants.csv` — owner per-token data.
- `accent_coach/reference/rp_norms.py:RP_VOWEL_F1_F2_MALE_MODERN` — current
  modern RP targets (Phase 0.7).
- Deterding 1997 numbers from `RP_VOWEL_F1_F2_MALE_LEGACY` — used as the
  "old RP" reference cluster.

---

## The four-cluster acceptance criterion

A normalisation scheme passes Phase 0.8 **only if** the chosen distance
metric produces all four of the following simultaneously:

| Cluster check | Required |
|---------------|----------|
| C1: Modern-RP speakers cluster together | mean pairwise dist {Fry, Lindsey, BBC-male, real BC} ≤ 50 (in the metric's units) |
| C2: Modern RP separated from Deterding | dist(modern_rp_avg, Deterding) ≥ 1.5× mean within-cluster dist from C1 |
| C3: Owner separated from modern RP | dist(owner, modern_rp_avg) ≥ 2× mean within-cluster dist from C1 |
| C4: Owner separated from Deterding | dist(owner, Deterding) ≥ 1.5× mean within-cluster dist from C1 — confirms owner ≠ "speaks old RP" |

C3 is the headline: the metric must put the owner visibly outside the modern-RP
cluster. C4 rules out the failure mode where a metric flatters the owner by
calling him an old-RP speaker.

---

## Hypotheses to test

Five normalisation schemes, ordered cheapest-first. Each is implemented as a
small pure function in `accent_coach/diagnostics/` that takes centroids and
returns transformed centroids; the four-cluster check above is a single
`evaluate_clustering(centroids_dict)` helper.

### H1 — F0-based VTL scaling

Estimate vocal-tract length from mean F0 per speaker (`VTL ∝ 1 / F0`); scale
each formant by `F0_speaker / F0_ref` where `F0_ref = mean F0 across modern
RP speakers`. Result: speakers with higher F0 have their formants pulled down,
removing ~5-10% of the formant bias attributable to vocal tract size.

**Module**: `accent_coach/diagnostics/vtl_normalize.py:vtl_normalize`

**Expected**: tightens the modern-RP cluster slightly (BBC-male and Lindsey
move closer to Fry). Owner's deviations shrink by ~10-20 Hz on average; the
150-250 Hz gaps survive.

**FAIL if**: any of C1-C4 not met. Specifically, if owner's distance to
modern RP shrinks below the C3 threshold, H1 is over-correcting (it's
removing real accent signal, not just VTL bias).

### H2 — Nearey log-mean normalisation

`F'_i = log(F_i) - mean(log(F_iː, F_ɑː, F_uː))` per speaker, using only the
three corner vowels in the centering set. Removes overall scale; keeps
relative positions and compression intact.

**Module**: `accent_coach/diagnostics/nearey_normalize.py:nearey_corner_normalize`

**Expected**: stronger correction than H1. Modern-RP cluster tightens
meaningfully. Owner stays outside if his accent shows up as relative
*shape* differences (centralised /iː/, monophthong-like /əʊ/), but is
flattered if his accent is purely VTL bias (unlikely).

**FAIL if**: C3 not met (owner pulled into modern-RP cluster). This would
mean owner's deviation is dominated by overall vocal-tract scale rather than
articulation — interesting finding but means we cannot use Nearey for
coaching.

### H3 — Anchor-relative coordinates (primary candidate)

For each speaker, compute the centroid of /iː ɑː uː/ as a local origin and
the iː-uː F2 distance as a local scale unit. Project every other vowel into
this coordinate frame: `x' = (F2 - F2_origin) / F2_scale`,
`y' = (F1 - F1_origin) / F1_scale`. Pure geometry. No mean variance
scaling. Closest to how phoneticians read vowel quadrilaterals.

**Module**: `accent_coach/diagnostics/anchor_normalize.py:anchor_normalize`

**Expected**: best separator. Modern RP cluster tight (corner vowels are
stable across native speakers). Owner's centralised /iː/ shifts the anchor
itself, but his non-corner vowels (/æ/, /eɪ/, /əʊ/) should still land far
from modern RP because their *relative* position to the corners is wrong.

**FAIL if**: anchor itself is the problem — i.e. owner's /iː ɑː uː/
centroids are so far from modern RP's that everything else looks normal once
projected. In that case Phase 0.8 conclusion is "the corner vowels *are* the
accent" and we need a different anchor set (perhaps {STRUT, FOOT, NURSE} as
secondary anchors).

### H4 — Bark/ERB-scale Euclidean distance

Convert F1/F2 to Bark (`Bark = 13 arctan(0.00076 F) + 3.5 arctan((F/7500)²)`)
or ERB scale; compute distances in that space. Not a normalisation per se —
it's a perceptually-motivated rescaling that compresses high-F2 differences
that are partly VTL bias.

**Module**: `accent_coach/diagnostics/bark_distance.py:bark_distance`

**Expected**: helps a little. Probably not enough alone to clear C1-C4 but
useful as a tiebreaker if two other hypotheses are close.

**FAIL if**: distances behave qualitatively the same as raw Hz (Bark is
nearly linear with Hz up to ~500 Hz, so we expect modest changes). If H4
just rescales without changing relative orderings, skip the deeper analysis
and don't write it up in the findings.

### H5 — F0-binned norms (fallback only)

Split modern RP norms by speaker F0 (110/140/170 Hz bins). Compare each test
speaker against the bin matching their own F0. Skip if H1-H3 produce a
working metric; the sample sizes are small (~1.8k clips pooled) and binning
makes them smaller.

**Module**: not implemented in Phase 0.8 unless H1-H3 all fail.

**FAIL if**: any bin has fewer than 100 tokens for any phoneme — the
per-phoneme variance estimate becomes unreliable.

---

## Replacement composite score

Independent of normalisation choice, replace the per-phoneme exponential
decay with a **piecewise-linear score** tied to the within-modern-RP
variance (call it σ_RP, computed empirically from the four-speaker cluster
in C1):

| Per-phoneme deviation | Score |
|-----------------------|-------|
| ≤ 1·σ_RP | 100 (native range) |
| 1·σ_RP to 2·σ_RP | linear 100 → 70 (lightly accented) |
| 2·σ_RP to 4·σ_RP | linear 70 → 30 (strongly accented) |
| > 4·σ_RP | 30 (out-of-distribution) |

The vowel composite is the mean across phonemes, weighted equally. No
exponential flattening.

**Module**: `accent_coach/comparison/vowels.py:score_vowels_piecewise`
(new function; do not remove the existing exponential one yet — both run
in parallel for Phase 0.8 so we can compare).

**Acceptance**: with the chosen normalisation, the piecewise score gives
owner a clearly lower number than real BC (gap ≥ 20 points), not the 2.5
gap that the exponential gives now.

---

## Execution order

```
H1 → H3 → H2 → H4 → (H5 only if needed)
```

H1 is the cheapest sanity check. H3 is the primary candidate per the
Phase 0.7 discussion. H2 is a backstop in case H3's anchors are themselves
shifted by the owner's accent. H4 is a perceptual cross-check. H5 is a last
resort if everything else fails.

After each hypothesis: run `evaluate_clustering`, log results to
`tts_output/accent_coach/phase0_8/<hypothesis>.json`, then decide whether to
stop or continue per the rules below.

---

## STOP rules

This is what "we are done" looks like, and what "this is going wrong" looks
like.

### GREEN — stop here, write up Phase 0.8 findings

- Any hypothesis passes all four cluster checks C1-C4.
- Piecewise score with that normalisation gives owner ≤ 60 and real BC ≥ 80.
- Per-phoneme Hz table shows the predicted Slavic-L1 markers (/əʊ/, /æ/,
  /ɑː/, /iː/, /ɛ/, /ʌ/) as the largest deviations for owner, with smaller
  deviations for BC.

→ Document the chosen scheme in `docs/accent_coach_phase0_8_findings.md`.
Update `rp_norms.py` and `accent_coach/comparison/vowels.py`. Move to
Phase 0.9 (re-open aspiration scorer) or Phase 1 (UI), depending on owner
priority.

### YELLOW — partial pass, judgement call

- One hypothesis passes C1, C2, C3 but fails C4 (owner ≈ Deterding).
- One hypothesis passes C1, C3, C4 but fails C2 (modern RP ≈ Deterding).
- Two hypotheses pass three criteria each but on different criteria.

→ Stop and ask the owner. Either pick the best-three-of-four metric and
explicitly accept the missing criterion, or run H5 as the tiebreaker.
Do **not** silently pick the best-looking metric — the failure mode here is
selecting on noise.

### RED — stop and reconsider the whole approach

Trigger any of the following and **stop immediately, do not run further
hypotheses**:

1. **No hypothesis passes C1.** Modern-RP speakers do not cluster under any
   normalisation. Implication: the formant extraction pipeline is producing
   inconsistent values across sources (alignment bug, Praat parameter drift,
   different sampling rates). This is a data quality problem, not a metric
   problem. Phase 0.8 cannot proceed — escalate to a Phase 0.8a data audit.

2. **All hypotheses pass C1, C2 but fail C3.** Owner indistinguishable from
   modern RP under every normalisation. Implication: either the owner has
   converged on modern RP and the project's premise (he has a Slavic accent)
   is wrong for vowels, OR the F1/F2 extraction is mis-aligning his vowels
   to wrong phonemes. Listen to 5 owner clips and verify the
   `phoneme_alignment.csv` matches what he actually said. If alignment is
   correct → conclude "vowels alone are insufficient to coach this owner,
   need consonant/duration features" → escalate to Phase 0.9 with aspiration
   scorer as the lead.

3. **C3 passes but the per-phoneme dump shows nonsensical advice.** E.g.
   "raise tongue on /iː/" when the owner's /iː/ F1 is already at 320 Hz
   (already high-tongue native value). Means the advice generator is reading
   the wrong column or the centroid is corrupted. Stop, debug, do not write
   up findings.

4. **Owner's F0 is outside the male range (≤ 100 or ≥ 180 Hz).** All
   normalisation hypotheses assume male reference norms. If the owner's
   audio extracts an F0 in the female/ambiguous range there's a recording
   or pitch-detection bug. Re-extract before running any hypothesis.

5. **Bench run wall-clock > 1 hour.** Each hypothesis should take ≤ 5 min
   wall-clock (centroids JSON is small, transformations are pure NumPy).
   Anything longer means accidental re-extraction or recomputation of
   formants — stop, find the loop.

---

## File layout

```
accent_coach/diagnostics/
  vtl_normalize.py        # H1
  nearey_normalize.py     # H2
  anchor_normalize.py     # H3 (primary)
  bark_distance.py        # H4
  cluster_eval.py         # evaluate_clustering(centroids_dict) -> dict

accent_coach/comparison/
  vowels.py               # add score_vowels_piecewise (keep existing)

scripts/
  accent_coach_phase0_8_run.py  # one entrypoint; loops H1..H4, writes JSON

tts_output/accent_coach/phase0_8/
  h1_vtl.json
  h2_nearey.json
  h3_anchor.json
  h4_bark.json
  cluster_metrics_summary.csv
  per_phoneme_distances_<winner>.csv

docs/
  accent_coach_phase0_8_plan.md      # this file
  accent_coach_phase0_8_findings.md  # written at end
  img/
    accent_coach_phase0_8_vowel_space_<winner>.png
    accent_coach_phase0_8_cluster_dendro.png
```

---

## Acceptance for Phase 0.8 sign-off

Phase 0.8 is done when:

1. `accent_coach/diagnostics/cluster_eval.py:evaluate_clustering` is
   implemented and tested on the seven speaker groups.
2. At least one of H1-H4 passes the four-cluster criterion (or H5 was run
   and passes).
3. `score_vowels_piecewise` is implemented and reproduces the gap on the
   bench.
4. `docs/accent_coach_phase0_8_findings.md` exists, names the winning
   metric, shows the cluster numbers, and explicitly addresses whether
   Phase 0 gate is now clearable on vowels alone.
5. `uv run ruff check accent_coach/ scripts/accent_coach_phase0_8_*.py` is
   clean.

If none of H1-H5 pass and a RED rule above fires, Phase 0.8 is also "done"
— the findings doc documents the failure mode and the recommended next
investigation. Do not paper over a failure.

---

## Estimated effort

| Phase | Task | Time |
|-------|------|------|
| Setup | `cluster_eval.py` + load centroids JSON | 30 min |
| H1 | `vtl_normalize.py` + run + log | 30 min |
| H3 | `anchor_normalize.py` + run + log | 45 min |
| H2 | `nearey_normalize.py` + run + log | 30 min |
| H4 | `bark_distance.py` + run + log | 30 min |
| Piecewise score | `score_vowels_piecewise` + bench rerun | 45 min |
| Findings | write-up + dendrogram + vowel-space plot of winner | 1 hr |
| **Total** | | **~4 hr** |

H5 adds another ~1 hr only if reached. If a RED rule fires the budget caps
at whatever was spent up to the trigger.
