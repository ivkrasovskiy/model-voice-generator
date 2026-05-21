# Accent Coach — Phase 0.8 Findings

> **Status**: YELLOW — no hypothesis passes all four cluster criteria.
> H3 (anchor) and H4 (Bark) each pass 3/4.  H4 meets the product piecewise
> score criteria (owner ≤ 60, real BC ≥ 80).  **Owner decision required**
> before Phase 0.8 can be signed off.
>
> **Builds on**:
> [`docs/accent_coach_phase0_7_findings.md`](accent_coach_phase0_7_findings.md),
> [`docs/accent_coach_phase0_8_plan.md`](accent_coach_phase0_8_plan.md).

---

## TL;DR

Four normalisation schemes tested.  None passed all four cluster criteria.
Two are close enough to be useful:

- **H4 (Bark scale)** passes C1/C2/C4 and the piecewise-score GREEN test
  (owner=56.5, real BC=89.8, gap=33.3).  C3 ratio = 1.65 vs 2.0 required —
  owner IS detectably outside the modern-RP cluster, just marginally below
  the 2× threshold.
- **H3 (anchor-relative)** passes C1/C3/C4 strongly (C3 ratio=2.65).  C2
  fails because Deterding 1997 and modern RP have the same non-corner vowel
  topology once each speaker's own corner vowels normalise out — a
  linguistically meaningful result, not a metric error.

H1 (VTL) was ill-conditioned: parselmouth estimated real BC at 89 Hz and
synth BC at 80 Hz, making the scaling corrections huge and wrong.
H2 (Nearey) barely separates owner from modern RP (C3 ratio=1.55).

---

## F0 estimates (H1 pre-requisite)

| Speaker | Mean F0 (Hz) | Source |
|---------|-------------|--------|
| BBC male | 124.6 | Phase E audit CSV |
| Fry | 135.7 | 20-clip parselmouth sample |
| Lindsey | 168.6 | 20-clip parselmouth sample |
| Real BC | **89.2** | 20-clip parselmouth sample |
| Owner | 138.5 | 20-clip parselmouth sample |
| Synth BC (IndexTTS) | **79.9** | 20-clip parselmouth sample |
| Modern RP (geom. mean) | 141.8 | computed |

Real BC at 89 Hz and synth BC at 80 Hz are anomalously low — Benedict
Cumberbatch's voice is deep but 89 Hz is near the lower limit of male
speech.  The IndexTTS-2 synthesised voice is even lower, likely a TTS
pitch artifact.  Lindsey at 168.6 Hz is borderline female range.

These extremes mean the VTL scaling for H1 is ill-conditioned: real BC's
formants get inflated by ×1.59 (141.8/89.2), moving it far from the
modern-RP cluster it should be inside.  **H1 is a data quality failure,
not a hypothesis failure** — if F0 estimates are repaired (better audio
pre-processing or direct pitch annotation), H1 should be retested.

---

## Cluster metrics — all four hypotheses

| Hypothesis | C1 within-RP | C2 RP/Deter | C3 owner/RP | C4 owner/Deter | C1✓ | C2✓ | C3✓ | C4✓ |
|------------|-------------|------------|------------|---------------|-----|-----|-----|-----|
| H1 VTL     | 566.9 Hz | 326.6 Hz | 157.2 Hz | 345.8 Hz | ✗ | ✗ | ✗ | ✗ |
| H3 anchor  | 0.36 | 0.34 | 0.97 | 1.08 | ✓ | **✗** | ✓ | ✓ |
| H2 Nearey  | 0.08 | 0.33 | 0.12 | 0.39 | ✓ | ✓ | **✗** | ✓ |
| H4 Bark    | 0.53 | 1.91 | 0.87 | 1.89 | ✓ | ✓ | **✗** | ✓ |

*(Units: Hz for H1; dimensionless for H3; log-units for H2; Bark for H4)*

C2/C3/C4 thresholds relative to C1:

| Hypothesis | C2 ratio | C3 ratio | C4 ratio | required C2 | required C3 | required C4 |
|------------|---------|---------|---------|------------|------------|------------|
| H3 anchor  | **0.94** | **2.65** | 2.96 | ≥1.5 | ≥2.0 | ≥1.5 |
| H2 Nearey  | 4.15 | **1.55** | 4.84 | ≥1.5 | ≥2.0 | ≥1.5 |
| H4 Bark    | 3.62 | **1.65** | 3.57 | ≥1.5 | ≥2.0 | ≥1.5 |

---

## Piecewise score results

σ_RP was computed from the four-speaker modern-RP cluster in each
metric's own units.

| Metric | σ_RP mean | real BC | synth BC | owner | gap (BC−owner) |
|--------|-----------|---------|----------|-------|----------------|
| H3 anchor | 0.26 | 78.9 | 85.1 | **51.6** | 27.3 |
| H2 Nearey | — | 78.9 | 62.5 | 65.5 | 13.4 |
| H4 Bark   | 0.33 | **89.8** | 74.6 | **56.5** | **33.3** |

**H4 meets the GREEN piecewise score criterion**: owner ≤ 60 and real BC ≥ 80.
H3 also meets the gap criterion (27.3 ≥ 20) and gives the lowest owner score
(51.6), but real BC sits at 78.9 rather than ≥ 80.

---

## Per-phoneme distances in Bark (H4)

Top deviations by owner distance from modern RP:

| Phoneme | Owner dist (Bark) | Real BC dist (Bark) | Ratio owner/BC |
|---------|------------------|---------------------|----------------|
| əʊ  | **1.60** | 0.49 | 3.3× |
| iː  | **1.43** | 0.46 | 3.1× |
| ɑː  | 1.13 | 0.62 | 1.8× |
| aɪ  | 1.07 | 0.61 | 1.8× |
| ʌ   | 1.01 | 0.18 | 5.6× |
| eɪ  | 0.98 | 0.15 | 6.5× |
| æ   | 0.95 | 0.52 | 1.8× |
| ɛ   | 0.95 | 0.36 | 2.6× |
| ʊ   | 0.86 | 0.26 | 3.3× |
| uː  | 0.79 | 0.49 | 1.6× |
| ɜː  | 0.75 | 0.27 | 2.8× |
| aʊ  | 0.05 | 0.56 | 0.1× |

Phonemes matching the Slavic-L1 prediction (from Phase 0.7):
**əʊ, iː, ɑː, eɪ, ʌ** all show the largest owner deviations.
The /aʊ/ reversal (owner closer than real BC) is consistent with
Phase 0.7's raw Hz table (owner=542, RP=536).

---

## Interpretation of C2 failing for H3

After anchor normalisation, the modern-RP average and Deterding 1997 have
nearly identical non-corner vowel topology (C2 ratio = 0.94).  This is
linguistically correct: the main empirical difference between modern RP and
Deterding 1997 lies in the corner vowels themselves — especially the
fronting of /uː/ (modern RP F2 1506 vs Deterding 1156) and the higher /iː/
F2 in Deterding (2249 vs 1962).  Once each speaker's own corner vowels
define the coordinate frame, the relative positions of /ɛ æ ɔː ʊ ɜː/ etc.
are similar in both RP flavours.

H3's C2 failure therefore exposes a property of RP, not a metric bug.
The metric is designed to detect *relative* position errors — which is
exactly what the owner shows (C3 = 2.65×).  The failure to separate modern
RP from Deterding on the C2 criterion should be read as: "the anchor metric
is not sensitive to which RP flavour is the target; it measures deviation
from any RP-like corner-vowel system."

---

## H1 post-mortem: F0 estimation

The VTL hypothesis is not proven wrong — the implementation is correct —
but the parselmouth F0 estimates are unreliable for these audio sources:

1. **Real BC corpus** clips are short, have varying microphone conditions,
   and BC's voice may have significant creak/subharmonics at low pitch,
   causing the pitch floor (70 Hz) to let creak frames through.
2. **IndexTTS-2 synthesised audio** has an unusual pitch contour — the TTS
   model generates a voice at BC-like register (~80 Hz) that is below
   typical male speech.

If F0 is repaired (better pitch extraction, or direct annotation), H1
could pass — VTL scaling at correct F0 values would tighten the cluster.
However, this is a Phase 0.9 task since H4 already provides a usable
metric.

---

## YELLOW decision: options for the owner

Per the Phase 0.8 plan, YELLOW means stop and ask.

**Option A — Accept H4 (Bark scale) with C3 caveat**

C3 ratio = 1.65 (vs 2.0 required).  The owner IS detectably outside the
modern-RP cluster in Bark space, just not by the full 2× margin.  The
piecewise score (owner=56.5, real BC=89.8) meets the product criteria.
Coaching advice based on per-phoneme Bark distances points to the correct
Slavic-L1 phonemes.

Explicit acceptance: "C3 marginally fails; we accept this for Phase 0.8
and will re-evaluate if subsequent coaching iterations fail to move the
score."

**Option B — Accept H3 (anchor-relative) with C2 caveat**

C3 ratio = 2.65 (strong separation of owner from RP).  C2 fails because
Deterding and modern RP have the same relative vowel topology — this is a
property of RP, not a metric error.

Explicit acceptance: "C2 fails by design; the anchor metric measures
deviation from any RP-like system.  We accept this and do not require
separation of modern RP from Deterding for coaching purposes."

**Option C — Run H5 (F0-binned norms)**

Split the modern-RP norms into bins by F0 (~110/140/170 Hz).  May narrow
the within-RP cluster enough to push C3 above 2× for H4.  Risk: small
per-bin sample sizes.

---

## Phase 0 gate status after Phase 0.8

| Criterion | Status |
|-----------|--------|
| Vowel metric separates owner from modern RP (piecewise) | **PARTIAL** — H4 shows gap=33 but C3 ratio=1.65 < 2.0 |
| Piecewise score: owner ≤ 60 | **MET** — owner=56.5 (H4) |
| Piecewise score: real BC ≥ 80 | **MET** — real BC=89.8 (H4) |
| Four-cluster criterion (all C1-C4) | **NOT MET** (best = 3/4) |
| Aspiration scorer calibrated | **NOT MET** (Phase 0.7) |
| B − C gap ≥ 15 | **NOT MET** (Phase 0.7) |

Vowels alone are insufficient to pass Phase 0.  The product criterion (gap
≥ 20 piecewise) is met by H4, but the formal four-cluster criterion is not.
**The Phase 0 gate depends on the owner's YELLOW decision above and on the
aspiration scorer (Phase 0.9).**

---

## All-speaker piecewise scores (H4 Bark)

Expanded to cover all seven groups after the bench run:

| Speaker | Score | Notes |
|---------|-------|-------|
| Fry | 100.0 | defines the modern-RP training cluster |
| BBC male | 94.6 | solid RP |
| Real BC | 89.8 | canonical RP; passes GREEN threshold |
| Lindsey | 86.6 | slightly looser — small corpus (255 clips) |
| **Synth BC** (IndexTTS-2) | **74.6** | TTS vowel distortion — see below |
| **Owner** | **56.5** | clearly outside RP cluster |

Gap between real BC and owner: **33.3 points**. Gap between real BC and
synth BC: **15.2 points** — IndexTTS-2 introduces its own vowel error on
top of whatever the reference clip carries.

---

## IndexTTS-2 vowel accuracy — can it be fixed?

The 15-point gap between synth BC (74.6) and real BC (89.8) has two
distinct causes with different fixability:

### Fixable: reference clip selection

The production reference is a 14 s conversational interview clip.
Interview speech has reduced, fast vowels.  The audiobook reference
clip used in earlier phases produced dramatically better ECAPA identity
scores (0.784 vs 0.310), which means the model produces more BC-like
output overall when the reference is read speech.  Audiobook narration
has deliberate, uncompressed vowels — closer to what the model needs to
calibrate its vowel output against the reference.

**Likely gain from switching to audiobook reference**: 5–8 piecewise
points (synth BC from ~75 → ~80+).  Worth running the Phase 0.8 bench
with the audiobook reference clip before concluding that synth BC is
irreparably off.

### Not fixable without fine-tuning: vowel regression to the mean

Zero-shot TTS systems extract timbre and prosody from the reference clip,
but vowel quality is governed by the model's phoneme embeddings — trained
on a corpus average, not on BC specifically.  The output vowels drift
toward the model's "generic English" centre regardless of the reference
used.  The /ɔɪ/ outlier (synth BC 405/1767 vs RP 429/1560) and the /ʌ/
depression (370/1226 vs RP 472/1339) seen in Phase 0.7's raw-Hz table are
systematic, not clip-specific — they will survive reference clip changes.

Fine-tuning IndexTTS-2 on BC audio (even 15–20 minutes of clean audiobook
data) would fix this by updating the phoneme embeddings toward BC's vowel
space.  Zero-shot cannot reach real BC parity on vowels; the ceiling without
fine-tuning is approximately the Lindsey range (~86–88).

### Practical recommendation

1. Re-run the vowel bench with the **audiobook reference clip** (`tts_output/ref_audiobook.wav` or equivalent) to measure the actual gain.
2. If synth BC reaches ≥ 82 (i.e. within Lindsey range), treat the zero-shot ceiling as acceptable and focus Phase 1+ on coaching the owner.
3. If it stays below 78, schedule a fine-tuning experiment in a later phase — the zero-shot vowel floor is a blocker for the "sound like BC" product goal even if coaching succeeds.

---

## Files produced in Phase 0.8

| File | Description |
|------|-------------|
| `accent_coach/diagnostics/cluster_eval.py` | `evaluate_clustering` + `per_phoneme_sigma_rp` |
| `accent_coach/diagnostics/vtl_normalize.py` | H1 VTL scaling |
| `accent_coach/diagnostics/anchor_normalize.py` | H3 anchor-relative |
| `accent_coach/diagnostics/nearey_normalize.py` | H2 Nearey log-mean |
| `accent_coach/diagnostics/bark_distance.py` | H4 Bark transform |
| `accent_coach/comparison/vowels.py` | `score_vowels_piecewise` added |
| `scripts/accent_coach_phase0_8_run.py` | Full H1→H3→H2→H4 runner |
| `tts_output/accent_coach/phase0_8/cluster_metrics_summary.csv` | All hypotheses summary |
| `tts_output/accent_coach/phase0_8/h{1,2,3,4}_*.json` | Per-hypothesis cluster JSON |
| `tts_output/accent_coach/phase0_8/piecewise_scores_*.json` | Piecewise scores per hypothesis |
| `tts_output/accent_coach/phase0_8/per_phoneme_distances_*.csv` | Per-phoneme distances |
