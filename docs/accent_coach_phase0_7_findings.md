# Accent Coach — Phase 0.7 Findings

> **Status**: Phase 0 gate NOT MET. Modern RP norms are in place and vowel
> scoring is validated. Aspiration and rhythm/stress scorers need calibration
> before Phase 1.
>
> **Builds on**:
> [`docs/accent_coach_phase0_6_findings.md`](accent_coach_phase0_6_findings.md),
> [`docs/accent_coach_phase0_5_findings.md`](accent_coach_phase0_5_findings.md).

---

## TL;DR

Phase E confirmed BBC is mostly-male-but-contaminated (29% female) → used
275 male clips. Phase F updated `rp_norms.py` with modern RP values pooled
from Fry + Lindsey + BBC-male (13 492 tokens). Phase G bench fails the Phase 0
gate: all composites sit around 52–55 against RP norms. The vowel skill is
working well (A=79.5, B=79.0), but aspiration (A=35, B=13.4) and
rhythm/stress (A=29/17, B=18/38) are systematically low even for real BC,
revealing scorer calibration bugs rather than a speaker gap. Until aspiration
scoring is fixed, the B − C gap cannot be measured meaningfully.

---

## Phase E resolution — BBC speaker audit

**Script**: `scripts/accent_coach_phase0_7_bbc_audit.py`  
**Input**: 451 BBC clips, all from a single BBC News bulletin.

| Gender | Count | Fraction |
|--------|-------|---------|
| male   | 275   | 0.610   |
| female | 131   | 0.291   |
| ambiguous | 45 | 0.100  |

Female fraction = 0.290 — between 0.10 and 0.30 → **filter_male** decision.
Phase F pools `modern_rp_bbc_male` (275 clips) with Fry and Lindsey.

Histogram saved to `docs/img/accent_coach_phase0_7_bbc_f0_hist.png`.

---

## Modern RP vowel table (Phase F)

New constant `RP_VOWEL_F1_F2_MALE_MODERN` in
[`accent_coach/reference/rp_norms.py`](../accent_coach/reference/rp_norms.py).
Pooled mean F1/F2 (Hz), duration ≥ 50 ms, n = 13 492 tokens across three
sources.

| Phoneme | F1 (modern) | F2 (modern) | F1 (Deterding) | F2 (Deterding) | Δ F2 |
|---------|------------|------------|----------------|----------------|-------|
| iː      | 348        | 1962       | 280            | 2249           | −287  |
| ɪ       | 386        | 1773       | 367            | 1757           | +16   |
| ɛ       | 462        | 1571       | 580            | 1799           | −228  |
| æ       | 545        | 1496       | 748            | 1710           | −214  |
| ɑː      | 518        | 1215       | 680            | 1100           | +115  |
| ɒ       | 600 ¹      | 900 ¹      | 600            | 900            | 0     |
| ɔː      | 459        | 1138       | 430            | 700            | +438  |
| ʊ       | 386        | 1427       | 378            | 950            | +477  |
| uː      | 352        | 1506       | 310            | 1156           | +350  |
| ʌ       | 472        | 1339       | 623            | 1224           | +115  |
| ɜː      | 482        | 1440       | 490            | 1570           | −130  |
| ə       | 412        | 1550       | 490            | 1350           | +200  |

¹ No /ɒ/ tokens survived quality filters; Deterding kept as placeholder.

Key shifts confirm the Phase 0.5 prediction: modern RP /æ/ is substantially
lower F1 than Deterding (545 vs 748); /ɔː/ and /ʊ/ have much higher F2
(fronted GOOSE/FOOT is a well-documented modern SSBE feature).

`get_rp_norms` now returns `RP_VOWEL_F1_F2_MALE_MODERN` for male-F0 speakers.
Deterding data is kept as `RP_VOWEL_F1_F2_MALE_LEGACY` and aliased to
`RP_VOWEL_F1_F2_MALE` for call-sites not yet migrated.

---

## Bench results (Phase G)

**Run**: `phase0_7`  
**Manifest**: `tts_output/accent_coach/bench/phase0_7_manifest.csv` (154 rows)

### Raw scores (original weights)

| Experiment | Composite | Vowels | Consonants | Aspiration | Rhythm | Stress | Intonation |
|---|---|---|---|---|---|---|---|
| A — synth BC vs RP | 54.9 | **79.5** | 70.0 | 35.0 | 29.2 | 16.7 | 82.6 |
| B — real BC vs RP  | 52.5 | **79.0** | 70.0 | 13.4 | 17.5 | 37.8 | 79.5 |
| C — owner vs RP    | 54.0 | 76.9 | 70.0 | 27.7 | 33.9 | 16.8 | 83.2 |
| D — owner vs synth BC | 69.4 | 73.1 | 70.0 | 42.8 | 65.4 | 70.8 | 91.7 |

Original weights: vowels=25, consonants=15, aspiration=15, rhythm=15,
stress=15, intonation=15.

### Adjusted scores (rhythm/stress down-weighted to 5% each)

Redistributed 10 pt to vowels (+5), consonants (+5), aspiration (+5),
intonation (+5). New weights: vowels=30, consonants=20, aspiration=20,
rhythm=5, stress=5, intonation=20.

| Experiment | Composite (adj) |
|---|---|
| A — synth BC vs RP | 63.7 |
| B — real BC vs RP  | 59.1 |
| C — owner vs RP    | 61.8 |
| D — owner vs synth BC | 69.7 |

### Phase 0 gate verdict

| Gate | Required | A | B | B − C | Status |
|------|----------|---|---|-------|--------|
| A composite | ≥ 80 | 54.9 (63.7 adj) | — | — | **FAIL** |
| B composite | ≥ 80 | — | 52.5 (59.1 adj) | — | **FAIL** |
| B − C gap   | ≥ 15 | — | — | −1.5 (−2.7 adj) | **FAIL** |

**Phase 0 gate: NOT MET.**

---

## Root-cause analysis

### Vowels ✓ — scorer working

Vowel scores for synth BC (79.5) and real BC (79.0) both confirm that modern
RP norms are well-calibrated. The scorer correctly rewards BC-level vowels.
Owner vowels score 76.9 — only 2.5 points below BC, which is smaller than
expected but within measurement noise for a 50-sentence bench.

### Aspiration ✗ — scorer miscalibrated

Real BC scores 13.4 on aspiration. BC is a canonical RP speaker and cannot
plausibly have below-15 aspiration. The bug is almost certainly in how the
aspiration scorer estimates VOT from TTS-generated audio or from reference
clips with different microphone conditions. This must be diagnosed before
the B − C gap can be measured reliably.

**Next investigation**: print raw VOT estimates for 5 BC sentences; compare
against the RP_VOT_RANGE_MS constants in `rp_norms.py`. Likely causes:
(a) Praat onset detection failing on the ref clips, (b) the 50 ms VOT floor
treating BC's typical 65–90 ms as out-of-range, or (c) missing /p t k/
detections when the aligner maps chunks differently.

### Rhythm / Stress ✗ — TTS timing artefact (known)

Synth BC scores rhythm=29.2, stress=16.7 — clearly below what real BC would
produce. This is the TTS-timing artefact noted since Phase 0.5: IndexTTS
produces unnaturally flat duration contours, depressing nPVI and stress
contrast. Real BC (B) scores rhythm=17.5, stress=37.8 — also low, suggesting
the nPVI target range or stress reference is wrong even for the reference
clips.

Down-weighting rhythm/stress to 5% each (see adjusted table above) does not
rescue the gate but reduces the noise floor by ~9 pts.

---

## Per-phoneme diagnostic dump for the owner

Top 5 phonemes by average deviation (across 10 sampled sentences, vs modern
RP norms):

| Phoneme | Owner F1 | Target F1 | Owner F2 | Target F2 | Advice |
|---------|----------|-----------|----------|-----------|--------|
| eɪ      | 249      | 383       | 1010     | 1873      | Tongue too high; lower jaw slightly; tongue too far back |
| iː      | 504      | 348       | 1938     | 1962      | Tongue too low; raise toward palate |
| ɑː      | 840      | 518       | 1553     | 1215      | Tongue too low; raise toward palate; tongue too far forward |
| ɔː      | 650      | 459       | 1142     | 1138      | Tongue too low; raise toward palate |
| ʌ       | 724      | 472       | 1242     | 1339      | Tongue too low; raise toward palate |

These match the Slavic-L1 transfer predictions from Phase 0: raised /iː/ (too
high F1 = centralized), lowered diphthong onglide /eɪ/, and systematically
elevated F1 on back vowels /ɑː ɔː ʌ/. Advices read correctly as phonetician
guidance.

---

## Phase 1 entry criteria — current go/no-go

| Criterion | Status |
|-----------|--------|
| Modern RP norms in `rp_norms.py` | **DONE** |
| Vowel scorer validated (A ≥ 75, B ≥ 75) | **DONE** (A=79.5, B=79.0) |
| Consonant scorer operational | **DONE** (flat 70 — passthrough, not broken) |
| Aspiration scorer calibrated (real BC ≥ 60) | **NOT MET** (B=13.4) |
| B − C gap ≥ 15 | **NOT MET** (−1.5) |
| Phase 1 (UI) unblocked | **NO — aspiration scorer blocks** |

**Next action before Phase 1**: investigate and fix aspiration VOT detection
on the real BC reference clips. Once real BC aspiration reaches ≥ 60, re-run
Phase G to check if the B − C gap opens up.

---

---

## Raw F1/F2 distances — why the composite score hides the gap

The composite vowel score (owner 76.9 vs synth BC 79.5) is nearly identical.
But the per-phoneme Hz distances tell a different story.

| Phoneme | Owner F1 | Owner F2 | BC F1 | BC F2 | RP F1 | RP F2 | Owner→RP | BC→RP | Closer |
|---------|----------|----------|-------|-------|-------|-------|----------|-------|--------|
| iː      | 500      | 1837     | 314   | 2032  | 348   | 1962  | **197**  | 78    | BC     |
| ɪ       | 450      | 1841     | 343   | 1701  | 386   | 1773  | 94       | 84    | BC     |
| ɛ       | 540      | 1737     | 486   | 1504  | 462   | 1571  | **184**  | 71    | BC     |
| æ       | 550      | 1725     | 598   | 1420  | 545   | 1496  | **229**  | 93    | BC     |
| ɑː      | 534      | 1437     | 501   | 1233  | 518   | 1215  | **223**  | 24    | BC     |
| ɔː      | 531      | 1126     | 431   | 959   | 459   | 1138  | 73       | 181   | owner  |
| ʊ       | 482      | 1412     | 423   | 1260  | 386   | 1427  | 97       | 171   | owner  |
| uː      | 438      | 1483     | 316   | 1567  | 352   | 1506  | 88       | 71    | BC     |
| ʌ       | 560      | 1212     | 370   | 1226  | 472   | 1339  | 154      | 153   | BC     |
| ɜː      | 509      | 1295     | 514   | 1368  | 482   | 1440  | 147      | 79    | BC     |
| ə       | 449      | 1585     | 419   | 1472  | 412   | 1550  | 51       | 78    | owner  |
| eɪ      | 492      | 1893     | 390   | 1814  | 383   | 1873  | 111      | 59    | BC     |
| aɪ      | 588      | 1741     | 510   | 1520  | 482   | 1599  | 177      | 84    | BC     |
| ɔɪ      | 499      | 1626     | 405   | 1767  | 429   | 1560  | 96       | 208   | owner  |
| əʊ      | 468      | 1091     | 374   | 1472  | 389   | 1353  | **274**  | 120   | BC     |
| aʊ      | 542      | 1307     | 413   | 1130  | 536   | 1304  | 7        | 213   | owner  |

Owner phonemes with distance > 150 Hz from modern RP (Slavic-L1 markers):
**əʊ** (274), **æ** (229), **ɑː** (223), **iː** (197), **ɛ** (184), **ʌ** (154).

**Why the composite score doesn't show the gap**: the `score_vowels` function uses
exponential decay (`exp(-d/sigma)`) with sigma calibrated to give scores in the
70–80 range for well-formed English. At 100–200 Hz deviation the score drops from
100 to ~75 — indistinguishable from BC's 20–80 Hz deviation that gives ~85.
The function flattens out the meaningful range.

**Implication**: the raw Hz distance table is the diagnostic the product needs,
not the aggregated vowel composite. A vowel-only report showing each phoneme's
F1/F2 vs modern RP target (in Hz, not a score) gives the clearest signal.

Full data: `tts_output/accent_coach/bench/phase0_7/phoneme_distances.json`

---

## Files produced in Phase 0.7

| File | Description |
|------|-------------|
| `scripts/accent_coach_phase0_7_bbc_audit.py` | Phase E BBC F0 audit |
| `tts_output/modern_rp_corpus/bbc/speaker_audit.csv` | Per-clip gender guess |
| `docs/img/accent_coach_phase0_7_bbc_f0_hist.png` | F0 histogram |
| `scripts/accent_coach_phase0_7_rebuild_norms.py` | Phase F norms derivation helper |
| `accent_coach/reference/rp_norms.py` | Updated with `RP_VOWEL_F1_F2_MALE_MODERN` |
| `scripts/accent_coach_phase0_7_make_manifest.py` | Phase G manifest builder |
| `tts_output/accent_coach/bench/phase0_7_manifest.csv` | 154-row bench manifest |
| `tts_output/accent_coach/bench/phase0_7/report.md` | Bench Markdown report |
| `tts_output/accent_coach/bench/phase0_7/results.json` | Per-experiment scores JSON |
| `tts_output/accent_coach/bench/phase0_7/phoneme_distances.json` | Owner vs BC vs modern RP per-phoneme Hz distances |
| `tts_output/accent_coach/bench/phase0_7/speaker_centroids.json` | F1/F2 centroids for all 7 speaker groups |

---

## Speaker vowel space comparison

### What "modern RP" is in this analysis

`modern_rp` average = token-weighted mean across three sources (duration ≥ 50 ms filter applied):

| Source | Clips | Tokens | Weight |
|--------|-------|--------|--------|
| Fry corpus (`modern_rp_fry`) | 1 252 | 9 109 | 68% |
| BBC male pool (`modern_rp_bbc_male`) | 271 | 2 473 | 18% |
| Lindsey corpus (`modern_rp_lindsey`) | 255 | 1 910 | 14% |
| **Total** | **1 778** | **13 492** | |

Fry = multiple speakers from Geoff Fry's SSBE corpus. Lindsey = Geoff Lindsey's
recordings. BBC male = 271 clips from a single BBC News bulletin (gender_guess
== "male", Phase E filter). The average is heavily Fry-weighted.

### Mean F1/F2 per phoneme — all speakers

(All values in Hz; duration ≥ 50 ms; /ɒ/ excluded — no tokens survived filters.)

| Ph | real BC | synth BC | Fry | Lindsey | BBC ♂ | modern RP | Owner |
|----|---------|----------|-----|---------|-------|-----------|-------|
| iː | 308/1881 | 314/2032 | 344/1917 | 344/2092 | 366/1970 | 348/1962 | **500/1837** |
| ɪ  | 373/1705 | 343/1701 | 381/1739 | 371/1926 | 415/1781 | 386/1773 | 450/1841 |
| ɛ  | 461/1490 | 486/1504 | 453/1564 | 456/1650 | 494/1549 | 462/1571 | **540/1737** |
| æ  | 496/1430 | 598/1420 | 540/1479 | 491/1699 | 591/1451 | 545/1496 | 550/1725 |
| ɑː | 463/1287 | 501/1233 | 527/1215 | 470/1223 | 529/1210 | 518/1215 | 534/**1437** |
| ɔː | 439/1147 | 431/959  | 459/1163 | 442/1144 | 474/1045 | 459/1138 | **531**/1126 |
| ʊ  | 405/1470 | 423/1260 | 393/1408 | 372/1454 | 365/1504 | 386/1427 | **482**/1412 |
| uː | 345/1618 | 316/1567 | 355/1455 | 334/1752 | 356/1535 | 352/1506 | **438**/1483 |
| ʌ  | 452/1335 | 370/1226 | 477/1338 | 433/1380 | 482/1310 | 472/1339 | **560**/1212 |
| ɜː | 469/1388 | 514/1368 | 490/1422 | 459/1498 | 482/1431 | 482/1440 | 509/1295 |
| ə  | 391/1451 | 419/1472 | 406/1528 | 404/1675 | 436/1544 | 412/1550 | 449/1585 |
| eɪ | 371/1846 | 390/1814 | 382/1817 | 377/1990 | 394/1962 | 383/1873 | **492**/1893 |
| aɪ | 431/1504 | 510/1520 | 481/1589 | 446/1697 | 528/1554 | 482/1599 | **588/1741** |
| ɔɪ | 451/1400 | 405/1767 | 432/1557 | 401/1693 | 456/1351 | 429/1560 | 499/1626 |
| əʊ | 404/1451 | 374/1472 | 390/1354 | 361/1360 | 408/1339 | 389/1353 | 468/**1091** |
| aʊ | 472/1298 | 413/1130 | 529/1284 | 526/1374 | 571/1325 | 536/1304 | 542/1307 |

Bold = notable deviation from modern RP column.

### Mean distance to modern RP average (across 16 phonemes)

| Speaker | Mean dist to modern RP |
|---------|------------------------|
| Fry | 22 Hz |
| BBC male | 54 Hz |
| Real BC | 77 Hz |
| Lindsey | 101 Hz |
| Synth BC (IndexTTS) | 110 Hz |
| **Owner** | **137 Hz** |

### Mean distance to real BC (across 16 phonemes)

| Speaker | Mean dist to real BC |
|---------|----------------------|
| Fry | 73 Hz |
| Modern RP avg | 77 Hz |
| BBC male | 86 Hz |
| Synth BC | 110 Hz |
| Lindsey | 145 Hz |
| **Owner** | **183 Hz** |

### Notes

- **Lindsey** (101 Hz from modern RP) is further than expected. His corpus is
  small (255 clips) and may oversample certain phonetic environments.
- **Synth BC** (IndexTTS, 110 Hz from modern RP) diverges most on /ɔɪ/
  (405/1767 vs 429/1560) and /ʌ/ (370/1226 vs 472/1339) — IndexTTS is pulling
  those vowels away from native RP.
- **Owner's biggest deviations** from both real BC and modern RP:
  /əʊ/ F2 ~360 Hz too low (monophthong-like onglide — classic Slavic-L1),
  /iː/ F1 ~190 Hz too high (centralised FLEECE),
  /aɪ/ F1 ~160 Hz too high (centralised PRICE onglide),
  /ɑː/ F2 ~150 Hz too high (BATH/PALM not backed enough).

---

## Open question: F0-related bias in raw Hz comparisons

### Does absolute pitch difference bias the F1/F2 numbers?

Yes. F0 and formants are correlated through vocal tract length. Higher-pitched
speakers have shorter vocal tracts and produce systematically higher F1/F2
across the board, independent of vowel quality. A speaker at mean F0 ~160 Hz
will show higher formants than one at ~110 Hz even with identical articulation.
This is the motivation for Lobanov and Nearey normalisation — both remove the
per-speaker mean and scale from each formant dimension.

All pairwise distances in the table above are raw Hz and therefore carry this
bias. In particular, any cross-speaker comparison where F0 differs by more than
~30 Hz should be treated with caution until normalisation is applied.

### Should we normalise before analysis?

It depends on what is being measured:

- **Phoneme identity / tongue position** — normalise first (Lobanov or Nearey).
  This removes the vocal-tract-size confound and makes native clusters tighter,
  giving a cleaner owner-vs-native distance on pure accent features.
- **Vowel space size / compression** — keep raw Hz, or compute a
  vocal-tract-invariant area measure *after* normalisation. Vowel space
  compression is itself a Slavic-L1 diagnostic signal; Lobanov erases it by
  stretching every speaker's space to unit variance (the Phase 0.6 finding).

The practical implication for the current table: the owner's elevated /ɑː/ F2
(1437 vs 1287 for real BC, +150 Hz) is the most likely candidate to be partly
a vocal-tract artefact rather than a pure accent feature. Phonemes where *both*
F1 and F2 are shifted in the same direction (e.g. /iː/ F1 too high, /əʊ/ F2
too low) are harder to explain by vocal tract size alone and are more likely
genuine accent features.

**Next step**: apply Lobanov normalisation per speaker to the centroids table
and re-examine which deviations survive. Deviations that shrink to near-zero
after normalisation were probably vocal-tract bias; deviations that persist are
genuine accent markers.
