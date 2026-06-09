# Consonant & Prosody Scoring — consolidated reference

Single source of truth for the consonant/prosody scoring pipeline: what's done,
what's open, the design principles, and the measurement references that the code
points at. Merges the former `consonant_scoring_audit.md` and `vot_bug_diagnosis.md`;
the blow-by-blow session history is summarised in [accent_coach_history.md](accent_coach_history.md).

## Status at a glance

| area | state |
|---|---|
| Silent fallbacks | **DONE** — removed everywhere (alignment, liquids, aggregator, prosody, compare); code fails explicitly instead of fabricating neutral/uniform values. |
| Audio comparability | **DONE** — one canonical `normalize_audio` (mono + 16 kHz + common bandwidth cap at 7.6 kHz) on the bench AND product path; vowels (formants ≤ ~4.5 kHz) unaffected. |
| Fricatives | **DONE** — bandwidth confound fixed; reference re-derived **in-domain** from native speakers; owner no longer scores above natives. |
| Rhotic /r/ | **DONE** — strong discriminator (native−owner gap ≈ +34). |
| Lateral /l/ | works; single-token variance (backlog). |
| VOT / aspiration | **FUNCTIONAL, not production-grade** — collapse-to-0 fixed, discriminates directionally (real natives > owner); real-speech absolute values still under-measure → AutoVOT is the robust path. |
| /θ ð/ detection | **OPEN (gate bug)** — the words DO contain /θ ð/ and alignment finds them, but the frication gate rejects ~100% of them for ALL groups (weak low-frequency dental frication), so /θ ð/ currently contribute nothing to anyone's score. |

## Open items (priority order)

1. **Fix the frication gate for weak dentals /θ ð/ (and /f v/).** Diagnosed via an
   expected→aligned→measured trace: the transcripts DO contain /θ ð/ (owner: 31 /ð/ in
   16 clips — every "the"), alignment finds all of them, but the HF>3 kHz frication gate
   rejects ~100% — for owner AND natives — because dental frication is weak and
   low-frequency. So /θ ð/ score nothing for anyone. Make the gate phoneme-aware (relaxed
   HF threshold for /θ ð f v/). Once measured, the owner-vs-native difference emerges from
   comparison: /θ/→/s/ substitution shows as too-high CoG (existing th-fronting detector),
   /ð/→/d/ as stop-like, plus a duration cue (owner /ð/ median 120 ms vs native 50 ms).
   Highest value.
2. **Up-weight absent-in-Russian markers** (/r/, dental /θ ð/, aspiration) vs Russian-shared
   phonemes. Equal-per-phoneme weighting is NOT the fix (it dilutes the few discriminators —
   see Per-phoneme findings).
3. **VOT real-speech accuracy → AutoVOT / Dr.VOT** (trained models; refs below).
4. **Lateral single-token confidence weighting** — flag/down-weight sub-scores from < ~3 tokens.

## Design principles (these calibrate the whole pipeline — also CLAUDE.md rules)

- **Comparison over absolutes; calibrate from the native distribution.** Absolute reference
  tables cannot separate groups that share a measurement artefact; compare against the same
  measurement on the native/BC target through the identical pipeline.
- **Never tune a constant — or a test — to a score *level*.** Validate by the native−owner
  *gap*, never the absolute level. Canonical failure: widening the fricative CoG decay to land
  natives in a "believable band" inverted the owner from lowest to highest (every group shared
  the same alignment artefact). Reject any constant change that shrinks the gap.
- **No silent fallbacks.** "Couldn't measure" propagates as `None`/skip or raises — never a
  fabricated neutral (50/65), uniform split, or swallowed exception. The archetype: a silent
  uniform G2P split once made the whole consonant bench meaningless while looking plausible.

## Fricatives — bandwidth confound + in-domain reference

**Root cause of the early inversion (owner scored *above* natives):** an audio-bandwidth
confound, not articulation. Native corpora are 16 kHz at source (8 kHz Nyquist — /s/ frication
above 8 kHz is gone); the owner is 44.1 kHz studio. Spectral CoG therefore measured *recording
bandwidth*: Jongman's /s/ = 7000 Hz is a 22 kHz-band figure, uncapturable in our 16 kHz band, so
natives scored "too low" while the wider-band owner scored high.

**Fix (three parts):**
1. **Canonical normalization** (`accent_coach/pipeline/audio_io.py`) caps every source to a
   common bandwidth so no clip can win on bandwidth alone.
2. **Frication gate** — reject fricative windows that mis-aligned onto a vowel/closure
   (HF<3 kHz energy ratio), so they don't pollute the mean. (KNOWN BUG: too aggressive for
   *weak* fricatives — rejects ~100% of dental /θ ð/ for ALL groups, verified by an
   expected→aligned→measured trace, so /θ ð/ score nothing. Needs a phoneme-aware
   threshold; see Open items #1.)
3. **In-domain reference** — measured through THIS pipeline on native speakers
   (`scripts/tools/measure_fricative_cog.py`; RP = Fry/Lindsey/BBC/real-BC,
   GA = Huberman/Harris/Sapolsky/vsauce), pooled (CoG is place-driven, accent-neutral):

   | ph | n | CoG (Hz) | | ph | n | CoG (Hz) |
   |---|---|---|---|---|---|---|
   | s | 455 | 5200 | | f | 23 | 4700 |
   | z | 331 | 5300 | | v | 36 | 4850 |
   | ʃ | 41 | 3970 | | θ | (sparse) | 4500* |
   | ʒ | (sparse) | 3480 | | ð | (sparse) | 4400* |

   *θ ð are gate-biased + sparse → conservative dental estimates, low confidence.

**Outcome:** natives lifted ~40 → ~77; composite ranks owner lowest. Fricative CoG then ties
across groups (~77–80) — which is *correct*: Russian /s z ʃ f/ ≈ English, so fricative CoG is
not an accent discriminator for this L1.

## VOT / aspiration — what was broken, what's fixed, path to production

**The old extractor collapsed to ≈0 ms for everyone** (greedy two-threshold heuristic): the
voicing loop read the *preceding* vowel's periodicity as the stop's onset; the burst threshold's
median included the following vowel; results were locked to a 5 ms grid with a hard 0 floor.

**Rewrite** (`accent_coach/pipeline/vot.py`, Lisker & Abramson via parselmouth, no new deps):
closure baseline → broadband burst transient searched *after* the closure minimum →
first F0-constrained voiced run (pitch 75–400 Hz) *strictly after* the burst. Returns `None`
(not a fabricated 0) when unmeasurable. TDD on hand-checkable synthetic ground truth
(`tests/accent_coach/test_vot.py`).

**Plus context filtering** (`filter_aspirating_stops`): English aspirates /p t k/ only when
stressed + prevocalic + NOT post-/s/. Measuring VOT in clusters (/str/, /pl/) or unreleased
finals dilutes the accent signal with tokens that are short for everyone. After filtering, every
*real* native scores above the owner on aspiration.

**Current limit:** synthetic is exact; real connected speech still under-measures (weak/ambiguous
bursts, closure voicing). Directionally native > owner, but not production-accurate.

**Path to production-grade VOT — trained models:**
- AutoVOT: https://github.com/MLSpeech/AutoVOT — Sonderegger & Keshet (2012) *JASA* 132(6):3965-3979
  (https://pubs.aip.org/asa/jasa/article-abstract/132/6/3965/915621)
- DeepVOT / Dr.VOT: https://github.com/adiyoss/DeepVOT
- Lisker & Abramson (1964) definition: https://www.haskinslaboratories.org/vot ;
  "VOT at 50" review (measurement pitfalls): https://pmc.ncbi.nlm.nih.gov/articles/PMC5665574/

## Per-phoneme discrimination (Phase 1)

`scripts/tools/consonant_per_phoneme.py` (CSV: `tts_output/accent_coach/consonant_per_phoneme.csv`).
Gap = native_pooled − owner:

```
TIE  (Russian-shared, frequent → dominate the class): s +2, z -3, ʃ -3, f -3, l -0
DISCRIMINATE:  r +34 (trill vs approximant),  p +9 (aspiration)
GATE-BUG:      θ, ð, v  — measured=0 for owner AND natives (frication gate rejects ~all
               weak dentals; transcripts DO contain them, owner has 31 /ð/ → fix the gate)
```

Composites (both rank owner lowest): class-weighted owner 36.8 vs natives 40–56;
equal-per-phoneme owner 44.9 vs natives 51–57. **Equal-weighting gives a *smaller* gap** —
it dilutes the few discriminators (/r/, /p/) among many tied Russian-shared phonemes. So the
fix is to fix the /θ ð/ gate (Open items #1) and up-weight the absent-in-Russian markers, not
to flatten the weights. The tool keeps both composites for comparison.
