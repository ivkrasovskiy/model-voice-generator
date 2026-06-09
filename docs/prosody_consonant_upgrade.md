# Prosody & Consonant Module Upgrade Plan

Living record of the rhythm/intonation/consonant scoring fixes (alignment,
silent-fallback removal, bandwidth normalization, in-domain fricative reference,
VOT rewrite, per-phoneme analysis). The original implementation spec it grew from
has been retired now that the modules are built.

---

## Current state (updated Phase 0.18 audit)

| Module | File | Status |
|--------|------|--------|
| Rhythm | `accent_coach/comparison/rhythm.py` | **DONE** — see Phase 0.17 in history |
| Intonation | `accent_coach/comparison/intonation.py` | Stub: DTW full-contour shape only |
| Consonants | `accent_coach/comparison/consonants.py` | Stub: CoG for /s z ʃ ʒ/ vs RP reference |
| VOT | `accent_coach/pipeline/vot.py` | Computes VOT but not wired into consonant scoring |

---

## Module 1 — Rhythm upgrades ✅ COMPLETE (Phase 0.17)

Three signals implemented and validated:

### 1A. nPVI — done
Scores against target speaker nPVI in comparison mode; against corpus-derived RP reference
(`RP_NPVI_MIN=40, RP_NPVI_MAX=62`, centre=51) in absolute mode. Reference supersedes
Grabe & Low 2002 (lab read-aloud, too high for natural speech).

### 1B. Per-syllable pattern correlation — done
Pearson r of mean-normalised duration vectors. Returns neutral 50 when syllable counts
diverge > 30% (would otherwise correlate misaligned phonological positions).

### 1C. Function word inflation detection — done
Uses WhisperX word-level timestamps (reliable even for 25ms function words). Ratio > 1.7×
target = inflated. When phoneme data unavailable, FW weight is redistributed to nPVI+pattern
rather than defaulting to a fictional perfect score.

### Syllable extraction — done
`extract_syllable_durations_from_words()` (hybrid): word-level timestamps from WhisperX
for inter-word contrast + per-word acoustic nucleus detection for within-word stress contrast.
Fixes the G2P-uniform distribution bug that gave nPVI ≈ 25 for all speakers.

### Output fields (all in `RhythmBreakdown`)
- `inflated_function_words: list[str]` ✅
- `pattern_correlation: float | None` ✅
- `function_word_score: float | None` ✅
- `outlier_syllables: list[tuple[int, str]]` ✅ (Phase 0.20 — per-position "long"/"short")
- `diagnostics: list[str]` ✅
- `sentence_type_scores` — deferred to `analyze_session()` (session-level aggregation)

### Diagnostic signals
- **Signal 1** (nPVI too flat, `user < ref − 10`): fires for syllable-timed L2 accents (French, Mandarin, Italian)
- **Signal 2** (function word inflation, ratio > 1.7×): fires for Slavic-type non-reduction
- **Signal 3** (pattern mismatch, correlation < 60): fires for wrong stress placement
- **Signal 4** (over-stressing, `user > ref + 10`): fires for lecture/hyper-articulated speech ✅ Phase 0.20
- **Signal 5** (per-syllable outliers, normalised diff > 0.5): pinpoints which positions are too long/short ✅ Phase 0.20

### Bench result (hybrid mode, Phase 0.20)
Conversational RP/GenAm: 78–86. TTS: 74.2. Owner: 52.0.
Full table in `docs/accent_coach_history.md#phase-020`.

---

## Phase 0.18 bugs — ALL FIXED (Phase 0.19) ✅

All 8 bugs documented in Phase 0.18 are resolved. 23/23 tests pass.
Full details in `docs/accent_coach_history.md#phase-019`.

| Bug | Fix | Status |
|---|---|---|
| GenAm BATH/LOT override always fires | `accent_target` param in `alignment.py`; gated on `== "rp"` | ✅ |
| `extract_formants()` never passes `--target` | Added `target` param, threaded to subprocess | ✅ |
| `compare()` always uses RP norms | Added `accent_target` param, branches `get_genam_norms` | ✅ |
| `score_vowels()` always uses RP norms | Added `accent_target` param, same fallback branch | ✅ |
| LOT /ɒ/ centroid was Deterding 1997 stub (600, 900) | Corpus re-measured: **(532, 1114)** n=153 | ✅ |
| `RP_VOWEL_F1_F2_MALE` alias → LEGACY | Alias → `RP_VOWEL_F1_F2_MALE_MODERN` | ✅ |
| Rhythm bench used acoustic-only vs hybrid norms | Bench accurate mode now uses WhisperX hybrid | ✅ |
| `_function_word_analysis()` returns 100.0 when no FW | Returns `(None, [])` → weight redistributed | ✅ |

**Also deleted**: `extract_syllable_durations()` (legacy G2P-uniform function) — zero callers.
Replacements: `extract_syllable_durations_from_words()` (hybrid) and
`extract_syllable_durations_acoustic()` (fast fallback).

**Next work**: intonation and consonant module upgrades (Modules 2 and 3 below).

---

## Module 2 — Intonation upgrades

### 2A. F0 extraction in semitones — NEW normalisation
Currently normalises pitch contour to 0–1. Instead: extract raw F0 (librosa YIN), convert to semitones relative to speaker's own median pitch. This makes user and target comparable across speaker pitch ranges without shape distortion.

### 2B. Boundary tone slope — NEW, highest priority signal
Measure linear slope of pitch in the final 250ms of the utterance (semitones/second). Classify as rising (>+1.5 st/s), falling (<-1.5 st/s), or flat.

Expected directions per sentence type:
- statement → falling
- wh_question → falling
- yes_no_question → rising
- list items → rising, final item falling
- exclamation → falling

Boundary direction wrong = major error, score 20 regardless of DTW. Boundary correct = 100. Weight: 50% of composite.

### 2C. DTW on semitone contour — upgrade existing
Keep DTW but run it on the semitone-normalised contour (not 0–1 scaled). Use `librosa.sequence.dtw`. Weight: 50% of composite.

### Output additions
- `boundary_slope_user: float`, `boundary_slope_target: float`
- `boundary_correct: bool`
- `sentence_type: str`
- Specific diagnostic strings per error type (wrong direction, flat ending, shape mismatch)

---

## Module 3 — Consonants upgrades

Split by consonant class — each needs a different algorithm.

### 3A. Fricatives — extend existing
Already have /s z ʃ ʒ/. Add:
- /TH DH/ (dental fricatives) — CoG ~2000–6000 Hz, low energy; most L2 speakers substitute /s z/. Diagnosis: if CoG > 5500 Hz → sounds like /s/ → tongue placement instruction.
- /F V/ — labiodental, broad spectrum ~3500–8000 Hz.
- /HH/ — skip CoG (aspiration, not a true fricative CoG signal).

Primary comparison: user CoG vs target CoG (not RP reference) — target IS the reference. Cross-check against reference range for sanity.

### 3B. VOT for stops — wire existing pipeline/vot.py into comparison
`pipeline/vot.py` already computes VOT. What's missing: the comparison/scoring layer.
- Only score voiceless stops (/P T K/) in word-initial stressed position — that's where English aspiration is contrastive.
- Score: exponential decay on (error vs target VOT) × (penalty for being outside English range 50–125ms).
- Under-aspiration (VOT < 35ms) = strongest non-native marker → specific diagnosis with "add a puff of breath" instruction.

### 3C. Rhotic /r/ — F3 depression — NEW
English /r/ (retroflex/bunched) drops F3 to ~1800–2200 Hz. Other languages' /r/ (tapped, trilled, uvular) leave F3 at 2400–2800 Hz. This is the single most reliable acoustic signature of rhoticity.

Use parselmouth Burg formant tracker at phoneme midpoint. F3 < 2200 Hz = rhotic. F3 > 2500 Hz = non-rhotic substitution.
Diagnosis: tongue tip position instruction or "bunch the tongue sides against upper molars."

### 3D. Lateral /l/ — dark vs clear — NEW
Syllable-final /l/ (dark /l/) requires velarization: F2 drops to ~800–1300 Hz. Many L2 speakers use clear /l/ everywhere (F2 stays ~1400–1800 Hz).
Only flag errors in syllable-final position — that's where it's perceptually distinctive.
Use parselmouth F2 at midpoint. Compare user vs target F2.

### 3E. Consonant aggregator — upgrade
Current `score_consonants` returns a single float. Replace with `ConsonantScore` dataclass:
- `score: float` (composite)
- `fricative_score`, `stop_aspiration_score`, `rhotic_score`, `lateral_score`
- Per-phoneme analysis lists (for drill-down UI)
- `diagnostics: list[str]` — top 3–4 issues sorted by severity

Weights: fricatives 35%, stops 35%, rhotics 20%, laterals 10%.

---

## Integration — analyze.py

Need a top-level `analyze_session()` that orchestrates all three modules per sentence then aggregates across the session:

Inputs: `user_audio`, `target_audio`, alignment (phoneme-level timestamps), `sentence_meta` (sentence_type per sentence).

Per sentence:
1. Extract syllable durations from alignment → rhythm module
2. Extract word-level timestamps → function word analysis
3. Run F0 extraction → intonation module
4. Iterate phonemes from alignment:
   - FRICATIVES → fricative analysis
   - VOICELESS_STOPS (word-initial, stressed only) → VOT analysis
   - R → rhotic analysis
   - L → lateral analysis (with syllable position flag)
5. Aggregate per-sentence consonant scores.

Output: `ComparisonResult` with rhythm, intonation, consonant sub-scores + diagnostics, averaged across sentences with per-sentence-type breakdowns.

---

## File structure (proposed)

```
accent_coach/comparison/
  rhythm.py          — extend existing
  intonation.py      — extend existing
  consonants/        — split into subpackage
    __init__.py
    fricatives.py    — CoG analysis for all 8 fricatives
    stops.py         — VOT scoring (wraps pipeline/vot.py)
    liquids.py       — /r/ F3, /l/ F2
    aggregator.py    — ConsonantScore dataclass + score_consonants()
accent_coach/pipeline/
  analyze.py         — NEW: top-level session orchestrator
```

---

## Dependencies

All within what's already used: `numpy`, `scipy`, `librosa`, `praat-parselmouth`.
`pipeline/vot.py` already exists — stops.py just wraps it.

---

## Stop criteria

- GREEN: `uv run pytest tests/ -q` passes, `uv run ruff check scripts/ accent_coach/` clean
- Each module has at least one unit test with synthetic audio or fixture data
- `analyze_session()` runs end-to-end on a real sentence without exception

---

---

## Phase 0.21 — Consonant module implementation (completed 2026-06-05)

### What was built

`accent_coach/comparison/consonants/` subpackage replacing the old single-float stub:

| File | What it does |
|---|---|
| `fricatives.py` | CoG scoring for all 8 English fricatives (`/s z ʃ ʒ θ ð f v/`); TH→/s/ substitution diagnosis (CoG > 5500 Hz) |
| `stops.py` | VOT scoring from pre-extracted `StopFeatures`; under-aspiration penalty (VOT < 35 ms) |
| `liquids.py` | `/r/` F3 depression via parselmouth Burg (5-formant, gender-aware ceiling); `/l/` dark vs clear F2 with syllable-final gate |
| `aggregator.py` | `ConsonantScore` dataclass; weight redistribution when sub-class has no tokens |

Models added: `ConsonantScore` (pydantic), `ComparisonResult.consonant_breakdown`.
Reference norms added to `rp_norms.py` and `genam_norms.py`: fricative CoG, rhotic F3, lateral F2, GA VOT.

Tests: 32 new tests across `test_consonants.py` (interface/TDD) and `test_consonants_quality.py` (business-logic calibration). All 183 tests green.

### Bench results (2026-06-05, n=10 clips/group, accent=rp)

```
Group              composite  fricative  stop_vot   rhotic  lateral
RP Fry                  25.1       22.3       6.1     31.8     64.5
RP Lindsey              27.6       30.5       3.2     32.2     64.8
Real BC                 28.4       22.0      23.4     33.2     60.7
TTS BC (IndexTTS-2)     25.6       19.9       6.2     28.8     59.4
Owner                   19.9       18.8       2.7     23.4     45.0
GA natives (Harris)     35.0       27.0       9.8     70.9     52.0
```

### What works

**Laterals discriminate correctly.** RP natives score 64–65 (best dark /l/), owner lowest at 45.
The F2-based dark-vs-clear scoring with syllable-final gate is acoustically sound.

**Owner is the lowest overall (19.9) on composite.** The primary ranking requirement holds
even with broken sub-scores — owner's low lateral score (45) and low rhotic (23) drives it down.

**GA rhotics dominate (70.9).** Harris/Huberman/Sapolsky produce retroflex /r/ in every position;
F3 depression fires cleanly. Matches linguistic expectation.

### Three broken sub-scores — root causes and fixes needed

#### VOT (all groups near 0–10, RP natives = 6)

The `extract_vot()` burst detector fails on natural speech. The burst search window is
40 ms after the G2P-derived phoneme start timestamp. G2P assigns phoneme boundaries by
**uniform splitting of word duration**, so the stop timestamp is rarely acoustically
accurate — the burst can fall outside the 40 ms window and is not detected.

Fix: widen burst search to 80–100 ms, or switch to a proper forced-alignment model
(WhisperX phone-level CTC) that gives per-phoneme acoustic boundaries. Until then
`stop_aspiration_score` is unreliable in the consonant composite; `score_aspiration()`
in `scoring.py` (which also uses `StopFeatures`) has the same problem.

#### Rhotics — RP non-rhotic position penalty

RP Fry/Lindsey score 31–32 on rhotics despite being native speakers. RP is
**non-rhotic**: post-vocalic /r/ (e.g., "together", "over", "bird") is not pronounced.
But the G2P pipeline always emits an /r/ phoneme for those words; the scorer measures F3
at that position, finds no depressed F3 (because no /r/ was produced), and penalises.

Fix: gate rhotic scoring on pre-vocalic position only. A phoneme followed by a vowel
phoneme in `user.phonemes` should be scored; all other /r/ tokens should be skipped for
non-rhotic target accents (`accent_target == "rp"`). Keep scoring all positions for GA
(`accent_target == "genam"`) since GA is fully rhotic.

#### Fricative CoG reference values are citation-form, not conversational

All groups score 18–30 on fricatives. The reference values in `rp_norms.py`
(`/s/` = 7000 Hz, `/ʃ/` = 3800 Hz, etc.) come from Jongman et al. (2000) citation-form
readings in quiet conditions. In natural conversational speech, CoG is typically 500–1000 Hz
lower due to coarticulation and speaking rate. A speaker producing a perfect /s/ in
running speech might measure at 6000–6500 Hz — a 500–1000 Hz gap from the reference
produces an exponential penalty even for native speakers.

Fix (two options, in priority order):
1. **Use comparison mode** (user CoG vs target speaker CoG) rather than absolute mode.
   This requires passing target audio and target phoneme timestamps to `score_fricatives()`.
   Target IS the reference — absolute reference values become only a sanity cross-check.
2. **Recalibrate corpus constants** by measuring CoG from the modern_rp_corpus / genam_lecture_corpus
   clips that already have transcripts and can be aligned. Replace the Jongman citation values
   with corpus-derived conversational means.

### Aspiration double-counting (known, deferred)

`scoring.py` calls `score_aspiration()` at 15% weight AND the consonant composite includes
stops at 35% of its 15% weight (≈ 5.25% total). Aspiration contributes ~20% of the total
composite vs the intended 15%. Tracked in `aggregator.py` docstring. Will be resolved when
`analyze.py` is built and `score_aspiration()` is retired.

### Next steps for consonants (priority order)

1. **Fix rhotic position gate** — 1 day. Gate RP rhotic scoring to pre-vocalic tokens only.
   Will fix RP native scores from ~32 to something close to GA levels.
2. **Fix VOT** — switch from G2P timestamps to WhisperX phone-level CTC alignment, OR widen
   burst search window in `extract_vot()`. High impact: VOT is the clearest L2 marker for
   Slavic/Romance speakers.
3. **Recalibrate fricative CoG** — measure conversational CoG from existing corpora and replace
   Jongman citation values. Or: default to comparison mode (target CoG) when target audio is
   available.
4. **Retire double-counting** — fold `score_aspiration()` into consonant weight when `analyze.py`
   is built.

---

## Constants inventory (Phase 0.18 audit — move to config)

All scoring constants are currently module-level globals, making them hard to tune
without hunting through source. Target: a single `accent_coach/config.py` (or
`accent_coach/reference/scoring_config.py`) that is the single source of truth.

| Constant | Current location | Value | Meaning |
|---|---|---|---|
| `_DECAY` | `comparison/rhythm.py:13` | 30.0 | nPVI score e^{-1} width (nPVI units) |
| `_NPVI_REF` | `comparison/rhythm.py:12` | 51.0 | nPVI corpus centre (derived from RP_NPVI_MIN/MAX) |
| `FW_THRESHOLD` | `comparison/rhythm.py:90` | 1.7 | Function-word inflation ratio |
| `PATTERN_DIVERGENCE_GUARD` | `comparison/rhythm.py:45` | 0.30 | Max count-mismatch fraction before neutralising pattern score |
| `_SCALE` | `comparison/vowels.py:13` | 1.5 | Vowel distance → score exponential scale |
| `_WEIGHTS` | `comparison/scoring.py:17` | vowels 25%, others 15% | Per-skill composite weights |
| `RP_NPVI_MIN/MAX` | `reference/rp_norms.py:133` | 40, 62 | Hybrid-adjusted nPVI native range |
| `RP_VOT_RANGE_MS` | `reference/rp_norms.py:99` | p,t,k ranges | English VOT reference ranges |
| `RP_VOT_SD_MS` | `reference/rp_norms.py:112` | 10 ms each | VOT SD (estimated, not corpus-derived) |

---

## VOT extractor — critical bug diagnosis (2026-06-07)

Symptom: VOT ≈ 0–3 ms for **every** speaker group (should be ~50–125 ms for
aspirated English /p t k/). Persists with char-aligned boundaries → this is an
**extractor-algorithm** problem, not only alignment. Findings code-traced AND
reproduced numerically on synthetic /pa/, /ka/ and connected-speech tokens.
Full detail + sources in [vot_bug_diagnosis.md](vot_bug_diagnosis.md).

### Critical bugs (`accent_coach/pipeline/vot.py`)

| line | bug | why VOT → ≈0 |
|---|---|---|
| 67 | voicing loop `range(burst_frame, …)` starts **inclusive**, first frame > threshold | periodicity at burst frame → `voice_frame == burst_frame` → VOT 0 |
| 43-47,67-72 | 20 ms `_PRE_MS` + `start_time` jitter pulls the **preceding voiced sound** into the window (English /p t k/ are post-vocalic) | residual voicing read as onset at frame 0 (reproduced ac=0.58 → VOT 0) |
| 62 vs 72 | VOT = `(i − burst_frame)·5 ms` — locked to 5 ms **hop grid**, hard 0 floor | no sub-frame resolution; same-frame = 0 |
| 53-58 | burst = first frame HF > `3×median`, median over a 100 ms window **containing the vowel** | threshold crossed at the vowel → burst ≈ voicing |
| 71 | voicing gate `autocorr > 0.35` **too low** (modal ~0.6–0.85) | formant ringing crosses 0.35 early → onset pulled to burst |
| 38 | `min_lag` ≈ 1 ms (~1000 Hz) | formant-band periodicity fakes voicing (should constrain F0 75–400 Hz) |

### Gaps vs validated methods (AutoVOT / Dr.VOT / Praat–Lisker&Abramson)
- No **closure-first anchoring** (no "burst after silence"); baseline must come from the closure, not a vowel-straddling median.
- Burst should be a **broadband energy transient/derivative**, not HF-energy-vs-median (fires on the vowel).
- Voicing onset via **F0-constrained periodicity** (parselmouth `To Pitch (ac)`, 75/400) strictly **after** the burst, persisting ≥20 ms.
- No **negative-VOT (prevoicing)** handling — code clamps `<0 → None`.
- Greedy first-crossing (vs AutoVOT joint optimisation) yields the degenerate same-frame solution.

### Recommended fix
Rewrite around a validated method (the greedy fixes interact):
- **Option A** AutoVOT/Dr.VOT (export stop windows from existing MMS alignment as TextGrids).
- **Option B** correct parselmouth/Praat pipeline (no new deps): closure → broadband-derivative burst (sub-frame) → F0-constrained voicing onset strictly after burst, signed for prevoicing; `vot_ms = (voice_sample − burst_sample)/sr·1000`.

Non-negotiables: (a) anchor burst to the closure→release transient, not an HF median that includes the vowel; (b) detect voicing strictly after the burst with an F0-constrained high-threshold periodicity test. Validate vs hand-measured tokens; expect native > TTS > owner, /p t k/ in 50–125 ms.

---

## Fricative inversion — root cause: audio-bandwidth confound (2026-06-07)

Owner (L2) scores HIGHEST on fricatives, native RP lowest. Worked the
data-correctness checklist; reproduced through the exact bench path.

**Item 1 (alignment/transcripts):** transcripts verbatim; alignment imperfect for
short fricatives but **symmetric across groups** → not the cause.
**Item 2 (phoneme ID / like-for-like):** correct — each /x/ compared to ref for the
same /x/; no cross-phoneme contamination.
**Item 3 (audio properties): ROOT CAUSE, confirmed with measurements.**

| group | source SR | energy ceiling | /s/ CoG @16k |
|---|---|---|---|
| rp_fry (native) | 16000 | 8000 Hz | 4013 ← lowest |
| rp_lindsey | 16000 | 8000 Hz | 5339 |
| real_bc | 16000 | 8000 Hz | 4590 |
| genam_harris | 16000 | 8000 Hz | 4580 |
| tts_bc | 22050 | 10717 Hz | 4687 |
| **owner** | **44100** | **19536 Hz** | **5316** |
| Jongman ref | (22 kHz src) | — | **7000** |

Mechanism: Jongman /s/ ref = 7000 Hz (from ≥22 kHz recordings, /s/ to 11+ kHz).
Native corpora are band-limited to 8 kHz **at source** (YouTube/lecture), so their
measured /s/ CoG sits ~4000–5300 Hz — a bandwidth artefact, not articulation. The
owner's 44.1 kHz /s/ keeps more in-band HF mass (CoG 5316 even after 16k
downsample), so `100·exp(-|CoG−7000|/2000)` rewards owner over natives. The metric
measures **source bandwidth, not accent**. F0/pitch irrelevant (HP@2 kHz + CoG is
F0-invariant).

**Item 4 (logic bug):** secondary — the 7000 Hz ref is uncapturable at 16 kHz
(Nyquist 8 kHz); reference band ≠ analysis band.

### Fix (priority order)
1. **Bandwidth-equalise every clip** to a common ceiling all corpora actually
   contain (≤8 kHz) before measuring CoG — bring owner/TTS DOWN to match natives
   (cannot recover HF natives never had).
2. **Re-derive the reference CoG in the same 0–8 kHz band** from the native corpus
   itself (Jongman 7000 Hz is a 22 kHz-band figure → meaningless at 8 kHz).
3. **Comparison mode** (`--target-manifest`) vs a bandwidth-matched BC target of the
   same phrase cancels the artefact.
4. Reject fricative tokens whose in-window HF(>3 kHz) energy ratio < ~0.2
   (mis-aligned /s z/ that landed on a vowel/closure).
Do NOT widen the decay (already rejected — hides the inversion).

Sources: Jongman, Wayland & Wong (2000) JASA 108(3):1252-1263; later sibilant
studies downsample to 22050 Hz to stay Jongman-comparable; high-freq fricative
refinements (PMC10540850); SR/anti-alias effects on fricatives (PMC7056453).

### In-domain fricative reference — result (2026-06-08)

Replaced Jongman (7000 Hz /s/) with native means measured through THIS pipeline
(scripts/tools/measure_fricative_cog.py; pooled RP+GA natives): /s/=5200, /z/=5300,
/ʃ/=3970, /ʒ/=3480, /f/=4700, /v/=4850, dentals 4400–4500 (gate-biased, conservative).

Bench fricative score, before → after:

| group | Jongman 7000 | in-domain 5200 |
|---|---|---|
| rp_fry (native) | 40.5 | **77.1** |
| rp_lindsey | 80.5 | 80.5 |
| real_bc | ~44 | 77.1 |
| genam_harris | ~45 | 77.2 |
| **owner** | **58.2 (highest)** | **78.2 (tied)** |

Outcome: natives lifted ~40 → ~77 (band-limiting bias removed). **Composite is now
owner-lowest ✓.** Fricative now shows all groups ≈ 77–80 — the owner sits WITH
natives, not above. Residual −1.1 "inversion" (owner 78.2 vs real_bc 77.1) is within
a 3-pt spread = noise.

**Honest finding:** fricative CoG does NOT discriminate a Russian-L2 speaker from
natives — Russian /s/ ≈ English /s/ in CoG (same place of articulation). The
"owner strictly lowest on fricatives" invariant is therefore too strong for this
L1; fricatives are not an accent discriminator here. Discrimination for this
speaker lives in vowels, rhotics, laterals, and (once fixed) VOT.

### VOT rewrite — result (2026-06-08)

Rewrote `extract_vot` (Lisker & Abramson via parselmouth, no new deps):
closure baseline → broadband burst transient (searched only AFTER the closure
minimum) → first F0-constrained voiced run (pitch floor 75 / ceiling 400)
STRICTLY after the burst, persisting ≥20 ms. Returns None (not a fabricated 0)
when no burst / no post-burst voicing.

TDD: business-logic tests on hand-checkable synthetic ground truth
(tests/accent_coach/test_vot.py) — known-VOT connected /apa/, short-vs-long
separation, English aspirated range. All pass; the old §2B/§2C tests that pinned
the deleted median-window/autocorr internals were retired.

**Synthetic:** fixed — isolated AND connected (post-vocalic) stops measure the
true VOT; the preceding-vowel collapse to ~0 is gone.

**Real clips:** much improved but not production-grade. Stop sub-score (was ≈0
for all groups): real_bc 11.6, tts_bc 7.7, genam 6.1 now register vs owner 0.7;
but rp_fry (0.5) is still low — real connected speech is messier than synthetic
(weak/ambiguous bursts, closure voicing), so many native tokens still
under-measure below the 50–125 ms aspirated range. Directionally native > owner
is emerging but noisy.

**Verdict:** the collapse bug is fixed and the metric is honest (None, not 0);
robust production-grade real-speech VOT needs a trained model (AutoVOT / Dr.VOT)
per docs/vot_bug_diagnosis.md — the heuristic's ceiling on messy audio.

### Overall consonant bench after all fixes (n=12, accent=rp)

```
metric      owner   lowest-other   gap    owner-lowest?
composite    33.0    38.7 (tts)    +5.7   ✓   (was inverted at session start)
fricative    78.2    77.1          -1.1   ✗ noise (all 77–80; RU /s/≈EN /s/, non-discriminating)
rhotic       37.6    40.7          +3.0   ✓
stop_vot      0.7     0.5          -0.2   ✗ noise (VOT still under-measured on real speech)
lateral      64.5    46.9         -17.6   ✗ single-token noise (backlog)
```

Composite now correctly ranks owner lowest. Remaining real inversion is lateral
(single-token variance) — see accent_coach_potential_improvements.md.

### VOT real-speech robustness — aspirating-context filter (2026-06-08)

Root cause of native≈owner VOT was NOT detection — it was MEASURING THE WRONG
STOPS. English aspirates /p t k/ (long VOT) only when stressed + syllable-initial
+ PREVOCALIC and NOT post-/s/. The bench measured all stressed stops, including
/str/, /pl/, /kl/ clusters and unreleased final stops — short for everyone, no
accent signal. Added `filter_aspirating_stops` (prevocalic, non-post-/s/) and
tightened the implausible-VOT cap to 150 ms.

Result (n=16): every REAL native now scores above owner on stop_vot
(real_bc 12.6, genam 7.6, rp 3.7–4.6 vs owner 1.9); only the TTS *clone* (1.8)
dips below (synthetic aspiration). Composite gap +11.2 ✓, rhotic +18.7 ✓.
Remaining VOT ceiling on messy real audio → AutoVOT (backlog).

### Per-phoneme consonant discrimination — Phase 1 result (2026-06-09)

`scripts/tools/consonant_per_phoneme.py` (CSV: tts_output/accent_coach/consonant_per_phoneme.csv).
Per-phoneme mean score (n tokens), gap = native_pooled − owner:

```
ph    owner  nat_pool  gap     note
s     76(8)    78      +2   TIE  — Russian /s/≈English /s/ (frequent → dominates fricative class)
z     81(3)    78      -3   TIE
ʃ     90(1)    87      -3   TIE  — Russian has /ʃ/
f     76(2)    73      -3   TIE  — Russian has /f/
l     53(11)   52      -0   TIE
θ      -(0)    68       -   owner produces NO measurable /θ/ (substitutes → /s/ or gate-rejected)
ð      -(0)    77       -   owner produces NO measurable /ð/
v      -(0)    71       -   owner produces NO measurable /v/ token in sample
p      1(3)    10      +9   DISCRIMINATES — aspiration (Russian unaspirated)
t      2       4       +2   (VOT detection still weak on real speech)
k      2       1       -1
r     23(9)    57     +34   DISCRIMINATES STRONGLY — Russian trill vs English approximant
```

Composites (both rank owner lowest):
```
group         class_weighted   equal_phoneme
owner             36.8            44.9   ← lowest in both
rp_fry            40.3            53.1
rp_lindsey        55.6            53.3
real_bc           48.4            51.3
genam_harris      52.3            56.5
tts_bc            46.3            53.6
```

**Findings (all three user hypotheses confirmed):**
1. **Russian-close sounds tie** (/s z ʃ f l/, gap≈0) and DOMINATE by token frequency
   (/s/ owner n=8, natives n=24–45) → they drag the fricative class to a tie.
2. **The real discriminators are /r/ (+34) and /p/ aspiration (+9)** — exactly the
   sounds absent/different in Russian. They work, but are outnumbered.
3. **The strongest markers /θ ð v/ are INVISIBLE**: the owner produces zero
   measurable /θ ð/ tokens — the Russian /θ/→/s/, /ð/→/z d/ substitution means the
   error never registers as a /θ/ score (it's measured as the substituted sound or
   gate-rejected). The biggest accent tell is silently absent from the score.

**Equal-phoneme weighting is NOT the fix** — it gives a SMALLER owner−native gap
(owner 44.9 vs native ~53, gap ~8) than class-weighting (36.8 vs ~49, gap ~12),
because weighting every phoneme equally dilutes the few discriminators (/r/, /p/)
among many tied Russian-shared phonemes. Real fixes: (a) capture the /θ ð/
substitution as a /θ ð/ penalty so it stops vanishing; (b) up-weight the
absent-in-Russian markers (/r/, /θ ð/, aspiration) rather than equal-weighting.
Both composites preserved in the tool for comparison.
