# Prosody & Consonant Module Upgrade Plan

Extracted from `docs/further_improvements.md`.
Context: three comparison modules exist but are thin stubs. This doc lists what to add to each.

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
- `diagnostics: list[str]` ✅
- `sentence_type_scores` — deferred to `analyze_session()` (session-level aggregation)

### Bench result
tts_self=100, owner_vs_tts=73.3, native_rp(absolute)=70.0

---

## Known bugs blocking further work (Phase 0.18 audit)

Eight bugs were found via code-review agents and documented with failing tests.
**Fix these before adding intonation or consonant modules** — otherwise the scoring
pipeline produces silently wrong results for GenAm targets, and the rhythm baseline
inverts native vs non-native.

Tests: `tests/accent_coach/test_norms_and_routing.py` and
`tests/accent_coach/test_score_ordering.py` — 8 of 23 tests currently fail.
Run with `uv run pytest tests/accent_coach/test_norms_and_routing.py tests/accent_coach/test_score_ordering.py -v`.

### Priority 1 — GenAm accent routing (vowel bugs 1, 2, 3, 6)

GenAm scoring is structurally broken: the aligner applies RP-specific BATH/LOT phoneme
overrides unconditionally, the formant extractor subprocess always uses `target="rp"`,
and `compare()` / `score_vowels()` always fall back to `get_rp_norms()`. The net effect
is that every GenAm speaker is evaluated as if they were an RP speaker — with wrong
phoneme labels on BATH/LOT words and RP norm distances.

**Fix sequence**:
1. Add `accent_target: str = "rp"` to `_word_to_phoneme_instances()` in `alignment.py`,
   thread it through `_whisperx_align()`, `_mms_align()`, `align_audio()`.
   Gate the BATH override on `accent_target == "rp"`; gate the LOT override identically.
2. Add `target: str = "rp"` to `pipeline/formants.py::extract_formants()` and pass
   `"--target", target` in the subprocess call.
3. Add `accent_target: str = "rp"` to `compare()` in `scoring.py`. Branch:
   `get_rp_norms(mean_f0)` for `"rp"`, `get_genam_norms(mean_f0)` for `"genam"`.
4. Same param to `score_vowels()` in `vowels.py` — use `get_genam_norms()` fallback
   when `accent_target == "genam"` and `reference_norms is None`.

### Priority 2 — Data correctness (vowel bugs 5, 7)

- **LOT /ɒ/ centroid** (`rp_norms.py:24`): still `(600, 900)` — Deterding 1997 stub.
  The LOT override is live so tokens now reach this centroid. Re-measure:
  run `accent_coach_extract_formants.py` on `tts_output/modern_rp_corpus/` with the
  Phase 0.17 LOT override active and replace the stub with the corpus median.
  Discuss result with owner before committing — the measured value might shift scores.

- **`RP_VOWEL_F1_F2_MALE` alias** (`rp_norms.py:66`): points to `LEGACY` (Deterding 1997).
  Change to `RP_VOWEL_F1_F2_MALE_MODERN`. Then update the 4 test call-sites in
  `test_comparison.py` and `test_integration.py` that import this alias — their
  "perfect score" baselines are currently computed against wrong norms.

### Priority 3 — Rhythm acoustic/hybrid calibration mismatch (rhythm bug 1)

The bench measures native Fry clips with `extract_syllable_durations_acoustic()`, but
`RP_NPVI_MIN=40`, `RP_NPVI_MAX=62`, and `_DECAY=30` were calibrated from **hybrid**
measurements (+11 nPVI above acoustic). Acoustic native nPVI ≈ 29–41; with
`_NPVI_REF=51` and `_DECAY=30`, native at p25 scores 48 — below non-native owner (73.3).

**Do NOT just change thresholds** — discuss with owner first. Two valid fix paths:
- **Path A (consistent detection)**: wire `extract_syllable_durations_from_words()`
  (WhisperX hybrid) into the bench for native reference clips. Requires word-level
  alignment data for the reference clips.
- **Path B (split norms)**: keep acoustic detection in bench, maintain a separate
  acoustic-only norm set (`RP_NPVI_MIN_ACOUSTIC ≈ 29`, `RP_NPVI_MAX_ACOUSTIC ≈ 51`).
  Only use the hybrid-adjusted norms when the hybrid detector was used.

### Priority 4 — Small fixes (rhythm bugs 4, 5)

- **`_function_word_analysis()` returns 100.0 when no FW matched** (`rhythm.py:93`):
  Change to `return None, []` and treat it identically to the no-phoneme path
  (redistribute the 0.2 weight to nPVI+pattern).

- **`check_ranking()` has no native-beats-owner assertion** (`rhythm_bench.py:205`):
  Add a soft assertion that `RP mean score >= owner mean score - 10`, fail with
  explicit message if violated.

### Priority 5 — Config layer for constants

Move scoring constants out of module-level globals into a shared config file.
See the constants inventory at the end of this file.

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
