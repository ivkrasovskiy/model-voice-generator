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
