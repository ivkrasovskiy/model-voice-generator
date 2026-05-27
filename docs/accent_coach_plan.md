# Accent Coach — Phase 0 Implementation Plan

This plan tells an executing agent (Claude Sonnet) **exactly** how to build the
backend pipeline for the accent coach feature. The full feature spec lives at
[`docs/accent_coach_technical_spec.md`](accent_coach_technical_spec.md). This
document is **not a replacement** for that spec — it is a sequenced,
guardrailed execution plan for Phase 0 only (backend pipeline, no UI, no API).

Phase 0 ends when the validation experiments below pass on real audio.
**Do not start Phase 1 (FastAPI, React, SQLite) until Phase 0 is signed off.**

---

## Current status — 2026-05-21

> **Phase history (0.5–0.12)**: [`docs/accent_coach_history.md`](accent_coach_history.md)
> — consolidated repro summary for all retired phases. Active work is Phase 0.13.

**Phase 0 acceptance gate: NOT MET.** Vowel scoring validated (BC gets 79/100
on vowels against modern RP). Aspiration scorer miscalibrated — real BC gets
13.4; needs VOT detection fix before the B − C gap is meaningful.

| Step | Status |
|---|---|
| 1 Bootstrap (deps, skeleton, models) | ✅ Done |
| 2 Calibration sentences (50) | ✅ Done |
| 3 Forced alignment (WhisperX + cmudict G2P) | ✅ Done — stress-aware AH0/AH1 |
| 4 Formant extraction (parselmouth) | ✅ Done — voiced-fraction filter, F1 ceiling |
| 5 VOT extraction | ✅ Done |
| 6 Prosody (pitch, nPVI) | ✅ Done |
| 7 Feature aggregation | ✅ Done |
| 8 Reference norms (Deterding 1997) | ✅ Done — all values cited |
| 9 Comparison & scoring (6 modules) | ✅ Done — **vowel scale = 1.5** |
| 10 Diagnostics / articulatory advice | ✅ Done — output is meaningful |
| 11 CLI scripts | ✅ Done |
| 12 Unit tests (8 passing) | ✅ Done |
| 12b Integration tests (3, skip without audio) | ✅ Done |
| 13 Validation notebook (3-way: BC/you/RP) | ✅ Done |
| 14 CLAUDE.md update | ✅ Done |
| Exp A — BC synth ≥ 85 | ❌ Actual: ~54. Broken by rhythm/stress (see findings) |
| Exp B — Owner ≤ 65 | ⚠️ Actual: ~53. Correct direction, gate not met |
| Exp C — diff ≤ ±5 | ❌ Actual: +17. TTS biases in BC reference vowels |
| Accent classification (per-phoneme) | ✅ Meaningful — see findings |

### What is broken and why

**Composite score doesn't separate BC from owner (~54 vs ~53).**
Root causes — both documented in the findings doc:

1. **Rhythm score** (15% weight): TTS generates uniform timing → nPVI ≈ 37 vs RP
   target 55–75. BC gets penalised for a TTS artefact, not an accent feature.

2. **Stress score** (15% weight): G2P assigns uniform duration within words, so
   all syllables have identical duration by construction. The duration component
   of stress detection is always zero → scores ≈ 8–12/100 for everyone.

3. **TTS vowel biases**: IndexTTS generates /æ/, /ʌ/, /ɛ/ differently from
   Deterding 1997 norms. On those three phonemes BC scores *worse* than the owner,
   cancelling the /iː/ advantage (+29 gap) in the aggregate.

### What IS working and meaningful

The per-phoneme vowel analysis separates BC from owner on the diagnostically
important phonemes:

| Phoneme | BC score | Owner | Gap |
|---|---|---|---|
| /iː/ | 87 | 58 | +29 ✓ |
| /ʊ/ | 80 | 67 | +13 ✓ |
| /ɔː/ | 82 | 73 | +9 ✓ |

Owner's /iː/ F2 is 447 Hz from RP (vs BC's 133 Hz) — **3× worse**. This is the
primary Slavic accent marker and it IS being detected correctly.

The articulatory diagnostic strings are phonetically plausible and match what
a phonetician would say (see findings doc for details).

### What needs to change before the gate can pass

1. Fix syllable timing — replace uniform G2P with vowel-nucleus onset detection.
   This fixes both the stress score and improves formant window accuracy.
2. Update RP norms to post-2000 SSBE data (Deterding 1997 is 30 years old).
3. Either use real BC recordings as reference (removing TTS biases) or add a
   per-phoneme TTS-bias correction.
4. Consider down-weighting rhythm/stress (to 5% each) until timing is fixed.

---

## What Phase 0 must prove

Three claims, in order of importance:

1. The pipeline **reliably extracts** phoneme-level acoustic features
   (F1/F2 per vowel, VOT per stressed stop, pitch contour per sentence,
   syllable PVI per sentence) from arbitrary clean English speech.
2. The pipeline produces a **separating score**: BC's voice (real or
   cloned via existing IndexTTS-2) scores ≥ 85/100 vs canonical RP; an
   L2-English-Slavic accented speaker scores ≤ 65/100 against the same
   reference. The gap must be wide, monotonic, and explainable per-skill.
3. Per-vowel diagnostics map deviations to **plausible articulatory advice**
   (e.g. "your /æ/ has higher F1 than RP → tongue lower than target") that
   matches what a phonetician would say by ear.

If any of these claims fail after honest tuning, **stop and escalate**. Don't
paper over a broken pipeline by inflating scores.

---

## Hard rules — read before writing any code

These are non-negotiable. Violating any of them means restarting the affected
work.

1. **Do not touch `vendor/`.** The pinned IndexTTS-2 install is sacred.
   Don't edit, don't reinstall, don't bump pins.
2. **Do not modify `scripts/indextts_smoke_test.py`, `scripts/indextts_gen.py`,
   `scripts/posthoc_eval.py`, or `scripts/build_podcast_ref.py`.** They guard
   the existing TTS baseline.
3. **Run the smoke test before you start and after any change that touches
   shared Python deps:**
   `vendor/index-tts/.venv/bin/python scripts/indextts_smoke_test.py`.
   It must still pass (`cas_01` WER ≤ 0.10, ECAPA ≥ 0.74; `sher_03` same).
   If it stops passing, you broke something — find and revert.
4. **Use `uv` for every dependency change.**
   `uv add <pkg>` for runtime, `uv add --group dev <pkg>` for dev tools.
   Never `pip install` into `.venv/` directly.
5. **No module may exceed 500 lines** (including docstrings and blank lines).
   If a module is approaching the limit, split it by concern into
   sibling files in the same subpackage.
6. **`uv run ruff check accent_coach/ scripts/accent_coach_*.py tests/`
   must report 0 errors and 0 warnings before any commit.** Run
   `uv run ruff format` on the same paths.
7. **No FastAPI, no React, no SQLite, no WebSockets in Phase 0.** Even if
   the spec mentions them. Phase 0 is a Python package + CLI scripts +
   pytest tests + a validation notebook. Nothing else.
8. **No new top-level directories** beyond `accent_coach/`, `tests/`,
   `notebooks/` additions, and `docs/` updates. Do not create `backend/`,
   `frontend/`, `data/`, or anything else.
9. **Do not commit audio files.** `.gitignore` already excludes
   `tts_output/`, `data/`, `runs/`. Put user-recorded calibration audio
   under `tts_output/accent_coach/users/<speaker>/...` so it stays
   ignored.
10. **Do not invent reference values.** Every numeric constant in
    `accent_coach/reference/rp_norms.py` must have a citation comment
    pointing at a published source (Deterding 1997, Wells 1982,
    Cruttenden 2014, Lisker & Abramson 1964 for VOT, Grabe & Low 2002
    for PVI). If a value cannot be cited, mark it `# TODO(cite)` — do
    not silently make one up.
11. **Don't add error handling, fallbacks, or validation for scenarios
    that can't happen.** Trust internal code. Only validate at boundaries
    (audio file load, transcript input). Don't add try/except around
    every numpy call.
12. **No comments explaining what code does.** Only `# Why:` comments for
    non-obvious decisions (e.g., "Why max_formant=5500 for adult male:
    Praat docs recommend …"). Citations for numeric constants are
    encouraged.

---

## Target directory layout

```
accent_coach/
├── __init__.py
├── calibration/
│   ├── __init__.py
│   └── sentences.py           # phonetically-balanced sentence set
├── pipeline/
│   ├── __init__.py
│   ├── alignment.py           # WhisperX phoneme alignment + MMS fallback
│   ├── formants.py            # parselmouth F1/F2 at vowel midpoint
│   ├── vot.py                 # custom VOT for /p t k/
│   ├── prosody.py             # pitch, energy, syllable PVI
│   └── features.py            # aggregates into SentenceAnalysis
├── reference/
│   ├── __init__.py
│   ├── rp_norms.py            # canonical RP F1/F2/VOT/PVI (cited)
│   └── normalize.py           # Lobanov z-score across a speaker's vowels
├── comparison/
│   ├── __init__.py
│   ├── vowels.py
│   ├── aspiration.py
│   ├── rhythm.py
│   ├── stress.py
│   ├── intonation.py
│   ├── consonants.py          # fricative spectral centroid (light)
│   └── scoring.py             # composite + skill aggregation
├── diagnostics/
│   ├── __init__.py
│   └── advice.py              # F1/F2 deltas → articulatory advice text
├── models.py                  # pydantic dataclasses from the spec
└── cli.py                     # entry points used by scripts/accent_coach_*.py

scripts/
├── accent_coach_analyze.py    # analyze a single (user.wav, target.wav, transcript)
└── accent_coach_bench.py      # runs Validation Experiments A–D

tests/
└── accent_coach/
    ├── conftest.py            # audio fixtures (synthetic + real)
    ├── test_alignment.py
    ├── test_formants.py
    ├── test_vot.py
    ├── test_prosody.py
    └── test_comparison.py

notebooks/
└── accent_coach_validation.ipynb   # human-readable run of Experiments A–D

docs/
└── accent_coach_plan.md       # this file (already created)
```

**No other paths.** Do not create `backend/`, `accent_coach/api/`,
`accent_coach/storage/`, `frontend/`, or anything else mentioned in the
full spec — those are Phase 1.

---

## Dependencies (added via `uv add`)

```bash
uv add whisperx                   # forced phoneme alignment
uv add praat-parselmouth          # F1/F2 via Praat
uv add scipy                      # already pulled by sklearn but pin it
uv add dtw-python                 # DTW for pitch contour comparison
uv add --group dev pytest pytest-cov
```

Already in `pyproject.toml`: `librosa`, `numpy`, `torch`, `torchaudio`,
`openai-whisper`, `jiwer`, `soundfile`. Reuse these — don't add duplicates.

**After adding deps**: re-run the smoke test. If WhisperX pulls a conflicting
torch/transformers version, **stop**. Don't force the install. Tell the user.

---

## Calibration sentence set

Spec lists 25. We can run longer — 50 is fine; the user can read for a few
minutes. Extend `calibration/sentences.py` to **50 sentences** structured as:

| Bucket | Count | Purpose |
|---|---|---|
| Pure monophthongs | 11 | one sentence each for FLEECE, KIT, DRESS, TRAP, PALM/BATH, LOT, THOUGHT, FOOT, GOOSE, STRUT, NURSE |
| Schwa / weak forms | 3 | unstressed vowel reduction |
| Diphthongs | 5 | FACE, PRICE, CHOICE, GOAT, MOUTH |
| TH consonants | 2 | voiced + voiceless |
| Aspiration stops | 6 | two each for /p/, /t/, /k/ (word-initial stressed) |
| Fricatives | 4 | /s/ /z/ /ʃ/ /ʒ/ |
| Approximants / liquids | 3 | /r/, /l/ (clear+dark), /w/ |
| Sentence types for intonation | 6 | 2 yes/no, 2 wh, 1 declarative, 1 complex |
| Rhythm / linking / weak forms | 6 | longer connected-speech sentences |
| Stress-shift triplets | 4 | photograph/photographer/photography style |

Each sentence is a `Sentence` pydantic model with `id`, `text`,
`sentence_type`, `targets: list[str]`. Use the 25 from the spec as the
starting point and add 25 more.

---

## Reference data: RP canonical norms

`accent_coach/reference/rp_norms.py` ships **hardcoded** acoustic
references for modern Standard Southern British English (SSBE / modern RP):

- **Vowel F1/F2 (Hz)** for adult male and adult female reference speakers.
  Primary source: Deterding (1997), "The formants of monophthong vowels in
  Standard Southern British English pronunciation", JIPA 27(1-2).
  Include both sexes; pick at runtime based on estimated f0 of the analysed
  speaker.
- **VOT ranges (ms)** for stressed word-initial /p t k/. Source: Docherty
  (1992), Lisker & Abramson (1964). Expected: /p/ 55–80, /t/ 65–90,
  /k/ 75–100.
- **PVI (nPVI) ranges** for English. Source: Grabe & Low (2002). English
  nPVI typically 55–75; syllable-timed languages fall closer to 40.
- **Pitch contour templates** per sentence type: rising for yes/no,
  falling for wh, plateau-then-fall for declaratives. Represent as
  normalised 50-point arrays so DTW can match user contours.

**Every numeric constant must carry a citation comment.** Example:

```python
# Deterding 1997, Table 1, adult male SSBE
RP_VOWEL_F1_F2_MALE: dict[str, tuple[float, float]] = {
    "iː": (280, 2249),   # FLEECE
    "ɪ":  (367, 1757),   # KIT
    ...
}
```

If a constant cannot be cited, label it `# TODO(cite)` and flag it in
the Phase 0 sign-off report.

---

## Step-by-step plan with acceptance criteria

Each step ends with an explicit check. **Do not move on until the check
passes.** If a check fails, fix the prior step — don't push forward and
hope.

### Step 1 — Bootstrap

1. Run `vendor/index-tts/.venv/bin/python scripts/indextts_smoke_test.py`.
   Record the baseline scores.
2. `uv add whisperx praat-parselmouth dtw-python` and
   `uv add --group dev pytest pytest-cov`.
3. Create `accent_coach/` skeleton (all `__init__.py` files, empty
   subpackages).
4. Create `accent_coach/models.py` with the pydantic models from the
   spec: `PhonemeInstance`, `VowelFeatures`, `StopFeatures`,
   `SentenceAnalysis`, `ComparisonResult`, `VowelDiagnostic`,
   `RhythmBreakdown`, `AspirationBreakdown`, `IntonationBreakdown`.

**Check**: `uv run python -c "from accent_coach.models import *"`
imports cleanly. Smoke test still passes after deps added.

### Step 2 — Calibration sentences

Implement `accent_coach/calibration/sentences.py` with the 50-sentence
set (see "Calibration sentence set" section). Provide a typed list
`CALIBRATION_SENTENCES: list[Sentence]` and `get_by_id(i)`.

**Check**: `uv run python -c "from accent_coach.calibration.sentences
import CALIBRATION_SENTENCES; assert len(CALIBRATION_SENTENCES) == 50"`.

### Step 3 — Forced alignment

Implement `accent_coach/pipeline/alignment.py`:

- Primary: WhisperX `load_align_model("en")` returning a wav2vec2-based
  phoneme aligner. Function signature
  `align_audio(audio_path: Path, transcript: str) -> list[PhonemeInstance]`.
- Fallback: `torchaudio.pipelines.MMS_FA` if WhisperX fails to load
  (e.g. missing model download).
- Convert ARPABET ↔ IPA mapping in the same file (single dict). Set
  `is_stressed` from a word-stress lookup (use `g2p_en` or hand-rolled
  CMU dict subset for the calibration sentence words only — don't
  pull in a full lexicon).

**Check**:
- Run on `tts_output/ref_interview.wav` with its known transcript
  (transcribe via existing Whisper if needed).
- Assert: at least 80% of expected phonemes detected, timestamps
  monotonic, no zero-length phonemes, all timestamps within
  `[0, audio_duration]`.

### Step 4 — Formant extraction

Implement `accent_coach/pipeline/formants.py`:

```python
def extract_vowel_features(
    audio: np.ndarray, sr: int, phonemes: list[PhonemeInstance]
) -> list[VowelFeatures]
```

- Estimate speaker pitch range from a 5-second window of the audio
  (librosa.yin), pick `maximum_formant` from a small lookup
  (5000 if mean f0 < 165, else 5500).
- Sample F1/F2 at the **midpoint** of each vowel phoneme using
  `parselmouth.Sound.to_formant_burg(max_number_of_formants=5, ...)`.
- Reject any vowel under 30 ms (alignment noise).

**Check**: Extract F1/F2 from BC vowels in `ref_interview.wav`.
Compare to RP norms (Deterding 1997). At least 70% of detected
monophthongs should fall within ±150 Hz of the published RP centroid
for their category. If they don't, the formant ceiling is wrong —
re-tune the pitch→ceiling mapping. **Do not** loosen the tolerance
to make the test pass.

### Step 5 — VOT extraction

Implement `accent_coach/pipeline/vot.py`:

```python
def extract_vot(
    audio: np.ndarray, sr: int, stop: PhonemeInstance
) -> float
```

Algorithm (custom DSP):
1. Window: `[stop.start - 20ms, stop.end + 80ms]`.
2. Burst detection: high-frequency energy peak (2–8 kHz band, 5 ms
   hop) within the first 40 ms — first sample > 3× median.
3. Voicing onset: first sample where autocorrelation peak at lag
   corresponding to local f0 exceeds 0.5.
4. VOT = (voicing_onset_time − burst_time) × 1000 ms.

Only run on **word-initial stressed** /p t k/ (filter by
`is_stressed` and word-position metadata from alignment).

**Check**: Extract VOT from BC's stressed /p t k/ in
`ref_interview.wav` and `ref_sherlock.wav`. Expected: /p/ ≈ 55–80 ms,
/t/ ≈ 65–90, /k/ ≈ 75–100 (Lisker & Abramson 1964 RP-adjacent
ranges). At least 70% of detected stops must fall in range.

### Step 6 — Prosody

Implement `accent_coach/pipeline/prosody.py`:

- f0 via `librosa.yin(audio, fmin=70, fmax=400)`.
- RMS energy per frame.
- Syllable boundaries from alignment (collapse consonant clusters).
- Per-sentence: `pitch_contour` (downsampled to 50 points by
  interpolation), `syllable_durations` (list of seconds),
  `stress_pattern` (bool per syllable: duration + energy + pitch
  peak score above syllable mean).
- `nPVI` computed per sentence from `syllable_durations`.

**Check**: BC sentences should yield nPVI in 55–75 range (Grabe &
Low 2002, English).

### Step 7 — Feature aggregation

Implement `accent_coach/pipeline/features.py`:

```python
def analyse_audio(
    audio_path: Path, transcript: str, sentence_meta: Sentence
) -> SentenceAnalysis
```

Composes alignment → formants → VOT → prosody into a single
`SentenceAnalysis`.

**Check**: Run on one BC sentence; output validates against the
pydantic schema and contains non-empty `vowels`, `stops`,
`pitch_contour`.

### Step 8 — Reference norms

Implement `accent_coach/reference/rp_norms.py` and
`accent_coach/reference/normalize.py`:

- Hardcoded RP F1/F2 per vowel (male + female), VOT ranges per stop,
  nPVI range, pitch-contour templates per sentence type.
- Lobanov z-score normalization: given a speaker's full vowel set,
  z-score each F1 and F2 against the speaker's own mean and SD.

**Check**: `lobanov_normalize` on BC's vowels collapses the inter-
speaker variation: after normalization, BC's /iː/ centroid sits
within ±0.3 z of the published normalized RP /iː/ centroid.

### Step 9 — Comparison & scoring

Implement the six comparison modules and `scoring.py`:

- `vowels.py`: per-vowel normalized Euclidean (F1, F2) distance →
  `100 * exp(-d / scale)` per vowel, weighted average.
- `aspiration.py`: per stop, z-score VOT against RP distribution
  (mean + SD per phoneme), score = `100 * exp(-|z| / 1.5)`.
- `rhythm.py`: |user_nPVI − reference_nPVI| per sentence type, mapped
  to score via decay.
- `stress.py`: per multi-syllable word, boolean stress-position
  match → averaged.
- `intonation.py`: per-sentence DTW distance between user pitch
  contour and reference contour, normalised by speaker pitch range.
- `consonants.py`: fricative spectral centroid distance to RP
  reference (lightweight — fricatives only).
- `scoring.py`: composite weighted average. Default weights from
  spec: vowels 25 / consonants 15 / aspiration 15 / rhythm 15 /
  stress 15 / intonation 15.

**Check**: Construct synthetic perfect-match inputs (user = target)
in a test fixture; scoring returns 100/100 on every skill.
Construct deliberately-shifted inputs (every F1 shifted +200 Hz);
vowel score drops below 60.

### Step 10 — Diagnostics

Implement `accent_coach/diagnostics/advice.py`. For each vowel
deviation:

- F1 user > F1 ref by > 50 Hz → "tongue too low; raise toward palate"
- F1 user < F1 ref by > 50 Hz → "tongue too high; lower jaw slightly"
- F2 user > F2 ref by > 100 Hz → "tongue too far forward; retract"
- F2 user < F2 ref by > 100 Hz → "tongue too far back; advance"
- Combine F1 + F2 advice into one sentence per vowel.

**Check**: Hand-construct a `VowelFeatures` with `/iː/` that has
F1 +150 Hz and F2 −200 Hz vs RP; advice string contains "raise"
and "back".

### Step 11 — CLI

`scripts/accent_coach_analyze.py`: argparse with
`--user-wav`, `--target-wav` (optional; if omitted, use RP norms
directly), `--transcript`, `--sentence-id` (optional, picks
calibration sentence by id), `--out-json`. Emits a full
`ComparisonResult` as pretty-printed JSON.

`scripts/accent_coach_bench.py`: runs the four validation
experiments below. Reads a manifest CSV listing
`(speaker, label, wav_dir)` rows and emits a Markdown report
to `tts_output/accent_coach/bench/<run_id>/report.md`.

**Check**: `uv run python scripts/accent_coach_analyze.py
--user-wav tts_output/ref_interview.wav --transcript "..."
--out-json /tmp/result.json` produces a valid JSON output.

### Step 12 — Tests

Add the following pytest tests under `tests/accent_coach/`. **Keep
the suite small** — the user asked for "a couple of tests", not full
coverage:

1. `test_alignment.py` — load a short synthetic sine + speech
   fixture, run `align_audio`, assert phoneme count and time
   ordering.
2. `test_formants.py` — synthesize a vowel at known F1/F2 with
   `scipy.signal` (or use a tiny real BC vowel clip), run
   `extract_vowel_features`, assert within ±100 Hz.
3. `test_vot.py` — synthetic burst+voicing pair at known offset,
   assert recovered VOT within ±5 ms.
4. `test_comparison.py` — identity case returns 100; +1 z deviation
   on every vowel drops score below 60.

`tests/accent_coach/conftest.py` builds the synthetic fixtures in
memory — **do not commit binary audio files** for tests.

**Check**: `uv run pytest tests/accent_coach -v` → all green.

### Step 13 — Validation notebook

`notebooks/accent_coach_validation.ipynb` runs Experiments A–D
(below) end-to-end. Outputs:

- Per-experiment score table.
- Per-vowel F1/F2 scatter (BC vs RP, user vs RP, user vs BC).
- Per-stop VOT bar chart.
- Diagnostic text dump for the user vs RP experiment.

Kernel: project `.venv` (not vendor's). Notebook must run top-to-
bottom without manual intervention.

### Step 14 — CLAUDE.md update

Add **at most three sentences** to `CLAUDE.md` pointing at the
spec and this plan. Format suggestion (the orchestrator will write
this — Sonnet should leave a TODO comment, not author it):

> **Accent coach (in development)**: per-phoneme accent assessment
> pipeline lives under `accent_coach/`. Full spec at
> [docs/accent_coach_technical_spec.md]. Phase 0 plan at
> [docs/accent_coach_plan.md].

---

## Validation experiments (the only thing that defines "done")

These four experiments are the Phase 0 acceptance gate. Run them via
`scripts/accent_coach_bench.py`. **All four must produce the expected
result** before declaring Phase 0 complete.

### Experiment A — Synth BC voice against RP

- Target: canonical RP norms (no target audio).
- User: IndexTTS-2 synthesised BC voice on all 50 calibration sentences.
  Generate by extending the existing `scripts/indextts_gen.py` flow —
  do **not** modify that script; instead write the calibration
  sentences to a CSV `tts_output/accent_coach/cal_50.csv` and pass it
  to the existing CLI.
- **Expected**: composite score ≥ 85/100. Vowel and intonation
  skills both ≥ 85.
- **If fails**: most likely the formant ceiling auto-pick is wrong,
  or the synthesised audio has muddy /æ/ /ɛ/ that the alignment is
  mislabelling. Investigate before tuning thresholds.

### Experiment B — Real BC against RP

- Target: canonical RP norms.
- User: BC's real audio. Use the existing reference clips
  (`ref_interview.wav`, `ref_sherlock.wav`, `ref_narrator.wav`,
  `ref_combined.wav`) and run alignment + analysis against their
  transcripts (transcribe with existing Whisper if no transcript on
  disk). The clips don't match the 50 calibration sentences — that's
  fine; we're testing the pipeline on real RP-adjacent speech, not
  using the calibration set.
- **Expected**: composite score ≥ 80/100. Slightly lower than
  Experiment A is OK (real audio has more variance), but the gap
  vs Experiment C must still be large.

### Experiment C — User against RP

- Target: canonical RP norms.
- User: real user audio. **Required input from project owner**: the
  user (project maintainer) records all 50 calibration sentences and
  drops them in `tts_output/accent_coach/users/owner/`. Until those
  exist, this experiment is blocked.
- **Expected**: composite score in 40–65 range. Specific predictions
  (Slavic-L1-on-English transfer): VOT under-aspiration (aspiration
  score ≤ 50), vowel inventory collapse on /æ/ vs /ɛ/, /ɪ/ vs /iː/
  (vowel score ≤ 60), syllable-timed rhythm (nPVI 40–50, rhythm
  score ≤ 60).
- **If user scores > 75**, the pipeline isn't picking up real
  accent — likely formant extraction failing on the user's audio.
  Don't celebrate. Investigate.

### Experiment D — User against BC (synth or real)

- Target: BC synth audio for the same sentences (from Experiment A).
- User: same user audio as Experiment C.
- **Expected**: scores within ±5 points of Experiment C. This
  confirms that using BC-as-reference instead of RP-norms-as-
  reference produces consistent results.

### Acceptance gate

Phase 0 is **done** iff:

- (A) ≥ 85 AND (B) ≥ 80 AND (C) ≤ 65 AND (D) within ±5 of (C).
- Per-vowel diagnostic strings on Experiment C are non-empty and
  pass a quick human eyeball ("does this sound like advice a
  phonetician would give?").
- The smoke test still passes.
- `uv run ruff check accent_coach/ scripts/accent_coach_*.py
  tests/` reports 0 issues.
- `uv run pytest tests/accent_coach -v` is green.

If any of those fails, **do not move to Phase 1**. Open a short
write-up in `docs/accent_coach_phase0_findings.md` describing what
broke and what was tried, and stop.

---

## Things that are explicitly **not** in Phase 0

- FastAPI server, REST endpoints, WebSocket streaming.
- React frontend, charts, audio recording UI.
- SQLite session storage, history view.
- Voice library selector — Phase 0 always uses BC.
- Real-time analysis — batch only.
- Custom calibration sets — fixed 50.
- Multi-language support — English only.

Don't build any of these. Don't add stubs for them. Don't import
FastAPI "just in case". Phase 1 will pick them up.

---

## What to do at the end of Phase 0

1. Commit each step as a small, focused commit. Don't bundle.
2. Write `docs/accent_coach_phase0_findings.md` summarising:
   - Which RP norms ended up working / needed adjustment.
   - Surprises (e.g., "/æ/ formant detection failed on noisy mic
     recordings until I added a 80 Hz HPF").
   - The four experiment scores.
   - Open questions for Phase 1.
3. Open a PR titled `accent-coach: Phase 0 pipeline` with the
   findings doc as PR body.
4. Stop. Wait for sign-off before touching Phase 1.
