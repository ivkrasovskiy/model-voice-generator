# Accent Coach — Phase 0 Findings

> **Related**: execution plan with step-by-step status →
> [`docs/accent_coach_plan.md`](accent_coach_plan.md)  
> Full feature spec → [`docs/accent_coach_technical_spec.md`](accent_coach_technical_spec.md)

Status as of 2026-05-20. Written after running all 50 calibration sentences
through the full pipeline with real recordings (owner) and IndexTTS-2 BC synth.

---

## What is done and working

### Pipeline steps
- **Forced alignment** (`pipeline/alignment.py`): WhisperX word-level timestamps +
  NLTK cmudict G2P. Stress-aware ARPABET: AH0→/ə/ (schwa), AH1→/ʌ/ (STRUT).
  Fallback to torchaudio MMS_FA. Gives ~8-10 vowel tokens per calibration sentence.
- **Formant extraction** (`pipeline/formants.py`): Parselmouth Burg LPC with
  voiced-fraction filter (≥35% voiced frames, fmin=50 Hz) and F1 ceiling at 900 Hz.
  Eliminates unvoiced outliers. Speaker pitch ceiling auto-selected (5000/5500 Hz).
- **Prosody** (`pipeline/prosody.py`): pitch contour (50-point), nPVI, syllable
  durations, stress pattern.
- **VOT** (`pipeline/vot.py`): burst+voicing-onset DSP for word-initial /p t k/.
- **Reference norms** (`reference/rp_norms.py`): Deterding 1997 F1/F2 (male+female),
  Lisker & Abramson 1964 VOT, Grabe & Low 2002 nPVI. All values cited.
- **Per-phoneme diagnostics** (`diagnostics/advice.py`): F1/F2 delta →
  articulatory advice strings. These are the most meaningful output of Phase 0.
- **Recording studio** (`notebooks/accent_coach_record.ipynb`): mic recording
  widget for all 50 calibration sentences. Works.
- **All unit tests** (`tests/accent_coach/`): 8 tests green.
- **Integration tests** (`tests/accent_coach/test_integration.py`): 3 tests that
  assert /iː/ separation and /æ/ raising. Skipped when audio not present.

### Key finding: accent classification
The per-phoneme diagnostic table is **the meaningful output of Phase 0**.
The owner's deviations from RP are consistent with **East Slavic (Russian/Ukrainian) L1**:

| Phoneme | ΔF1 Hz | ΔF2 Hz | Slavic feature |
|---|---|---|---|
| /iː/ FLEECE | +194 | **−447** | Russian /и/ is more central; strongest marker |
| /æ/ TRAP | **−186** | −34 | No /æ/ in Russian; speakers use /ɛ/-like substitute |
| /ʌ/ STRUT | −157 | +312 | /a/ substitution, common in Slavic |
| /ʊ/ FOOT | +117 | **+488** | Back vowels fronted; Slavic general |
| /ɔː/ THOUGHT | +116 | +594 | Same pattern |

This matches the plan's prediction exactly ("vowel inventory collapse on /æ/ vs /ɛ/,
/ɪ/ vs /iː/").

---

## What is done but does not work

### Composite score (broken — do not report to users)
Phase 0 target: BC ≥ 85, owner ≤ 65. Actual: BC ≈ 54, owner ≈ 53.

Two skills are broken and drag both speakers to ~50:

**1. Rhythm score (15% weight)**
- nPVI target range: 55–75 (Grabe & Low 2002 English).
- BC synth nPVI ≈ 37 — well below range.
- Root cause: TTS systems produce more uniform timing than natural speech.
  The rhythm scorer penalises BC for a TTS artefact, not an accent feature.
- Fix needed: use real BC recordings for reference, or down-weight rhythm when
  the reference is TTS audio.

**2. Stress score (15% weight)**
- BC stress score ≈ 8/100, owner ≈ 12/100. Both near-random.
- Root cause: the stress detector relies partly on syllable **duration** to
  identify stressed syllables. The G2P alignment distributes word duration
  uniformly across phonemes, so all syllables have identical duration by
  construction. The duration component of stress detection is always zero.
- Fix needed: use vowel-onset timestamps from Parselmouth or a vowel-nucleus
  detector to get real syllable duration, not G2P-derived uniform chunks.

**3. Composite score separation**
Even with vowel scores (BC=75.5 vs owner=73.3), the 30% combined weight of
broken rhythm+stress pulls both composites to ≈53. The 2-point vowel gap
contributes only 0.5 composite points.

### Vowel scoring — partial separation only
Per-phoneme vowel scores (scale=1.5, raw Hz, Deterding 1997 norms):

| Phoneme | BC score | Owner score | Gap | Reliable? |
|---|---|---|---|---|
| /iː/ | 87 | 58 | +29 | ✓ strong signal |
| /ɔː/ | 82 | 73 | +8 | ✓ |
| /ʊ/ | 80 | 67 | +13 | ✓ |
| /ɪ/ | 78 | 72 | +6 | ✓ |
| /æ/ | 74 | 78 | −4 | ✗ BC worse (see note) |
| /ʌ/ | 68 | 79 | −11 | ✗ BC worse (see note) |
| /ɛ/ | 73 | 88 | −15 | ✗ BC worse (see note) |

Note: BC scores *worse* than owner on /æ/, /ʌ/, /ɛ/. Root cause: the IndexTTS
model generates these vowels differently from Deterding 1997 norms. BC's real
voice produces /æ/ with F2≈1366 Hz vs RP 1710 Hz (−344 Hz backed), and /ʌ/ with
F1≈382 Hz vs RP 623 Hz (−241 Hz raised). These are TTS biases, not real BC accent
features. They partially cancel the /iː/ advantage in the aggregate vowel score.

When averaging over all phonemes equally, BC ≈ 75.5 and owner ≈ 73.3 — only 2
points apart. Not useful as a single "accent score".

### Lobanov normalisation (abandoned for now)
Implemented in `reference/normalize.py` and wired into `comparison/vowels.py`
via `speaker_params` / `ref_params` arguments. However: Lobanov normalization
requires a balanced phoneme distribution to compute a meaningful grand mean/SD.
The RP "speaker" is only 17 data points (one per phoneme in the norms table),
making its SD unreliable. Lobanov mode produces z-score distances too large for
the current scale and made scores worse. Left in the code for future use with a
proper RP corpus; disabled in the notebook.

---

## What is not done (Phase 0 incomplete items)

- **Experiment A acceptance gate not met**: BC composite should be ≥ 85, is ≈ 54.
  Blocked by broken rhythm/stress scores.
- **Experiment C gap too large**: You vs BC diff = +17 (target ±5). Partly because
  BC synth has TTS-biased vowels that aren't a fair phonetic target.
- **Smoke test after all changes**: need to re-run to confirm IndexTTS still intact.

---

## What needs to change for Phase 1 to make scoring meaningful

1. **Fix syllable timing**: replace uniform G2P duration with a vowel-nucleus
   detector (Parselmouth intensity + F0 to find vowel onsets). This fixes the
   stress score and improves formant window accuracy.

2. **Use real BC recordings as reference**: record or source real BC speech on the
   same 50 calibration sentences, replacing the IndexTTS synth for Experiments A/D.
   This removes TTS biases from /æ/, /ʌ/, /ɛ/ and would give a fair reference.

3. **Update RP norms**: Deterding 1997 norms are nearly 30 years old. Modern SSBE
   has shifted significantly (especially /uː/ fronting). Use Hawkins & Midgley 2005
   or more recent corpus data. TODO(cite) entries in `rp_norms.py` need filling.

4. **Per-phoneme weighted score**: weight diagnostically important phonemes higher
   in the vowel aggregate. /iː/ has a 29-point BC vs owner gap — it should count
   more than /ɛ/ (which currently makes BC look worse).

5. **Down-weight or replace rhythm/stress**: either fix the syllable duration issue
   or reduce the rhythm/stress weights from 15%+15% to 5%+5% until the timing
   is fixed, redistributing to vowels.

---

## Scores to report (as of Phase 0)

| Metric | BC synth | Owner | Separation? |
|---|---|---|---|
| Composite | 54 | 53 | ✗ |
| Vowel skill | 76 | 73 | Marginal |
| /iː/ score alone | 87 | 58 | ✓ (+29) |
| /ʊ/ score | 80 | 67 | ✓ (+13) |
| /ɔː/ score | 82 | 73 | ✓ (+9) |
| /iː/ F2 gap from RP | 133 Hz | 447 Hz | ✓ (3× worse) |
| /æ/ F1 gap from RP | 74 Hz | 186 Hz | ✓ (2.5× worse) |

The per-phoneme separation IS real and consistent. The composite is not ready.
