# Consonant Scoring Audit — Implementation Brief

**Date:** 2026-06-05 (rewritten after critical review)
**Audience:** implementer (Sonnet). This is a work order, not a finished design.
**Method:** TDD. For every item below, **write the business-logic test first**,
confirm it fails for the right reason (wrong score / wrong behaviour, not an
import error), then change the code until it passes. Do not implement a fix
before its test exists.

**No shortcuts.** Acoustic accuracy wins over speed everywhere. Compute time is
not a constraint in the scoring pipeline (CLAUDE.md "No speed-quality
trade-off"). Do not approximate formant/CoG/VOT measurement to go faster.

## Ground rules for this work

- Tests live in `tests/accent_coach/test_consonants_quality.py` (business logic)
  and `tests/accent_coach/test_consonants.py` (interface/TDD). Use synthetic or
  fixture audio with known properties so the expected score is derivable by hand.
- The invariant that every test ultimately defends: **native RP/GA speakers >
  TTS BC > owner** on each sub-score, and absolute native scores should land in
  a believable band (roughly 60–80 for a correct production), not 20–30.
- `accent_target` must change behaviour. A test that passes identically for
  `"rp"` and `"genam"` is not exercising the dual-accent path.
- After each fix: `uv run pytest tests/ -q` green and
  `uv run ruff check scripts/ accent_coach/` clean.

---

## Bench baseline for reference

```
Group              composite  fricative  stop_vot   rhotic  lateral
RP Fry                  25.1       22.3       6.1     31.8     64.5
RP Lindsey              27.6       30.5       3.2     32.2     64.8
Real BC                 28.4       22.0      23.4     33.2     60.7
TTS BC (IndexTTS-2)     25.6       19.9       6.2     28.8     59.4
Owner                   19.9       18.8       2.7     23.4     45.0
GA natives (Harris)     35.0       27.0       9.8     70.9     52.0
```

Only **laterals** currently rank natives above owner correctly. Fricatives are
compressed (everyone 18–30), VOT is near-zero for all groups (non-functional),
rhotics are broken for RP (32 vs GA 71). Composite ordering mostly holds, but
absolute scores are not yet usable for per-phoneme coaching.

---

## 0. RP vs GA provenance — fix the references before tuning anything

**This is the highest-priority correctness issue and was not in the original
audit.**

`accent_coach/reference/rp_norms.py:174` comments the fricative CoG table as
"RP/SSBE adult male — Jongman et al. (2000)". Jongman's corpus is **American
English**. `genam_norms.py:111` cites the *same* Jongman table, and the values
are essentially identical:

| Phoneme | RP_FRICATIVE_COG_HZ | GA_FRICATIVE_COG_HZ |
|---|---|---|
| s | 7000 | 7000 |
| z | 6500 | 6400 |
| ʃ | 3800 | 3700 |
| ʒ | 3300 | 3200 |

So the `accent_target` switch for fricatives is cosmetic — both branches resolve
to the same GA numbers under a false RP citation.

**What to do (pick ONE, document the choice):**

1. **Preferred — measure RP CoG from corpus.** Align the `modern_rp_corpus`
   clips (they have transcripts), extract CoG per fricative with the *corrected*
   measurement path (§1), and replace the RP table with corpus-derived
   conversational means. This is the no-shortcut option.
2. **Acceptable fallback — declare fricative CoG accent-neutral.** If you keep
   one shared table, *remove the false "RP/SSBE" citation* and state in the
   docstring that fricative CoG is place-of-articulation driven and not
   accent-distinguishing (the genam:115 comment already hints this). Then the
   `accent_target` branch for fricatives is honest about being a no-op.

**Test first:** assert that the RP and GA fricative tables are either (a)
measurably different where a corpus difference exists, or (b) explicitly the
same object with a docstring asserting accent-neutrality — not two copies
pretending to be independent.

Laterals (RP dark 1050 vs GA 1000) and rhotics (RP F3 1950 vs GA 1900) *are*
genuinely differentiated and sourced from Wells/Hagiwara. Leave those targets
alone except where §3 says otherwise.

---

## 1. Fricatives — `accent_coach/comparison/consonants/fricatives.py`

### 1A. Magnitude spectrum, not power spectrum (line 48) — CONFIRMED BUG

```python
spectrum = np.abs(np.fft.rfft(segment))   # magnitude
```

The docstring on line 42 says "using power spectrum" but the code computes
magnitude. Jongman et al. (2000), the source of the reference CoG values, used
the **power spectrum**. Code contradicts its own docstring and its reference.

**Fix:** `spectrum = np.abs(np.fft.rfft(segment)) ** 2`

**Test first:** synth a fricative-like signal (band-limited noise with a known
high-frequency peak). With a power spectrum the measured CoG must sit closer to
the energy peak than with magnitude. Assert the power-spectrum CoG on a
known-peak signal lands within tolerance of the analytic CoG.

### 1B. 2000 Hz hard high-pass mask over-suppresses low-CoG fricatives (lines 38, 50)

```python
_COG_HP_HZ: float = 2000.0
spectrum = spectrum * (freqs >= _COG_HP_HZ)
```

Zeroing every bin below 2 kHz removes the left shoulder of broad-spectrum
fricatives (/θ ð f v/), pushing their measured CoG above the reference and
producing false low scores for native speakers. /s z/ are unaffected (their
energy is far above 2 kHz).

**Do NOT tune 1A, 1B, and the decay (§1C) independently in one pass.** Two
biases currently partly cancel: the Jongman references are citation-form (too
high for conversational speech, per Phase 0.21 history) and the HP mask pushes
measured CoG up. Correct the measurement first (1A + 1B), re-measure native
clips, *then* decide on references/decay from data.

**Fix (no-shortcut option):** replace the hard frequency mask with a proper
4th-order Butterworth high-pass applied to the segment before the FFT, so the
transition is gradual rather than a brick-wall bin kill. Per-phoneme adaptive
cutoffs are a cheaper alternative but a Butterworth is the accurate one — use
it.

**Test first:** on a synthetic /θ/-like broadband signal whose analytic CoG is
known, assert that the corrected path measures CoG within tolerance of the true
value and that a native /θ/ no longer scores below ~60.

### 1C. Decay constant — tune from data, AFTER 1A+1B

`RP_FRICATIVE_COG_DECAY_HZ = 2000.0` (rp_norms.py:192). At decay=2000 a one-SD
delta (Jongman speaker SD ≈ 400–800 Hz) already costs ~33 points. This is
plausibly too strict, but **do not change it until the measurement fixes land
and you have re-measured native CoG**. If natives still fall below ~60 after
1A+1B with honest references, widen decay and justify it with the re-measured
gap, not a guess.

**Test first:** parametrised test asserting a native fricative produced within
one corpus SD of the reference scores ≥ ~65 with whatever decay you choose.

### 1D. Comparison mode is the right default when target audio exists (no bug)

When `target_audio` is passed the scorer already compares user CoG vs target CoG
(lines 79–105), which sidesteps the citation-form reference problem entirely.
The bench never passes target audio, so it only exercises the weaker absolute
path. **Action:** ensure the bench and `analyze_session()` pass target audio so
comparison mode is the default; absolute references become a sanity
cross-check, not the primary signal.

---

## 2. VOT (stops) — `accent_coach/pipeline/vot.py`

### 2A. Alignment is the real problem — fix this first

VOT is near-zero for every group because the burst detector runs on **G2P
uniform-split phoneme timestamps**, which rarely land on the acoustic stop
boundary. No window/threshold tuning recovers a burst that isn't in the search
window. Phase 0.22 already added `_char_timestamps_to_phoneme_instances`
(acoustic char boundaries) for the **MMS** path.

**Action:** wire stop extraction to the char-aligned timestamps, not uniform
G2P. The `_whisperx_align` path still uniform-splits; when WhisperX is run with
`return_char_alignments=True`, route it through the same helper. Until stops are
fed acoustic boundaries, treat any VOT score as untrusted.

**Test first:** a fixture clip with a hand-labelled stop boundary. Assert that
`extract_stop_features` fed char-aligned timestamps detects a burst and returns
a VOT in the plausible English range (voiceless aspirated /p t k/ ≈ 50–125 ms);
fed deliberately-offset uniform timestamps, it returns `None` (documents the
failure mode rather than emitting a garbage number).

### 2B. Median computed over full post-window, not search window (line 55) — CONFIRMED BUG

```python
median_e = np.median(hf_energy)   # uses all frames
burst_candidates = np.where(search > 3 * median_e)[0]
```

The 3× threshold is computed against the median of the whole 220 ms window
(burst + aspiration + following vowel), not the search window the burst is
actually sought in. Inconsistent reference for the threshold.

**Fix:** `median_e = np.median(search)`

**Test first:** synth a closure-then-burst-then-voicing signal where the burst
is a known frame. Assert the corrected median yields detection of that burst
frame; assert the old full-window median misses it. This bug is real but is a
*second-order* effect behind 2A — it only matters once boundaries are correct.

### 2C. Voicing-onset threshold 0.5 → 0.35 (line 71) — tuning, defensible

At true voicing onset the first few ms have autocorrelation peaks of ~0.35–0.45,
so a 0.5 gate can miss the real onset by several ms and inflate VOT. Lowering to
0.35 is reasonable.

**Test first:** a synthetic voiced segment whose onset frame is known; assert
the detected onset frame matches within one hop at threshold 0.35 and is late at
0.5.

> **Removed from the original audit:** the claim that the autocorrelation
> `min_lag` window is "wrong for 16 kHz" and should move from ~16 to 64 samples.
> `min_lag` only skips the zero-lag self-peak; at 16 samples (~1 ms) the male
> pitch peak (133–160 samples for 100–120 Hz) is already well inside the search
> range. Changing it does nothing for male speech. Do not implement this.

### 2D. RMS-normalise before energy (lines 17–26) — only if 2A–2C are insufficient

Absolute bandpass energy makes the 3×median threshold sensitive to recording
level. RMS-normalising the filtered signal before framing removes that
sensitivity. This is a robustness improvement; add it only if VOT detection is
still level-dependent after 2A–2C, and back it with a test that runs the same
clip at two gains and asserts the same VOT.

---

## 3. Rhotics — `accent_coach/comparison/consonants/liquids.py`

**Diagnose before changing any constant.** RP rhotics score ~32 while GA scores
~71, even though the scorer only fires on **pre-vocalic /r/** (liquids.py:219
gate) — the one position where RP [ɹ] and GA bunched/retroflex /r/ are
acoustically most alike (both strong approximants with deeply depressed F3). A
50-point gap there is a **measurement artefact, not a target-value problem.**

The scorer rewards *low* F3 (`if f3 <= target: 100`). For RP to score low, F3
must be measured *too high* — i.e. **overestimation**, from one of:

### 3A. Midpoint may land in the adjacent vowel (line 85)

```python
mid = (p.start_time + p.end_time) / 2
```

With uniform G2P timestamps the window centre is the word-time midpoint, which
can fall in the following vowel where F3 has already risen. **Fix:** take the
**minimum F3 across several sample points** (e.g. 25/33/50/67%) to find the true
constriction trough rather than a single midpoint.

### 3B. F3/F4 confusion at the 5000 Hz male ceiling (line 50)

`_MAX_FORMANT_MALE_HZ = 5000.0` with 5 formants in a 25 ms window can let Burg
insert a spurious pole, pushing measured F3 up. **Fix candidate:** raise the
male ceiling (e.g. 5500 Hz) so the tracker has headroom to place F4/F5
correctly. Validate, don't assume.

**Test first (covers 3A+3B together):** a fixture native RP /r/ clip with a
hand-measured F3 (or a synthetic vowel-/r/-vowel with a known F3 trough). Assert
the measured F3 lands within ~150 Hz of truth and the score is ≥ ~65. Build a
small diagnostic that prints raw measured F3 per /r/ token on native clips and
commit the numbers — the constant changes in §3C are only justified once these
prints show whether the error is over- or under-estimation.

### 3C. Target/decay constants — change ONLY if 3A+3B don't close the gap

> **Removed from the original audit:** the recommendation to raise
> `RP_RHOTIC_F3_TARGET_HZ` 1950 → 2050 and `RP_RHOTIC_F3_DECAY_HZ` 350 → 450 was
> justified by a self-contradictory premise (it claimed LPC *underestimates* F3
> while illustrating an *over*estimate). The code already encodes RP>GA via 1950
> vs GA's 1900. Do not bump these constants to mask a measurement bug. If, after
> 3A+3B, correctly-measured native F3 still scores low, revisit decay with the
> measured native F3 distribution as justification.

### 3D. Pre-vocalic gate reduces RP token count (diagnostic only)

The gate correctly skips post-vocalic /r/ for non-rhotic RP, leaving ~4–6 tokens
per clip vs GA's 12–15, so the RP rhotic mean is higher-variance. Not a bug;
**log per-group rhotic token counts in bench output** for visibility.

---

## 4. Laterals — `accent_coach/comparison/consonants/liquids.py`

Laterals are the one working sub-score. The fixes here are small and safe.

### 4A. Coda-cluster heuristic misses syllable-final-but-not-word-final /l/ (line 230) — CONFIRMED BUG

```python
is_final = bool(word_phones) and word_phones[-1] == "l"
```

Marks /l/ syllable-final only when it is the *last* phoneme of the word. Coda
clusters keep /l/ syllable-final without being word-final:

| Word | CMU phones | /l/ | Current | Correct |
|---|---|---|---|---|
| call | K AO1 L | final | True | ✓ |
| milk | M IH1 L K | pre-cons | False | ✗ |
| help | HH EH1 L P | pre-cons | False | ✗ |
| felt | F EH1 L T | pre-cons | False | ✗ |

Affected /l/ tokens are scored with the initial-position formula (no dark-/l/
penalty), so clear /l/ passes. Systematic for ~30% of /l/ tokens.

**Fix:** /l/ is syllable-final if the **next phoneme in the word is a consonant,
or there is no next phoneme.** Determine position by **iterating to find this
specific instance's index** — do **not** use `word_phones.index("l")`, which
returns the first /l/ and breaks on words with two (e.g. "little" L IH T AH L).
Reuse `IPA_VOWELS` from `pipeline/alignment.py` for the vowel test.

**Test first:** parametrise over {call, milk, help, felt, little} with synthetic
F2 set to a *clear* value, and assert each /l/ that is syllable-final gets the
dark-/l/ penalty (low score), while a genuinely word-initial clear /l/ (e.g.
"light") does not.

### 4B. Hardcoded `clear_target = 1550.0` ignores accent_target (line 153) — minor

The initial-position branch uses a literal 1550 Hz for both RP and GA. Low
impact (initial /l/ is lightly scored) but inconsistent with the dual-accent
rule.

**Fix:** add `GA_LATERAL_CLEAR_F2_TARGET_HZ` to `genam_norms.py` (and the RP
equivalent to `rp_norms.py`) and branch on `accent_target` instead of the
literal.

**Test first:** assert the initial-/l/ score for an identical F2 differs between
`"rp"` and `"genam"` when the two clear targets differ.

### What is already correct (do not touch)

- Dark /l/ exponential-decay scoring with the velarisation threshold.
- Clear-/l/ detection in final position (F2 above threshold → penalty).
- Ordering RP (64–65) > owner (45).
- GA dark target 50 Hz below RP (linguistically motivated).

---

## 5. Alignment status (context, no action here)

`_mms_align` now uses `_char_timestamps_to_phoneme_instances` (linear
interpolation over acoustic char boundaries) — see
`tests/accent_coach/test_alignment.py` Section 3. `_whisperx_align` still
uniform-splits; upgrading it with `return_char_alignments=True` through the same
helper is the dependency for §2A (VOT) and helps §3A (rhotic midpoint).

---

## Suggested implementation order (TDD, each item: test → fix → green)

| Step | Area | Item | Why this order |
|---|---|---|---|
| 1 | References | §0 fricative RP/GA provenance | Don't tune against false references |
| 2 | Lateral | §4A coda-cluster heuristic | Cheap, correct, improves the working sub-score |
| 3 | Lateral | §4B accent-aware clear target | Same file, finishes dual-accent coverage |
| 4 | Fricative | §1A power spectrum + §1B Butterworth HP | Fix measurement before decay/refs |
| 5 | Fricative | §1C decay (data-driven) + §1D comparison-mode default | Only after re-measuring natives |
| 6 | Alignment | §2A char-aligned stop boundaries | Unblocks VOT and rhotic midpoint |
| 7 | VOT | §2B median + §2C threshold (+ §2D if needed) | Second-order once boundaries are real |
| 8 | Rhotic | §3A min-F3 sampling + §3B ceiling, with diagnostic prints | Settle over/under-estimation with data |
| 9 | Rhotic | §3C constants — only if step 8 leaves a gap | Avoid masking measurement bugs |

**Success criterion:** after steps 1–8, on the bench, native RP/GA sub-scores
land in a believable band (target ~60–80 for correct productions), the ordering
natives > TTS > owner holds on every sub-score, and no sub-score is uniformly
near-zero. Composite ordering with owner lowest must be preserved throughout.
```