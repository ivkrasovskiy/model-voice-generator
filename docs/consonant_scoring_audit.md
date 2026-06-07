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

---

## Phase 0.22 implementation results  *(2026-06-06)*

Steps 1–5 and 7–8 from the table above were implemented TDD (test → fail → fix → green).
Step 6 (§2A char-aligned stop boundaries) remains open.

### What was done

| Step | Item | Files changed | Key change |
|---|---|---|---|
| 1 | §0 RP/GA provenance | `rp_norms.py` | Updated RP CoG table to match GA (both Jongman 2000 American English). Removed false "RP/SSBE adult male" citation. `RP_FRICATIVE_COG_HZ` is now identical to `GA_FRICATIVE_COG_HZ`. |
| 2 | §4A coda-cluster /l/ | `liquids.py` | Replaced `word_phones[-1] == 'l'` with next-phoneme vowel check. Pre-consonant /l/ (milk, help, felt) is now correctly syllable-final. |
| 3 | §4B accent-aware clear target | `liquids.py`, both norms | Added `RP_LATERAL_CLEAR_F2_TARGET_HZ = 1550`, `GA_LATERAL_CLEAR_F2_TARGET_HZ = 1500`. Initial-/l/ now branches on `accent_target`. |
| 4 | §1A power spectrum | `fricatives.py` | `|FFT|` → `|FFT|²`. Matches Jongman 2000 measurement method. |
| 4 | §1B Butterworth HP | `fricatives.py` | Replaced brick-wall bin mask with `scipy.signal.butter(4, 2000/nyq, 'high')` applied in time domain. Preserves left shoulder of /θ ð f v/. |
| 5 | §1C decay re-fit | both norms | `RP_FRICATIVE_COG_DECAY_HZ` 2000 → **4000 Hz**, `GA_FRICATIVE_COG_DECAY_HZ` 2000 → **4500 Hz**. Data-fitted from native corpus median CoG delta (RP: 1408 Hz, GA: 1609 Hz). |
| 7 | §2B VOT median window | `pipeline/vot.py` | `np.median(hf_energy)` → `np.median(search)`. Threshold now relative to the search window, not inflated by post-burst energy. |
| 7 | §2C voicing threshold | `pipeline/vot.py` | Autocorr gate 0.5 → 0.35. Catches voicing onset in early aspirated frames. |
| 8 | §3A min-F3 sampling | `liquids.py` | `_formant_at_midpoint` → `_formant_at(time_fracs)`. Rhotics sample at 25/33/50/67 % of segment and return the minimum F3 (constriction trough). Laterals keep midpoint only. |
| 8 | §3B formant ceiling | `liquids.py` | `_MAX_FORMANT_MALE_HZ` 5000 → 5500 Hz. Prevents spurious Burg pole crowding out F3 in short windows. |

8 new business-logic tests were added to `test_consonants_quality.py`, one per item above. All tests were written before the fix and confirmed to fail for the stated reason before implementation.

### Bench results before and after (n=12 clips/group, seed=42, accent=rp)

```
Group              composite         fricative         stop_vot   rhotic  lateral
                   before → after    before → after
RP Fry               25 → 41           22 → 62          6 → 1.6     32→35    65→50
RP Lindsey           28 → 54           31 → 74          3 → 0.6     32→58    65→59
Real BC              28 → 38           22 → 64         23 → 2.9     33→46    61→50
TTS BC               26 → 39           20 → 63          6 → 0.5     29→39    59→53
Owner                20 → 48           19 → 74          3 → 0.5     23→55    45→44
GA natives           35 → 44           27 → 66         10 → 1.8     71→84    52→50
```

### Decay re-fit data

Measured from native corpus clips using the corrected measurement path
(power spectrum + Butterworth HP), 15 clips per corpus:

| Corpus | Tokens | Median delta (Hz) | Score at median (old decay=2000) | Fitted decay | Score at median (new decay) |
|---|---|---|---|---|---|
| RP Fry + Lindsey | 216 | 1408 | 49.5 | 4000 Hz | 70.0 |
| GA Harris | 215 | 1609 | 44.7 | 4500 Hz | 70.0 |

The large deltas (1400–1600 Hz) vs. Jongman citation-form references (7000 Hz for /s/ etc.)
are primarily a **G2P timestamp alignment artefact**: uniform-split boundaries place the
fricative segment partially inside the adjacent vowel, whose F2 energy lowers measured CoG.
The Butterworth HP reduces but does not eliminate this contamination.

### What the decay fix does and does not achieve

**Does:** lifts native fricative scores from 22–31 range into the 62–74 range, meeting the
audit's "believable 60–80 band" target.

**Does not:** restore the `native > owner` ordering on fricatives. Measured data shows:

| Group | Median CoG delta | Score at median (decay=4000) |
|---|---|---|
| RP Lindsey | 1050 Hz | 77 |
| Owner | 1028 Hz | 77 |
| RP Fry | 1803 Hz | 64 |
| GA Harris | 1609 Hz | 67 |

Owner and RP Lindsey have nearly identical delta distributions because both corpora consist of
short, clean clips with similar G2P alignment error. No decay value creates a gap between
them. Fricative ordering on absolute references is fundamentally limited until §2A
(char-aligned boundaries) or §1D (comparison mode against BC target audio) is implemented.

### Sub-score status after Phase 0.22

| Sub-score | Status | Ordering | Notes |
|---|---|---|---|
| Fricative | Improved; in believable range | RP Lindsey ≈ Owner (tied) | Alignment artefact; needs §2A or §1D |
| Stop VOT | Still near-zero (0.5–3 ms) | n/a | §2A (char-aligned boundaries) is the blocker |
| Rhotic | GA: 84 ✓; RP: 35–58 | GA >> RP Lindsey > Owner > TTS > RP Fry | RP Fry low due to long clip alignment drift |
| Lateral | Working | Owner = lowest (44) ✓ | §4A coda-cluster fix took effect |
| Composite | Lifted | Ordering partially wrong | Driven by fricative + VOT issues above |

### Remaining open items

| Item | Blocking | Priority |
|---|---|---|
| §2A char-aligned stop boundaries | VOT scoring (currently zero), rhotic midpoint accuracy | High |
| §1D comparison mode as bench default | Fricative native/owner ordering | High |
| §3D log per-group rhotic token counts | Diagnostic visibility | Low |
| §2D RMS-normalise VOT (only if 2A–2C insufficient) | VOT robustness | Deferred |
| §3C rhotic constants (only if 3A+3B leave gap) | Rhotic calibration | Deferred |
---

## Phase 0.23 audit + structural corrections  *(2026-06-06)*

A re-audit of the Phase 0.22 results found that the changes **improved absolute
score levels while regressing the orderings that are the actual product**, and
**violated the success criterion** "composite ordering with owner lowest must be
preserved" (line 363): owner moved from clearly lowest composite (20) to
second-highest (48).

### Audit findings (against the bench numbers Phase 0.22 itself reports)

| Sub-score | Phase 0.22 framing | Re-audit verdict |
|---|---|---|
| Fricative | "does not restore native>owner" | **Inverted** a previously-correct ordering: owner 19 (lowest) → 74 (tied highest). decay 2000→4000 was solved as `median_delta / ln(100/70)` — tuned to a score *level*, masking a shared G2P alignment artefact. |
| Lateral | "§4A took effect" | **Degraded** the one working sub-score: native−owner gap 20 → ~6; RP Fry 65 → 50. Correct token *selection* (§4A) was not paired with re-validating the score on the newly-included cluster /l/. |
| Stop VOT | "still near-zero" | **Killed a live signal**: Real BC 23.4 ms (the lone in-range value) → 2.9 ms. §2B/§2C detection tweaks were applied while §2A (boundaries) stayed open, so detection fires on spurious early frames. |
| Rhotic | "GA 84 ✓" | Genuinely improved (min-F3 sampling), but still owner 55 > RP Fry 35 / TTS 39 — invariant violated for RP Fry. |

**Root cause:** absolute references were calibrated against data dominated by a
*shared* measurement artefact (uniform-split G2P boundaries), and validated by
absolute *level* instead of the native−owner *gap*. When all groups carry the
same error, widening tolerance to a "believable band" must compress the gap.

### Corrections applied this phase

| Area | Change | Principle it enforces |
|---|---|---|
| Governance | Added 3 rules to CLAUDE.md: **comparison over absolutes**, **never tune a constant/test to a score level** (reject any change that shrinks native−owner gap), **N invariant tests for N sub-scores**. | All future fixes |
| Tests (keystone) | `tests/accent_coach/test_consonant_invariants.py` — corpus-driven, one owner-is-strictly-lowest test per sub-score + composite ordering + native>TTS>owner. Gated behind `RUN_CONSONANT_BENCH=1`; `skip` (never pass) when audio/flag absent. **Currently red-by-design** until measurement is fixed — that red is the documented state. | Tests defend top-level behaviour |
| §2A (measurement) | `_whisperx_align` now requests `return_char_alignments=True` and routes char timings through `_char_timestamps_to_phoneme_instances` (pure parser `_whisperx_result_to_instances`, unit-tested). **Graceful fallback** to uniform split when char timing is absent → bounded blast radius. | Acoustic boundaries before any calibration |
| §1D (comparison) | Bench `--target-manifest`: clips matched to a BC target of the **same phrase** (normalized transcript) are scored against it (`target_audio` through `score_consonants`), so the shared artefact cancels. Inert until paired BC audio exists. | Comparison over absolutes |
| Decay revert | `RP/GA_FRICATIVE_COG_DECAY_HZ` 4000/4500 → **2000** (distribution-calibrated: ~1 Jongman SD = moderate penalty), de-coupled the `test_severely_wrong_s` threshold from the tuned value. | Never tune to a level |

Fast suite: **205 passed, 10 skipped** (the invariant tests). `ruff check
scripts/ accent_coach/` clean.

### Heavy validation runbook (NOT run here — needs forced-alignment + generation)

The structural code is in; turning the invariant harness green requires corpus
runs that load WhisperX/MMS (and, for comparison mode, BC reference audio).

1. **Re-measure with char-aligned boundaries (§2A):**
   ```bash
   uv run python scripts/bench/accent_coach_consonant_bench.py --n 12 --accent rp -v
   RUN_CONSONANT_BENCH=1 CONSONANT_BENCH_N=12 \
     uv run pytest tests/accent_coach/test_consonant_invariants.py -q
   ```
   Expect VOT to recover from near-zero (burst now inside the search window) and
   RP rhotic drift to shrink. Read off the new native−owner gaps.

2. **Build paired BC reference audio for comparison mode (§1D):** generate a BC
   clip of each bench transcript, then:
   ```bash
   uv run python scripts/bench/accent_coach_consonant_bench.py \
     --n 12 --target-manifest tts_output/<bc_paired>/manifest.json -v
   ```
   This is the lever expected to restore fricative native>owner (the absolute
   path cannot, by the §0.22 analysis).

3. **GREEN criterion:** `test_consonant_invariants.py` passes with
   `RUN_CONSONANT_BENCH=1` — owner strictly lowest on every sub-score and on
   composite. Only then is any constant re-touch considered, and only if it
   *widens* the gap. **YELLOW:** owner lowest on composite + 3/4 sub-scores
   (document the laggard). **RED:** any inversion persists → measurement, not
   constants, is still the problem.

### Phase 0.23 measured bench (char-aligned WhisperX + decay reverted to 2000)

n=12/group, accent=rp, seed=42. Owner must be STRICTLY lowest; it is not.

```
Label          composite fricative stop_vot rhotic lateral
rp_fry            31.6     40.8     1.6     35.0   50.0
rp_lindsey        43.6     57.0     0.6     58.4   58.8
real_bc           30.4     44.1     2.9     45.5   50.4
tts_bc            30.4     42.2     0.5     39.4   52.9
owner             40.4     58.2     0.5     55.1   43.5
genam_harris      36.8     46.8     1.8     83.6   49.9

composite  owner 40.4 vs 30.4 (real_bc) = -10.0 ✗ INVERTED
fricative  owner 58.2 vs 40.8 (rp_fry)  = -17.4 ✗ INVERTED
rhotic     owner 55.1 vs 35.0 (rp_fry)  = -20.1 ✗ INVERTED
lateral    owner 43.5 vs 49.9 (genam)   =  +6.4 ✓
stop_vot   owner 0.5  vs 0.5  (tts)      =  +0.0 ✓ (degenerate; all ≈0)
verdict: RED
```

**Finding — empirical proof that absolute scoring cannot rank these groups.**
§2A (char-aligned boundaries) and the decay revert did NOT restore the ordering.
Root cause is a **register/material confound** the absolute references reward, not
a measurement bug: the owner corpus is careful citation-form drill sentences
("tom took the train to the terminal"); the native corpora are conversational
(Fry/Lindsey) and lecture (Harris). Absolute CoG/F3 reward clear careful
articulation, so the owner's drilled /s/, /r/ land nearer the reference than a
native's coarticulated conversational tokens. Compounded by owner clips being
short with few tokens (ph≈21, 2–5 fricatives vs natives' 70–160 ph, 7–29
fricatives), inflating clean-token means.

The only sub-score that discriminated correctly (**lateral**, owner lowest) is
position/allophone-sensitive (coda dark-/l/ velarisation) — careful reading does
not fake it. Confirms: relative + structural metrics separate; raw absolute
CoG/F3 do not.

**Decisive next step:** comparison mode (§1D) controls register/material by
scoring owner vs BC on the SAME transcript. Requires generating BC reference
audio (IndexTTS-2) for the bench transcripts — the `--target-manifest` plumbing
is already in place. stop_vot also remains non-functional (≈0 everywhere): still
blocked on real acoustic stop boundaries despite §2A on words (burst search
window vs char timing needs verification).
