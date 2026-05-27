# Accent Coach — Phase 0.13 Execution Brief: Levers A and B

> **Audience**: executing agent. **Do not reason about scope.** Decisions are made.
> Write code per the tasks, verify each by running the listed commands, stop on
> any failed verification.

---

## 0. Context (one paragraph)

Phase 0.10 ([findings](accent_coach_history.md#phase-010)) swept GPT + CFM params
(N=3 Optuna, 35 trials). Verdict: RED. Best `composite_lindsey_mean = 62.10`
(gate 82), best `composite_BC_mean = 70.80` (gate 75). Phase 0.11 swept
full-sentence Fry emo clips at `emo_alpha ∈ {0.3, 0.5, 0.7, 1.0}` and Phase 0.12
rebuilt Lindsey/Fry/modern_rp centroids via ECAPA-filtered corpus (sim ≥ 0.50).
After cleanup, the cross-speaker matrix shows `synth_BC vs modern_rp = 75.4`,
`real_BC vs modern_rp = 88.9` — a **13.5-point gap** concentrated in 5
phonemes (`/ʌ/, /ʊ/, /ɔː/, /aʊ/, /ɜː/`) where real BC scores 86–100 but synth
scores 30–59. Param sweep + emo sweep have shown a ~2–3 pt ceiling. Phase 0.13
tests two remaining levers from the
[Phase 0.10 findings §"Remaining TTS accent levers"](accent_coach_history.md#phase-010#remaining-tts-accent-levers).

**Order chosen: Lever B before Lever A.** Lever B is deterministic, cheap to
build (no generation budget), and yields a ceiling number that gates Lever A.
If signal-level edits cannot close ≥5 pts of the gap, the gap is in the model
and Lever A's emo conditioning (also a signal-shaping mechanism) will not crack
it either. If the ceiling is ≥10 pts, Lever A's production-viable form is worth
the build cost.

**Phase 0.13c (F3-based normalization bench) runs independently of A and B.**
It is purely diagnostic — it does NOT change the production scoring path. Can
run in parallel with A and B, in any order, and is gated on nothing.

---

## 1. Hard constraints — DO NOT change these

| Constraint | Value | Reason |
|---|---|---|
| Speaker reference clip | `tts_output/ref_interview.wav` | Locked since Phase 0.9 |
| Base gen params | `num_beams=5, temperature=0.8, top_p=0.8, cfg_rate=0.7, diffusion_steps=25` | Phase 0.9 baseline (NOT Phase 0.10 trial #0 — that was N=1, unreliable) |
| Phrases | `tts_output/accent_coach/cal_25.csv` | Same as Phase 0.10/0.11 — apples-to-apples |
| Replicates per cell | **N=3** | No N=1 anywhere |
| Centroid overlay | `tts_output/accent_coach/cleaned_corpus/speaker_centroids_cleaned.json` | Phase 0.12 output; `load_baseline_centroids()` picks it up automatically |
| Failing phonemes (5) | `/ʌ/, /ʊ/, /ɔː/, /aʊ/, /ɜː/` (STRUT, FOOT, THOUGHT, MOUTH, NURSE) | From [Phase 0.10 findings §Cross-speaker validation](accent_coach_history.md#phase-010#cross-speaker-validation-matrix-phase-012) |
| sigma_rp | from `{fry, lindsey, bbc_male}` cluster | Same as Phase 0.10 |
| ECAPA reference | `tts_output/ref_narrator.wav` | Per-clip refs unavailable |
| Primary target | `modern_rp` (cleaned) | Use lindsey + real_BC + fry as diagnostic targets |
| Device | CPU (MPS blocked by BigVGAN) | Per Phase 0.10 Appendix 9 |

---

## 2. Code reuse map — USE these, do NOT reimplement

| Need | Existing symbol | Location |
|---|---|---|
| Clip generation (CLI subprocess) | `generate_clips(...)` | [accent_coach/pipeline/experiment.py:60](../accent_coach/pipeline/experiment.py#L60) |
| Formant extraction | `extract_formants(...)` | [accent_coach/pipeline/experiment.py:133](../accent_coach/pipeline/experiment.py#L133) |
| Per-phoneme centroid build | `build_centroids_from_formants(...)` | [accent_coach/pipeline/experiment.py:167](../accent_coach/pipeline/experiment.py#L167) |
| Load baseline (cleaned overlay) | `load_baseline_centroids()` | [accent_coach/pipeline/experiment.py:204](../accent_coach/pipeline/experiment.py#L204) |
| Score with per-phoneme breakdown | `score_against(...)` returns `{'composite', 'per_phoneme'}` | [accent_coach/pipeline/experiment.py:232](../accent_coach/pipeline/experiment.py#L232) |
| Posthoc WER/ECAPA/DNSMOS | `score_posthoc_clips(...)` | [accent_coach/pipeline/experiment.py:303](../accent_coach/pipeline/experiment.py#L303) |
| Phase 0.12 corpus audit (per-clip ECAPA sim) | `tts_output/accent_coach/corpus_audit/{lindsey,fry}_audit.json` | already produced |
| Phase 0.12 transcripts (cached) | `tts_output/accent_coach/cleaned_corpus/{lindsey,fry}/transcripts.json` | already produced |
| Per-token formant rows (with timing) | `tts_output/accent_coach/cleaned_corpus/{lindsey,fry}/formants.csv` | already produced — has `start_s`, `end_s`, `phoneme`, `F1`, `F2`, `clip_id` columns |

Anti-pattern: do NOT write a new Whisper alignment loop, a new centroid
builder, a new piecewise scorer, or a new bark transform. They exist.

---

## 3. Phase 0.13a — Lever B (formant shift ceiling)

### 3.1. Goal

Quantify how much of the 13.5-pt `modern_rp` gap is recoverable by directly
editing F1/F2 of the 5 failing phonemes in already-generated synth_BC clips
toward the cleaned `modern_rp` centroid. This is a **ceiling measurement**, not
a production path.

### 3.2. Hypothesis

H1 (signal-bound): if the gap is in the acoustic placement of vowels, formant
editing closes most of it; per-phoneme scores for the edited 5 jump from
30–59 → 80+, composite lifts 8–12 pts, DNSMOS drops < 0.5.

H2 (model-bound): if the gap is in duration, spectral tilt, formant bandwidths,
or VOT, formant-only editing barely moves the score (≤ 3 pts) and DNSMOS drops
hard (> 1.0) from artifacts.

### 3.3. Dependencies

```bash
uv add pyworld parselmouth
# Note: parselmouth is the praat Python binding (already used by
# accent_coach_extract_formants.py — verify presence first; only add if missing)
```

Verify:
```bash
.venv/bin/python -c "import pyworld; import parselmouth; print(pyworld.__version__, parselmouth.__version__)"
```

### 3.4. Design — 7 cells, N=3 each

| Cell ID | Description |
|---|---|
| `cell_baseline_b` | No edit — re-generate Phase 0.9 baseline on cal_25 (cross-check: should match Phase 0.11 cell `00_baseline`) |
| `cell_strut` | Shift /ʌ/ only → `modern_rp` centroid |
| `cell_foot` | Shift /ʊ/ only |
| `cell_thought` | Shift /ɔː/ only |
| `cell_mouth` | Shift /aʊ/ only |
| `cell_nurse` | Shift /ɜː/ only |
| `cell_all5` | Shift all 5 simultaneously |

Per-cell metrics: `composite_modern_rp`, per-phoneme scores for the 5 targets,
WER, ECAPA, DNSMOS_OVR (mean ± std over N=3 reps).

### 3.5. Tasks

#### B1. Generate the synth_BC base clips (3 replicates)

**What**: Reuse `generate_clips()` to produce 3 replicate runs of cal_25 at
Phase 0.9 baseline params. If `cell_baseline_b/rep_{i}/manifest.json` already
covers Phase 0.11 cell `00_baseline` (same params), symlink instead of
re-generating to save ~30 minutes.

**Files to create**:
- `scripts/accent_coach_phase0_13_lever_b.py` — the cell driver
- `tts_output/accent_coach/phase0_13/cells/cell_baseline_b/rep_{0,1,2}/...`

**Replicate seeding**: IndexTTS-2 with `do_sample=True` is non-deterministic;
each rep is a fresh subprocess call to `indextts_gen.py`. Output dirs differ
(`rep_0`, `rep_1`, `rep_2`) so the idempotency check in `generate_clips()`
does not skip later reps.

**Verify**:
```bash
.venv/bin/python scripts/accent_coach_phase0_13_lever_b.py --cell cell_baseline_b --replicates 3
```
Expected: 3 × 25 = 75 WAVs under `cells/cell_baseline_b/rep_{0,1,2}/clips/`.
Mean `composite_modern_rp` over 3 reps should land in `[72, 78]` (Phase 0.12
cross-speaker reported synth_BC = 75.4).

If outside this band, stop and report — something has drifted from Phase 0.11.

#### B2. Per-token alignment from existing formants CSV

**What**: For each generated WAV (post B1), produce a per-token segment table.
**Reuse**: `extract_formants()` already runs whisper-based forced alignment via
[scripts/accent_coach_extract_formants.py](../scripts/accent_coach_extract_formants.py).
Its CSV output has columns `clip_id, start_s, end_s, phoneme, F1, F2, duration_s`.
**Verify** the columns by reading one row of an existing formants CSV (e.g.
`tts_output/accent_coach/cleaned_corpus/fry/formants.csv`) before writing the
shift code. If columns differ, fix the column names in the shift script — do
NOT rewrite the extractor.

**Critical**: no new alignment code. If a column you need (e.g. token mid-time)
is missing, derive it (`mid = (start_s + end_s) / 2`).

#### B3. Implement `shift_vowels_to_centroid`

**Files to create**:
- `accent_coach/dsp/formant_shift.py`

**Signature**:
```python
def shift_vowels_to_centroid(
    wav_in: Path,
    wav_out: Path,
    segments: list[dict],   # rows from extract_formants CSV: [{phoneme, start_s, end_s, F1, F2}, ...]
    target_phonemes: set[str],   # e.g. {"ʌ", "ʊ", "ɔː", "aʊ", "ɜː"}
    target_centroid: dict,   # {phoneme: {"f1": float, "f2": float}}
) -> dict:
    """Read wav_in, for each segment whose phoneme is in target_phonemes:
       compute (Δf1, Δf2) from current (segment_F1, segment_F2) → target.
       Apply a smooth spectral envelope warp via WORLD vocoder on that segment.
       Resynthesize, splice into the output WAV. Return {edits_applied, segments_skipped}."""
```

**Algorithm**:
1. `f0, sp, ap = pyworld.wav2world(audio.astype(np.float64), sr)`
2. For each target segment `[start_s, end_s]`:
   - Convert times → frame indices using `pyworld.default_frame_period` (5 ms).
   - For frames in segment: compute current (mean) F1/F2 from `parselmouth.Sound(segment).to_formant_burg()`.
   - Compute Δ in Hz: `δf1 = target.f1 - measured_f1`, same for f2.
   - Build a frequency warp function for those frames: piecewise-linear map
     anchored at (0, 0), (F1_old, F1_new), (F2_old, F2_new), (sr/2, sr/2). Apply
     via `scipy.interpolate.interp1d` on the log-spectrum or via direct
     spectral envelope shift (shift the formant peaks by interpolation along
     the frequency axis of `sp[frame, :]`).
3. `out_audio = pyworld.synthesize(f0, sp_shifted, ap, sr)`
4. Write WAV at `wav_out`.

**Critical**:
- Only edit frames inside target segments. Outside frames: copy `sp` unchanged.
- Smooth the warp across segment boundaries (overlap-add the spectral
  envelope by 10 ms each side) so the splice is not audibly clicked.
- Sample rate: the IndexTTS-2 output is 24 kHz mono float; preserve that.
- pyworld accepts float64 — convert on input, cast to float32 on output.

**Verify**:
```bash
.venv/bin/python -c "
from pathlib import Path
from accent_coach.dsp.formant_shift import shift_vowels_to_centroid
# Take one synth_BC clip from cell_baseline_b/rep_0, one segment row from its
# formants CSV (the /ʌ/ in some short word), shift by Δf1=+50, Δf2=-100.
# Confirm the output WAV plays, is the same duration ±5 ms, and a re-extracted
# formant on the edited segment shows F1, F2 measurably moved.
"
```

#### B4. Score against modern_rp per phoneme — single phoneme cells

**What**: For each of `cell_strut`, `cell_foot`, `cell_thought`, `cell_mouth`,
`cell_nurse`:
1. Read `cell_baseline_b/rep_{i}/manifest.json` (the 25 baseline clips).
2. For each clip, run `shift_vowels_to_centroid` editing ONLY the one target
   phoneme. Write outputs under `cells/cell_<name>/rep_{i}/clips/` with a new
   manifest pointing to the patched WAVs.
3. Re-run `extract_formants` on the patched manifest.
4. `build_centroids_from_formants` → synth centroids.
5. `score_against(synth, "modern_rp", baseline_centroids)` → composite + per_phoneme.
6. Average across N=3.

**Output**: per-cell `result.json` with:
```json
{
  "cell_id": "cell_strut",
  "edited_phoneme": "ʌ",
  "composite_modern_rp_mean": <float>,
  "composite_modern_rp_std": <float>,
  "per_phoneme_target_mean": <float>,    // score for /ʌ/ alone, averaged
  "per_phoneme_target_std": <float>,
  "wer_mean": <float>, "ecapa_mean": <float>, "dnsmos_ovr_mean": <float>,
  "elapsed_s": <float>
}
```

#### B5. cell_all5 — combined shift + decision

**What**: Same as B4 but edit all 5 phonemes simultaneously per clip. This is
the headline number.

**Then**: write `docs/accent_coach_phase0_13_lever_b_findings.md` with:

1. Table: cell, per-phoneme target score (baseline → edited), composite Δ, DNSMOS Δ.
2. **Decision row** evaluating stop criteria (§3.6).
3. A 4-speaker plot using `accent_coach_phase0_10_plot.py` — strip and re-fit
   to show `owner, real_BC, synth_BC[baseline], synth_BC[all5_edited]` on
   Bark-normalised vowel space, highlighting the 5 target phonemes.

### 3.6. Stop criteria for Phase 0.13a (Lever B)

| Outcome | Trigger | Action |
|---|---|---|
| **GREEN** | `cell_all5` composite lift ≥ 10 pts AND DNSMOS drop ≤ 0.5 AND no per-target phoneme lift < 15 | Pursue Lever A in Phase 0.13b. Document Lever B as production-blocked by artifact level. |
| **YELLOW** | `cell_all5` lift ∈ [5, 10) OR DNSMOS drop ∈ (0.5, 1.0] | Pursue Lever A in Phase 0.13b. Note ceiling is bounded; reconsider whether neural vocoder follow-up is justified. |
| **RED** | `cell_all5` lift < 5 OR DNSMOS drop > 1.0 OR ≥3 per-target phonemes show lift < 5 | Gap is model-bound. STOP. Skip Phase 0.13b. Recommend fine-tuning path in findings doc. |

---

## 4. Phase 0.13b — Lever A (targeted emo clips)

**Only run after Lever B verdict is GREEN or YELLOW.** If RED, write the
findings doc and stop.

### 4.1. Goal

Test whether constructing emo clips that are **phonetically dense in the 5
failing phonemes** moves per-phoneme scores for those phonemes more than
Phase 0.11's broadband Fry sentences did.

### 4.2. Hypothesis

H3: the CFM stage's emo embedding can steer phoneme placement when the emo
clip is dominated by exactly those phonemes; per-target lift > 5 pts each.

H4: the emo embedding is too sentence-level to discriminate, and per-target
lift mirrors Phase 0.11 (≤2 pts, identity-preserving) regardless of clip
phonetic density.

### 4.3. Design — build phoneme-dense emo clips

#### A1. Mine corpus_audit for phoneme-dense segments

**Files to create**:
- `scripts/accent_coach_phase0_13_mine_emo.py`

**Algorithm**:
1. Load `tts_output/accent_coach/cleaned_corpus/fry/formants.csv` (per-token rows
   with timing and phoneme labels). Also load `lindsey/formants.csv` as a fallback
   source.
2. For each clip in those CSVs, count tokens whose phoneme ∈ {/ʌ/, /ʊ/, /ɔː/, /aʊ/, /ɜː/}.
3. Greedy: pick the clip with the highest target-density (target_tokens /
   total_tokens). If duration > 5 s, trim to a contiguous 5 s window that
   maximises target density.
4. Repeat to build 3 candidates (independent — different clips).

**Output**: `tts_output/accent_coach/phase0_13/emo_dense/cand_{0,1,2}.wav` plus
`emo_dense/index.json` describing source clip, time window, target-phoneme
counts in the window, and ECAPA cosine vs `ref_narrator` (just for logging).

**Verify** (A1):
```bash
.venv/bin/python scripts/accent_coach_phase0_13_mine_emo.py
```
Each candidate must contain ≥ 8 tokens of target phonemes (otherwise emo
density is no better than Phase 0.11's broadband Fry; abort and report).

#### A2. Phase 0.11-style sweep on dense candidates

**What**: For each candidate `cand_{0,1,2}` × `emo_alpha ∈ {0.3, 0.5, 0.7}` ×
N=3 reps, run cal_25 generation with the candidate as `emo_audio`. Use
`generate_clips()` from `pipeline.experiment`.

**Files to create**:
- `scripts/accent_coach_phase0_13_lever_a.py`

**Total cells**: 3 candidates × 3 alphas = 9 cells, plus 1 baseline (re-use
`cell_baseline_b` from Lever B) → 10 cells × N=3 = 30 generation runs of 25
phrases. Cost estimate: ~30 hours CPU. Run in background.

#### A3. Per-phoneme score table + decision

Average across reps. Produce table:

| Cell | `cfg_rate` | `α` | composite_modern_rp | /ʌ/ | /ʊ/ | /ɔː/ | /aʊ/ | /ɜː/ | WER | ECAPA | DNSMOS |
|---|---|---|---|---|---|---|---|---|---|---|---|

Headline metric: **per-target phoneme mean lift over baseline** (averaged across
the 5 targets).

### 4.4. Stop criteria for Phase 0.13b (Lever A)

| Outcome | Trigger | Action |
|---|---|---|
| **GREEN** | Best cell shows per-target mean lift ≥ 8 pts AND composite_modern_rp ≥ 80 AND WER ≤ 0.05 | Lever A is a production path — productionise the emo clip selection. |
| **YELLOW** | Per-target mean lift ∈ [3, 8) AND composite ≥ 77 AND WER ≤ 0.06 | Partial win — emo conditioning helps but won't close the full gap. Combine with fine-tuning. |
| **RED** | Per-target mean lift < 3 OR composite drops below 73 OR WER > 0.06 | Emo embedding cannot steer at phoneme granularity. Confirms H4. Path forward: fine-tuning. |

---

## 5. Phase 0.13c — F3-based normalization bench (INDEPENDENT of A and B)

### 5.1. Goal

Test whether F3-based, **speaker-intrinsic, per-token** normalization tightens
the native RP cluster `{fry, lindsey, bbc_male}` more than the current Bark
transform — **without** erasing the L2 vowel-space compression signal that
Lobanov destroyed (see [feedback memory](../.claude/projects/-Users-ivkrasovskii-model-voice-generator/memory/project_phase0_6_lobanov_limit.md)).

**This phase is a measurement experiment only.** It does not change the
production scoring path. Output is a number per method; no Phase 0.13a/b
artifacts are touched and no Phase 0.10/0.11/0.12 scores are recomputed under
the new metric in this phase.

### 5.2. Independence

- Inputs from A/B: **none**.
- Outputs consumed by A/B: **none**.
- Can be executed in parallel with A and B, in any order, by any worker.

### 5.3. Hypothesis

H5: A speaker-intrinsic F3 method (Syrdal-Gopal, Nearey intrinsic, or F-ratios)
reduces native intra-cluster variance ≥ 20 % relative to Bark while preserving
owner-vs-RP discrimination within 15 % of Bark.

H6 (negative): All F3 methods either fail to tighten the cluster meaningfully
or erase L2 discrimination the way Lobanov did.

### 5.4. Hard constraints — special to Phase 0.13c

| Constraint | Rule |
|---|---|
| Production scoring path | **Untouched.** `accent_coach.diagnostics.bark_distance`, `accent_coach.diagnostics.cluster_eval`, `accent_coach.comparison.vowels`, `accent_coach.pipeline.experiment` MUST NOT be edited. |
| New normalization module | **Starts from raw F1/F2/F3 in Hz.** Implements each method from first principles. **MUST NOT import** `bark_transform`, `per_phoneme_sigma_rp`, `score_vowels_piecewise`, or any other existing-pipeline normalization symbol. No copy-paste of the existing Bark code into the new module. |
| Existing CSVs / centroids | `tts_output/accent_coach/cleaned_corpus/` and `tts_output/accent_coach/bench/phase0_7/` are **read-only**. Re-extraction goes to a new directory. |
| Opt-in | New normalization is exposed ONLY through the bench script. `load_baseline_centroids()`, `score_against()`, `generate_clips()`, and every other production function continues to use Bark unchanged. |
| Default behaviour | Unchanged everywhere. No CLI flag added to production scripts in Phase 0.13c. |

### 5.5. Tasks

#### C1. Add F3 column to formant CSV writer

**What**: One small backwards-compatible edit to
[scripts/accent_coach_extract_formants.py](../scripts/accent_coach_extract_formants.py).
F3 is already computed by [accent_coach/pipeline/formants.py:62](../accent_coach/pipeline/formants.py#L62)
and stored on `VowelFeatures.f3` ([accent_coach/models.py:22](../accent_coach/models.py#L22))
— it is only dropped at the CSV writer.

Changes:
- In the row dict (around line 95-103): add `"F3": (round(vf.f3, 1) if vf.f3 is not None else "")`.
- In `fieldnames` (around line 111): insert `"F3"` between `"F2"` and `"voiced_fraction"`.

**Backwards compatibility check** before editing:
```bash
rtk grep -rn "fieldnames.*F2\|F1.*F2.*voiced_fraction\|reader\.fieldnames" scripts/ accent_coach/ tests/
```
If any reader relies on positional column order (not name-based), STOP and
report. `csv.DictReader` access by column name is safe.

**Verify** (after editing): re-extract one small manifest and confirm the
CSV has 8 columns including `F3`, and `csv.DictReader` reads non-empty
F3 for ≥ 80 % of rows.

---

#### C2. Re-extract formants with F3 to a new directory

**What**: Run the (now F3-aware) extractor on the existing manifests for the
six speakers needed by the bench. Output goes to a NEW directory; existing
CSVs are not touched.

**Files / dirs**:
- `tts_output/accent_coach/phase0_13c/formants_with_f3/{fry,lindsey,bbc_male,real_BC,owner,synth_BC}.csv` [new]

**Source manifests** (Sonnet to resolve exact paths from the existing layout —
they are the same manifests used by Phase 0.12 corpus rebuild and Phase 0.7
bench). Hints:
- fry, lindsey → `tts_output/accent_coach/cleaned_corpus/{spk}/manifest.json` (already produced by Phase 0.12)
- bbc_male, real_BC, owner → look under `tts_output/accent_coach/bench/phase0_7/` for the corresponding manifests
- synth_BC → re-use the frozen Phase 0.7 synth_BC manifest OR generate fresh on cal_25 if not available

**Critical**: do NOT overwrite or move the source manifests. New CSV outputs only.

**Verify**:
```bash
.venv/bin/python -c "
import csv
from pathlib import Path
for f in Path('tts_output/accent_coach/phase0_13c/formants_with_f3').glob('*.csv'):
    rows = list(csv.DictReader(f.open()))
    n_f3 = sum(1 for r in rows if r.get('F3', '').strip() not in ('', 'nan'))
    print(f'{f.name}: {len(rows)} rows, F3 present in {n_f3} ({100*n_f3/max(len(rows),1):.0f}%)')
"
```
Expected: F3 present in ≥ 80 % of rows per speaker.

---

#### C3. Implement F3 normalization methods (fresh module, NO Bark imports)

**Files to create**:
- `accent_coach/diagnostics/f3_normalization.py`

**Critical**:
- This module MUST NOT contain `from accent_coach.diagnostics.bark_distance import ...`, MUST NOT contain `from accent_coach.diagnostics.cluster_eval import ...`, MUST NOT contain `from accent_coach.comparison.vowels import ...`.
- Inputs: raw F1, F2, F3 in Hz (numpy arrays or scalars).
- Outputs: 2-tuples `(dim1, dim2)` in the new coordinate space.
- No centroid math, no per-speaker statistics, no aggregation — pure point functions.

**Methods to implement**:

```python
def syrdal_gopal(f1_hz, f2_hz, f3_hz):
    """Syrdal & Gopal 1986 Bark-difference metric.
    Implements bark from Hz locally (Traunmüller 1990):
      Z = (26.81 * f_hz) / (1960 + f_hz) - 0.53
    Returns (Z3 - Z1, Z3 - Z2)."""

def nearey_intrinsic(f1_hz, f2_hz, f3_hz):
    """Nearey 1978 CLIH intrinsic.
    Per token: log_gm = mean(log(F1), log(F2), log(F3))
    Returns (log(F1) - log_gm, log(F2) - log_gm)."""

def f_ratios(f1_hz, f2_hz, f3_hz):
    """Simple speaker-size-invariant ratios.
    Returns (F1 / F3, F2 / F3)."""
```

**Verify** with hand-computed reference values. For an adult male /ɑ/ at
F1=730, F2=1090, F3=2440 Hz:

- Syrdal-Gopal: Z1 ≈ 6.78, Z2 ≈ 8.97, Z3 ≈ 14.74; (Z3-Z1, Z3-Z2) ≈ (7.96, 5.77)
- Nearey intrinsic: log_gm ≈ 7.241; (log F1 − log_gm, log F2 − log_gm) ≈ (-0.648, -0.241)
- F-ratios: (730/2440, 1090/2440) ≈ (0.299, 0.447)

If any output is off by more than 1 % from the reference, STOP and re-check the formula. (Sonnet: recompute the references yourself; do not trust the comment above blindly — comment is a sanity guide, not a test fixture.)

---

#### C4. Bench script

**Files to create**:
- `scripts/accent_coach_phase0_13c_f3_bench.py`

**Algorithm**:
1. Load the six F3-bearing CSVs from C2. Drop rows where F3 is missing,
   F1 > 900 Hz (already filtered upstream), or duration_s < 0.05.
2. For each method `M ∈ {bark_control, syrdal_gopal, nearey_intrinsic, f_ratios}`:
   a. Apply M per token → `(dim1, dim2)` columns added to a working dataframe.
   b. Build per-(phoneme, speaker) centroids: mean over tokens.
   c. **Native tightness**: for each of the 17 target phonemes, compute the
      stddev of `(dim1, dim2)` across the three native centroids
      `{fry, lindsey, bbc_male}`. Aggregate via mean across phonemes.
   d. **L2 discrimination**: per phoneme, Euclidean distance
      `|owner_centroid − modern_rp_centroid|` where `modern_rp` here is the
      mean of native 3 in this method's space. Aggregate via mean across phonemes.
   e. **BC gap**: same but `|real_BC − modern_rp|`.
3. The `bark_control` row is the ONLY caller of the existing
   `accent_coach.diagnostics.bark_distance.bark_transform`. Import it in the
   bench script ONLY (never in `f3_normalization.py`). Use it read-only — do
   not write any Bark-derived output back to the production paths.
4. Output to `docs/accent_coach_phase0_13c_norm_bench.csv` with columns:
   `method, native_tightness, l2_discrimination, bc_gap, native_tightness_rel_bark, l2_discrimination_rel_bark, bc_gap_rel_bark`.

**Verify**:
```bash
.venv/bin/python scripts/accent_coach_phase0_13c_f3_bench.py
```
Expected: 4 rows in the output CSV (one per method); `bark_control` row has
`*_rel_bark == 1.000` for all three; other rows have ratios reported to 3 dp.

---

#### C5. Findings doc

**Files to create**:
- `docs/accent_coach_phase0_13c_norm_findings.md`

**Required sections**:
1. **Verdict** — apply §5.6 stop criteria to the bench table.
2. **Bench table** — render `accent_coach_phase0_13c_norm_bench.csv` as a
   markdown table.
3. **Per-phoneme breakdown** — for the winning method (if any), show per-phoneme
   native stddev, L2 distance, BC gap. Highlight phonemes where the winning
   method does notably better or worse than Bark.
4. **Vowel space plot** — one plot per method showing the 4 speakers
   `{owner, real_BC, synth_BC, modern_rp}` in that method's coordinate space.
   Output: `docs/img/phase0_13c_norm_{method}.png`. Adapt
   [scripts/accent_coach_phase0_10_plot.py](../scripts/accent_coach_phase0_10_plot.py)
   for the new coordinate axes — do NOT modify the original plot script.
5. **Recommendation** — explicit GREEN/YELLOW/RED decision per §5.6 and the
   suggested follow-up (e.g. "switch in Phase 0.14" or "stay on Bark").

### 5.6. Stop criteria for Phase 0.13c

Let `T(M), D(M), G(M)` be native tightness, L2 discrimination, BC gap of method
`M`, and let `T_b, D_b, G_b` be the same for `bark_control`. Smaller `T` is
better; larger `D` and `G` are better.

| Outcome | Trigger | Action |
|---|---|---|
| **GREEN** | ∃ method M with `T(M) ≤ 0.80 · T_b` AND `D(M) ≥ 0.85 · D_b` AND `G(M) ≥ 0.75 · G_b` | Document the winner. **Propose** switching the production normalization in a follow-up Phase 0.14. **Do NOT switch in 0.13.** |
| **YELLOW** | ∃ method M with `T(M) ≤ 0.90 · T_b` AND `D(M) ≥ 0.70 · D_b` but not GREEN | Worth exploring a mixed scoring (e.g. Bark for coaching, F3 for cluster tolerance). Document the trade-off table; no production change. |
| **RED** | No method satisfies even YELLOW (every method either tightens < 10 % or loses > 30 % discrimination) | Bark remains the production normalization. Lobanov-style failure mode confirmed across F3 methods too. Findings doc still gets written — the negative result is valuable. |

### 5.7. Special anti-patterns for Phase 0.13c

1. **DO NOT** import `bark_transform`, `per_phoneme_sigma_rp`, or `score_vowels_piecewise` from anywhere into `accent_coach/diagnostics/f3_normalization.py`. The new module is independent and starts from raw Hz.
2. **DO NOT** modify `accent_coach/diagnostics/bark_distance.py` or `accent_coach/comparison/vowels.py` or `accent_coach/pipeline/experiment.py`.
3. **DO NOT** add the new normalization as an option in production scripts (`indextts_gen.py`, `accent_coach_phase0_13_lever_b.py`, `accent_coach_phase0_13_lever_a.py`, etc.) in this phase. New methods exist only inside the bench.
4. **DO NOT** overwrite anything under `tts_output/accent_coach/cleaned_corpus/` or `tts_output/accent_coach/bench/phase0_7/`. Re-extraction outputs go to `tts_output/accent_coach/phase0_13c/`.
5. **DO NOT** recompute Phase 0.10/0.11/0.12 historical scores under the new metric. That recomputation, if needed, belongs in a follow-up phase.
6. **DO NOT** rewrite the CSV writer to drop existing columns; F3 is ADDED, F1/F2/voiced_fraction/etc. stay where they are.
7. **DO NOT** assume F3 is always present — handle `None`/NaN gracefully; the bench filters such rows out, it does not crash.

---

## 6. File outputs (final state)

```
scripts/
  accent_coach_phase0_13_lever_b.py           [new — Lever B driver]
  accent_coach_phase0_13_lever_a.py           [new — Lever A driver]
  accent_coach_phase0_13_mine_emo.py          [new — phoneme-dense emo clip selection]
  accent_coach_phase0_13c_f3_bench.py         [new — F3 normalization bench (independent)]
  accent_coach_extract_formants.py            [modified — adds F3 column to CSV (~3 lines)]

accent_coach/dsp/
  __init__.py                                 [new if missing]
  formant_shift.py                            [new — pyworld + parselmouth warp]

accent_coach/diagnostics/
  f3_normalization.py                         [new — fresh module, NO imports from existing pipeline]

tts_output/accent_coach/phase0_13/
  cells/cell_baseline_b/rep_{0,1,2}/clips/    [new — N=3 baselines]
  cells/cell_strut/rep_{0,1,2}/clips/         [new]
  cells/cell_foot/rep_{0,1,2}/clips/          [new]
  cells/cell_thought/rep_{0,1,2}/clips/       [new]
  cells/cell_mouth/rep_{0,1,2}/clips/         [new]
  cells/cell_nurse/rep_{0,1,2}/clips/         [new]
  cells/cell_all5/rep_{0,1,2}/clips/          [new]
  emo_dense/cand_{0,1,2}.wav                  [new — only if Lever A runs]
  emo_dense/index.json                        [new — only if Lever A runs]
  lever_a_grid.csv                            [new — only if Lever A runs]
  lever_b_grid.csv                            [new]

tts_output/accent_coach/phase0_13c/
  formants_with_f3/{fry,lindsey,bbc_male,real_BC,owner,synth_BC}.csv  [new — F3-bearing extracts]

docs/
  accent_coach_phase0_13_plan.md              [this file]
  accent_coach_phase0_13_lever_b_findings.md  [new]
  accent_coach_phase0_13_lever_a_findings.md  [new — only if Lever A runs]
  accent_coach_phase0_13_findings.md          [new — combined summary, GREEN/YELLOW/RED]
  accent_coach_phase0_13c_norm_bench.csv      [new — F3 bench output table]
  accent_coach_phase0_13c_norm_findings.md    [new — F3 bench findings]
  img/phase0_13_4speaker_lever_b.png          [new]
  img/phase0_13_4speaker_lever_a.png          [new — only if Lever A runs]
  img/phase0_13c_norm_{bark_control,syrdal_gopal,nearey_intrinsic,f_ratios}.png  [new — F3 bench plots]
```

---

## 6. Anti-patterns — do not do these

1. **Do not re-tune `cfg_rate`, `diffusion_steps`, `num_beams`** during Phase
   0.13. Phase 0.10 covered this; new variance dilutes the signal from the
   actual levers under test.
2. **Do not use N=1 anywhere.** Every cell is N=3.
3. **Do not write a new forced aligner.** `extract_formants` already produces
   per-token timing.
4. **Do not write a new vowel-space plotter.** Adapt
   [scripts/accent_coach_phase0_10_plot.py](../scripts/accent_coach_phase0_10_plot.py).
5. **Do not edit `vendor/index-tts/`** — Lever A reuses the existing
   `--emo-audio` / `--emo-alpha` CLI flags already exposed in
   [scripts/indextts_gen.py:64-72](../scripts/indextts_gen.py#L64).
6. **Do not modify** `tts_output/accent_coach/cleaned_corpus/` —
   it is the Phase 0.12 output and is the read-only source of truth for
   centroids and corpus audit.
7. **Do not change `cal_25.csv`.** Same phrases everywhere for comparability.
8. **Do not skip Lever B's baseline cell.** It is the anchor; without it
   the lifts are uninterpretable.
9. **Do not run Lever A if Lever B verdict is RED.** Write the findings doc
   and stop; recommend fine-tuning.
10. **Do not poll background jobs.** When `run_in_background=true`, you get a
    notification when done. Use the wait.
11. **Do not let the F3 bench (Phase 0.13c) leak into production code.** The
    new normalization methods live ONLY in `accent_coach/diagnostics/f3_normalization.py`
    and are exercised ONLY by `scripts/accent_coach_phase0_13c_f3_bench.py`.
    `pipeline/experiment.py`, `score_against`, `bark_distance`, and every
    Lever-A/B script continue to use Bark unchanged. See §5.4 / §5.7.
12. **Do not reimplement the existing Bark transform inside `f3_normalization.py`.**
    The new module is for new methods; for the `bark_control` comparison row,
    the bench script imports the existing `bark_transform` ONCE (read-only).

---

## 7. Quick reference — verification commands

```bash
# Dependencies
.venv/bin/python -c "import pyworld; import parselmouth; print(pyworld.__version__, parselmouth.__version__)"

# Lever B — baseline first
.venv/bin/python scripts/accent_coach_phase0_13_lever_b.py --cell cell_baseline_b --replicates 3
# Expected: composite_modern_rp ∈ [72, 78]

# Lever B — single-phoneme smoke
.venv/bin/python scripts/accent_coach_phase0_13_lever_b.py --cell cell_strut --replicates 3
# Expected: /ʌ/ per-phoneme score rises measurably (>10 pts) vs baseline

# Lever B — all 5
.venv/bin/python scripts/accent_coach_phase0_13_lever_b.py --cell cell_all5 --replicates 3
# Evaluate §3.6 stop criteria.

# Lever A — only if Lever B is GREEN/YELLOW
.venv/bin/python scripts/accent_coach_phase0_13_mine_emo.py
.venv/bin/python scripts/accent_coach_phase0_13_lever_a.py --replicates 3
# Evaluate §4.4 stop criteria.

# Phase 0.13c — F3 normalization bench (INDEPENDENT — can run anytime)
# C1: backwards-compat check before editing the CSV writer
rtk grep -rn "fieldnames.*F2\|F1.*F2.*voiced_fraction\|reader\.fieldnames" scripts/ accent_coach/ tests/

# C2: re-extract corpus with F3 column (after C1 edit)
mkdir -p tts_output/accent_coach/phase0_13c/formants_with_f3
# (per-speaker extract commands — see §5.5 C2)

# C4: bench (after C3 module is in place)
.venv/bin/python scripts/accent_coach_phase0_13c_f3_bench.py
# Evaluate §5.6 stop criteria.
```

---

## 8. Notes on what comes after

- **GREEN both levers**: productionise Lever A (build emo-clip selection into
  the inference path); keep Lever B as a research artefact (vocoder artifacts
  block production).
- **YELLOW either**: bundle the partial gains; move to fine-tuning as the
  primary path to close the remaining gap. Lever A's emo clips can be reused
  in the fine-tuning corpus.
- **RED both**: gap is fundamentally in the model. Path forward is one of:
  (a) LoRA fine-tune on a BC-leaning RP corpus; (b) switch base model; (c)
  accept the 13.5-pt gap and ship as-is. Out of scope for this plan.

### Phase 0.13c outcomes

- **GREEN**: a winning F3 method is documented. **Open Phase 0.14** to switch
  the production normalization, recompute Phase 0.10–0.12 historical scores
  under the new metric, and re-evaluate Lever A/B stop criteria under the new
  baseline. Do NOT flip the switch inside 0.13.
- **YELLOW**: trade-off is real but not clean. Consider a hybrid scheme (Bark
  for coaching, F3 for cluster tolerance) in a follow-up. Production stays
  on Bark.
- **RED**: Bark is confirmed as the best speaker-blind normalization available
  to us with current per-token F3 estimates. The negative result is itself
  useful — close the door and move on.
