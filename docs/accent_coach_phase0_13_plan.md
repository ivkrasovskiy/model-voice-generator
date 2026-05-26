# Accent Coach — Phase 0.13 Execution Brief: Levers A and B

> **Audience**: executing agent. **Do not reason about scope.** Decisions are made.
> Write code per the tasks, verify each by running the listed commands, stop on
> any failed verification.

---

## 0. Context (one paragraph)

Phase 0.10 ([findings](accent_coach_phase0_10_findings.md)) swept GPT + CFM params
(N=3 Optuna, 35 trials). Verdict: RED. Best `composite_lindsey_mean = 62.10`
(gate 82), best `composite_BC_mean = 70.80` (gate 75). Phase 0.11 swept
full-sentence Fry emo clips at `emo_alpha ∈ {0.3, 0.5, 0.7, 1.0}` and Phase 0.12
rebuilt Lindsey/Fry/modern_rp centroids via ECAPA-filtered corpus (sim ≥ 0.50).
After cleanup, the cross-speaker matrix shows `synth_BC vs modern_rp = 75.4`,
`real_BC vs modern_rp = 88.9` — a **13.5-point gap** concentrated in 5
phonemes (`/ʌ/, /ʊ/, /ɔː/, /aʊ/, /ɜː/`) where real BC scores 86–100 but synth
scores 30–59. Param sweep + emo sweep have shown a ~2–3 pt ceiling. Phase 0.13
tests two remaining levers from the
[Phase 0.10 findings §"Remaining TTS accent levers"](accent_coach_phase0_10_findings.md#remaining-tts-accent-levers).

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
| Failing phonemes (5) | `/ʌ/, /ʊ/, /ɔː/, /aʊ/, /ɜː/` (STRUT, FOOT, THOUGHT, MOUTH, NURSE) | From [Phase 0.10 findings §Cross-speaker validation](accent_coach_phase0_10_findings.md#cross-speaker-validation-matrix-phase-012) |
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

## 5. File outputs (final state)

```
scripts/
  accent_coach_phase0_13_lever_b.py           [new — Lever B driver]
  accent_coach_phase0_13_lever_a.py           [new — Lever A driver]
  accent_coach_phase0_13_mine_emo.py          [new — phoneme-dense emo clip selection]

accent_coach/dsp/
  __init__.py                                 [new if missing]
  formant_shift.py                            [new — pyworld + parselmouth warp]

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

docs/
  accent_coach_phase0_13_plan.md              [this file]
  accent_coach_phase0_13_lever_b_findings.md  [new]
  accent_coach_phase0_13_lever_a_findings.md  [new — only if Lever A runs]
  accent_coach_phase0_13_findings.md          [new — combined summary, GREEN/YELLOW/RED]
  img/phase0_13_4speaker_lever_b.png          [new]
  img/phase0_13_4speaker_lever_a.png          [new — only if Lever A runs]
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
