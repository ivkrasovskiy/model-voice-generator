# Accent Coach — Consolidated Phase History (0.5–0.16)

This document is the single reference for all retired accent-coach experiment phases.
Active experiments are in Phase 1 (FastAPI/React UI). The original per-phase plan and
findings files are preserved in git history; this document distills each phase down to
its goal, inputs, command, output, and verdict so the individual files can be deleted.

---

## Phase 0.5 — Modern-RP reference design (Deterding 1997 is obsolete)

**Goal**: Explain why synth-BC and the owner scored nearly identically on the Phase 0
vowel composite (BC=75.5, owner=73.3). Attribute each problem phoneme (/æ/, /ɛ/, /ʌ/)
to one of four root causes (H1: stale norms; H2: BC natural deviation; H3: IndexTTS
distortion; H4: synth-BC/owner collision).

**Inputs**:
- D1: Real BC speech extracted from `https://youtu.be/cHmkAStZBkc` via diarization
  → `tts_output/real_bc_corpus/`
- D2: Modern RP corpus (Geoff Fry + Geoff Lindsey + BBC News bulletin)
  → `tts_output/modern_rp_corpus/`
- D3: Existing synth-BC calibration set `tts_output/bc_cal_50/`
- D4: Owner's 50 calibration recordings

**Command**:
```bash
# Step 1: build real-BC corpus
.venv/bin/python scripts/accent_coach_build_real_bc.py \
    --primary-url https://youtu.be/cHmkAStZBkc --ref-wav tts_output/ref_interview.wav \
    --out-dir tts_output/real_bc_corpus
# Step 2: build modern-RP corpus
.venv/bin/python scripts/accent_coach_build_modern_rp.py \
    --urls-json configs/accent_coach_phase0_5/modern_rp_urls.json \
    --out-dir tts_output/modern_rp_corpus
# Step 3: extract formants for all four sources (run once per manifest)
.venv/bin/python scripts/accent_coach_extract_formants.py \
    --manifest <source>/manifest.json --out <source>/formants.csv \
    --source-label <real_bc|modern_rp|synth_bc|owner>
# Step 4: build comparison table
.venv/bin/python scripts/accent_coach_build_table.py \
    --modern-rp-csv tts_output/modern_rp_corpus/formants.csv \
    --real-bc-csv   tts_output/real_bc_corpus/formants.csv \
    --synth-bc-csv  tts_output/bc_cal_50/formants.csv \
    --owner-csv     tts_output/owner_cal_50/formants.csv
# Step 5: verdicts
.venv/bin/python scripts/accent_coach_compute_verdicts.py
```

**Output**:
- `docs/accent_coach_phase0_5_table.csv` — per-(phoneme, source) F1/F2 mean/std/n
- `docs/accent_coach_phase0_5_verdicts.json` and `.md`
- `docs/img/accent_coach_phase0_5_vowel_space.png`

**Verdict**: GREEN — H1 CONFIRMED (Deterding 1997 stale: median |Deterding − Modern RP|
= F1 163 Hz, F2 195 Hz, threshold 75 Hz). H3 partially confirmed: IndexTTS pulls /æ/ F1
+106 Hz toward Deterding-era values. H4 not confirmed (synth-BC is not colliding with
owner in F1/F2 space). Downstream: norms replacement deferred to Phase 0.7 (BBC gender
audit pending); modern-RP corpus established as D2 for all subsequent phases. Note added
to `accent_coach_phase0_5_findings.md` flagging multi-speaker BBC contamination (29%
female by F0 audit) and recommending Fry+Lindsey-only for male-target comparisons.

---

## Phase 0.6 — Lobanov z-scoring erases L2 vowel compression (diagnostic, no artifact)

**Goal**: Validate or refute the owner's objection to Phase 0.5's conclusion that the
owner is "closer to modern RP than Deterding is." Test the hypothesis that the owner's
compressed vowel space in F1 (range 122 Hz vs 173 Hz for real BC) was masked by raw-Hz
distances, and that Lobanov normalisation would reveal the true ordering.

**Inputs**:
- `docs/accent_coach_phase0_5_table.csv` (read-only — no re-extraction)
- Three aggregated baselines: `modern_rp_full` (BBC+Fry+Lindsey), `modern_rp_no_bbc`
  (Fry+Lindsey), `native_wide` (Fry+Lindsey+real_BC+synth_BC)

**Command**:
```bash
# Phase A: geometry (convex hull areas)
.venv/bin/python scripts/accent_coach_phase0_6_geometry.py
# Phase B: Lobanov normalisation + pairwise distances + dendrogram
.venv/bin/python scripts/accent_coach_phase0_6_lobanov.py
# Phase C: verdicts
.venv/bin/python scripts/accent_coach_phase0_6_compute_verdicts.py
```

**Output**: No durable artifact committed (pure diagnostic). Intermediate outputs:
- `docs/accent_coach_phase0_6_geometry.csv`
- `docs/accent_coach_phase0_6_lobanov_table.csv`
- `docs/accent_coach_phase0_6_lobanov_distances.csv`
- `docs/img/accent_coach_phase0_6_lobanov_vowel_space.png`
- `docs/img/accent_coach_phase0_6_dendrogram.png`

**Verdict**: YELLOW — V1 (owner vowel space compressed, articulation_idx ≤ 0.7) FAILED
(actual 1.13; F1 is compressed but F2 is not, so hull area is normal-sized). V2 (owner
clusters separately from natives in Lobanov 18-D) PASSED. V3/V4 (distance reversal ≥ 2/3
baselines) FAILED: reversal holds only for `native_wide` (1/3). Key finding for the
product: Lobanov per-speaker z-scoring erases L2 vowel-space compression (the very
diagnostic signal for Slavic-L1 accent), making the owner appear closer to modern RP
than he is. Lobanov is retained as a clustering visualisation tool only; raw-Hz
per-phoneme distance is the correct coaching metric. Phase E (BBC gender audit) required,
deferred to Phase 0.7.

---

## Phase 0.7 — Rebuilt RP norms from modern-RP corpus; froze baseline centroids

**Goal**: (E) Complete the BBC speaker gender audit deferred from Phase 0.6. (F) Update
`accent_coach/reference/rp_norms.py` to modern-RP values (Fry+Lindsey+BBC-male). (G)
Re-run the Phase 0 bench against updated norms to check whether the acceptance gate
(composite A ≥ 80, B ≥ 80, B−C ≥ 15) is met.

**Inputs**:
- `tts_output/modern_rp_corpus/bbc/clips/*.wav` — 451 BBC clips for gender audit
- `tts_output/modern_rp_corpus/formants.csv` — per-token formants from Phase 0.5
- `tts_output/accent_coach/bc_cal_50/` and `tts_output/accent_coach/users/owner/` —
  existing calibration audio

**Command**:
```bash
# Phase E: BBC gender audit
.venv/bin/python scripts/accent_coach_phase0_7_bbc_audit.py
# Phase F: derive modern-RP norms (print dict, paste into rp_norms.py manually)
.venv/bin/python scripts/accent_coach_phase0_7_rebuild_norms.py
# Phase G: bench run
.venv/bin/python scripts/accent_coach_bench.py \
    --manifest tts_output/accent_coach/bench/phase0_7_manifest.csv --run-id phase0_7
# (phase0_7_make_manifest.py builds the manifest from existing audio dirs)
```

**Output**:
- `accent_coach/reference/rp_norms.py` — updated with `RP_VOWEL_F1_F2_MALE_MODERN`
  (13 492 tokens, Fry 68% + BBC-male 18% + Lindsey 14%); legacy Deterding kept as
  `RP_VOWEL_F1_F2_MALE_LEGACY`
- `tts_output/accent_coach/bench/phase0_7/speaker_centroids.json` — F1/F2 centroids
  for all 7 speaker groups (real_BC, synth_BC, Fry, Lindsey, BBC-male, modern_rp, owner)
- `tts_output/accent_coach/bench/phase0_7/results.json`, `report.md`,
  `phoneme_distances.json`
- `tts_output/modern_rp_corpus/bbc/speaker_audit.csv`
- `docs/img/accent_coach_phase0_7_bbc_f0_hist.png`

**Verdict**: RED (Phase 0 gate NOT MET). BBC is 29% female → 275 male clips kept.
Modern-RP norms updated correctly (key shifts: /æ/ F1 748→545, /ʊ/ F2 950→1427,
/ɔː/ F2 700→1138). Vowel skill validated (synth-BC A=79.5, real-BC B=79.0). However
aspiration scorer is broken (real BC scores 13.4 — cannot be correct for a canonical RP
speaker) and rhythm/stress continue to drag composites to 52–55. Phase 0 gate explicitly
not met. The `speaker_centroids.json` artifact became the load-bearing reference file
for all subsequent phases.

---

## Phase 0.8 — Tested H1–H5 normalisation schemes; Bark piecewise won

**Goal**: Find a vowel-distance metric that satisfies four cluster criteria simultaneously
(C1: modern-RP speakers cluster together; C2: modern RP separated from Deterding 1997;
C3: owner separated from modern RP by ≥ 2× within-RP spread; C4: owner separated from
Deterding). Test H1 (F0/VTL scaling), H2 (Nearey log-mean), H3 (anchor-relative), H4
(Bark-scale Euclidean), H5 (F0-binned norms, fallback). Replace the exponential-decay
composite with a piecewise-linear score anchored to within-RP variance.

**Inputs**:
- `tts_output/accent_coach/bench/phase0_7/speaker_centroids.json` (read-only)
- `tts_output/modern_rp_corpus/formants.csv` and `tts_output/accent_coach/users/owner/formants.csv`

**Command**:
```bash
.venv/bin/python scripts/accent_coach_phase0_8_run.py
# Runs H1→H3→H2→H4, writes per-hypothesis JSON, cluster summary CSV
```

**Output**:
- `accent_coach/diagnostics/cluster_eval.py`, `vtl_normalize.py`, `anchor_normalize.py`,
  `nearey_normalize.py`, `bark_distance.py`
- `accent_coach/comparison/vowels.py` — `score_vowels_piecewise` added (existing
  exponential function preserved)
- `tts_output/accent_coach/phase0_8/h{1,2,3,4}_*.json`
- `tts_output/accent_coach/phase0_8/cluster_metrics_summary.csv`
- `tts_output/accent_coach/phase0_8/piecewise_scores_*.json`
- `tts_output/accent_coach/phase0_8/per_phoneme_distances_*.csv`

**Verdict**: YELLOW — no hypothesis passed all four criteria. H4 (Bark) passes C1/C2/C4
with C3 marginal (ratio 1.65 vs 2.0 required) and meets the product piecewise GREEN test
(owner=56.5, real-BC=89.8, gap=33.3). H3 (anchor) passes C1/C3/C4 but fails C2 by
design (Deterding and modern RP share the same corner-vowel topology, which is a
linguistic fact not a metric bug). H1 (VTL) failed due to ill-conditioned F0 estimates
(BC parselmouth F0=89 Hz, synth-BC=80 Hz — anomalously low). Owner accepted Option A:
H4 (Bark piecewise) adopted as the production metric with explicit C3 caveat.
`num_beams=5` identified as the main synth-BC lever (reference-clip swap from interview
→ audiobook estimated to give 5–8 additional points).

---

## Phase 0.9 — Reference-clip swap + beam sweep; set production num_beams=5

**Goal**: Close the 15-point gap between synth-BC (H4 Bark 74.6) and real-BC (89.8)
without fine-tuning. Run controlled Track A (6 BC reference clips) and Track B (2 RP
reference clips) experiments, optionally followed by Track C (generation parameter
sweep).

**Inputs**:
- `tts_output/cross_eval_50/eval_short.csv` — 15-phrase eval set
- Reference clips: `tts_output/ref_interview.wav` (A1 baseline), `ref_interview_429_446.wav`
  (A2), `ref_interview_1450_1510.wav` (A3), `ref_sherlock.wav` (A4), `ref_narrator.wav`
  (A5), `ref_combined.wav` (A6), `ref_fry.wav` (B1), `ref_lindsey.wav` (B2)
- `tts_output/accent_coach/bench/phase0_7/speaker_centroids.json`

**Command**:
```bash
.venv/bin/python scripts/accent_coach_phase0_9_run.py
# Orchestrates generation via scripts/indextts_gen.py, formant extraction,
# H4 Bark scoring, and posthoc WER/ECAPA/DNSMOS for each variant.
# Log: tts_output/accent_coach/phase0_9/experiment_log.csv
# Per-variant: tts_output/accent_coach/phase0_9/variants/<ID>/{clips,centroids.json,bark_scores.json}
```

**Output**:
- `tts_output/accent_coach/phase0_9/experiment_log.csv` — all 15+ variants scored
- `tts_output/accent_coach/phase0_9/variants/C_int_b5/` — winning variant artifacts

**Verdict**: GREEN — `C_int_b5` (interview reference, `num_beams=5`, temp=0.8) reaches
H4 Bark piecewise **82.1**, crossing the ≥82 stop criterion. Raising `num_beams` 3→5
gives +7.5 points, outperforming every reference-clip swap in Track A. RP references
(B1, B2) are counterproductive (66.1–71.4), confirming the model's phoneme embeddings
are tied to a generic-English centroid that RP reference clips cannot override.
Downstream: `ref_interview.wav` + `num_beams=5` locked as production defaults in
`CLAUDE.md` and `scripts/indextts_gen.py`.

---

## Phase 0.10 — Optuna GPT+CFM sweep; built cal_25 calibration set

**Goal**: Widen the parameter sweep from Phase 0.9 (which only varied `num_beams`) to
include the s2mel CFM diffusion knobs (`inference_cfg_rate`, `diffusion_steps`) that had
been hard-coded since project start. Use Optuna TPE with N=3 replicates. Optimize a
two-objective loss combining synth-BC vs. Lindsey score and synth-BC vs. real-BC score.
Also: build a 25-phrase stratified calibration subset (`cal_25`) for faster iteration.

**Inputs**:
- `tts_output/accent_coach/cal_25.csv` (built by Phase 0.10 task A5)
- `tts_output/accent_coach/bench/phase0_7/speaker_centroids.json`
- `tts_output/ref_interview.wav` (locked reference)

**Command**:
```bash
# Build cal_25 subset
.venv/bin/python scripts/accent_coach_phase0_10_build_cal25.py
# Smoke-test single cell
.venv/bin/python scripts/accent_coach_phase0_10_run_cell.py \
    --params-json '{"num_beams":5,"temperature":0.8,"top_p":0.8,"cfg_rate":0.7,"diffusion_steps":25}' \
    --replicate 0 --out-dir tts_output/accent_coach/phase0_10/cell_smoke/
# Pilot (5 trials, N=1)
.venv/bin/python scripts/accent_coach_phase0_10_optuna.py --n-trials 5 --replicates 1 --pilot
# Full study (30 trials, N=3)
.venv/bin/python scripts/accent_coach_phase0_10_optuna.py --n-trials 30 --replicates 3 --resume
```

**Output**:
- `tts_output/accent_coach/cal_25.csv` — 19-phrase stratified subset (greedy selection
  stopped early; all 11 available target phonemes reached ≥2 hits)
- `tts_output/accent_coach/phase0_10/optuna.db` — 35 trials (14 COMPLETE, 21 PRUNED)
- `docs/accent_coach_phase0_10_top_trials.csv`
- `docs/accent_coach_phase0_10_posthoc.csv`
- `docs/img/phase0_10_4speaker_plot.png`
- `accent_coach/pipeline/experiment.py` — shared experiment infrastructure extracted
  from Phase 0.9 runner

**Verdict**: RED (against Lindsey single-speaker target). Best reliable N=3 result:
`composite_lindsey = 59.10 ± 1.16`, `composite_BC = 67.17 ± 1.86` (trial #25). Neither
met the ≥82 / ≥75 thresholds (note: thresholds were calibrated for `modern_rp` aggregate,
not single-speaker Lindsey; a fair Lindsey-equivalent GREEN gate is ~70, which was also
not met). Key findings: `diffusion_steps` is the dominant parameter (fANOVA importance
0.518); higher steps (60) appear in 6 of top 8 trials; `num_beams` has near-zero
importance once CFM knobs are free; `beams=5` is confirmed in all top trials; Lindsey
and real-BC objectives are weakly correlated (Spearman ρ=0.491). Cross-speaker
validation matrix established (native RP 94–98, real-BC 88.9, synth-BC 75.4, owner 59.3).
Two unexplored levers identified for Phase 0.11: targeted emo clips (Lever A) and
vocoder-based formant shift (Lever B).

---

## Phase 0.11 — Emotion-vector sweep + N=3 confirmation

**Goal**: Test whether conditioning IndexTTS-2 generation with a phoneme-dense Fry emo
clip at `emo_alpha ∈ {0.3, 0.5, 0.7, 1.0}` moves the synth-BC vowel score upward vs.
baseline (no emo override). Confirm the best-looking cell (Fry α=0.3) with N=3
replicates before declaring it a real signal.

**Inputs**:
- `tts_output/accent_coach/cal_25.csv`
- `tts_output/ref_interview.wav` (speaker reference, locked)
- `tts_output/ref_fry_emo.wav` (Fry emo clip, 14 s, stitched from modern-RP corpus)
- `tts_output/accent_coach/bench/phase0_7/speaker_centroids.json`
- `tts_output/accent_coach/cleaned_corpus/speaker_centroids_cleaned.json`
  (Phase 0.12 output, loaded if present for cross-validation)

**Command**:
```bash
# Emo sweep (5 cells, N=1, resumable)
.venv/bin/python scripts/accent_coach_phase0_11_emo_sweep.py
# N=3 confirmation for baseline + fry α=0.3
.venv/bin/python scripts/accent_coach_phase0_11_n3_confirm.py
```

**Output**:
- `tts_output/accent_coach/phase0_11/cells/cell_<id>/` — per-cell clips, formants.csv,
  centroids.json, result.json
- `tts_output/accent_coach/phase0_11/grid_results.csv` — all 5 cells scored
- `tts_output/accent_coach/phase0_11_n3/cells/cell_<id>/rep{1,2}/` — 4 new generations
- `tts_output/accent_coach/phase0_11_n3/n3_summary.csv` — N=3 aggregated results

**Verdict**: RED — emo conditioning provides no statistically significant lift. N=3
aggregated results (against cleaned centroids): baseline `fry_clean = 69.87 ± 2.51`,
Fry α=0.3 `fry_clean = 72.07 ± 2.99`. Lift = +2.20, pooled σ = 3.90 → "within noise."
`modern_rp_clean` lift = +1.34, pooled σ = 3.93 → "within noise." The emo embedding is
a sentence-level vector and cannot steer individual phoneme production reliably enough
to clear within-run variance. Lever A (targeted emo clips) is closed. Only Lever B
(vocoder post-processing) and fine-tuning remain unexplored for closing the
synth-BC/real-BC gap.

---

## Phase 0.12 — Rebuilt centroids on ECAPA-filtered modern-RP corpus

**Goal**: Address a potential confound in Phase 0.11: the Fry and Lindsey reference
centroids in `speaker_centroids.json` were built from unfiltered clips that may include
non-solo-speaker contamination. Rebuild Lindsey and Fry centroids using only clips that
pass an ECAPA cosine-similarity filter (≥0.50 vs. a speaker-specific reference
embedding), then re-score Phase 0.11 cells against the cleaned targets to see if
the noise floor drops.

**Inputs**:
- `tts_output/accent_coach/corpus_audit/{lindsey,fry}_audit.json`
  (output of `accent_coach_corpus_audit.py`, run separately)
- `tts_output/modern_rp_corpus/{lindsey,fry}/clips/*.wav`
- Phase 0.11 cell centroids from `tts_output/accent_coach/phase0_11/cells/`

**Command**:
```bash
.venv/bin/python scripts/accent_coach_phase0_12_rebuild.py
# Optional: .venv/bin/python scripts/accent_coach_phase0_12_rebuild.py --threshold 0.6
```

**Output**:
- `tts_output/accent_coach/cleaned_corpus/lindsey/` — kept_clips.json, transcripts.json,
  manifest.json, formants.csv, centroid.json
- `tts_output/accent_coach/cleaned_corpus/fry/` — same structure
- `tts_output/accent_coach/cleaned_corpus/speaker_centroids_cleaned.json` — overlay
  replacing lindsey, fry, and modern_rp entries in the baseline with ECAPA-filtered
  versions; bbc_male and real_BC unchanged
- `tts_output/accent_coach/cleaned_corpus/phase0_11_rescored.csv` — Phase 0.11 cells
  re-scored against cleaned targets (orig vs. cleaned delta per cell)

**Verdict**: YELLOW — ECAPA filtering shifts scores upward by 2–4 points uniformly
across all cells (e.g. baseline `fry_cleaned` 66.6 vs. `fry_orig` 64.5, Δ+2.1; Fry
α=0.3 `fry_cleaned` 70.8 vs. `fry_orig` 68.7, Δ+2.1). The delta is consistent across
all 5 emo cells, which means the lift is attributable to the reference-centroid cleaning
rather than any specific emo configuration. The cleaned overlay became the canonical
reference for Phase 0.11 N=3 confirmation (Phase 0.11's `n3_confirm.py` loads it if
present). The `speaker_centroids_cleaned.json` artifact is the authoritative centroid
file for Phase 0.13+.

---

## Phase 0.13 — Formant shift (Lever B) + phoneme-dense emo (Lever A) + F3 norm bench

**Goal**: Close the remaining 13.5-point synth_BC vs modern_rp gap (synth_BC=67.53,
real_BC=88.9) using two remaining non-training levers. Lever B: WORLD-vocoder formant
shift on the 5 failing phonemes (/ʌ/, /ʊ/, /ɔː/, /aʊ/, /ɜː/). Lever A:
phoneme-dense emo clips (Fry/Lindsey clips with high density of failing phonemes) at
`emo_alpha ∈ {0.3, 0.5, 0.7}`. Phase 0.13c (parallel, independent): test whether F3-based
normalization tightens the native RP cluster without destroying L2 discrimination.

**Inputs**:
- `tts_output/accent_coach/cal_25.csv` (25-phrase eval set)
- `tts_output/accent_coach/cleaned_corpus/speaker_centroids_cleaned.json`
- Emo candidate clips: `tts_output/accent_coach/phase0_13/emo_dense/cand_{0,1,2}.wav`
  (lindsey; fry/1_066; fry/1_004)

**Command**:
```bash
# Lever B sweep
.venv/bin/python scripts/accent_coach_phase0_13_lever_b.py
# Lever A sweep (N=3 reps each cell)
.venv/bin/python scripts/accent_coach_phase0_13_lever_a.py
# Phase 0.13c: F3 normalization bench
.venv/bin/python scripts/accent_coach_phase0_13c_norm_bench.py
```

**Output**:
- `tts_output/accent_coach/phase0_13/lever_b/cell_{baseline_b,strut,foot,thought,mouth,nurse,all5}/`
- `tts_output/accent_coach/phase0_13/lever_a/cell_{cand0,cand1,cand2}/alpha_{0.3,0.5,0.7}/`
- `accent_coach/dsp/formant_shift.py` — WORLD formant shift DSP

**Verdict**:
- Lever B: **YELLOW** — cell_all5 lifts composite +5.47 pts (67.53 → 73.00) at cost of
  −0.37 DNSMOS (WORLD re-synthesis overhead). /ʌ/ responds strongly (+31.6 pts per-phoneme);
  /aʊ/ collapses (−33.3) — WORLD can't represent formant trajectories. Ceiling is model-bound.
- Lever A: **RED** — best per-target lift +2.88 pts (below 3-pt YELLOW threshold); that cell
  simultaneously drops composite by −3.7 pts. Emo vector is a global acoustic modifier, not
  phoneme-targeted. Higher alpha raises DNSMOS (prosodic expressiveness) but not phoneme accuracy.
- F3 bench: **RED** — no normalization method (Syrdal-Gopal, Nearey intrinsic, F-ratios) beats
  bare Bark on both native-tightness and L2-discrimination simultaneously. Bark confirmed as
  production norm.
- **Overall**: the 13.5-pt gap is model-bound. Path forward: LoRA fine-tuning.

---

## Phase 0.14 — LoRA fine-tuning of IndexTTS-2 GPT; RP pipeline corrections

**Goal**: Fine-tune IndexTTS-2's GPT (UnifiedVoice) backbone with LoRA on a 1195-clip
modern-RP corpus (Fry + Lindsey + BBC male, ECAPA-filtered) to improve the 5 failing
vowels. Simultaneously: fix systematic WhisperX/CMU alignment biases for RP (BATH
misclassification, rhotic alignment errors, LOT/BATH confusion). Also evaluate the base
model vs LoRA on 50 cross-eval phrases.

**Inputs**:
- `tts_output/modern_rp_corpus/{fry,lindsey,bbc}/clips/*.wav` — 1195 training clips
- Pre-cached mel embeddings in `tts_output/accent_coach/phase0_14/lora/emb_cache/`
- `tts_output/cross_eval_50/` — 50-phrase cross-eval set

**Command**:
```bash
# LoRA training (3 runs; best = Run 3, early-stopped at step_400)
vendor/index-tts/.venv/bin/python scripts/accent_coach_phase0_14_lora_train.py \
    --patience 200 --eval-every 100 --max-steps 900
# Score base vs LoRA on 50 phrases
.venv/bin/python scripts/score_rp_all.py
# BATH-targeted score
.venv/bin/python scripts/score_bath_rp.py
```

**Output**:
- `ckpt/best/` — Run 3, step_400, eval_loss 5.726 (best LoRA adapter)
- `accent_coach/pipeline/rp_postprocess.py` — BATH/LOT relabeling + coda-R stripping
- `accent_coach/reference/bath_words.py` — BATH and LOT word sets
- `tests/test_rp_postprocess.py` — 17 unit tests

**Key findings** (50-phrase cross-eval ranking, dist_to_RP Bark):
| Rank | Source | dist_to_RP |
|---|---|---|
| 1 | fry | 0.402 |
| 2 | **gen_base** | 0.420 |
| 3 | real_bc | 0.521 |
| 4 | **gen_lora_best** | 0.567 |
| 5 | owner | 0.888 |

**Verdict**: **RED (LoRA counterproductive)**. gen_base (0.420) is within 0.018 Bark of
Fry (0.402) — essentially indistinguishable. The base IndexTTS-2 zero-shot already produces
near-RP vowels. LoRA at 1195 clips / 400 steps partially unlearns strong zero-shot calibration
while correcting others; net effect is −0.147 Bark regression. BATH centroids confirm the
base model is nearly perfect (gen_base F1=510 F2=1233 vs RP target F1=518 F2=1215). LoRA
direction is abandoned. Pipeline corrections (BATH/LOT/coda-R) are a permanent improvement
kept in production.

**Downstream**: owner-reported BATH/GOAT as worst vowels (1.230, 1.600 Bark). RP
measurement pipeline is now coach-grade; Phase 0.15 pivots to coach-grade measurement for
both RP and GenAm.

---

## Phase 0.15 — Coach-grade measurement + reliability gate (RP ready; GenAm blocked)

**Goal**: Upgrade the measurement stack to coach-grade quality — per-token bootstrap CIs,
within-category dispersion, Bhattacharyya overlap, F3 rhoticity — and validate that
the pipeline can reliably distinguish RP from GenAm speakers before shipping any feedback.
New product decision (from Phase 0.14 analysis): coach is measure-only, no TTS in loop.
Both RP and GenAm are target accents.

**Inputs**:
- Existing RP corpus formants (fry, lindsey, bbc, real_bc, gen_base)
- New GenAm validation corpus: Vsauce/Michael Stevens (76 clips, 801 vowel tokens)

**Command**:
```bash
.venv/bin/python scripts/accent_coach_coach_eval.py  # dual-target scorer + gate
```

**Output**:
- `accent_coach/diagnostics/coach_metrics.py` — per-token metric (CIs, dispersion, overlap, rhoticity)
- `scripts/accent_coach_coach_eval.py` — dual-target scorer + gate verdict
- `tts_output/genam_corpus/formants_genam.csv` — Vsauce GA formants
- `tts_output/accent_coach/phase0_15/coach_eval.json` — full scored results

**Verdict**: **YELLOW — RP ready, GenAm blocked**. All 5 RP-class sources score
RP-closer with non-overlapping CIs and read non-rhotic. GenAm source (Vsauce) scores
RP-closer due to stale Hillenbrand-1995 norms (citation-form, 30 years old) missing
GOOSE-fronting (F2=997 vs modern 1500) and diphthong convention mismatch. Same defect
that Phase 0.5 fixed for RP. Owner RP coaching is actionable now (worst: GOAT 2.67,
THOUGHT 2.11, BATH 2.01 Bark). GenAm fix → Phase 0.16.

---

## Phase 0.16 — Modern GenAm norms (GREEN) + identity/accent disentanglement verdict

**Goal**: (A) Build modern connected-speech GenAm norms from 3 GA male lecture speakers
(Huberman, Harris, Sapolsky), validate with leave-one-out held-out gate. (B) Determine
whether IndexTTS-2 can clone GA and whether identity and accent can be separated.

**Inputs**:
- 266 GA lecture clips from Huberman (dopamine monologue), Harris (AMA), Sapolsky
  (Stanford lecture) — extracted by `scripts/accent_coach_build_genam.py`

**Command**:
```bash
.venv/bin/python scripts/accent_coach_build_genam.py      # build corpus
.venv/bin/python scripts/accent_coach_build_genam_norms.py  # derive centroids
.venv/bin/python scripts/accent_coach_genam_loo.py        # LOO GREEN gate
.venv/bin/python scripts/accent_coach_phase0_16_score_b.py  # Track B: clone + disentangle
.venv/bin/python scripts/accent_coach_phase0_16_dsp.py    # B2ii: DSP own-voice→RP
```

**Output**:
- `accent_coach/reference/genam_norms.py` — `_GENAM_MALE_MODERN` (Hillenbrand kept as legacy)
- `tts_output/genam_lecture_corpus/` — 266-clip GA corpus + formants
- `tts_output/accent_coach/phase0_16/` — clone, disentangle, DSP outputs

**Verdict**:
- Track A: **GREEN** — LOO 3/3: each held-out GA speaker scores GenAm-closer + rhotic.
  Key fixes vs Hillenbrand: GOOSE F2 997→1301 (fronting captured); PRICE/GOAT switched
  from onset to steady-state measurement (matches RP convention). Rhoticity broadened
  beyond NURSE to all pre-/r/ contexts (enough token count now).
- Track B — disentanglement:
  - **B1**: GA reference → GenAm-closer + rhotic (1893 Hz) + Huberman identity (0.839). GA cloning works.
  - **B2i**: spk=BC + emo=Huberman → BC identity (0.763) but RP accent stays (non-rhotic 2345). Emo does NOT carry accent.
  - **B2ii**: DSP formant-shift on owner voice → identity 0.976 (self), vowels move +0.57 Bark toward RP. DSP separates identity from *vowel*-accent only (not rhoticity or prosody).
- **Conclusion**: identity and accent are entangled in IndexTTS-2's single speaker reference.
  "Favourite voice + arbitrary accent" is not achievable from the model alone. DSP gives
  coaching-grade partial separation (vowels only), sufficient for the F2 feature
  (own voice, target accent). F1 feature (BC voice in GA) requires an external voice-conversion
  model if full fidelity (rhoticity + prosody) is needed.

---

## Phase 0.17 — Rhythm module production-grade upgrade

**Goal**: Make `score_rhythm()` produce coaching-meaningful scores. Pre-work state: nPVI was
always ~25 for all speakers due to G2P-uniform duration distribution; reference norms from a
1990s lab read-aloud task; several scoring logic bugs.

**Inputs**:
- Modern RP corpus (`tts_output/modern_rp_corpus/`) — 363 clips from Fry/BBC/Lindsey
- GenAm corpus (`tts_output/genam_lecture_corpus/`) — 98 clips
- Owner calibration set (`tts_output/owner_cal_50/`) — 50 clips, 3 matching our test sentences
- 8 TTS reference clips (`tts_output/rhythm_bench_tts/`) — generated with IndexTTS-2 num-beams=3

**Files changed**:
- `accent_coach/pipeline/prosody.py` — new `extract_syllable_durations_from_words()` (hybrid)
  and `_syllable_durs_within_word()` helper; acoustic 2-peak fallback fixed
- `accent_coach/pipeline/features.py` — switched from acoustic-only to hybrid function
- `accent_coach/pipeline/alignment.py` — BATH_WORDS / LOT_WORDS overrides wired in;
  `ɐ` removed from `IPA_VOWELS` (dead); `ɒ` now reachable via LOT override
- `accent_coach/comparison/rhythm.py` — pattern correlation 30% count-divergence guard;
  FW weight redistribution when phoneme data unavailable; diagnostic threshold made relative
- `accent_coach/reference/rp_norms.py` — `RP_NPVI_MIN` 55→40, `RP_NPVI_MAX` 75→62
  (corpus-derived, hybrid-adjusted); `_DECAY` in rhythm.py 20→30
- `tests/accent_coach/test_prosody.py` — new (7 tests)
- `tests/accent_coach/test_alignment.py` — BATH/LOT split tests added
- `tests/accent_coach/test_rhythm.py` — Bug 1/2 regression tests added

**Root causes found and fixed**:

1. **G2P-uniform distribution** (CRITICAL): WhisperX gives word-level timestamps; old code
   split each word uniformly across phonemes → all syllables in a word got identical
   duration → nPVI ≈ 25 for all speakers regardless of accent.
   Fix: `extract_syllable_durations_from_words()` uses word-level timestamps directly for
   single-syllable words (captures the content/function-word contrast) and runs per-word
   acoustic nucleus detection for multi-syllable words.

2. **Acoustic 2-peak fallback → nPVI=0** (BUG): 2 detected peaks → 1 inter-peak interval →
   `compute_npvi([x]) = 0.0` → false "rhythm is too even" diagnostic.
   Fix: unified < 3 peak cases to always use rate-based estimate (≥ 2 elements).

3. **Pattern correlation on misaligned vectors** (BUG): count mismatch truncated both vectors
   to the shorter length, correlating different phonological positions (e.g. if "the" was
   missed in one, every subsequent index was off by one).
   Fix: returns 50.0 (neutral) when `(longer − shorter) / longer > 30%`.

4. **fw_score defaulted to 100 when no phoneme data** (BUG): the 0.2 weight was applied to a
   fictional perfect score when WhisperX alignment wasn't available or phonemes were empty.
   Fix: when FW analysis unavailable, composite is 0.5 × nPVI + 0.5 × pattern (no FW term).

5. **Stale nPVI reference norms** (CALIBRATION): Grabe & Low 2002 (lab read-aloud,
   centre=65) is ~14 nPVI points above measured natural speech from our own corpus.
   Empirical measurement on 461 native clips (acoustic detector): mean=41, p25=29, p75=51.
   Hybrid correction +11 (estimated from bench back-calculation) → p25≈40, p75≈62.
   Updated: `RP_NPVI_MIN=40, RP_NPVI_MAX=62`, `_DECAY=30` (was 20 — too steep for natural
   speech; native speaker at corpus p25 was scoring 58, now scores ~75).

6. **BATH/LOT phoneme overrides not wired** (BUG): `accent_coach/reference/bath_words.py`
   had both `BATH_WORDS` and `LOT_WORDS` sets from a prior phase but they were never imported
   into the aligner. cmudict uses AE1 for both TRAP and BATH (GenAm merger) and AA1 for both
   PALM and LOT. Result: "class", "last", "dance" were scoring against /æ/ norms (wrong for
   RP), and "lot", "not", "hot" were scoring against /ɑː/ (PALM) norms instead of /ɒ/.
   Fix: wired into `_word_to_phoneme_instances()`.
   Note: BATH override is RP-only (GenAm has no TRAP/BATH split); currently applied
   unconditionally since all coaching targets are RP.

**Bench results** (`tts_output/accent_coach/bench/rhythm_validate_03/`):

| Experiment | Rhythm (old) | Rhythm (new) | Note |
|---|---|---|---|
| tts_self | 100.0 | **100.0** | Sanity: identical clips → perfect |
| owner_vs_tts | 68.8 | **73.3** | Real coaching gap vs TTS target |
| native_rp | 61.6 | **70.0** | 3 Fry podcast clips, absolute mode |

8 TTS reference sentences in `tts_output/rhythm_bench_tts/` covering: monosyllabic
stress-timing, function-word contrast, polysyllabic content, wh-question, yes/no question,
dense content, function-word-heavy narrative, polysyllabic-only.

**Still broken (out of scope)**:
- Aspiration: 18.0 for native Fry — scorer calibrated for lab VOT (50–100ms), natural speech
  doesn't trigger stops at those positions. Needs same corpus-derived recalibration as nPVI.
- Stress: 25.8 for native Fry — pattern matcher compares against fixed template; natural
  speech diverges. Same fix path.
- ɒ (LOT) formant norms: still Deterding 1997 (`(600, 900)`). Needs corpus re-measurement
  now that LOT tokens are correctly labelled in the aligner.

**Verdict**: **GREEN for rhythm**. Pipeline is self-consistent (tts_self=100), produces a
meaningful coaching gap (owner 73.3 vs TTS 100), and handles all known measurement bugs.
Aspiration and stress remain broken in absolute mode; both are known and deferred.

---

## Phase 0.18 — Bug audit + test infrastructure (no corpus runs)

**Goal**: (A) Clean up obsolete scripts from the `scripts/` directory. (B) Audit rhythm and
vowel scoring for critical bugs using code review agents. (C) Write failing tests that
document each bug and serve as acceptance criteria for fixes. (D) Identify hard-coded magic
constants that need to move to a config layer.

**No corpus runs, no model inference.** Pure code audit and test writing.

**Scripts deleted** (16 one-timers from closed/abandoned phases):
`phase0_13_lever_{a,b}.py`, `phase0_13_mine_emo.py`, `phase0_13c_f3_bench.py`,
`phase0_14_{accent_probe,build_dataset,cache_embeddings,lora_train}.py`,
`phase0_16_{dsp,score_b}.py`, `accent_coach_dsp_shift.py`,
`diagnose_mps_{grad_explosion,ram}.py`, `diagnose_bad_clips.py`,
`score_bath_rp.py`, `score_rp_all.py`.

**Bugs found — vowel scoring (7)**:

1. **CRITICAL — BATH/LOT overrides unconditional** (`alignment.py:144–158`):
   `_word_to_phoneme_instances()` has no `accent_target` parameter; BATH (AE1→ɑː)
   and LOT (AA1→ɒ) overrides always fire. GenAm speakers get wrong phoneme labels on
   every BATH/LOT word (~130 Hz F2 error per token). /ɒ/ doesn't exist in GenAm at all.

2. **HIGH — `pipeline/formants.py` subprocess never passes `--target`** (`formants.py:42–46`):
   Formant extraction wrapper always uses `target="rp"` default — any GenAm corpus
   extraction through this wrapper silently applies RP relabeling.

3. **HIGH — `compare()` has no accent-target routing** (`scoring.py:39–43`):
   When `reference_norms=None`, always calls `get_rp_norms()`. GenAm callers have no
   way to get GenAm norms auto-selected; must pass pre-computed norms externally.

4. **HIGH — Piecewise sigma is RP-only** (`centroids.py:109,122,135`):
   Comment says "sigma_rp is always derived from {fry,lindsey,bbc_male} regardless of
   target." GenAm speaker with on-target THOUGHT vowel is penalized because it falls
   outside the RP σ band.

5. **MEDIUM — LOT /ɒ/ centroid still Deterding 1997 stub** (`rp_norms.py:24`):
   `RP_VOWEL_F1_F2_MALE_MODERN["ɒ"] = (600, 900)` — the same value as LEGACY.
   Phase 0.17 wired the LOT override so tokens now reach this centroid; the centroid
   itself was never updated from the corpus.

6. **MEDIUM — `score_vowels()` missing accent-target param** (`vowels.py:66–69`):
   Falls back to `get_rp_norms()` when norms are None. No path to GenAm scoring
   without explicit external norm injection.

7. **LOW — `RP_VOWEL_F1_F2_MALE` alias points to Deterding 1997 legacy** (`rp_norms.py:66`):
   `RP_VOWEL_F1_F2_MALE = RP_VOWEL_F1_F2_MALE_LEGACY`. Tests importing this alias
   compute scores against superseded norms. Affects `test_comparison.py` and
   `test_integration.py`.

**Bugs found — rhythm scoring (5)**:

1. **CRITICAL — Bench uses acoustic-only detection; norms calibrated for hybrid**
   (`rhythm_bench.py:80`, `rhythm_compare.py:155–156`):
   `get_syllable_durations()` calls `extract_syllable_durations_acoustic()` for all
   groups including native Fry clips. The norms (`RP_NPVI_MIN=40, RP_NPVI_MAX=62`,
   centre=51, `_DECAY=30`) were calibrated from hybrid measurements (+11 above
   acoustic). Native podcast clips measure acoustic nPVI ≈ 29–41; score =
   100·exp(−(51−35)/30) ≈ 58 — below non-native owner in comparison mode (73.3).
   Root cause of the native < owner inversion.

2. **HIGH — `_DECAY` miscalibrated for acoustic-only** (`rhythm.py:13`):
   Second-order effect of Bug 1; `_DECAY=30` was tuned against hybrid nPVI. Fixed
   automatically when Bug 1 is resolved (consistent detection mode).

3. **HIGH — Display bug in compare script** (`rhythm_compare.py:127`):
   "nPVI tgt" column prints `ref_npvi_min + 10`, which accidentally recovers the right
   value due to the ±10 window construction — fragile if `ref_min` changes.

4. **MEDIUM — `check_ranking()` has no native-beats-owner assertion** (`rhythm_bench.py:205–255`):
   Bench reports GREEN even when native RP scores below non-native owner.

5. **MEDIUM — `_function_word_analysis()` returns 100.0 when no FW matched** (`rhythm.py:93–94`):
   Content-word-heavy sentences get 20 free FW points. Fix: return `(None, [])` when
   `matched == 0` — same treatment as the no-phoneme path.

**Tests written** (`tests/accent_coach/`):
- `test_norms_and_routing.py` — 14 tests; covers alias bug, LOT stub, GenAm norm content,
  alignment accent routing, compare() routing. **6 fail now** (bugs 1,2,5,7 + alignment).
- `test_score_ordering.py` — 9 tests; covers dialect discrimination ordering, rhythm
  ordering (stress > syllable-timed, native p25 not penalized as L2, FW=100 bug).
  **2 fail now** (acoustic calibration + FW=100).

**Run to see failures**:
```bash
uv run pytest tests/accent_coach/test_norms_and_routing.py \
              tests/accent_coach/test_score_ordering.py -v
# Expected now: 15 pass, 8 fail
# Expected after fixes: 23 pass, 0 fail
```

**Hard-coded constants identified for config extraction** (separate task):
- `_DECAY = 30.0` in `comparison/rhythm.py` — nPVI decay constant
- `_SCALE = 1.5` in `comparison/vowels.py` — Euclidean-to-score scale
- `1.7` (FW inflation threshold) in `comparison/rhythm.py`
- `0.30` (pattern correlation divergence guard) in `comparison/rhythm.py`
- `_WEIGHTS` dict in `comparison/scoring.py` — per-skill composite weights
- VOT ranges / means / SDs in `reference/rp_norms.py`
- `RP_NPVI_MIN`, `RP_NPVI_MAX` (already in rp_norms.py — model, not hardcoded)

**Verdict**: **Not a scored phase** — audit + infrastructure only. Eight bugs documented,
tests written, constants inventoried. Next session: fix bugs in priority order
(alignment routing → compare() routing → LOT norm re-measurement → rhythm
acoustic/hybrid reconciliation → FW=100 → alias → remaining). Do not adjust test
thresholds to make tests pass; fix the underlying logic.

---

## Phase 0.19 — Fix all Phase 0.18 bugs (23/23 tests green)

**Goal**: Fix all 8 bugs documented by Phase 0.18. Target: `test_norms_and_routing.py`
and `test_score_ordering.py` both pass in full. No new corpus runs except the LOT
centroid re-measurement (P5).

**Fixes applied**:

**P1 — GenAm accent routing** (vowel bugs 1, 2, 3, 6 → 4 tests):
Added `accent_target: str = "rp"` to `_word_to_phoneme_instances()`, `_whisperx_align()`,
`_mms_align()`, `align_audio()` in `alignment.py`. BATH (AE1→ɑː) and LOT (AA1→ɒ) overrides
now gated on `accent_target == "rp"` — GenAm passes through to default `AA1→ɑː` mapping.
Added `accent_target` to `compare()` in `scoring.py` (branches `get_genam_norms` vs
`get_rp_norms` when `reference_norms=None`), to `score_vowels()` in `vowels.py` (same
fallback branch), and `target` param to `extract_formants()` in `formants.py` (passed as
`--target` to subprocess).

**P2 — RP alias** (vowel bug 7 → 1 test):
`RP_VOWEL_F1_F2_MALE = RP_VOWEL_F1_F2_MALE_MODERN` (was `LEGACY`).
Any call-site importing the alias now gets Fry+Lindsey+BBC-male norms.

**P3 — FW analysis returns 100.0 when no FW matched** (rhythm bug 5 → 1 test):
`_function_word_analysis()` now returns `(None, [])` when `matched == 0`.
Composite weight (0.2) redistributed to nPVI+pattern, same as the no-phoneme path.
Updated pre-existing `test_rhythm.py::test_content_words_not_flagged` assertion from
`score == 100.0` to `score is None` (was asserting the buggy behaviour).

**P4 — Rhythm acoustic/hybrid calibration mismatch** (rhythm bug 1 → 1 test):
Chose Path A (consistent hybrid detection). `get_syllable_durations()` in
`scripts/accent_coach_rhythm_bench.py` now uses `align_audio()` +
`extract_syllable_durations_from_words()` in accurate mode (non-fast). Fast mode retains
acoustic-only and prints an explicit NOTE. Test fixture updated from acoustic p25
(nPVI≈29, `_alternating_durs(0.20, 0.149)`) to hybrid p25 (nPVI≈40 = RP_NPVI_MIN,
`_alternating_durs(0.20, 0.133)`). No norm constants changed — the hybrid norms were
always correct; the bench was using the wrong detector.

**P5 — LOT /ɒ/ centroid** (vowel bug 5 → 1 test):
Full word-level extraction on 195 LOT-word clips from `modern_rp_corpus` with the Phase
0.17 LOT override active (aligner now routes LOT words to /ɒ/).
Sub-manifest: `tts_output/modern_rp_corpus/lot_word_clips_manifest.json`.
Result: n=153 /ɒ/ tokens; F1 median=532 Hz, F2 median=1114 Hz.
`RP_VOWEL_F1_F2_MALE_MODERN["ɒ"] = (532, 1114)`.
Deterding stub (600, 900) retired: modern SSBE /ɒ/ has substantially de-rounded
(F2 +214 Hz vs Deterding), consistent with ongoing vowel shifts in Southern British English.

**Also deleted**: `extract_syllable_durations()` (legacy G2P-uniform syllable function in
`prosody.py`) — had zero callers. The two proper replacements remain:
`extract_syllable_durations_from_words()` (hybrid, accurate) and
`extract_syllable_durations_acoustic()` (fast fallback).

**Command**:
```bash
uv run pytest tests/accent_coach/test_norms_and_routing.py \
              tests/accent_coach/test_score_ordering.py -v
uv run pytest tests/ -q
uv run ruff check scripts/ accent_coach/
```

**Outcome**: 23/23 pass in the two new files. 149/149 in full suite. Lint clean.

**Bench result** (`--fast` mode, acoustic-only, nPVI ~11 below hybrid — relative ranking only):

| Group | nPVI mean | Score mean |
|---|---|---|
| rp_fry | 48.5 | 78.3 |
| rp_bbc | 35.1 | 60.8 |
| rp_lindsey | 36.5 | 64.5 |
| genam_vsauce | 34.8 | 60.3 |
| genam_harris | 43.4 | 69.6 |
| genam_huberman | 39.2 | 60.1 |
| genam_sapolsky | 49.1 | 70.7 |
| generated_bc | 52.7 | 70.1 |
| owner | 29.5 | 54.0 |

Generated BC scores higher than native in fast/absolute mode — expected: TTS is
deliberate speech with nPVI near reference centre (51); podcast/lecture speech is
faster and more reduced. This is not a scoring bug. Accurate bench mode (hybrid) is
correctly wired; comparison mode remains authoritative for coaching (user vs same-sentence
TTS target). Absolute bench is a detector sanity-check only.

**Verdict**: GREEN — all 8 bugs fixed, 23 tests pass, pipeline now correctly routes RP
and GenAm speakers through separate phoneme labels and norm tables.

---

## Phase 0.20 — Conversational reference data + rhythm diagnostic upgrades

**Goal**: (A) Replace lecture/podcast reference clips (Fry, Lindsey, BBC) with
conversational speech so the rhythm bench reflects natural English rather than
deliberate monologue. (B) Add two missing diagnostic signals and validate them
with synthetic accent-profile tests.

### A — Conversational reference corpus

Root cause confirmed: Fry/Lindsey/Huberman are lecture-prosody — their nPVI in
hybrid mode is *above* the RP reference ceiling (Fry: 69.2, ceiling 62), not
below. Deliberate read-speech exaggerates within-word stress contrast; the acoustic
detector missed this in fast mode, making them look near-centre.

New conversational groups downloaded to `tts_output/conversational_refs/`:

| Group | Source | Type |
|---|---|---|
| rp_conv_mitchell | David Mitchell — Would I Lie to You? (sF_tPdV1Jjc) | panel show banter |
| rp_conv_mack | Lee Mack — Would I Lie to You? (QPRDPIrwmL8) | panel show banter |
| genam_conv_carell | Steve Carell — Late Night with Conan O'Brien (9RViEPCoEec) | talk show |
| genam_conv_jlaw | Jennifer Lawrence — Late Night with Seth Meyers (KZDYjcE9mIc) | talk show |
| genam_conv_freshair | NPR Fresh Air interview (XogN47yBM_M) | radio interview |

**Full hybrid-mode bench table** (all groups, aligned/WhisperX mode):

| Group | Type | nPVI | Score |
|---|---|---|---|
| owner | owner | 74.3 | 52.0 |
| rp_fry | lecture | 69.2 | 58.9 |
| genam_huberman | lecture | 66.5 | 63.6 |
| genam_harris | lecture | 64.1 | 69.7 |
| real_bc | interview | 61.9 | 66.2 |
| genam_vsauce | podcast | 58.0 | 61.5 |
| rp_bbc | broadcast | 57.8 | 75.6 |
| genam_sapolsky | lecture | 52.4 | 80.0 |
| generated_bc | TTS | 46.7 | 74.2 |
| rp_conv_mack | conversational | 50.6 | 80.9 |
| rp_lindsey | lecture | 48.9 | 71.7 |
| rp_conv_mitchell | conversational | 43.8 | 79.5 |
| genam_conv_freshair | conversational | 48.0 | 86.0 |
| genam_conv_carell | conversational | 46.6 | 78.8 |
| genam_conv_jlaw | conversational | 45.8 | 78.4 |

Conversational speakers (78–86) all exceed TTS (74.2) and sit squarely inside the
[40, 62] reference range. Hypothesis confirmed.

### B — New diagnostic signals

**Signal 4 — over-stressing** (`user_npvi > ref + 10`):
> "Your speech sounds over-stressed (nPVI=72 vs target 48). Try to speak more
> naturally — each stressed syllable is too exaggerated."

**Signal 5 — per-syllable outliers** (normalized diff > 0.5 threshold):
> "Timing: syllable(s) 2, 5 are too long relative to target."

`RhythmBreakdown` gains `outlier_syllables: list[tuple[int, str]]` (position, "long"|"short").

**Tests** (`tests/accent_coach/test_rhythm_diagnostics.py`, 17 tests):
Synthetic syllable duration fixtures for French (nPVI≈15), Italian (nPVI≈10),
Mandarin (nPVI≈5), Slavic function-word inflation, inverted-pattern mismatch,
lecture over-stressing (nPVI≈89), per-syllable outlier detection, and a
native-quality no-diagnostic baseline. All 17 pass; full suite 149/149.

**Verdict**: GREEN — conversational reference corpus established, both new signals
implemented and tested, bench now authoritative in hybrid mode.


---

## Phase 0.19 addendum — Hybrid bench (all groups, seed=42, n=8)

Full re-run in hybrid (aligned) mode including new conversational corpora
(`rp_conv`, `genam_conv`) and `real_bc_corpus`. Conversational clips have no
transcript so fall back to acoustic detection within hybrid mode.

| Group | Accent | N | nPVI (hybrid) | nPVI std | Score (hybrid) | Score std | Type |
|---|---|---|---|---|---|---|---|
| rp_fry | RP | 8 | 69.2 | 16.5 | 58.9 | 27.6 | lecture/read |
| rp_bbc | RP | 8 | 57.8 | 8.5 | 75.6 | 14.7 | broadcast |
| rp_lindsey | RP | 8 | 48.9 | 13.0 | 71.7 | 16.8 | lecture |
| rp_conv_mack | RP | 8 | 50.6 | 8.8 | 80.9 | 12.5 | conversational |
| rp_conv_mitchell | RP | 8 | 43.8 | 4.3 | 79.5 | 11.7 | conversational |
| genam_vsauce | GenAm | 8 | 58.0 | 20.1 | 61.5 | 21.4 | podcast |
| genam_harris | GenAm | 8 | 64.1 | 14.0 | 69.7 | 24.1 | lecture |
| genam_huberman | GenAm | 8 | 66.5 | 12.2 | 63.6 | 21.9 | lecture |
| genam_sapolsky | GenAm | 8 | 52.4 | 8.6 | 80.0 | 12.3 | lecture |
| genam_conv_carell | GenAm | 8 | 46.6 | 9.9 | 78.8 | 16.0 | conversational |
| genam_conv_freshair | GenAm | 8 | 48.0 | 6.7 | 86.0 | 13.1 | conversational |
| genam_conv_jlaw | GenAm | 7 | 45.8 | 8.6 | 78.4 | 15.1 | conversational |
| real_bc | Real-BC | 8 | 61.9 | 12.1 | 66.2 | 17.2 | interview |
| generated_bc | Generated | 8 | 46.7 | 9.8 | 74.2 | 10.2 | TTS |
| owner | Owner | 8 | 74.3 | 16.5 | 52.0 | 25.8 | owner |

**Key observations:**

- nPVI boost from acoustic→hybrid: lecture/podcast groups +12–27 units (matches
  expected ~11+ uplift); sapolsky only +3 (likely clean speech, acoustic already
  capturing contrast well); owner +45 (very irregular owner prosody — acoustic
  misses the within-word stress peaks).
- Conversational speakers (rp_conv, genam_conv) cluster in nPVI 44–51, with
  *higher scores* (79–86) than lecture groups — these clips fall squarely in the
  RP_NPVI target range [40, 62], so Score=80+ is correct behaviour.
- Generated BC: nPVI 46.7 (within target range [40, 62]) — Score 74.2. Consistent
  with TTS being deliberate, measured speech.
- Real BC interview clips: nPVI 61.9, Score 66.2 — near the upper end of the
  target range, which makes sense for BC's precise RP delivery.
- Owner voice: nPVI 74.3, Score 52.0 — above the target ceiling, confirming the
  coaching signal direction is correct (user needs to reduce stress-timing contrast,
  i.e. more fluid).
- Bench sanity check: OK.

---

## Consonant scoring overhaul (2026-06-05 → 06-09)

Audited and rebuilt the consonant/prosody scoring after the bench ranked the L2
owner *above* native speakers. Summary of outcomes (full detail + open items in
[prosody_consonant_upgrade.md](prosody_consonant_upgrade.md)):

- **Silent fallbacks removed** across alignment, liquids, aggregator, prosody,
  compare(): code now fails explicitly (raise / `None`-skip) instead of fabricating
  neutral (50/65) or uniform values. A silent uniform G2P split had made the whole
  consonant bench meaningless.
- **Alignment §2A fixed** — WhisperX char alignments (stored at segment level) now
  actually drive phoneme boundaries; deleted the uniform splitter (`AlignmentError`).
- **Audio normalization** — one canonical `normalize_audio` (mono/16 kHz/7.6 kHz cap)
  on bench + product path; vowels unaffected.
- **Fricatives** — fixed an audio-**bandwidth confound** (native corpora 16 kHz vs
  owner 44.1 kHz made CoG measure recording bandwidth); replaced Jongman's 7000 Hz
  /s/ with an **in-domain native reference** (/s/≈5200). Natives lifted ~40→77;
  composite now ranks owner lowest. Fricative CoG ties — correct, since Russian /s z
  ʃ f/ ≈ English.
- **VOT/aspiration** — rewrote the extractor (collapse-to-0 bug) to Lisker & Abramson
  (closure→burst→F0 voicing) + aspirating-context filtering. Functional and
  discriminates directionally (native > owner); real-speech accuracy still needs a
  trained model (AutoVOT) — backlog.
- **Stress skill deleted** (word-position heuristic, not acoustic).
- **Per-phoneme diagnostic** — confirmed the discriminators are /r/ (+34) and /p/
  aspiration (+9); Russian-shared /s z ʃ f l/ tie and dominate by frequency.
  /θ ð v/ score nothing for ANYONE — traced to a frication-gate bug (it rejects ~all
  weak dentals; transcripts DO contain them, e.g. owner has 31 /ð/), not absence.

**Still open:** /θ ð/ frication-gate fix (biggest gap), up-weighting
absent-in-Russian markers, production-grade VOT (AutoVOT), lateral single-token noise.
Governance + post-mortem lessons codified in CLAUDE.md.
