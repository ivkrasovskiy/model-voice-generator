# Accent Coach — Phase 0.10 Execution Brief (for Sonnet)

> **Audience**: Sonnet, running as the executing agent. **Do not reason about scope.**
> All decisions are made. Your job is to write code per the tasks below, verify each
> task by running the listed commands, and stop if a verification step fails.
>
> **If a verification fails**: report what failed, do NOT attempt a fix without
> a human in the loop. Do not retry with reduced rigour.

---

## 0. Context (one paragraph)

Phase 0.9 found `num_beams=5` raises H4 Bark piecewise from 74.6 → 82.1 on
`synth_BC` vs `modern_rp` ([docs/accent_coach_phase0_9_findings.md](accent_coach_phase0_9_findings.md)).
The result is N=1 and clears the GREEN gate (82) by 0.1 pt — too tight to trust.
Phase 0.10 widens the param sweep to include s2mel CFM diffusion knobs
(`inference_cfg_rate`, `diffusion_steps`) that have been hard-coded since the
project started, adds a second optimization target (distance to `real_BC` in
addition to `modern_rp`), runs **N=3 replicates everywhere**, and uses Optuna
TPE to search the continuous params efficiently.

---

## 1. Hard constraints — DO NOT change these

| Constraint | Value | Reason |
|---|---|---|
| Reference clip (TTS input) | `tts_output/ref_interview.wav` | Locked per Phase 0.9 finding + user directive |
| Phoneme set | 12 monophthongs + 5 sonorants (l, r, m, n, ŋ) = 17 | F1/F2 only meaningful for these |
| Replicates per cell | **N=3** | Every Optuna trial, every grid cell. No N=1 except the explicit pilot in Task B2. |
| Bark transform | H4 (existing) | `accent_coach.diagnostics.bark_distance.bark_transform` |
| Piecewise scoring | existing | `accent_coach.comparison.vowels.score_vowels_piecewise` |
| **RP target speaker** | **`lindsey`** (single named speaker) | User directive: only 4 speakers total — owner, real_BC, synth_BC, lindsey. Do NOT use `modern_rp` aggregate as target. |
| **BC target speaker** | **`real_BC`** | Cumberbatch's real recordings — second optimization target |
| sigma_rp tolerance | RP cluster (Fry+Lindsey+BBC) | `accent_coach.diagnostics.cluster_eval.per_phoneme_sigma_rp` — keep aggregate for variance estimate even though target is Lindsey alone |
| ECAPA reference | `tts_output/ref_narrator.wav` | per-clip real-BC refs absent on this machine |
| Phase 0.7 centroids | read-only | `tts_output/accent_coach/bench/phase0_7/speaker_centroids.json` — load-bearing for `real_BC`, `owner`, `lindsey`, `fry`, `bbc_male`, `modern_rp` data |

---

## 2. Code reuse map — USE these, do NOT reimplement

| Need | Existing symbol | Location |
|---|---|---|
| Bark transform | `bark_transform(centroids)` | [accent_coach/diagnostics/bark_distance.py](../accent_coach/diagnostics/bark_distance.py) |
| Piecewise scoring | `score_vowels_piecewise(user, ref, sigma)` | [accent_coach/comparison/vowels.py:85](../accent_coach/comparison/vowels.py#L85) |
| Per-phoneme sigma | `per_phoneme_sigma_rp(bark_centroids)` | [accent_coach/diagnostics/cluster_eval.py](../accent_coach/diagnostics/cluster_eval.py) |
| RP norms (legacy) | `RP_VOWEL_F1_F2_MALE_LEGACY` | [accent_coach/reference/rp_norms.py](../accent_coach/reference/rp_norms.py) |
| Formant extraction | `scripts/accent_coach_extract_formants.py` | CLI subprocess |
| TTS generation | `scripts/indextts_gen.py` | CLI subprocess |
| Whisper/ECAPA/DNSMOS load | `lib.transcribe.load_whisper`, `lib.identity.load_ecapa`, `lib.metrics.load_dnsmos` | reuse the singleton pattern from `accent_coach_phase0_9_run.py` lines 458-492 |
| Speaker centroids | `tts_output/accent_coach/bench/phase0_7/speaker_centroids.json` | already has `real_BC`, `owner`, `modern_rp`, `lindsey`, `fry`, `bbc_male` |

Anti-pattern: do NOT write a new bark transform, new scoring function, or a new
formant extractor. They exist. Call them.

---

## 3. Param search space — what gets toggled

| Param | Stage | Type | Range/Set | Currently |
|---|---|---|---|---|
| `num_beams` | GPT (LLM) | categorical | {3, 5, 7} | exposed via CLI |
| `temperature` | GPT (LLM) | uniform | [0.4, 0.9] | exposed via CLI |
| `top_p` | GPT (LLM) | uniform | [0.7, 0.95] | exposed via CLI |
| `inference_cfg_rate` | s2mel CFM | uniform | [0.3, 1.5] | **HARD-CODED at 0.7** in `vendor/index-tts/indextts/infer_v2.py:638` — must expose |
| `diffusion_steps` | s2mel CFM | categorical | {15, 25, 40, 60} | **HARD-CODED at 25** in `vendor/index-tts/indextts/infer_v2.py:637` — must expose |
| `repetition_penalty` | GPT (LLM) | LOCKED at 10.0 | — | already exposed but locked for sweep |
| `top_k` | GPT (LLM) | LOCKED at 30 | — | already exposed but locked for sweep |
| `length_penalty` | GPT (LLM) | LOCKED at 0.0 | — | leave as default |
| `ref_audio` | input | LOCKED at `ref_interview.wav` | — | hard constraint |

5 free dims; ~5e6 continuous combos. Optuna TPE with 30 trials × N=3 = 90 generations.

---

## 4. Optimization objective

Two-objective fitness, minimized:

```
loss = 0.5 * (100 - synth_vs_lindsey_piecewise) + 0.5 * (100 - synth_vs_real_BC_piecewise)
```

Where:
- `synth_vs_lindsey_piecewise` = `score_vowels_piecewise(synth_BC_bark, lindsey_bark, sigma_rp)["composite"]`
- `synth_vs_real_BC_piecewise` = `score_vowels_piecewise(synth_BC_bark, real_BC_bark, sigma_rp)["composite"]`

Both use the **same sigma_rp** (from the RP cluster Fry+Lindsey+BBC) as the per-phoneme
tolerance. Targets are single speakers (Lindsey and real_BC) — do NOT score against
`modern_rp` anywhere in Phase 0.10.

Higher composite scores are better → lower loss. Equal-weighted average is the default;
do not tune the weights in Phase 0.10.

A trial's reported score is the **mean of N=3 replicates** for each objective, plus the std.

---

## 5. TASKS — execute in order. Stop on any failed verification.

---

### A. Technical setup

These tasks have NO long-running generation. Do them all before Task B.

---

#### A1. Document IndexTTS-2 device support; check MPS feasibility

**What**: Inspect `vendor/index-tts/indextts/infer_v2.py` to confirm what `device` values are accepted by `IndexTTS2.__init__`. Try `device="mps"`. If MPS works for the smoke test, default it on Apple Silicon.

**Files to edit**:
- [scripts/indextts_gen.py](../scripts/indextts_gen.py) — change default device from `"cpu"` to auto-detect: `"mps" if torch.backends.mps.is_available() else "cpu"`.
- [scripts/indextts_smoke_test.py](../scripts/indextts_smoke_test.py) — same auto-detect.

**Critical**: if MPS fails (e.g. unsupported op), capture the exact error, REVERT the device default to `"cpu"`, and write a note to [docs/accent_coach_phase0_10_plan.md](accent_coach_phase0_10_plan.md) appendix explaining the failure. Do NOT try to patch ops in vendor/.

**Verify**:
```bash
vendor/index-tts/.venv/bin/python scripts/indextts_smoke_test.py
```
Expected: `cas_01` and `sher_03` hit WER ≤ 0.10, ECAPA ≥ 0.74 (per CLAUDE.md). Report wall-clock time; if MPS works, expect ~2-4× faster than CPU baseline (~2 min total → 30-60 s).

---

#### A2. Expose s2mel CFM params via CLI

**What**: Two params are hard-coded in `vendor/index-tts/indextts/infer_v2.py`:
- Line 637: `diffusion_steps = 25`
- Line 638: `inference_cfg_rate = 0.7`

These are inside `infer_generator`. Patch the function to accept them as kwargs (defaulting to 25 and 0.7), then thread them through `indextts_gen.py` as `--diffusion-steps` and `--cfg-rate` CLI flags.

**Files to edit**:
- [vendor/index-tts/indextts/infer_v2.py](../vendor/index-tts/indextts/infer_v2.py) — add `diffusion_steps` and `inference_cfg_rate` to `infer_generator` signature with current defaults, replace the hard-coded lines 637-638 with the new kwargs. Pop them from `generation_kwargs` if present (mirror the pattern at lines 518-526).
- [scripts/indextts_gen.py](../scripts/indextts_gen.py) — add `--diffusion-steps` (int, default 25) and `--cfg-rate` (float, default 0.7) CLI args. Thread into `gen_kwargs`.

**Critical**: do NOT modify any other line in `vendor/index-tts/`. The pin in CLAUDE.md is at SHA `830f6f8f`. Your diff to `infer_v2.py` must be minimal (≤10 lines).

**Verify**:
```bash
vendor/index-tts/.venv/bin/python scripts/indextts_gen.py \
  --phrases-csv tts_output/cross_eval_50/eval_short.csv \
  --out-dir /tmp/phase0_10_cfm_smoke \
  --cfg-rate 0.7 --diffusion-steps 25
```
Expected: 15 wavs generated, no errors. Sanity: re-run with `--cfg-rate 1.2 --diffusion-steps 40`; output should DIFFER from the first run (different bytes) but be playable.

Then run regression:
```bash
vendor/index-tts/.venv/bin/python scripts/indextts_smoke_test.py
```
Must pass — defaults (cfg=0.7, steps=25) reproduce the baseline.

---

#### A3. Refactor shared experiment infrastructure

**What**: Extract reusable functions from `scripts/accent_coach_phase0_9_run.py` into a new module so Phase 0.10 (and future phases) do not copy-paste.

**Files to create**:
- `accent_coach/pipeline/experiment.py`

**Files to edit**:
- [scripts/accent_coach_phase0_9_run.py](../scripts/accent_coach_phase0_9_run.py) — replace the inline implementations with imports from the new module. The Phase 0.9 run script must STILL WORK end-to-end after the refactor (smoke-test reproducibility in A4).

**Functions to extract into `accent_coach/pipeline/experiment.py`** (copy logic, do not rewrite the math):

```python
def generate_clips(
    phrases_csv: Path,
    ref_audio: Path,
    out_dir: Path,
    gen_params: dict,  # {temperature, top_p, top_k, num_beams, cfg_rate, diffusion_steps}
    label: str,
) -> Path:  # returns manifest.json path
    """Subprocess wrapper around indextts_gen.py. Idempotent (skips if manifest matches)."""

def extract_formants(
    manifest: Path,
    out_csv: Path,
    source_label: str = "synth_BC",
    n_workers: int | None = None,  # None → cpu_count()//2; pass 1 for debugging
) -> Path:
    """Subprocess wrapper around accent_coach_extract_formants.py.
    See §6.5 — parallelize per-phrase via multiprocessing.Pool inside the worker."""

def build_centroids_from_formants(
    formants_csv: Path,
    min_duration_s: float = 0.050,
) -> dict[str, dict]:
    """Per-phoneme mean F1/F2 grouped by phoneme column. Same logic as phase0_9_run.py:build_synth_bc_centroids."""

def load_baseline_centroids() -> dict:
    """Loads phase0_7/speaker_centroids.json + adds deterding pseudo-speaker. Same as phase0_9_run.py:_load_baseline_centroids."""

def score_against(
    synth_centroids: dict[str, dict],
    target: str,  # MUST be "lindsey" or "real_BC" in Phase 0.10. Other keys allowed for diagnostics only.
    baseline_centroids: dict | None = None,
) -> dict:
    """Apply bark_transform, compute sigma_rp from RP cluster once, score synth_BC vs target.
    Returns {'composite': float, 'per_phoneme': {phoneme: score}}.
    Patches synth_centroids into baseline under 'synth_BC' key before transform.
    sigma_rp is ALWAYS computed from the {fry, lindsey, bbc_male} cluster regardless of target."""

def score_posthoc_clips(
    manifest: Path,
    ecapa_ref: Path,
    out_csv: Path,
    n_workers: int | None = None,  # None → cpu_count()//2; pass 1 for debugging
) -> dict:  # {WER, ECAPA, DNSMOS_OVR}
    """Reuses Whisper/ECAPA/DNSMOS singletons. Same logic as phase0_9_run.py:score_posthoc.
    See §6.5 — parallelize per-clip via multiprocessing.Pool when n_workers > 1."""
```

**Module-level singletons** for Whisper/ECAPA/DNSMOS so multiple variants in the same process share model loads. Mirror the `_WHISPER` / `_ECAPA` / `_DNSMOS` pattern from `accent_coach_phase0_9_run.py:458-492` but inside the module.

**Critical**: do NOT change return types or default values of these functions in a way that breaks `accent_coach_phase0_9_run.py`. After refactor, Phase 0.9's A1 reproduction must still pass (Task A4).

**Verify**: see A4.

---

#### A4. Smoke test: reproduce Phase 0.9 A1 with refactored code

**What**: Run Phase 0.9 A1 variant. It must still produce H4 Bark piecewise = 74.6 ± 0.5 (frozen centroids path, no re-extraction).

**Command**:
```bash
.venv/bin/python scripts/accent_coach_phase0_9_run.py --variants A1 --skip-gen --skip-formants --skip-posthoc
```

**Expected output line**:
```
>>> synth_BC H4 Bark piecewise = 74.6
```
and
```
A1 validity check PASSED: 74.6 (expected 74.6±2, Δ=0.0)
```

**If this fails**: revert the refactor in A3, stop, ask for help.

---

#### A5. Build cal_25 stratified subset

**What**: The existing `tts_output/accent_coach/cal_50.csv` has 50 phrases. For Optuna screening (30 trials × N=3 = 90 generations) we need a faster cycle. Build a stratified subset with even per-phoneme coverage at ≥ 2 hits per target phoneme.

**Files to create**:
- `scripts/accent_coach_phase0_10_build_cal25.py`
- `tts_output/accent_coach/cal_25.csv` (the script's output)

**Algorithm**:
1. Run `accent_coach_extract_formants.py` on the existing `tts_output/accent_coach/bc_cal_50/manifest.json` (already exists per phase 0.7 setup).
2. For each row in cal_50.csv, count how many target phonemes (the 17 from §1) appear in the formants for that phrase.
3. Greedy selection: until you've picked 25 phrases, pick the one that maximizes the minimum per-phoneme count across the 17 targets. Stop when every target phoneme has ≥ 2 hits or 25 phrases reached.
4. Write the selected slugs+prompts to `tts_output/accent_coach/cal_25.csv` in the same CSV format as cal_50.csv (`slug,prompt`).

**Verify**:
```bash
.venv/bin/python scripts/accent_coach_phase0_10_build_cal25.py
.venv/bin/python -c "
import csv
from collections import Counter
# Re-extract formants on the cal_25 manifest (you must generate clips for it first
# OR use the cal_50 formants and filter to cal_25 slugs).
# Then assert per-phoneme counts.
"
```
Expected: 25 phrases written; per-phoneme coverage report printed; every one of the 17 target phonemes has ≥ 2 hits.

If coverage cannot be achieved with 25 phrases for some phoneme, INCREASE the budget to 30 or 35 — but keep the budget ≤ 35.

---

#### A6. Smoke-test the full Phase 0.10 cell pipeline (1 cell, N=1, cal_25)

**What**: Write a thin driver that, given a single set of `(num_beams, temperature, top_p, cfg_rate, diffusion_steps)` params and N=1, runs:
1. Generate clips on `cal_25.csv` (25 generations)
2. Extract formants
3. Build synth_BC centroids
4. Score against `lindsey` → composite_lindsey
5. Score against `real_BC` → composite_BC
6. Compute loss = 0.5*(100-composite_lindsey) + 0.5*(100-composite_BC)

**Files to create**:
- `scripts/accent_coach_phase0_10_run_cell.py` — single-cell runner. Signature: `python ... --params-json '{"num_beams":5, ...}' --replicate 0 --out-dir tts_output/accent_coach/phase0_10/cell_smoke/`

**Reuse**: every step uses functions from `accent_coach/pipeline/experiment.py` (built in A3). Do not duplicate code.

**Verify** with the Phase 0.9 winning params (`num_beams=5, temp=0.8, top_p=0.8, cfg_rate=0.7, diffusion_steps=25`):
```bash
.venv/bin/python scripts/accent_coach_phase0_10_run_cell.py \
  --params-json '{"num_beams":5,"temperature":0.8,"top_p":0.8,"cfg_rate":0.7,"diffusion_steps":25}' \
  --replicate 0 \
  --out-dir tts_output/accent_coach/phase0_10/cell_smoke/
```

Expected: prints `composite_lindsey=<float>`, `composite_BC=<float>`, `loss=<float>`. Scores on cal_25 will not match cal_50 numbers exactly (different phrases). Sanity: both composites should be in [60, 95]. If outside that range, stop and report.

---

### B. Optuna sweep

Only run B after every A verification passed.

---

#### B1. Optuna study skeleton

**What**: Write the Optuna driver. Single TPE study; objective minimizes the loss from §4 averaged over N=3 replicates.

**Files to create**:
- `scripts/accent_coach_phase0_10_optuna.py`

**Dependencies**: `uv add optuna` (use uv, not pip).

**Skeleton structure**:
```python
import optuna

STORAGE = "sqlite:///tts_output/accent_coach/phase0_10/optuna.db"
STUDY_NAME = "phase0_10"

def objective(trial: optuna.Trial) -> float:
    params = {
        "num_beams": trial.suggest_categorical("num_beams", [3, 5, 7]),
        "temperature": trial.suggest_float("temperature", 0.4, 0.9),
        "top_p": trial.suggest_float("top_p", 0.7, 0.95),
        "cfg_rate": trial.suggest_float("cfg_rate", 0.3, 1.5),
        "diffusion_steps": trial.suggest_categorical("diffusion_steps", [15, 25, 40, 60]),
    }
    losses = []
    lindsey_scores = []
    bc_scores = []
    for rep in range(3):
        result = run_cell(params, replicate=rep, ...)  # calls accent_coach.pipeline.experiment functions
        losses.append(result["loss"])
        lindsey_scores.append(result["composite_lindsey"])
        bc_scores.append(result["composite_BC"])
        # Allow Optuna to prune slow/bad trials after rep 1 if score is in worst quartile
        trial.report(np.mean(losses), step=rep)
        if trial.should_prune():
            raise optuna.TrialPruned()
    # Log full result set to trial.user_attrs for analysis
    trial.set_user_attr("composite_lindsey_mean", float(np.mean(lindsey_scores)))
    trial.set_user_attr("composite_lindsey_std",  float(np.std(lindsey_scores)))
    trial.set_user_attr("composite_BC_mean",      float(np.mean(bc_scores)))
    trial.set_user_attr("composite_BC_std",       float(np.std(bc_scores)))
    return float(np.mean(losses))

study = optuna.create_study(
    study_name=STUDY_NAME, storage=STORAGE, load_if_exists=True,
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=8),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
)
study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)
```

CLI flags: `--n-trials` (default 30), `--cal` (default `cal_25`), `--resume` (passes through to `load_if_exists=True`).

**Critical**:
- Each replicate gets a DIFFERENT seed for IndexTTS-2 sampling. Use `replicate` index to set `PYTHONHASHSEED` and pass `--seed` if available (check if `indextts_gen.py` supports it; if not, the natural randomness from `do_sample=True` provides per-replicate variance — that's fine).
- DO NOT change params between replicates. All N=3 reps of a trial must share the same `params` dict.
- The optuna.db must be the single source of truth — every trial result is in there. Never edit it by hand.

**Verify** (no run yet): import the module, instantiate the study, print the search space. No execution.

---

#### B2. Pilot: 5 trials × N=1 to validate the Optuna machinery

**What**: Run a 5-trial pilot with N=1 to confirm the loop works end-to-end. This is the ONLY place where N=1 is allowed.

**Command**:
```bash
.venv/bin/python scripts/accent_coach_phase0_10_optuna.py \
  --n-trials 5 \
  --cal cal_25 \
  --replicates 1 \
  --pilot
```

**Expected**:
- 5 trials complete
- `tts_output/accent_coach/phase0_10/optuna.db` is created
- `optuna trials list` (or equivalent) shows 5 COMPLETE trials with non-NaN losses
- No tracebacks, no missing files

**If any trial errors**: stop. Investigate and report. Do NOT continue to B3.

**Cost estimate**: 5 trials × 1 replicate × 25 phrases × ~30 s (assuming MPS works; if CPU, ~60 s) = ~60-150 min. If this takes more than 4 hours, stop and report — something is wrong.

---

#### B3. Full study: 30 trials × N=3

**What**: Run the full Optuna study. N=3 per trial (the hard constraint from §1). Resume the same optuna.db; pilot trials count toward total.

**Command**:
```bash
.venv/bin/python scripts/accent_coach_phase0_10_optuna.py \
  --n-trials 30 \
  --cal cal_25 \
  --replicates 3 \
  --resume
```

**Expected**: 30 total trials (5 pilot + 25 new); pruned trials report as PRUNED not COMPLETE. Median-pruner kicks in for trials whose rep-1 loss is worse than the median of completed trials.

**Cost estimate**: With MPS (best case): 25 trials × 3 reps × 25 phrases × ~30 s ≈ 16 h. With CPU: ~32 h. Run in background with `run_in_background`. Do NOT poll. Pull final results from the optuna.db when notified of completion.

**Critical**:
- This is the longest task. Do NOT attempt to "improve" anything mid-run. If pilot passed, the machinery is correct.
- DO NOT change the search space or weighting between pilot and full study. Same `objective()` function.

---

### C. Analysis + final validation

---

#### C1. Top-3 trial summary + Phase 0.9 baseline check

**What**: Pull top 3 trials by mean loss from the optuna.db. Verify Phase 0.9's params (`num_beams=5, temp=0.8, top_p=0.8, cfg_rate=0.7, diffusion_steps=25`) are within ±5 trials of the top (sanity that the new study reproduces the previous finding's strength).

**Files to create**:
- `scripts/accent_coach_phase0_10_summary.py` — reads optuna.db, prints top 10 with mean ± std for each objective, marks the closest match to Phase 0.9 config.

**Output to**: `docs/accent_coach_phase0_10_top_trials.csv`

---

#### C2. 4-speaker comparison plot

**What**: Plot exactly 4 speakers (owner, real_BC, synth_BC[best], lindsey) on a single Bark-normalized F1×F2 vowel-space chart for the 17 target phonemes.

**Files to create**:
- `scripts/accent_coach_phase0_10_plot.py` — fork/adapt [scripts/accent_coach_phase0_9_vowel_plot.py](../scripts/accent_coach_phase0_9_vowel_plot.py); strip the multi-variant logic, show exactly 4 speakers.

**Plot spec**:
- F2 axis inverted (left→right); F1 axis inverted (top→bottom)
- Bark scale (use `bark_transform` from `accent_coach.diagnostics.bark_distance`)
- 17 phonemes plotted: 12 monophthongs + l, r, m, n, ŋ
- Colors: owner=red, real_BC=blue, synth_BC=green, lindsey=gray
- IPA labels placed at lindsey's centroid for each phoneme (lindsey is the RP exemplar; do NOT label modern_rp)
- Output: `docs/img/phase0_10_4speaker_plot.png`

**Critical**: do NOT show `modern_rp` aggregate, `fry`, `bbc_male`, or any 5th series. Exactly 4 speakers on the chart, period.

---

#### C3. Posthoc WER/ECAPA/DNSMOS on best params

**What**: For the top-1 trial, run posthoc scoring on `eval_short.csv` (15 phrases) at N=3 replicates. Report mean ± std.

**Reuse**: `score_posthoc_clips` from `accent_coach/pipeline/experiment.py` (built in A3).

**Output to**: `docs/accent_coach_phase0_10_posthoc.csv` with columns `replicate,WER,ECAPA,DNSMOS_OVR`.

**Compare to**: Phase 0.9 `C_int_b5` baseline (WER=0.037, ECAPA=0.276, DNSMOS=2.67). Phase 0.10 best should be no worse on WER (≤ 0.05) and ideally better on DNSMOS.

---

#### C4. Findings doc

**What**: Write `docs/accent_coach_phase0_10_findings.md` following the format of [accent_coach_phase0_9_findings.md](accent_coach_phase0_9_findings.md). Required sections:

1. **Verdict** (one line: GREEN if best `composite_lindsey_mean ≥ 82 AND composite_lindsey_mean - std ≥ 80` AND `composite_BC_mean ≥ 75`; YELLOW if only one threshold met; RED otherwise)
2. **Scores table** — top 10 trials, columns: trial_id, num_beams, temp, top_p, cfg_rate, diff_steps, lindsey_mean±std, BC_mean±std, loss
3. **Stop criteria evaluation** — explicit pass/fail per threshold
4. **Finding 1** — does CFM tuning (`cfg_rate`, `diffusion_steps`) actually move the score, or is it dominated by GPT-side params? (Use `optuna.importance.get_param_importances(study)`.)
5. **Finding 2** — is `d(synth_BC, real_BC)` correlated with `d(synth_BC, lindsey)` or are they independent dimensions? (Compute Spearman ρ across trials' `composite_BC_mean` vs `composite_lindsey_mean`.)
6. **Finding 3** — does the new winning config beat Phase 0.9 `C_int_b5` on WER/ECAPA/DNSMOS too?
7. **Caveats** — list anything that bit you (MPS gotchas, pruned trials, etc).

---

## 6. Anti-patterns — do not do these

1. **Do not use N=1 anywhere except the explicit B2 pilot.** Every other cell is N=3.
2. **Do not edit `vendor/index-tts/` beyond the two-line change in A2.** Pin is at SHA `830f6f8f`; bigger diffs invalidate the pin contract in CLAUDE.md.
3. **Do not change the reference clip.** It is `ref_interview.wav` everywhere.
4. **Do not rewrite Bark transform, piecewise scoring, sigma_rp computation, or formant extraction.** They exist; import them.
5. **Do not modify `tts_output/accent_coach/bench/phase0_7/speaker_centroids.json`.** Read-only.
6. **Do not modify the 50-phrase `cal_50.csv`.** Build `cal_25.csv` as a separate file.
7. **Do not run B3 if B2 had any errors.** Stop, report.
8. **Do not poll background jobs.** When `run_in_background=true`, you get a callback when done. Use the wait.
9. **Do not switch sampler or weighting mid-study.** Same objective from pilot to full.
10. **Do not delete or overwrite `optuna.db` once trials have run.** It is the single source of truth.
11. **Do not score against `modern_rp` anywhere.** Only `lindsey` and `real_BC` are valid targets in Phase 0.10. `modern_rp` is used ONLY to derive `sigma_rp`.
12. **Do not add a 5th speaker series to the C2 plot.** Exactly: owner, real_BC, synth_BC, lindsey.

---

## 6.5. Parallelization — where it helps, where it doesn't

The dominant cost is IndexTTS-2 generation (1-2 GB resident; cannot reliably run >2
instances on an 18 GB Mac). The cheap-RAM stages CAN be parallelized. Use these
opportunities but only after Tasks A1-A6 verify correctness in serial.

### Cheap wins — implement during Task A3 refactor

| Stage | RAM/worker | Approach | Add to |
|---|---|---|---|
| Formant extraction (Whisper align + Parselmouth formants) per phrase | ~500 MB (Whisper-small or shared model) | `multiprocessing.Pool(n=os.cpu_count()//2)` over phrases inside `extract_formants`. Load Whisper inside each worker's init function. | `accent_coach/pipeline/experiment.py:extract_formants` |
| Posthoc WER/ECAPA/DNSMOS per clip | ~700 MB | `multiprocessing.Pool` over wav paths. Each worker loads its own Whisper+ECAPA+DNSMOS singletons (lazy on first call). | `accent_coach/pipeline/experiment.py:score_posthoc_clips` |

Add a `n_workers: int = None` (None → autodetect = `max(1, cpu_count() // 2)`) parameter to both functions. Default behaviour stays single-process if `n_workers=1` is passed — needed for debugging.

### Speculative win — try only after pilot (B2) passes

| Stage | RAM/worker | Approach | Risk |
|---|---|---|---|
| Optuna trial-level | ~2 GB (full IndexTTS-2 model) | `study.optimize(objective, n_jobs=2)` — runs 2 trials in parallel processes. Each loads its own IndexTTS-2. | RAM pressure: 2× IndexTTS-2 (~3-4 GB) + Whisper/ECAPA/DNSMOS (~1.5 GB) + system overhead ≈ 6-8 GB. Marginal on 18 GB. Try `n_jobs=2` after B2; rollback to `n_jobs=1` if any OOM or MPS contention. |

**Do NOT enable `n_jobs>1`** until:
1. B2 pilot completes cleanly at `n_jobs=1`
2. A single trial's RAM footprint is measured at peak (use `psutil.Process().memory_info().rss` logged during B2)
3. Estimated `n_jobs × peak_rss + 4 GB headroom < 18 GB`

If MPS is in use, also confirm MPS supports concurrent contexts (try a 2-process synthetic test first — generate one phrase from each of 2 Python processes simultaneously and confirm both finish).

### What NOT to parallelize

- **IndexTTS-2 within a single trial across replicates** — they share the same loaded model in-process; spawning a worker per replicate would just reload the model 3× and lose to serial. Keep replicates serial inside one trial.
- **The 25 phrases inside one generation pass** — IndexTTS-2 is internally single-stream; phrase-level parallelism does not exist at the model level.

---

## 7. File outputs (final state)

```
scripts/
  accent_coach_phase0_10_build_cal25.py        [new]
  accent_coach_phase0_10_run_cell.py           [new]
  accent_coach_phase0_10_optuna.py             [new]
  accent_coach_phase0_10_summary.py            [new]
  accent_coach_phase0_10_plot.py               [new]
  indextts_gen.py                              [modified: device auto + 2 new flags]
  indextts_smoke_test.py                       [modified: device auto]
  accent_coach_phase0_9_run.py                 [modified: imports from pipeline.experiment]

accent_coach/
  pipeline/experiment.py                       [new]

vendor/index-tts/indextts/
  infer_v2.py                                  [modified: 2 hard-coded values exposed as kwargs, ≤10 lines]

tts_output/accent_coach/
  cal_25.csv                                   [new]
  phase0_10/
    optuna.db                                  [new — Optuna trial store]
    cell_smoke/                                [from A6]
    trial_<id>/                                [per-trial outputs]

docs/
  accent_coach_phase0_10_plan.md               [this file]
  accent_coach_phase0_10_top_trials.csv        [new]
  accent_coach_phase0_10_posthoc.csv           [new]
  accent_coach_phase0_10_findings.md           [new]
  img/phase0_10_4speaker_plot.png              [new]
```

---

## 10. Appendix: A6 sanity-range caveat (Task A6 — 2026-05-23)

The A6 cell smoke with Phase 0.9 winning params produced:

```
composite_lindsey = 53.70   ← below the [60, 95] sanity range
composite_BC      = 60.50   ← within range
```

Root cause: the [60, 95] range was written assuming lindsey scores ≈ modern_rp
scores. Diagnostic on frozen Phase 0.7 synth_BC centroids shows:
- synth_BC vs modern_rp = 74.60 (Phase 0.9 basis)
- synth_BC vs lindsey   = 58.80 (already below 60 at baseline)
- synth_BC vs real_BC   = 73.50

Scoring against the single speaker `lindsey` is structurally stricter than
scoring against the `modern_rp` aggregate. The sanity floor of 60 for
`composite_lindsey` is too high; a real floor is ~50-55 for current synth_BC.
The warning is **expected on cal_25 with baseline params and is not a bug**.

**Resolution**: warning is left in place as a signal (not a hard stop). Actual
per-speaker scores will be interpreted relative to the multi-trial sweep
distribution, not against an absolute threshold calibrated for modern_rp.

---

## 9. Appendix: MPS failure (Task A1 — 2026-05-22)

MPS was auto-detected and loaded successfully (model weights moved to MPS device).
Inference crashed during the BigVGAN vocoder pass with:

```
NotImplementedError: Output channels > 65536 not supported at the MPS device.
  File ".../bigvgan/alias_free_activation/torch/resample.py", line 33, in forward
    x = self.ratio * F.conv_transpose1d(
```

BigVGAN uses a large anti-aliased convolution (alias_free_activation) whose output
channel dimension exceeds Apple's MPS limit of 65536. This is not patchable without
rewriting the vendor model.

**Resolution**: both `indextts_gen.py` and `indextts_smoke_test.py` have `_DEFAULT_DEVICE = "cpu"`.
All Phase 0.10 generation runs on CPU (~45-90 s per phrase, consistent with CLAUDE.md).

---

## 8. Quick reference — verification commands

```bash
# After A1 (MPS check):
vendor/index-tts/.venv/bin/python scripts/indextts_smoke_test.py

# After A2 (CFM exposed):
vendor/index-tts/.venv/bin/python scripts/indextts_gen.py \
  --phrases-csv tts_output/cross_eval_50/eval_short.csv \
  --out-dir /tmp/phase0_10_cfm_smoke --cfg-rate 0.7 --diffusion-steps 25
vendor/index-tts/.venv/bin/python scripts/indextts_smoke_test.py

# After A3+A4 (refactor + reproduction):
.venv/bin/python scripts/accent_coach_phase0_9_run.py --variants A1 --skip-gen --skip-formants --skip-posthoc
# Must print: A1 validity check PASSED: 74.6

# After A5 (cal_25):
.venv/bin/python scripts/accent_coach_phase0_10_build_cal25.py

# After A6 (cell smoke):
.venv/bin/python scripts/accent_coach_phase0_10_run_cell.py \
  --params-json '{"num_beams":5,"temperature":0.8,"top_p":0.8,"cfg_rate":0.7,"diffusion_steps":25}' \
  --replicate 0 --out-dir tts_output/accent_coach/phase0_10/cell_smoke/

# After B2 (pilot):
.venv/bin/python scripts/accent_coach_phase0_10_optuna.py --n-trials 5 --replicates 1 --pilot

# After B3 (full study, runs long):
.venv/bin/python scripts/accent_coach_phase0_10_optuna.py --n-trials 30 --replicates 3 --resume
```
