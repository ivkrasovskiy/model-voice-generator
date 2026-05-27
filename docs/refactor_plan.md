# Repo cleanup & refactor plan

**Audience**: an engineer (Sonnet) executing the cleanup. **Goal**: shrink the
repo to a reusable `accent_coach/` + `scripts/lib/` library plus thin
experiment drivers, summarize obsolete phases into one history doc, and reclaim
~1 GB of dead generated audio — without breaking the IndexTTS-2 production path
or the accent-coach phase 0.13 work (the only active experiment).

Owner decisions baked into this plan:
1. **Delete** superseded phase drivers (0.5–0.12) after writing their repro summary.
2. **Consolidate** all per-phase docs into one history doc; delete the originals (git keeps them).
3. **Delete** dead-experiment audio, but **preserve every centroid/result JSON + corpus formants CSV** with a written provenance note (which dataset / how many tracks each was built from).

---

## Baseline (verified 2026-05-27, before any change)

| Check | Command | Current state |
|---|---|---|
| Tests | `.venv/bin/python -m pytest tests/ -q` | **30 passed** (~81 s) |
| Lint | `.venv/bin/python -m ruff check scripts/ accent_coach/` | **29 errors** (20 auto-fixable) |
| Disk | `du -sh tts_output` | **1.6 G** |
| Smoke | `vendor/index-tts/.venv/bin/python scripts/indextts_smoke_test.py` | passes (~2 min, run once to confirm) |

Re-run these after every stage. The end state must be: **tests 30 passed,
ruff 0 errors, smoke test passes.**

## DO NOT TOUCH (locked artifacts — CLAUDE.md)

- `vendor/` (vendored IndexTTS-2 + its `.venv`)
- `tts_output/eval_indextts_v2/` and `…/scores.regression_baseline.csv`
- `tts_output/ref_interview.wav` (production reference clip) — its **content** is
  locked (never overwrite/regenerate). Stage 5e **moves** it to
  `tts_output/refs/production/ref_interview.wav` (owner-approved) and updates the
  CLAUDE.md path; moving ≠ modifying.
- `tts_output/cross_eval_50/` (the frozen eval CSVs `eval_short.csv` / `eval_long.csv`)
- IndexTTS-2 install pins in `CLAUDE.md`

---

## Target architecture

```
accent_coach/                 reusable accent-analysis library (already foldered)
  calibration/                calibration sentence sets
  comparison/                 per-feature scoring (vowels, consonants, …)
  diagnostics/                normalization schemes, bark, clustering, advice
  dsp/                        formant shifting
  pipeline/                   alignment, formants, prosody, vot, features
    generate.py     [new]     generate_clips()         (moved out of experiment.py)
    centroids.py    [new]     build/load/score centroids (moved out of experiment.py)
  reference/                  RP norms, normalization
scripts/
  lib/                        cross-cutting TTS + audio primitives
    scoring.py      [new]     score_clips() — the ONE WER/ECAPA/DNSMOS loop
  <thin experiment drivers>   phase 0.13 + production entry points only
docs/
  accent_coach_history.md [new]  3-sentence repro block per retired phase
```

Rule: **every shared concern lives in exactly one place** — audio generation,
WER/ECAPA/DNSMOS scoring, formant extraction, phoneme alignment, centroid
math. Experiments import them; they never re-implement them.

---

## Stage 0 — Safety net (do first, do not skip)

1. The working tree has uncommitted phase-0.13 work (`git status`: modified
   `pyproject.toml`, `scripts/accent_coach_extract_formants.py`, etc. + untracked
   `accent_coach/dsp/`, `accent_coach/diagnostics/f3_normalization.py`, the
   `phase0_13*` scripts and docs). **Commit that first** so the refactor is a
   clean, isolated, revertible diff:
   ```bash
   rtk git add accent_coach/ scripts/accent_coach_phase0_13* docs/accent_coach_phase0_13* \
       scripts/accent_coach_extract_formants.py pyproject.toml uv.lock
   rtk git commit -m "Phase 0.13 work-in-progress (pre-refactor snapshot)"
   ```
2. Work on the current branch `ivk-temp-branch`; **commit after each stage**
   below so any stage is independently revertible.
3. Run the four baseline checks above and record the numbers.

**Check**: `git status` clean except intended changes; baseline numbers recorded.

---

## Stage 1 — Summarize retired phases, then delete their docs

### 1a. Write `docs/accent_coach_history.md`

One block per retired phase (0.5, 0.6, 0.7, 0.8, 0.9, 0.10, 0.11, 0.12). Each
block is **exactly the 5 elements needed to reproduce**, ~3 sentences:

- **Goal** — the hypothesis/question.
- **Inputs** — data + which `tts_output/` artifacts it consumed.
- **Command** — the driver invocation (copy from the deleted script's docstring/`argparse`).
- **Output** — the artifact it produced (centroid JSON, CSV, doc) + where it lives.
- **Verdict** — GREEN/YELLOW/RED outcome and what it changed downstream.

Source material: the existing `docs/accent_coach_phase0_*_findings.md` and
`_plan.md`. **⚠ Phases 0.11 and 0.12 have no findings doc** — reconstruct their
blocks from the driver scripts (`accent_coach_phase0_11_emo_sweep.py`,
`_11_n3_confirm.py`, `_12_rebuild.py`) and `tts_output/accent_coach/phase0_11_n3_run.log`
*before* deleting those scripts in Stage 2.

Phase summary anchors (verify against sources, do not trust blindly):

| Phase | One-line | Key output (keep) |
|---|---|---|
| 0.5 | Modern-RP reference design (Fry/Lindsey cluster supersedes Deterding 1997) | RP norms → baked into `reference/rp_norms.py` |
| 0.6 | Lobanov z-scoring erases L2 vowel compression → use raw-Hz per-phoneme | (diagnostic, no artifact) |
| 0.7 | Rebuilt RP norms from modern_rp corpus; froze baseline centroids | `accent_coach/bench/phase0_7/speaker_centroids.json` |
| 0.8 | Tested H1–H5 normalization (VTL/Nearey/anchor/bark); bark piecewise won | `accent_coach/phase0_8/h*.json` |
| 0.9 | Reference-clip swap; set production `num_beams=5` (H4 74.6→82.1) | `accent_coach/phase0_9/experiment_log.csv` |
| 0.10 | Optuna GPT+CFM sweep; built `cal_25` calibration set | `accent_coach/phase0_10/` trials, `cal_25.csv` |
| 0.11 | Emotion-vector sweep + N=3 confirmation | `accent_coach/phase0_11*/` |
| 0.12 | Rebuilt centroids on ECAPA-cleaned corpus | `cleaned_corpus/speaker_centroids_cleaned.json` |

Keep these docs (current / cross-cutting): `accent_coach_plan.md`,
`accent_coach_technical_spec.md`, all `accent_coach_phase0_13*.md`, `history.md`.

### 1b. Delete the retired per-phase docs

```
docs/accent_coach_phase0_findings.md
docs/accent_coach_phase0_5_plan.md  _5_findings.md  _5_table.csv  _5_verdicts.json  _5_verdicts.md
docs/accent_coach_phase0_6_plan.md  _6_findings.md  _6_geometry.csv  _6_lobanov_distances.csv  _6_lobanov_table.csv  _6_verdicts.json  _6_verdicts.md
docs/accent_coach_phase0_7_plan.md  _7_findings.md
docs/accent_coach_phase0_8_plan.md  _8_findings.md
docs/accent_coach_phase0_9_plan.md  _9_findings.md
docs/accent_coach_phase0_10_plan.md  _10_findings.md  _10_posthoc.csv  _10_top_trials.csv
```

Then fix dangling links: `accent_coach_plan.md` and `rp_norms.py` reference
`phase0_5/6/7_findings.md`. Repoint them to `accent_coach_history.md#phase-0X`.

**Check**:
- `accent_coach_history.md` has 8 blocks, each with all 5 elements.
- `rtk grep -rn "phase0_\(5\|6\|7\|8\|9\|10\)_\(plan\|findings\)" docs/ accent_coach/ scripts/` → **0 matches** (no live links to deleted docs).
- `git log --stat` shows the deleted docs still in history.

---

## Stage 2 — Delete obsolete code (scripts)

No script imports another script (verified — they only import `accent_coach.*`
and `lib.*`), so these deletions are import-safe. **Delete:**

```
# phase 0.7 helpers
scripts/accent_coach_phase0_7_bbc_audit.py
scripts/accent_coach_phase0_7_make_manifest.py
scripts/accent_coach_phase0_7_rebuild_norms.py
# phase 0.8
scripts/accent_coach_phase0_8_run.py
# phase 0.9
scripts/accent_coach_phase0_9_run.py
scripts/accent_coach_phase0_9_vowel_plot.py
# phase 0.10
scripts/accent_coach_phase0_10_build_cal25.py
scripts/accent_coach_phase0_10_optuna.py
scripts/accent_coach_phase0_10_plot.py
scripts/accent_coach_phase0_10_posthoc.py
scripts/accent_coach_phase0_10_run_cell.py
scripts/accent_coach_phase0_10_summary.py
# phase 0.11
scripts/accent_coach_phase0_11_emo_sweep.py
scripts/accent_coach_phase0_11_n3_confirm.py
# phase 0.12
scripts/accent_coach_phase0_12_rebuild.py
# phase 0.5/0.6 table + verdict + plot generators (superseded by 0.9 plot)
scripts/accent_coach_build_table.py
scripts/accent_coach_compute_verdicts.py
scripts/accent_coach_vowel_space_plot.py
```

**Keep** (production TTS): `_dotenv_init.py`, `indextts_gen.py`,
`indextts_smoke_test.py`, `posthoc_eval.py`, `build_podcast_ref.py`, all of `scripts/lib/`.

**Keep** (active accent-coach — phase 0.13 + product + reusable):
`accent_coach_phase0_13_lever_a.py`, `_13_lever_b.py`, `_13_mine_emo.py`,
`_13c_f3_bench.py`, `accent_coach_extract_formants.py` (called by
`pipeline/experiment.py` via subprocess — **mandatory**), `accent_coach_analyze.py`
(single-recording product entry), `accent_coach_bench.py` (validation bench),
`accent_coach_build_modern_rp.py`, `accent_coach_build_real_bc.py`,
`accent_coach_corpus_audit.py` (corpus builders/audit — produce kept corpora).

**Check**:
- For each deleted filename, `rtk grep -rn "<basename without .py>" scripts/ accent_coach/ notebooks/ docs/accent_coach_phase0_13* CLAUDE.md` → **0 matches**.
- Every kept script still parses & imports:
  ```bash
  for f in scripts/accent_coach_phase0_13*.py scripts/accent_coach_analyze.py scripts/accent_coach_bench.py; do
    .venv/bin/python -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$f" && echo "OK $f"; done
  ```
- Smoke test still passes (guards `indextts_gen` + `extract_formants` subprocess paths).

---

## Stage 3 — De-duplicate shared utilities (no code duplicates)

WER/ECAPA/DNSMOS scoring is currently re-implemented in **5 places**:
`posthoc_eval.py`, `accent_coach/pipeline/experiment.py:score_posthoc_clips`,
`accent_coach_build_modern_rp.py`, `accent_coach_build_real_bc.py`,
`accent_coach_corpus_audit.py`. Collapse to one.

### 3a. Create `scripts/lib/scoring.py`

It owns the single scoring loop, built on the existing primitives
(`lib.transcribe.load_whisper`, `lib.identity.load_ecapa/embed_wav/cosine`,
`lib.metrics.compute_wer/compute_dnsmos`). Public API:

```python
def score_clips(manifest: Path, ecapa_ref: Path, out_csv: Path,
                cache: bool = True) -> dict:   # -> {"WER","ECAPA","DNSMOS_OVR"}
```

Lift the body almost verbatim from `experiment.py:score_posthoc_clips`
(lines ~303–399) — it is already the most complete version (lazy model
singletons, cached-CSV short-circuit, per-clip NaN handling).

### 3b. Repoint every caller

- `accent_coach/pipeline/experiment.py` → `score_posthoc_clips` becomes a 1-line
  delegate to `lib.scoring.score_clips` (or delete it and update importers:
  `accent_coach_phase0_13_lever_a.py`, `_13_lever_b.py` import it).
- `posthoc_eval.py` → its scoring helpers (`_score_one`, the WER/ECAPA/DNSMOS
  block ~166–230) call `lib.scoring`.
- `build_modern_rp.py`, `build_real_bc.py`, `corpus_audit.py` → replace their
  inline model-load + score blocks with `lib.scoring` calls.

### 3c. Split `experiment.py` (399 lines, at the limit) by concern

- `accent_coach/pipeline/generate.py` ← `generate_clips` (+ the `INDEXTTS_PYTHON`/param-mapping helper).
- `accent_coach/pipeline/centroids.py` ← `build_centroids_from_formants`,
  `load_baseline_centroids`, `score_against` (+ `BASELINE/CLEANED_CENTROIDS_PATH`).
- Keep `extract_formants` (subprocess wrapper) in `pipeline/formants.py` next to the real extractor, or in `generate.py`.
- Update phase-0.13 imports accordingly (they currently do
  `from accent_coach.pipeline.experiment import (...)`).

Other dedup to confirm while here:
- **Formants**: ensure nothing re-implements parselmouth extraction outside
  `accent_coach/pipeline/formants.py` (`phase0_8_run.py` did — it's deleted in Stage 2; verify no kept file calls `parselmouth` except `pipeline/formants.py`).
- **Optuna**: only the deleted `phase0_10_*` used it → no shared optuna util needed. If a future phase needs it, add `accent_coach/calibration/optuna_search.py` then.

**Check**:
- `rtk grep -rln "compute_wer\|compute_dnsmos\|load_whisper" scripts/ accent_coach/` → only `scripts/lib/scoring.py` and `scripts/lib/metrics.py`/`transcribe.py` (the primitives). No corpus builder or phase driver appears.
- `rtk grep -rn "parselmouth" accent_coach/ scripts/` → only `accent_coach/pipeline/formants.py`.
- **Numerical equivalence**: re-score one already-scored cached manifest with the
  new path and diff against the committed CSV — values must match to 4 dp, e.g.
  ```bash
  .venv/bin/python -c "from pathlib import Path; from scripts.lib.scoring import score_clips; \
    print(score_clips(Path('tts_output/eval_indextts_v2/manifest.json'), Path('tts_output/ref_interview.wav'), Path('/tmp/recheck.csv'), cache=False))"
  ```
  Compare to `tts_output/eval_indextts_v2/scores.regression_baseline.csv`.
- All imports resolve; tests pass.

---

## Stage 4 — Refactor oversized modules (≤ 400 code lines each)

"Code lines" = non-blank, non-comment. After Stages 2–3 the only files still
over budget are the kept ones below. Count with:

```bash
for f in $(git ls-files '*.py') $(find accent_coach scripts -name '*.py'); do
  n=$(grep -vcE '^[[:space:]]*(#.*)?$' "$f"); [ "$n" -gt 400 ] && echo "$n  $f"; done | sort -rn | sort -u
```

Targets (post-dedup line counts will already be lower):

| File | Raw lines | Action |
|---|---|---|
| `scripts/posthoc_eval.py` | 608 | After 3b dedup, split remaining orchestration: keep CLI + main pipeline; move posthoc-specific helpers (`_score_one`, sampling) into a small `scripts/lib/posthoc_helpers.py` if still >400. |
| `scripts/accent_coach_build_real_bc.py` | 539 | After 3b dedup, split YouTube download/segment logic into a reusable `scripts/lib/corpus_build.py` shared with `build_modern_rp.py` (both download → VAD/segment → score). |
| `scripts/accent_coach_phase0_13c_f3_bench.py` | 417 | Trim: move F3-normalization math already in `accent_coach/diagnostics/f3_normalization.py`; the driver should only orchestrate. |
| `scripts/accent_coach_build_modern_rp.py` | 403 | Falls under 400 once download/score moved to `lib/corpus_build.py` + `lib/scoring.py`. |
| `accent_coach/pipeline/experiment.py` | 399 | Resolved by the Stage 3c split. |

Note: `build_real_bc.py` and `build_modern_rp.py` share a download→segment→audit
shape → extracting `lib/corpus_build.py` removes duplication AND brings both
under budget (kills two birds).

**Check**:
- The line-count command above prints **nothing** (all `.py` ≤ 400 code lines).
- Folder layout matches the target tree (new modules in `accent_coach/pipeline/` and `scripts/lib/`).
- Tests pass; smoke test passes; ruff still 0 (see Stage 6).

---

## Stage 5 — Clean `tts_output/` (delete dead audio, keep results + provenance)

`tts_output/` is fully gitignored (0 tracked files) → deletions are local-only
and regenerable. **This is the one irreversible-on-disk step — keep the
keep/delete table exact.**

### 5a. KEEP (results, references, corpora, eval baselines)

| Path | Why keep |
|---|---|
| `eval_indextts_v2/`, `eval_indextts_v2_long/` | locked IndexTTS-2 baseline (regression) |
| `eval_indextts_interview_short/`, `…_long/` | current production-config eval |
| `cross_eval_50/` | frozen eval CSVs |
| `accent_coach/` | **keep all `*.json` centroids, `*.csv` formants/scores, manifests, logs** |
| `modern_rp_corpus/`, `real_bc_corpus/` | RP + BC corpora the centroids were built from |
| `bc_cal_50/`, `owner_cal_50/`, `cal_25.csv`, `cal_50.csv` | calibration sets |
| All `ref_*.wav` at `tts_output/` root | reference clips for generation |

### 5b. DELETE (dead non-IndexTTS experiments)

```
tts_output/_podcast_dl/                 (390M yt-dlp download cache)
tts_output/eval_xtts_v2/                (XTTS — retired model)
tts_output/openvoice_eval/              (OpenVoice — retired)
tts_output/kd_lam05_eval/  posthoc_8blk/  posthoc_8blk_1500/  posthoc_stella/   (F5 KD)
tts_output/posthoc/  posthoc_6phrase/   (deprecated 6-phrase/centroid-ECAPA — see history.md)
tts_output/sweep_v2_cfg10_subset/ …_cfg15 …_cfg15_subset …_cfg20 …_cfg30 …_cfg40  sweep_cfg30/   (F5 CFG sweeps)
tts_output/selective_cfg_smoke/  selective_cfg_t008_cfg20/  nfe64_selcfg_cfg20/                  (F5 CFG)
tts_output/eval_indextts_centroid_short/  eval_indextts_centroid_long/                            (failed centroid exp)
tts_output/eval_combined_v2/  cross_eval_v1/  compare/  validate/  ref_clip_ab/  exp_speedfix_imperative/
tts_output/f5_demo_1.wav  f5_demo_2.wav  f5_demo_3.wav
```

### 5c. Trim generated WAVs inside kept accent_coach phase dirs (optional, big win)

Inside `tts_output/accent_coach/phase0_*/`, the bulk (≈573 MB) is generated
`.wav`. The **results are the JSON/CSV**, which we keep. Delete only the
generated audio, preserving every `*.json`, `*.csv`, `manifest.json`, `*.log`:

```bash
find tts_output/accent_coach -path '*/phase0_*' -name '*.wav' -delete   # review with -print first
```

### 5d. Centroid provenance manifest (owner's explicit ask)

Write `tts_output/accent_coach/CENTROIDS.md` recording, for each preserved
centroid JSON, **which dataset and how many tracks** it was built from:

| Centroid file | Source corpus | N | Built by (phase) |
|---|---|---|---|
| `bench/phase0_7/speaker_centroids.json` | modern_rp fry+lindsey+bbc_male | n≈749/source (per-phoneme) | 0.7 |
| `cleaned_corpus/speaker_centroids_cleaned.json` | ECAPA-cleaned fry+lindsey | (record from manifest) | 0.12 |
| `cleaned_corpus/fry/centroid.json`, `…/lindsey/centroid.json` | per-speaker cleaned | (record) | 0.12 |
| `cal_50.csv` | calibration sentences | 50 rows | 0.10 |
| `cal_25.csv` | calibration subset | 19 rows | 0.10 |

(Fill exact N by reading each manifest; do not guess.)

### 5e. Reorganize reference clips into purpose-named folders

The root `tts_output/ref_*.wav` clips are scattered and opaque — you can't tell
what each is for without reading docs. **Fix this physically**: move them into
`tts_output/refs/<purpose>/` subfolders so the filesystem is self-documenting.
(Owner decision: folders over a README — the layout itself must convey purpose.)

**Run this step AFTER Stage 2.** Most clips are referenced only by the deleted
phase 0.9/0.10/0.11 drivers, so after Stage 2 only **four** clips have live
references to update: `ref_interview`, `ref_narrator`, `ref_fry_emo`,
`ref_lindsey_emo`. The rest are orphaned inputs (still filed for provenance).

Target layout:

```
tts_output/refs/
  production/
    ref_interview.wav            # CLAUDE.md production ref (interview 4:47–5:01)
  indextts_baseline/
    ref_narrator.wav             # Casanova audiobook 12 s — baseline + smoke test
    ref_sherlock.wav             # Sherlock audiobook
    ref_combined.wav             # combined cas+sher
  phase0_9_refswap/              # ref-clip-swap candidates (phase 0.9)
    ref_interview_429_446.wav
    ref_interview_1450_1510.wav
  rp_ceiling/                    # RP upper-bound refs (phase 0.9 / 0.11)
    ref_fry.wav   ref_fry_emo.wav
    ref_lindsey.wav  ref_lindsey_emo.wav
```

Mandatory path updates after moving (the only surviving live references):

| Clip → new path | Files to update |
|---|---|
| `ref_interview.wav` → `refs/production/` | `accent_coach_phase0_13_lever_b.py`, `accent_coach_build_real_bc.py`, **`CLAUDE.md`** (3 mentions: "What this project does", locked-artifacts list, "How to generate"), `notebooks/generate.ipynb` |
| `ref_narrator.wav` → `refs/indextts_baseline/` | `indextts_smoke_test.py`, `accent_coach_phase0_13_lever_a.py`, `accent_coach_phase0_13_lever_b.py`, `build_podcast_ref.py`, `posthoc_eval.py` |
| `ref_fry_emo.wav` → `refs/rp_ceiling/` | `accent_coach_corpus_audit.py` |
| `ref_lindsey_emo.wav` → `refs/rp_ceiling/` | `accent_coach_corpus_audit.py` |

The orphaned clips (`ref_sherlock`, `ref_combined`, `ref_fry`, `ref_lindsey`,
`ref_interview_429_446`, `ref_interview_1450_1510`) have **no live references
after Stage 2** — move them for provenance, no code edits needed.

Also add a one-line `tts_output/refs/README.md` index of the folders (the
folder names carry the meaning; the README is just a courtesy map).

**Check**:
- `rtk grep -rn "tts_output/ref_[a-z]" scripts/ accent_coach/ CLAUDE.md notebooks/` returns **0** root-level hits (all now under `refs/…`; ignore `__pycache__`).
- `CLAUDE.md`'s locked-artifacts list and generate command point at `tts_output/refs/production/ref_interview.wav`.
- **Smoke test passes** with the new `ref_narrator.wav` path (it is the production guard — re-run after the move).

**Check**:
- `du -sh tts_output` dropped by ~1 GB (was 1.6 G).
- Every path in the Stage-5a KEEP table still exists.
- `tts_output/accent_coach/CENTROIDS.md` exists with real N values.
- Smoke test passes (`ref_interview.wav` + eval baseline untouched).
- `rtk grep -rn "tts_output/" scripts/ accent_coach/` → no live code points at a deleted dir.

---

## Stage 6 — Lint & tests green

1. `.venv/bin/python -m ruff check scripts/ accent_coach/ --fix` (clears the 20
   auto-fixable). Then hand-fix the rest: `B023` (loop-var closure in the now-deleted
   `phase0_11_n3_confirm.py` — gone), `F841` unused var, `E741` ambiguous `l`,
   `B905` `zip(..., strict=…)` in `f3_normalization.py`, `F401`/`F811` in
   `experiment.py` (resolved by the Stage-3c split).
2. `.venv/bin/python -m ruff check scripts/ accent_coach/` → **0 errors**.
3. `.venv/bin/python -m pytest tests/ -q` → **30 passed**.
4. `vendor/index-tts/.venv/bin/python scripts/indextts_smoke_test.py` → passes.

---

## Stage 7 — Update CLAUDE.md to the post-cleanup state

CLAUDE.md must describe the repo as it exists *after* Stages 1–6. Update:

- **Production ref path**: every `tts_output/ref_interview.wav` →
  `tts_output/refs/production/ref_interview.wav` (the "What this project does"
  intro, the locked-artifacts list, and the "How to generate" command).
- **Accent-coach docs pointers**: the per-phase `*_plan.md`/`*_findings.md` are
  gone — point readers to `docs/accent_coach_history.md` (retired phases 0.5–0.12),
  the `accent_coach_phase0_13*.md` docs (active), plus the surviving
  `accent_coach_plan.md` and `accent_coach_technical_spec.md`.
- **Scoring**: note WER/ECAPA/DNSMOS now live once in `scripts/lib/scoring.py`
  (`posthoc_eval.py` stays the CLI entry).
- **Add a short "Code layout" subsection** describing the target architecture
  (the tree near the top of this plan): `accent_coach/` folders, `scripts/lib/`
  primitives, thin drivers in `scripts/`.
- **Remove** any mention of deleted scripts/venvs (e.g. `.venv_openvoice`).
- **Keep and verify the "For spawned agents" block** — it already mandates the
  RTK prefix on every shell command and CodeGraph for symbol navigation. Confirm
  it still lists the *correct* interpreters (`.venv/bin/python`,
  `vendor/index-tts/.venv/bin/python`) and no deleted venvs, and that the
  RTK-prefix + CodeGraph (`codegraph_search`/`callers`/`callees`/`impact`)
  guidance is intact and prominent (see Stage 9c — this is the same requirement
  enforced for the main session).

**Check**: `rtk grep -n "ref_interview.wav" CLAUDE.md` → only the `refs/production`
path; no deleted script/doc/venv names appear in CLAUDE.md; the agent block names
RTK + CodeGraph + the two live venvs.

## Stage 8 — Future-proofing: standards + automated guardrails

Stop the repo from re-accumulating long scripts, ad-hoc folders, untested code,
and duplicated logic.

### 8a. Add a "Code standards" section to CLAUDE.md

- New experiments are **thin drivers** in `scripts/` that import `accent_coach/*`
  and `scripts/lib/*`. Never re-implement generation, scoring, formant
  extraction, phoneme alignment, or centroid math — extend the library instead.
- **Hard cap: ≤ 400 code lines** (non-blank, non-comment) per `.py`. Over budget
  → split into a module under the appropriate `accent_coach/` or `scripts/lib/` folder.
- Every new `accent_coach/`/`lib/` module needs a test in `tests/`; any change to
  the IndexTTS path must keep `indextts_smoke_test.py` green.
- Experiment outputs go under `tts_output/<run-name>/` via `--out-dir`; never commit audio.

### 8b. Commit an enforcement check

Add `scripts/check_repo.sh` (and/or a `Makefile` `check` target) that runs:
1. `ruff check scripts/ accent_coach/`
2. the ≤400-line guard (the Stage-4 counting command; exit nonzero if any file is over)
3. `pytest tests/ -q`

Document "run `bash scripts/check_repo.sh` before every commit" in CLAUDE.md.

### 8c. (Recommended) wire it as a pre-commit hook

Add `.pre-commit-config.yaml` (or a committed hook installer) so `check_repo.sh`
runs automatically on commit — the surest way to keep the standards enforced.

**Check**: `scripts/check_repo.sh` exits 0 on the clean post-refactor tree; a
deliberately added 401-line file makes it exit nonzero; CLAUDE.md documents both
the standards and the command.

## Stage 9 — Fix Claude Code permissions (fewer prompts, no conflicts)

**Diagnosis.** The two files don't truly *conflict* — allow-lists union and
neither has deny rules. The real problems: (1) `settings.local.json` holds ~120
stale one-off entries (`.venv_xtts`/`.venv_openvoice`/`finetune_f5`/specific
`kill <PID>`/`nohup …` lines) that are dead weight; (2) prompts still fire
because a prefix rule like `Bash(.venv/bin/python *)` matches **only when that
token is the first token of a single command**. It does **not** match when:
- a leading env assignment shifts the first token —
  `PYTHONHASHSEED=random .venv/bin/python …` (first token is the assignment), or
- the command uses `$(…)` substitution or chains/pipes in a way the matcher
  can't safely decompose.

### 9a. Consolidate canonical rules into committed `.claude/settings.json`

```jsonc
"allow": [
  "Bash(uv *)", "Bash(uv run *)",
  "Bash(.venv/bin/python *)", "Bash(.venv/bin/ruff *)", "Bash(.venv/bin/pytest *)",
  "Bash(vendor/index-tts/.venv/bin/python *)",
  "Bash(/Users/ivkrasovskii/model-voice-generator/.venv/bin/python *)",
  "Bash(/Users/ivkrasovskii/model-voice-generator/vendor/index-tts/.venv/bin/python *)",
  "Bash(rtk *)",                       // highest leverage — the RTK hook prefixes most commands
  "WebSearch", "WebFetch(domain:arxiv.org)", "WebFetch(domain:github.com)",
  "Read", "Edit", "Write"
]
```
Remove from `settings.json`: `Bash(.venv_openvoice/bin/python *)` + its absolute
variant (venv deleted), the one-off `tee`/`mkdir`/`pkill`/`nohup lever_a`/
`python3 -m json.tool` entries, and the redundant `additionalDirectories` that
point *inside* the project root (`scripts/`, `accent_coach/`,
`tts_output/accent_coach` are already under the cwd). Keep `/private/tmp`, `/tmp`.

### 9b. Prune `settings.local.json` to machine-local essentials

Delete every entry referencing `.venv_xtts`, `.venv_openvoice`, `.venv_indextts`,
`finetune_f5`, `regen_missing`, `xtts_gen`, `gen_finetuned`, specific `kill <PID>`,
and the one-off `nohup`/`awk`/build-index-tts lines — all obsolete after cleanup.
Keep only generic read-only helpers not covered by the committed rules
(`ps *`, `vm_stat`, `ffprobe …`, `ffmpeg -version`) if still wanted.

### 9c. Behavioral guidance (add to CLAUDE.md, applies to main session AND agents)

- Run interpreters as the **first token** — don't prefix with `VAR=val`; set seeds
  inside the script or `export` them in a separate (allowed) command first.
- Avoid wrapping interpreter calls in `$(…)` or piping through `| tee`/`| tail`
  when not needed — these defeat prefix matching and force a prompt.
- **Agents**: every shell command must be RTK-prefixed (`rtk git …`, `rtk grep …`)
  and navigation should use CodeGraph (`codegraph_search`/`callers`/`callees`/
  `impact`/`context`/`node`) over grep/find — already in the "For spawned agents"
  block; Stage 7 verifies it survives the CLAUDE.md edits.

### 9d. (Optional) run `/fewer-permission-prompts` after a few sessions

Lets the harness auto-add any remaining common read-only calls it observes.

**Check**:
- `settings.json` contains the consolidated rules incl. `Bash(rtk *)`; no
  `.venv_openvoice`/`.venv_xtts`/`.venv_indextts` entries anywhere in either file.
- `settings.local.json` is down to a short machine-local list (~≤15 entries),
  none referencing deleted venvs/scripts.
- A representative command (`.venv/bin/python -m pytest tests/ -q`) runs without a
  permission prompt; a piped/chained variant is understood to still prompt (documented limitation).

## Final verification checklist

- [ ] `ruff check scripts/ accent_coach/` = 0 errors
- [ ] `pytest tests/ -q` = 30 passed
- [ ] `indextts_smoke_test.py` passes (production path intact)
- [ ] No `.py` in `scripts/` or `accent_coach/` exceeds 400 code lines (Stage-4 command prints nothing)
- [ ] Scoring loop exists once (`grep` finds only `lib/scoring.py` + primitives)
- [ ] No parselmouth extraction outside `pipeline/formants.py`
- [ ] `accent_coach_history.md` covers phases 0.5–0.12, 5 elements each
- [ ] No live link/import to any deleted doc or script
- [ ] `du -sh tts_output` ≈ 0.6 G; all KEEP-table paths present
- [ ] `tts_output/accent_coach/CENTROIDS.md` written; `ref_*.wav` moved into `tts_output/refs/<purpose>/` folders + `refs/README.md` index
- [ ] No root-level `tts_output/ref_*.wav` references remain in code/CLAUDE.md/notebooks (all repointed to `refs/…`)
- [ ] `eval_indextts_v2/`, `cross_eval_50/`, `vendor/` untouched; `ref_interview.wav` moved (not modified)
- [ ] CLAUDE.md updated to post-cleanup state (ref path, doc pointers, scoring, code-layout); "For spawned agents" block intact with RTK + CodeGraph + correct venvs
- [ ] `scripts/check_repo.sh` (ruff + ≤400-line guard + pytest) committed and exits 0; CLAUDE.md documents standards + the command
- [ ] `.claude/settings.json` consolidated (incl. `Bash(rtk *)`, no dead venvs); `settings.local.json` pruned to machine-local essentials; representative `.venv/bin/python …` runs prompt-free
- [ ] Each stage committed separately (revertible)

## Risk notes

- **Irreversible**: Stage 5 disk deletions (regenerable but re-downloading
  YouTube corpora is fragile — that's why 5c keeps formants CSVs). Run every
  `rm`/`find -delete` with a `-print`/dry-run first and confirm the count.
- **Highest-blast-radius file**: `pipeline/experiment.py` — imported by the
  active phase-0.13 drivers. Do Stage 3c carefully and run the phase-0.13
  import check after.
- **Smoke test is the production guard** — if it ever fails after a stage,
  revert that stage's commit before continuing.
- **Ref-clip move (5e) touches the production path**: `ref_interview.wav`
  (CLAUDE.md) and `ref_narrator.wav` (smoke test) are moved. Update every path
  reference in the table *and* CLAUDE.md, then re-run the smoke test before
  committing the stage. Doing 5e after Stage 2 minimizes the edit surface.
