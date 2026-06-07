# CLAUDE.md

Guidance for Claude Code when working in this repo.

## What this project does

Generate Benedict Cumberbatch's voice from text. Current production model is
**IndexTTS-2 zero-shot** with a 14 s interview reference clip
(`tts_output/refs/production/ref_interview.wav`, sourced from
[youtu.be/cHmkAStZBkc](https://youtu.be/cHmkAStZBkc) at 4:47-5:01) and
**`num_beams=5`** (Phase 0.9 finding — raises H4 Bark piecewise from 74.6 → 82.1).
No fine-tuning involved.

For the full backstory — F5-TTS / XTTS-v2 / OpenVoice / F5R-TTS / centroid
experiments, dataset construction, why each approach failed — see
[docs/history.md](docs/history.md).

## Accent coach (in development)

Per-phoneme accent assessment pipeline that scores a user's English against
RP norms and against the cloned BC target. Code lives under `accent_coach/`,
driven by `scripts/accent_coach_*.py`. Read these only if you're working on
that feature: full spec in
[docs/accent_coach_technical_spec.md](docs/accent_coach_technical_spec.md),
sequenced Phase 0 execution plan in
[docs/accent_coach_plan.md](docs/accent_coach_plan.md), phase history
(outcomes of all completed phases 0.5–0.12) in
[docs/accent_coach_history.md](docs/accent_coach_history.md).

**Active experiment**: Phase 0.13 (Lever B formant shifting + Lever A emo
conditioning). Drivers: `scripts/accent_coach_phase0_13_lever_b.py`,
`scripts/accent_coach_phase0_13_lever_a.py`.

## Hardware

macOS, Apple M3 Pro, 18 GB unified memory. IndexTTS-2 runs on **CPU**
(~45-90 s per generated clip on the 15-phrase short eval). MPS path not
implemented.

## Environments

```bash
# Project deps (Whisper, ECAPA, DNSMOS — used by scoring)
uv sync
uv sync --group dev          # adds ruff

# IndexTTS-2 lives in its own vendored venv (pinned versions)
cd vendor/index-tts && uv sync --no-dev
```

| venv | Created by | For |
|---|---|---|
| `.venv/` | `uv sync` | Scoring (Whisper, ECAPA, DNSMOS), notebooks |
| `vendor/index-tts/.venv/` | `cd vendor/index-tts && uv sync --no-dev` | IndexTTS-2 inference |

Lint: `uv run ruff check scripts/ accent_coach/` must be zero errors before commits.

## Code layout

```
accent_coach/
  pipeline/          # generate.py, formants.py, centroids.py, experiment.py (re-exporter)
  dsp/               # formant_shift.py (Lever B DSP)
  diagnostics/       # f3_normalization.py, bark_distance.py
  reference/         # rp_norms.py — RP vowel targets
scripts/
  lib/               # scoring.py (WER/ECAPA/DNSMOS), transcribe.py, identity.py, metrics.py
  indextts_gen.py    # batch TTS generation
  posthoc_eval.py    # score a manifest (calls lib/scoring.py)
  accent_coach_phase0_13_*.py  — active Phase 0.13 drivers
tts_output/
  refs/              # reference WAVs (never overwrite)
    production/      # ref_interview.wav (IndexTTS-2 spk ref)
    indextts_baseline/ # ref_narrator.wav (smoke test + ECAPA ref)
  eval_indextts_v2/  # locked baseline WAVs + scores.regression_baseline.csv
  cross_eval_50/     # eval_short.csv, eval_long.csv
  accent_coach/      # per-phase cell outputs, centroids (see CENTROIDS.md)
```

**Shared scoring**: all WER/ECAPA/DNSMOS logic lives in `scripts/lib/scoring.py`.
Use `score_clips(manifest, ecapa_ref, out_csv)` for batch scoring or
`load_scoring_models()` + `score_single_wav()` for live per-clip scoring.
Never load Whisper/ECAPA/DNSMOS primitives directly in driver scripts.

## IndexTTS-2 install pins (DO NOT bump without verifying smoke test)

These are the exact versions that produced the locked baseline. The smoke
test guards against regression here.

| Component | Pin |
|---|---|
| IndexTTS-2 repo SHA | `830f6f8f94a51fea23ab1d639027a86200075a4e` |
| HuggingFace weights revision | `740dcaff396282ffb241903d150ac011cd4b1ede` |
| Python | 3.10.20 (via uv) |
| torch | 2.8.0 |
| transformers | 4.52.1 |

**Rebuild from scratch** if `vendor/index-tts/` is lost:
```bash
git clone https://github.com/index-tts/index-tts.git vendor/index-tts
cd vendor/index-tts && git checkout 830f6f8f && uv sync --no-dev
.venv/bin/huggingface-cli download IndexTeam/IndexTTS-2 \
    --revision 740dcaff396282ffb241903d150ac011cd4b1ede \
    --local-dir checkpoints
cd ../.. && vendor/index-tts/.venv/bin/python scripts/indextts_smoke_test.py
```

## How to generate audio

**Interactive (recommended for one-off phrases)**: open
[`notebooks/generate.ipynb`](notebooks/generate.ipynb), kernel
`vendor/index-tts/.venv/bin/python`. Type a string, hit run, hear the result.

**Batch (CLI)**:
```bash
vendor/index-tts/.venv/bin/python scripts/indextts_gen.py \
    --phrases-csv tts_output/cross_eval_50/eval_short.csv \
    --out-dir tts_output/my_run \
    --num-beams 5
```

**Build a new reference clip from a YouTube URL**:
```bash
.venv/bin/python scripts/build_podcast_ref.py \
    --url <youtube-url> --start 00:04:31 --duration 14
```

## Scoring (optional)

`scripts/posthoc_eval.py` consumes a `manifest.json` (written by
`indextts_gen.py`) and emits WER / ECAPA / DNSMOS:

```bash
.venv/bin/python scripts/posthoc_eval.py --score-only \
    --phrases-csv tts_output/cross_eval_50/eval_short.csv \
    --out-dir tts_output/my_run
```

## Regression protection

Run before AND after any change that touches IndexTTS-2 paths:

```bash
vendor/index-tts/.venv/bin/python scripts/indextts_smoke_test.py
```

Asserts `cas_01` and `sher_03` hit WER ≤ 0.10 and ECAPA ≥ 0.74. ~2 minutes.

Locked artifacts (never overwrite):
- `tts_output/eval_indextts_v2/` — baseline generated WAVs
- `tts_output/eval_indextts_v2/scores.regression_baseline.csv` — baseline scores
- `tts_output/refs/production/ref_interview.wav` — production reference clip
- `tts_output/refs/indextts_baseline/ref_narrator.wav` — smoke test + ECAPA ref

## Eval results (current state)

Per-clip `ECAPA(generated, real_clip)` — each eval CSV row points to a real BC
recording for the phrase.

### 15-phrase short eval (3-5 s phrases)

| Configuration | WER ↓ | ECAPA ↑ | DNSMOS OVR ↑ |
|---|---|---|---|
| IndexTTS-2 baseline (audiobook ref) | 0.037 | 0.784 | 2.90 |
| IndexTTS-2 + interview ref (current) | 0.051 | 0.310¹ | 2.73 |

### 8-phrase long eval (9-12 s phrases)

| Configuration | WER ↓ | ECAPA ↑ | DNSMOS OVR ↑ |
|---|---|---|---|
| IndexTTS-2 baseline (audiobook ref) | 0.013 | 0.846 | 3.48 |
| IndexTTS-2 + interview ref (current) | 0.028 | 0.322¹ | 3.13 |

¹ ECAPA crash on interview-ref is a target-mismatch artifact: eval-set
reference clips are audiobook BC, so the metric measures distance from
audiobook register. Listen-test confirmed identity is preserved and the
audiobook creak is removed. See [docs/history.md](docs/history.md) for full
context.

## Code standards

- **≤ 400 code lines per file** — if a script grows past this, split into a lib module.
- **Ruff clean** — `uv run ruff check scripts/ accent_coach/` must pass.
- **Tests green** — `uv run pytest tests/ -q` must pass (30 tests).
- **Smoke test** — run `vendor/index-tts/.venv/bin/python scripts/indextts_smoke_test.py`
  before and after any change touching IndexTTS-2 paths.
- Run all three with `bash scripts/check_repo.sh` before declaring a stage done.
- **No hardcoded values in logic.** Calibration constants, thresholds, decay
  values, file paths, model revisions, and accent targets live in dedicated
  config/reference modules (`accent_coach/reference/*.py`, env vars, or a settings
  file) — **never as magic numbers inline** in pipeline/scoring/generation code.
  Every constant carries a provenance comment (corpus + citation it came from).
  A literal threshold buried in a scorer is a future silent-miscalibration bug.
- **No silent compromises or fallbacks.** Code that cannot do its real job must
  **raise an explicit, explainable exception** — never degrade to a fabricated/
  neutral/uniform result and continue. Specifically banned: silent uniform/
  evenly-spaced substitution for a measured value; returning a neutral score
  (`50.0`, `65.0`, `0.0`, perfect `100`) when a measurement failed; `except:
  pass` or `except Exception: return <default>` that swallows a real failure.
  Exceptions must be **caught at a level that can handle them meaningfully**
  (e.g. drop the clip and record why), not blanket-swallowed. `None` is allowed
  ONLY when the contract is "no data → caller redistributes/skips" AND the caller
  actually does so. The litmus test: *if this path runs, would the output look
  valid while being meaningless?* If yes, raise instead.

## Accent coach quality rules

- **TDD**: write failing tests before implementing any feature. Tests must fail for the right reason (missing feature, not import error) before implementation begins.
- **Scoring sanity**: consonant and vowel quality scores must rank **native RP/GenAm speakers > TTS > owner**. A metric that grades the owner above natives is broken.
- **Comparison over absolutes**: prefer *relative* scoring — the user's value vs. the **same measurement on the BC/native target through the identical pipeline** — over absolute reference thresholds. Shared measurement error (G2P boundary drift, LPC/CoG bias) cancels in the difference; absolute reference tables cannot separate groups that share that error. Absolute references are a weak cross-check, never the primary signal. When a target clip exists, comparison mode is the default. Do not "fix" a broken absolute score by re-tuning its constants — move it to comparison mode. (See [docs/consonant_scoring_audit.md](docs/consonant_scoring_audit.md): widening fricative decay to land natives in a "believable band" inverted owner from lowest to highest because every group shared the same alignment artefact.)
- **Never tune thresholds — or tests — to a score *level***: calibrate decay/target constants from the native-corpus *distribution* and validate them by the **native−owner gap**, never by absolute level. It is forbidden to widen a tolerance so scores reach a target band, and equally forbidden to relax a test threshold to accommodate a tuned constant (tests pin behaviour; constants do not get to move the test). **Reject any constant change that shrinks the native−owner gap, even if it raises the absolute scores.** Every constant change must cite the measured native/owner distribution that justifies it.
- **Tests defend top-level invariants, one per sub-score**: for N sub-scores there must be N business-logic tests asserting **owner is strictly lowest** (and `native > TTS > owner`) on that sub-score, plus a composite-ordering test. A change that inverts any ordering must fail CI. Per-item synthetic tests are necessary but not sufficient — they pass in isolation and cannot catch a bench-level ordering inversion. Invariant tests run on the real corpus and `skip` (not pass) when the audio is absent.
- **Dual-accent coverage**: every scoring module must handle both modern RP (Fry/Lindsey norms) and General American (Hillenbrand/modern corpus norms) via `accent_target` parameter.
- **Reuse first**: before writing a new module, check if `accent_coach/pipeline/`, `accent_coach/comparison/`, or `accent_coach/reference/` already implements the needed primitive. Wrap or extend; do not duplicate.
- **No speed-quality trade-off**: prefer acoustic accuracy (parselmouth Burg LPC, proper bandpass filters) over cheap approximations. Compute time is not a constraint in scoring pipelines.
- **Review gate**: after completing a tests batch and after completing a feature implementation, spawn a code-review agent to find critical mistakes before committing.

## Lessons learned (failure post-mortems — read before touching scoring)

These are real failures from this repo. Each cost a wrong conclusion we almost shipped.

1. **Silent uniform alignment made the whole consonant bench meaningless.**
   `align_audio` requested WhisperX char timestamps but the parser looked in the
   wrong place (chars are stored at *segment* level, not per word). It silently
   fell back to a uniform `word_dur/n_phonemes` split, so every stop/fricative/
   liquid was measured on the wrong audio slice. Scores looked plausible; the
   owner beat native speakers. **Lesson:** a silent fallback to fabricated data
   is worse than a crash — it produces confident garbage. Verify a feature
   *actually engaged* on real data (we proved char timing was non-uniform on a
   real clip) before trusting its output. Now alignment raises `AlignmentError`
   instead of fabricating boundaries.

2. **Tuning a constant to a target score level inverted the ordering.**
   Fricative decay was solved as `median_delta / ln(100/70)` to land natives at
   ~70 — which widened tolerance until the owner's careful speech scored *highest*.
   **Lesson:** calibrate from the distribution and validate by the native−owner
   gap, never by absolute level; and never relax a test to fit a tuned constant.

3. **Neutral fallbacks (`return 50.0`/`65.0`) silently corrupted aggregates.**
   Unmeasurable /r/, /l/, or a consonant-less clip returned a confident neutral
   that got averaged into group means as if measured, defeating the
   None-redistribute design used elsewhere. **Lesson:** "couldn't measure" must
   propagate as `None`/skip or raise — never as a number that looks real.

4. **Absolute references reward register/material, not just accent.** Owner clips
   are careful citation-form drills; native corpora are conversational. Absolute
   CoG/F3 tables ranked the careful owner above natives. **Lesson:** control the
   confound (comparison mode on the same transcript), don't trust absolute tables
   across mismatched material.

5. **Diagnostics must report discrimination, not just levels.** The bench's
   level-only table hid an ordering inversion (everything rose into a "believable
   band"). **Lesson:** report the native−owner gap + an inversion verdict so a
   regression screams instead of looking like progress.

## Conventions

- Use `uv` for everything Python-package related. Never raw pip.
- Always use `--run-name` / `--out-dir` to separate experiment outputs.
- Do not commit: `.env`, `vendor/`, `tts_output/`, `data/`, `runs/`, `logs/`,
  any `.venv*/`.
- Generated WAVs and large intermediates live under `tts_output/`. The repo
  ships scoring CSVs only; raw audio is regenerated on demand.

## For spawned agents (Explore, Plan, general-purpose, etc.)

Sub-agents start cold and must respect the tooling already set up here:

- **Python execution**: use `.venv/bin/python` or `uv run python` for the
  scoring stack; use `vendor/index-tts/.venv/bin/python` for anything that
  touches IndexTTS-2. Never `python3` bare, never `pip install`.
- **uv only** for dependency changes (`uv add`, `uv sync`). Do not edit
  `pyproject.toml` then run `pip`.
- **RTK prefix every shell command** — `rtk git status`, `rtk grep ...`,
  `rtk ls ...`, `rtk pytest ...`. The hook rewrites them; RTK passes through
  if no filter exists, so it is always safe. Token savings are 60-90% on
  most ops — meaningful for sub-agents whose context fills fast.
- **CodeGraph is initialised** (`.codegraph/` exists). For symbol lookup,
  callers/callees, and impact analysis prefer `codegraph_search`,
  `codegraph_callers`, `codegraph_callees`, `codegraph_impact`,
  `codegraph_context`, `codegraph_node` over grep / find scans. Falls back
  to grep only when the symbol is not indexed (e.g. fresh code).
- **Do not touch** `vendor/`, `tts_output/eval_indextts_v2/`,
  `tts_output/refs/` — locked baseline artifacts. If a sub-agent is asked
  to "look around" it should treat these as read-only.

<!-- rtk-instructions v2 -->
# RTK (Rust Token Killer) - Token-Optimized Commands

## Golden Rule

**Always prefix commands with `rtk`**. If RTK has a dedicated filter, it uses it. If not, it passes through unchanged. This means RTK is always safe to use.

**Important**: Even in command chains with `&&`, use `rtk`:
```bash
# ❌ Wrong
git add . && git commit -m "msg" && git push

# ✅ Correct
rtk git add . && rtk git commit -m "msg" && rtk git push
```

## RTK Commands by Workflow

### Build & Compile (80-90% savings)
```bash
rtk cargo build         # Cargo build output
rtk cargo check         # Cargo check output
rtk cargo clippy        # Clippy warnings grouped by file (80%)
rtk tsc                 # TypeScript errors grouped by file/code (83%)
rtk lint                # ESLint/Biome violations grouped (84%)
rtk prettier --check    # Files needing format only (70%)
rtk next build          # Next.js build with route metrics (87%)
```

### Test (60-99% savings)
```bash
rtk cargo test          # Cargo test failures only (90%)
rtk go test             # Go test failures only (90%)
rtk jest                # Jest failures only (99.5%)
rtk vitest              # Vitest failures only (99.5%)
rtk playwright test     # Playwright failures only (94%)
rtk pytest              # Python test failures only (90%)
rtk rake test           # Ruby test failures only (90%)
rtk rspec               # RSpec test failures only (60%)
rtk test <cmd>          # Generic test wrapper - failures only
```

### Git (59-80% savings)
```bash
rtk git status          # Compact status
rtk git log             # Compact log (works with all git flags)
rtk git diff            # Compact diff (80%)
rtk git show            # Compact show (80%)
rtk git add             # Ultra-compact confirmations (59%)
rtk git commit          # Ultra-compact confirmations (59%)
rtk git push            # Ultra-compact confirmations
rtk git pull            # Ultra-compact confirmations
rtk git branch          # Compact branch list
rtk git fetch           # Compact fetch
rtk git stash           # Compact stash
rtk git worktree        # Compact worktree
```

Note: Git passthrough works for ALL subcommands, even those not explicitly listed.

### GitHub (26-87% savings)
```bash
rtk gh pr view <num>    # Compact PR view (87%)
rtk gh pr checks        # Compact PR checks (79%)
rtk gh run list         # Compact workflow runs (82%)
rtk gh issue list       # Compact issue list (80%)
rtk gh api              # Compact API responses (26%)
```

### JavaScript/TypeScript Tooling (70-90% savings)
```bash
rtk pnpm list           # Compact dependency tree (70%)
rtk pnpm outdated       # Compact outdated packages (80%)
rtk pnpm install        # Compact install output (90%)
rtk npm run <script>    # Compact npm script output
rtk npx <cmd>           # Compact npx command output
rtk prisma              # Prisma without ASCII art (88%)
```

### Files & Search (60-75% savings)
```bash
rtk ls <path>           # Tree format, compact (65%)
rtk read <file>         # Code reading with filtering (60%)
rtk grep <pattern>      # Search grouped by file (75%). Format flags (-c, -l, -L, -o, -Z) run raw.
rtk find <pattern>      # Find grouped by directory (70%)
```

### Analysis & Debug (70-90% savings)
```bash
rtk err <cmd>           # Filter errors only from any command
rtk log <file>          # Deduplicated logs with counts
rtk json <file>         # JSON structure without values
rtk deps                # Dependency overview
rtk env                 # Environment variables compact
rtk summary <cmd>       # Smart summary of command output
rtk diff                # Ultra-compact diffs
```

### Infrastructure (85% savings)
```bash
rtk docker ps           # Compact container list
rtk docker images       # Compact image list
rtk docker logs <c>     # Deduplicated logs
rtk kubectl get         # Compact resource list
rtk kubectl logs        # Deduplicated pod logs
```

### Network (65-70% savings)
```bash
rtk curl <url>          # Compact HTTP responses (70%)
rtk wget <url>          # Compact download output (65%)
```

### Meta Commands
```bash
rtk gain                # View token savings statistics
rtk gain --history      # View command history with savings
rtk discover            # Analyze Claude Code sessions for missed RTK usage
rtk proxy <cmd>         # Run command without filtering (for debugging)
rtk init                # Add RTK instructions to CLAUDE.md
rtk init --global       # Add RTK to ~/.claude/CLAUDE.md
```

## Token Savings Overview

| Category | Commands | Typical Savings |
|----------|----------|-----------------|
| Tests | vitest, playwright, cargo test | 90-99% |
| Build | next, tsc, lint, prettier | 70-87% |
| Git | status, log, diff, add, commit | 59-80% |
| GitHub | gh pr, gh run, gh issue | 26-87% |
| Package Managers | pnpm, npm, npx | 70-90% |
| Files | ls, read, grep, find | 60-75% |
| Infrastructure | docker, kubectl | 85% |
| Network | curl, wget | 65-70% |

Overall average: **60-90% token reduction** on common development operations.
<!-- /rtk-instructions -->