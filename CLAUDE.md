# CLAUDE.md

Guidance for Claude Code when working in this repo.

## What this project does

Generate Benedict Cumberbatch's voice from text. Current production model is
**IndexTTS-2 zero-shot** with a 14 s interview reference clip
(`tts_output/ref_interview.wav`, sourced from
[youtu.be/cHmkAStZBkc](https://youtu.be/cHmkAStZBkc) at 4:47-5:01). No
fine-tuning involved.

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
[docs/accent_coach_plan.md](docs/accent_coach_plan.md).

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

Lint: `uv run ruff check scripts/` must be zero errors before commits.

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
    --out-dir tts_output/my_run
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
- `tts_output/ref_interview.wav` — production reference clip

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

## Conventions

- Use `uv` for everything Python-package related. Never raw pip.
- Always use `--run-name` / `--out-dir` to separate experiment outputs.
- Do not commit: `.env`, `vendor/`, `tts_output/`, `data/`, `runs/`, `logs/`,
  any `.venv*/`.
- Generated WAVs and large intermediates live under `tts_output/`. The repo
  ships scoring CSVs only; raw audio is regenerated on demand.
