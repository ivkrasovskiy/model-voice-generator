# Accent Coach — Phase 0.9 Plan

> **Status**: ready to execute. Investigates whether synth_BC quality can be
> moved closer to real_BC and modern RP **without fine-tuning**, using only
> reference-clip selection and (deferred) generation-parameter tuning.
>
> **Builds on**:
> [`docs/accent_coach_phase0_8_findings.md`](accent_coach_phase0_8_findings.md),
> [`docs/accent_coach_phase0_8_plan.md`](accent_coach_phase0_8_plan.md).

---

## Where we are

Phase 0.8 picked **H4 (Bark)** as the working vowel-distance metric (passes
C1/C2/C4; C3 marginal at 1.65 vs 2.0; piecewise scores meet GREEN). The
piecewise scores across all seven speaker groups are:

| Speaker | H4 Bark piecewise score |
|---------|-------------------------|
| Fry | 100.0 |
| BBC male | 94.6 |
| Real BC | 89.8 |
| Lindsey | 86.6 |
| **Synth BC (IndexTTS-2)** | **74.6** |
| **Owner** | **56.5** |

The 15-point gap between **real BC (89.8)** and **synth BC (74.6)** is the
single biggest, most addressable product problem right now. Phase 0.8's
findings hypothesised two causes:

1. **Reference-clip register** — the production reference
   (`tts_output/ref_interview.wav`, 14 s) is an interview clip with reduced,
   fast vowels and explanatory rhythm. The model extracts both timbre **and**
   prosody from the reference, so it may be learning interview-register vowels
   that don't match real BC's audiobook-register vowels.
2. **Model's "generic English" centroid** — zero-shot TTS vowel quality is
   governed by the phoneme embeddings (corpus average), which the reference
   clip cannot fully override.

Phase 0.9 disentangles the two by running controlled reference-clip swaps and
a non-BC ceiling experiment. **Rhythm, intonation, aspiration scoring is
paused** — we do not yet have trustworthy scorers and Phase 0.7 confirmed
this; we will not pretend to measure them in Phase 0.9.

---

## Goals

1. Establish whether changing the BC reference clip can move synth_BC
   piecewise score (H4 Bark) toward real_BC (89.8).
2. Establish the **zero-shot vowel ceiling** for IndexTTS-2 by feeding it
   canonical-RP references (Fry, Lindsey). If even RP references can't
   produce RP-like output, no BC-side change will help.
3. Produce a clean experiment log so subsequent phases (generation-param
   sweep, possible fine-tuning) can build on a known baseline.

---

## Methodology — scoring is fixed to Phase 0.8's H4 Bark pipeline

For every variant `<ID>`:

1. **Generate** 15-phrase eval set using the variant's reference clip
   (and optionally non-default gen params).
   - Phrases CSV: `tts_output/cross_eval_50/eval_short.csv`
   - Output dir: `tts_output/accent_coach/phase0_9/variants/<ID>/clips/`
   - Uses `vendor/index-tts/.venv/bin/python scripts/indextts_gen.py`.
2. **Extract formants** from the generated clips into a per-variant
   centroids JSON (parallel to
   `tts_output/accent_coach/bench/phase0_7/speaker_centroids.json` but with
   `synth_BC` replaced by this variant's clips).
   - Reuse `scripts/accent_coach_extract_formants.py` (or whatever the
     Phase 0.7 manifest pipeline calls) — do **not** invent a new extractor.
3. **Score** using the Phase 0.8 H4 Bark pipeline:
   - Run a thin wrapper around `accent_coach/diagnostics/bark_distance.py`
     + `accent_coach/comparison/vowels.py::score_vowels_piecewise`.
   - Compute piecewise score for `real_BC`, `synth_BC_<ID>`, `owner` against
     the unchanged modern-RP centroid.
4. **Also compute** WER, ECAPA (vs real_clip per row), DNSMOS via
   `scripts/posthoc_eval.py --score-only` so we keep the v2 baseline
   reporting consistent.
5. **Log one row** per variant to `tts_output/accent_coach/phase0_9/experiment_log.csv`.

A small driver script `scripts/accent_coach_phase0_9_run.py` should orchestrate
steps 1-4 per variant and append to the log.

---

## Track A — BC reference clip alternatives

Same generation config (current defaults — whatever `scripts/indextts_gen.py`
runs today). Vary only the reference clip.

| ID | Reference | Source | Duration | Notes |
|----|-----------|--------|---------:|-------|
| A1 | `tts_output/ref_interview.wav` | YouTube `cHmkAStZBkc` 4:47-5:01 | 14 s | Baseline (current production) — must reproduce 74.6 |
| A2 | NEW: `tts_output/ref_interview_429_446.wav` | Same URL 4:29-4:46 | 17 s | Non-stop speech, owner-flagged candidate |
| A3 | NEW: `tts_output/ref_interview_1450_1510.wav` | Same URL 14:50-15:10 | 20 s | Non-stop speech, owner-flagged candidate |
| A4 | `tts_output/ref_sherlock.wav` | Sherlock dialogue | 7 s | Character speech, scripted, deliberate |
| A5 | `tts_output/ref_narrator.wav` | Casanova audiobook narrator | 11 s | Read register, clean vowels |
| A6 | `tts_output/ref_combined.wav` | Casanova 11 s + Sherlock 7 s stitched | 18 s | Multi-register reference |

**Building A2 / A3**: use `scripts/build_podcast_ref.py` (per
[CLAUDE.md](../CLAUDE.md#how-to-generate-audio)):

```bash
.venv/bin/python scripts/build_podcast_ref.py \
    --url https://youtu.be/cHmkAStZBkc \
    --start 00:04:29 --duration 17 \
    --out tts_output/ref_interview_429_446.wav

.venv/bin/python scripts/build_podcast_ref.py \
    --url https://youtu.be/cHmkAStZBkc \
    --start 00:14:50 --duration 20 \
    --out tts_output/ref_interview_1450_1510.wav
```

(If `build_podcast_ref.py` has different flag names, match its actual CLI.)

---

## Track B — Non-BC reference ceiling (diagnostic, not product)

Determines whether the model can reach RP-like output at all, given the best
possible reference. Same gen config as Track A.

| ID | Reference | Source | Target duration |
|----|-----------|--------|----------------:|
| B1 | NEW: `tts_output/ref_fry.wav` | Stitched from `tts_output/modern_rp_corpus/fry/clips/*.wav` | ~15-20 s |
| B2 | NEW: `tts_output/ref_lindsey.wav` | Stitched from `tts_output/modern_rp_corpus/lindsey/clips/*.wav` | ~15-20 s |

Selection rule for stitching: pick clips with the cleanest signal (high
voiced fraction, no overlapping speech). Concatenate with ~100 ms silence
between. Document chosen source clips in the experiment log.

**Output identity won't be BC** — this is a ceiling diagnostic, not a
product candidate.

---

## Track C — Generation parameter sweep (DEFERRED)

Run **after** Tracks A and B complete. Requires extending
`scripts/indextts_gen.py` to expose `--cfg-scale`, `--t-threshold`, and
`--selective-cfg` (these already exist in `scripts/lib/inference.py` and
`scripts/posthoc_eval.py`; plumbing is mechanical).

Initial sweep (3 × 4 = 12 variants, run on **best ref from Tracks A+B**):

| Param | Values |
|-------|--------|
| selective_cfg + t_threshold | off; 0.08 (memory-best for ECAPA+WER); 0.15 |
| cfg_scale | 1.0, 1.5, 2.0, 3.0 |

**Caveat**: param tuning may re-rank A and B winners. If Track C lands GREEN
on the chosen reference, re-run the top-2 alternates from A and B with the
winning params before declaring a final reference choice.

---

## Logging spec

`tts_output/accent_coach/phase0_9/experiment_log.csv`:

| Column | Type | Notes |
|--------|------|-------|
| `experiment_id` | string | e.g. `A1`, `A2`, `B1`, `C03` |
| `track` | string | `A`, `B`, or `C` |
| `ref_clip` | path | Absolute path to the reference WAV |
| `ref_duration_s` | float | Reference clip length |
| `gen_params_json` | string | JSON: cfg_scale, t_threshold, selective_cfg, etc. (empty for defaults) |
| `synth_BC_piecewise_H4Bark` | float | The primary metric |
| `per_phoneme_csv` | path | Per-variant per-phoneme Bark distances |
| `WER` | float | Mean WER across 15-phrase eval |
| `ECAPA` | float | Mean cos sim vs real_clip per row |
| `DNSMOS_OVR` | float | Mean DNSMOS overall |
| `gen_time_min` | float | Generation wall-clock |
| `notes` | string | Any caveats |

Per-variant directory layout:

```
tts_output/accent_coach/phase0_9/variants/<ID>/
├── clips/                 # 15 generated WAVs + manifest.json
├── centroids.json         # speaker_centroids.json with this variant's synth_BC
├── bark_scores.json       # piecewise scores under H4 Bark
├── per_phoneme.csv        # per-phoneme Bark distances
└── posthoc_scores.csv     # WER / ECAPA / DNSMOS rows
```

---

## Stop criteria

| Outcome | Trigger | Action |
|---------|---------|--------|
| **GREEN** | Any A/B variant reaches synth_BC piecewise ≥ **82** (within Lindsey range; within ~5 pts of real BC) | Declare zero-shot ceiling acceptable. Write findings. Move to Phase 0.10 (owner-side coaching). |
| **YELLOW** | Best A/B variant scores 78–82 | Document the improvement, run Track C, hand decision to owner. |
| **RED** | Best A/B variant scores < 78 | Zero-shot floor reached. Run Track C as a last attempt; if still < 78, schedule fine-tuning experiment as Phase 0.10. |
| **B-floor RED** | B1 AND B2 score < 80 | Model's vowel floor is below RP cluster regardless of reference. No BC-side change can close the gap. Skip Track C for ref selection (still run for ECAPA/WER tuning); recommend fine-tuning. |

A1 must reproduce the Phase 0.8 score of **74.6 ± 2** as a pipeline-validity
check before any other variant's score is trusted. If A1 deviates by more
than 2 points, stop and diagnose the pipeline before proceeding.

---

## Out of scope (explicit — do not work on these in Phase 0.9)

- **Rhythm, intonation, aspiration scoring** — paused per owner direction;
  Phase 0.7 already showed these scorers are not trustworthy.
- **C3 ratio improvement** — Phase 0.8 accepted H4 Bark at C3=1.65; do not
  re-litigate.
- **Owner piecewise score** — Phase 0.9 is about synth_BC quality only.
- **Fine-tuning** — explicitly the next-phase fallback, not a Phase 0.9 lever.
- **H5 F0-binned norms** — Phase 0.8 Option C; not pursued here.
- **Touching locked artifacts**: `tts_output/eval_indextts_v2/`,
  `tts_output/ref_interview.wav` (A1 reads it but does not overwrite),
  `vendor/`.

---

## Execution order (suggested)

1. **A1 baseline reproduction** — validates the pipeline before anything else.
2. **A2, A3** — owner-flagged interview cuts (requires
   `build_podcast_ref.py` first).
3. **A4, A5, A6** — existing reference clips, no new audio needed.
4. **B1, B2** — build Fry/Lindsey stitched refs, then generate.
5. **Findings write-up** — Markdown summary tabulating all variants by H4
   Bark piecewise score, ECAPA, WER, DNSMOS.
6. (Deferred) **Track C** — only if A/B outcome is YELLOW or RED.

A1-A6 + B1-B2 = **8 generation runs**. At ~45-90 s per clip on M3 Pro × 15
phrases = ~12-22 min per variant. Total wall-clock ~1.5-3 hours of gen,
plus scoring.

---

## Caveats to record in findings

- **Reference duration is a co-variate** in Track A (7 s to 20 s). Not a
  fully controlled experiment — duration may independently affect output.
  Document the duration column in the log and discuss in findings.
- **Identity drift** in Track B is expected and not a failure — outputs
  there will sound like Fry/Lindsey, not BC. Useful only as a vowel ceiling.
- **A6 (combined ref)** mixes two registers (narrator + character). May help
  or hurt; document either way.

---

## Files to produce

| Path | Description |
|------|-------------|
| `tts_output/ref_interview_429_446.wav` | A2 reference (NEW) |
| `tts_output/ref_interview_1450_1510.wav` | A3 reference (NEW) |
| `tts_output/ref_fry.wav` | B1 reference (NEW) |
| `tts_output/ref_lindsey.wav` | B2 reference (NEW) |
| `scripts/accent_coach_phase0_9_run.py` | Driver: per-variant gen + score + log |
| `tts_output/accent_coach/phase0_9/experiment_log.csv` | Master log |
| `tts_output/accent_coach/phase0_9/variants/<ID>/...` | Per-variant artifacts |
| `docs/accent_coach_phase0_9_findings.md` | Final write-up |

---

## Hand-off note for the executing agent

This plan was scoped after Phase 0.8's YELLOW verdict, where the owner chose
to pivot from "which metric is right" (academic) to "fix the synth_BC gap"
(product). Keep that framing: every variant's score must be reported in H4
Bark piecewise terms so it is directly comparable to Phase 0.8's table.

If A1 fails to reproduce 74.6 ± 2, **stop and surface the discrepancy** —
do not silently proceed with a drifted pipeline.

If you discover that `scripts/indextts_gen.py` or the formant-extraction path
has changed since Phase 0.8, note it in the findings and freeze the version
for the duration of Phase 0.9 to keep variants comparable.
