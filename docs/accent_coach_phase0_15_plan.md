# Phase 0.15 — Coach-Grade Measurement (RP + GenAm, measure-only)

**Date**: 2026-05-29
**Branch**: ivk-temp-branch
**Supersedes the LoRA direction of** [phase0_14](accent_coach_lora_eval_phase0_14.md)

---

## Decision (from Phase 0.14 analysis)

Two requirements were confirmed with the owner:

1. **Coach output = measure + feedback only.** No audio generation in the loop.
   The coach scores a learner's vowels/rhoticity against a target accent and
   says *which phonemes to fix and in which direction*. This removes all TTS
   generation (45–90 s/clip) and the accent-conversion problem entirely.
2. **Targets = Modern RP *and* General American, both now.**

This means the deliverable is a pure **analysis pipeline**:
`learner audio → align → per-token formants (+F3) → distance to RP & GenAm → feedback`.

### Why this reframes everything

Phase 0.14 proved the *generation* side is done and over-engineered: IndexTTS-2
zero-shot already matches ground-truth RP (Fry) within measurement noise, and
LoRA makes it worse. **LoRA is frozen.** The remaining gap is not the model —
it is the **measurement**, which is not yet coach-grade:

- The headline metric collapses every token of a vowel into one mean centroid,
  hiding the learner's actual error (category *overlap*, e.g. BATH↔TRAP).
- No confidence intervals — the Phase 0.14 ranking is statistically
  indistinguishable across ranks 1–4 (gen_base 0.420 vs Fry 0.402 is noise).
- No rhoticity axis. F1/F2 centroids **cannot separate RP from GenAm** — the two
  are near-identical in vowel space; the separator is rhoticity (F3 lowering).
- The GenAm path was actively wrong: the pipeline strips coda-R and relabels
  BATH→ɑː, which destroys exactly the features that define GenAm.

## Key facts established by code inspection (2026-05-29)

- Per-token formant CSVs **already contain F3** (`formants_*.csv` schema:
  `clip_id, …, F1, F2, F3, …`). Fry NURSE F3 = 2459 Hz, gen_base 2412 Hz —
  both high ⇒ non-rhotic ⇒ correctly read as RP. **Rhoticity needs no
  re-extraction** for the RP corpora and generated audio.
- **Owner CSV has zero F3** (extracted before F3 was added). Owner rhoticity
  requires one cheap re-extraction (formant extraction is ~realtime).
- `genam_norms.py` (Hillenbrand 1995, male+female) and `rp_norms.py`
  (modern RP male) norm tables already exist — **scoring a learner needs no
  corpus**, only these published tables. Corpora are for *validation* only.
- CMU (used by WhisperX) **is American**, so the correct GenAm correction is
  *no correction* — the RP corrections exist precisely to undo CMU's Americanness.

---

## Plan

### Phase A — Freeze & reframe  ✅ (this doc)
LoRA frozen; `ckpt/best/` kept as artifact. Coach = measure-only.

### Phase B — Target-aware corrections
Add `apply_accent_corrections(phonemes, target)` dispatcher in
`accent_coach/pipeline/rp_postprocess.py`:
- `target="rp"`   → strip_coda_r + relabel_bath + relabel_lot (current behavior)
- `target="genam"`→ identity (CMU is already GenAm-appropriate; keep rhotic R, keep æ for BATH)
- `target="none"` → identity (raw measurement, e.g. owner)

Wire a `target` param through `scripts/accent_coach_extract_formants.py`
(`apply_rp_corrections=True` kept for backward compat).

### Phase C — Coach-grade metric
New module `accent_coach/diagnostics/coach_metrics.py`, operating on
**per-token** rows (not centroids):
- `bootstrap_ci` — kills sub-noise comparisons.
- `dispersion` — within-category σ (Bark); L2 vowels are diffuse.
- `bhattacharyya_overlap` — minimal-pair confusion (the real learner error).
- `rhoticity` — NURSE F3 + Syrdal-Gopal Z3−Z2; the RP-vs-GenAm separator.
- `score_source(tokens, target, mean_f0)` — assembles all of the above vs the
  chosen norm table.

New scorer `scripts/accent_coach_coach_eval.py` — scores every available source
against **both** RP and GenAm, prints a report, dumps JSON.

### Phase D — Reliability gate (next, needs small GenAm validation set)
- 🟢 GREEN: Fry scores RP-near & GenAm-far (high NURSE F3); a GenAm speaker the
  reverse; BATH↔TRAP overlap separates RP speakers from owner. → score owner, ship feedback.
- 🟡 YELLOW: vowels separate but rhoticity noisy on TTS-grade audio → tune F3 extraction.
- 🔴 RED: RP & GenAm still indistinguishable after F3 → formants insufficient;
  bring in consonant/prosody cues (`vot.py`, `prosody.py`).

GenAm validation set: ~30 clips via yt-dlp + Whisper (~30–45 min wall).
Owner re-extraction with F3 also happens here.

### Phase E — Owner feedback
Score owner vs both targets, rank fixes. Phase 0.14 predicts RP fixes:
GOAT, BATH, PRICE, TRAP. GenAm will reorder (no BATH split ⇒ BATH stops being an error).

---

## Compute (M3 Pro, 18 GB)
Phases B, C, E: minutes, <2 GB. Phase D GenAm set: ~30–45 min, Whisper-bound.
Zero GPU-hours, zero TTS generation. (Kill orphaned Python between runs — 18 GB.)
