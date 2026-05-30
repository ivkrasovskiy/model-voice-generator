# Phase 0.16 Plan — Modern GenAm Norms + GA Cloning + Identity/Accent Disentanglement

**Date**: 2026-05-30
**Branch**: ivk-temp-branch
**Follows**: [phase0_15 results](accent_coach_phase0_15_results.md) (gate = 🟡 YELLOW, GenAm blocked)

---

## Goals (from owner)

1. Fix GenAm measurement → turn the reliability gate 🟢 GREEN.
2. Confirm IndexTTS-2 can **clone General American well enough**.
3. Answer the open question: **can we separate voice identity from accent**, or is
   the single-reference zero-shot the limit?

These unlock two user-facing features:
- **F1 — favourite voice, custom accent**: e.g. Benedict Cumberbatch's voice in GA.
- **F2 — own voice, target accent**: the learner hears themselves in RP or GA.

Two tracks. **Track A (measurement) is committed and cheap.** **Track B
(generation/disentanglement) is a spike with an explicit go/no-go** — it may hit
the model's limit, and the plan is designed to find that out fast.

---

## GA speaker sources (modern, male, clean)

| Speaker | Content | Role | Audio |
|---|---|---|---|
| Andrew Huberman | Huberman Lab solo intros | norms + cloning ref | studio-clean, single-speaker |
| Sam Harris | Making Sense solo monologues | norms + held-out validation | clean, single-speaker |
| Robert Sapolsky | Stanford *Human Behavioral Biology* lectures | norms + **diarization test** | lecture hall, audience Q/applause |
| Vsauce (Phase 0.15) | already built (76 clips) | held-out validation | studio-clean |

Source URLs go in `configs/accent_coach_phase0_16/genam_lecture_urls.json`, with
per-entry `start_s`/`end_s` windows to grab ~5–8 min of the speaker talking.

---

## Track A — Modern connected-speech GenAm norms (→ GREEN)

### A1. Build a clean GA corpus (handles lectures: questions, applause, noise)

The Huberman/Harris monologues are already single-speaker. Sapolsky lectures are
multi-speaker + noisy, so the builder must isolate the lecturer:

- **Speaker isolation**: reuse the ECAPA speaker-clustering from
  [accent_coach_build_real_bc.py](../scripts/accent_coach_build_real_bc.py) (built
  to diarise BC interviews). Embed every Whisper segment with ECAPA, cluster, keep
  the **dominant** cluster (the lecturer is the majority speaker). Drop minority
  clusters (audience questions).
- **Non-speech / applause reject**: gate each segment on spectral flatness +
  harmonic ratio; applause/laughter is high-flatness, low-harmonicity → drop.
  (The existing `_silent_fraction` gate stays.)
- **Overlap reject**: drop segments where the ECAPA embedding is far from the
  cluster centroid (cross-talk / over-applause speech).
- Emit `samples_to_verify/*.wav` for an owner listen-check before norms are built.

Driver: new `scripts/accent_coach_build_genam.py` (or extend `build_real_bc`'s
diariser into a shared lib). Extract formants with `--target genam`.

### A2. Rebuild GenAm norms from the corpus

- Compute per-vowel male centroids from the pooled GA corpus (Huberman + Harris +
  Sapolsky), same pipeline/measurement point as RP, so the two norm sets are
  commensurable. **Measure diphthongs at steady-state**, not Hillenbrand's onset
  nucleus — fixes the PRICE/GOAT mismatch.
- Write `RP`-style `GENAM_VOWEL_F1_F2_MALE_MODERN` in
  [genam_norms.py](../accent_coach/reference/genam_norms.py); keep Hillenbrand as
  `_LEGACY` (mirrors the Deterding→modern-RP supersession from Phase 0.5).
- This directly fixes the diagnosed defects: fronted GOOSE, diphthong convention,
  citation-vs-connected peripherality.

### A3. Broaden rhoticity (NURSE n=7 was too sparse)

- Extend the rhoticity measure from NURSE-only to **all r-coloured contexts**:
  START (ɑr), NORTH/FORCE (ɔr), NEAR/SQUARE, and lettER/commA (ɚ, r-coloured
  schwa). Target ≥ ~20 r-coloured tokens/speaker.
- Pool their F3 / Syrdal-Gopal Z3−Z2 into one rhoticity index in
  [coach_metrics.py](../accent_coach/diagnostics/coach_metrics.py).

### A4. Re-run the gate (held-out validation)

Derive norms from {Huberman, Harris, Sapolsky}, validate on **held-out Vsauce**
(and leave-one-out across the lecture speakers).

- 🟢 **GREEN**: held-out GA speaker scores GenAm-closer (CI-separated) and reads
  rhotic; all RP sources stay RP-closer + non-rhotic. → GenAm coach ships.
- 🟡 **YELLOW**: vowels separate but rhoticity still weak → keep RP coach, ship GA
  vowels-only feedback, flag rhoticity as provisional.
- 🔴 **RED**: GA still not separable from RP after modern norms → formants
  insufficient; escalate to consonant/prosody cues (`vot.py`, `prosody.py`).

**Track A compute**: corpus build ~1–2 h wall (download + WhisperX + diarise,
CPU-bound); norms + re-gate minutes. <4 GB RAM. No GPU.

---

## Track B — GA cloning + identity/accent disentanglement (spike)

### Hypothesis

Single-reference zero-shot **entangles identity and accent**: the accent of the
output is inherited from the reference speaker. Evidence so far: BC-ref → near-RP
output; owner-ref → owner's L2 accent. If true, F1/F2 need a way to source accent
**independently** of timbre.

### B1. Can IndexTTS-2 clone GA? (and is accent reference-carried?)

- Build a 10–15 s clean cloning ref for each GA speaker (like `ref_interview.wav`).
- Zero-shot generate the 15-phrase short eval from each GA ref; score vowels +
  rhoticity against the **new** GA norms, and ECAPA-identity against the GA speaker.
- 🟢 if cloned-GA reads GA (rhotic, GA vowels) with good ECAPA → cloning works AND
  confirms accent is reference-carried (entanglement). 🔴 if cloned-GA still reads
  RP → the model can't produce GA → cloning is the limiter.

### B2. Disentanglement — three approaches, cheapest-first

| # | Approach | Identity from | Accent from | Cost | Fixes |
|---|---|---|---|---|---|
| i | IndexTTS-2 native split — audit whether the **emotion/style reference** (used in Phase 0.13 Lever A) is separable from timbre, and whether it carries accent. Test: timbre=BC, style-ref=GA speaker. | timbre ref | style/emo ref? | low (in-repo) | F1+F2 if it works |
| ii | **DSP formant-shift (Lever B)** the target speaker's own TTS output toward target-accent centroids. | target ref | DSP centroids | low (in-repo) | vowels only; no rhoticity/prosody |
| iii | **Cascade TTS→VC**: generate in target accent with a native ref (right accent, wrong identity), then voice-convert to target identity preserving accent. | VC model | native TTS ref | high (out-of-repo VC model) | full, if VC preserves accent |

- Start with (i): audit `vendor/index-tts` inference API for separable
  timbre-vs-style inputs (Phase 0.13 already used emotion conditioning). Measure
  identity (ECAPA vs BC) **and** accent (vowels+rhoticity vs GA) on the output.
- Fall through to (ii) if (i) doesn't move accent — it's the realistic
  identity-preserving path on this hardware, already partly built (Lever B).
- (iii) only if F1/F2 demand full-accent fidelity and (i)/(ii) fall short; it
  pulls in an external VC model (scope expansion — needs a separate decision).

### B3. Verdict on the open question

Write up: does the current stack separate identity from accent, and to what
degree? Concretely score each feature:
- **F2 (own voice + target accent)**: best served by (ii) DSP today — preserves
  identity, shifts vowels; honest about the rhoticity/prosody gap.
- **F1 (favourite voice + custom accent)**: needs (i) to work or (iii) — report
  which.

- 🟢 if (i) or (ii) gives identity-preserved, accent-shifted output that the
  metric scores as moved toward target → ship the feature with documented limits.
- 🟡 if only partial (e.g. vowels move, rhoticity doesn't) → ship F2 as a
  vowel-coach aid, defer F1.
- 🔴 if nothing separates them → **single-reference is the limit**; document it and
  stop (no VC scope-creep without owner sign-off).

**Track B compute**: IndexTTS-2 generation is the cost — 45–90 s/clip CPU × 15
phrases × N refs/variants. Budget ~20 min/reference. B1 (3 GA refs) ~1 h; B2(i)/(ii)
a few more reference-runs. ~6–8 GB RAM. Kill stale Python between runs (18 GB).

---

## Sequencing & guardrails

1. Track A first (committed, cheap, unblocks the GenAm coach).
2. Then Track B as a spike, cheapest approach first, stop at the first 🔴.
3. Locked artifacts untouched (`tts_output/eval_indextts_v2/`, `refs/`, `vendor/`).
4. Run `bash scripts/check_repo.sh` before declaring any stage done; run the
   IndexTTS smoke test before/after any Track B change that touches IndexTTS paths.
5. LoRA stays frozen (Phase 0.14).

---

## Open decision for the owner

Track B approach (iii) (cascade TTS→VC) would give the fullest F1 ("BC voice in
GA") but requires adopting an **external voice-conversion model** — out of the
current repo's pinned stack. Approaches (i) and (ii) stay in-repo. Recommend:
exhaust (i)+(ii) first; only consider (iii) if a feature demands it and you
approve the new dependency.
