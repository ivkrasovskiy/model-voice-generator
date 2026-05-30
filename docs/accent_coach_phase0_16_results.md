# Phase 0.16 Results — Modern GenAm Norms (GREEN) + Disentanglement Verdict

**Date**: 2026-05-30
**Branch**: ivk-temp-branch
**Plan**: [accent_coach_phase0_16_plan.md](accent_coach_phase0_16_plan.md)
**Resolves**: [phase0_15](accent_coach_phase0_15_results.md) YELLOW (GenAm blocked)

---

## Headline

- **Track A — GenAm measurement: 🟢 GREEN.** Modern connected-speech GenAm norms
  generalise to held-out GA speakers (leave-one-out 3/3 GenAm-closer + rhotic).
- **Track B — disentanglement verdict: identity and accent are ENTANGLED in
  IndexTTS-2's single speaker reference.** The model can clone GA, but its only
  other knob (emo prompt) does **not** carry accent. The one in-repo lever that
  separates identity from (vowel-)accent is **DSP formant-shifting**.

---

## Track A — Modern GenAm norms → GREEN

### What was wrong (Phase 0.15) and how it was fixed

Hillenbrand-1995 GenAm norms were citation-form + 30 years stale. Rebuilt from a
**266-clip connected-speech corpus** of 3 modern male GA speakers — Andrew
Huberman (dopamine monologue), Sam Harris (AMA), Robert Sapolsky (Stanford
lecture) — using our own pipeline, so the GenAm and RP norms are now
commensurable. Built by `scripts/accent_coach_build_genam.py` (unsupervised ECAPA
dominant-speaker isolation — drops audience questions/applause, no HF token).

Diagnosed defects, fixed:

| Vowel | Hillenbrand (old) | Modern corpus (new) | Fix |
|---|---|---|---|
| GOOSE uː | (378, **997**) | (316, **1301**) | captures modern GA fronting |
| PRICE aɪ | (727, 1184) onset | (545, 1550) steady | matches RP measurement convention |
| GOAT əʊ | (497, 910) onset | (429, 1095) steady | same |

### Reliability gate — leave-one-out (non-circular)

Norms built from 2 of the 3 lecture speakers; the **held-out** speaker scored
against them (`scripts/accent_coach_genam_loo.py`):

| Held-out | vs RP | vs GenAm | closer | rhoticity | |
|---|---|---|---|---|---|
| harris | 1.648 | **1.461** | GenAm | rhotic 1790 | ✅ |
| huberman | 1.844 | **1.795** | GenAm | rhotic 1918 | ✅ |
| sapolsky | 1.763 | **1.662** | GenAm | rhotic 1830 | ✅ |

**3/3 → 🟢 GREEN.** RP speakers stay RP-closer + non-rhotic (2400–2575). Vowel-space
CIs between RP and GenAm overlap — which is linguistically correct: the accents
differ mainly in **rhoticity**, which is now the load-bearing discriminator
(broadened beyond the too-sparse NURSE to all pre-/r/ contexts via a new
`next_phoneme` column).

### Data-quality note (owner-reported)
The Sapolsky sample opens with audience laughter. It does **not** pollute the
norms: extraction is alignment-based (laughter yields no vowel tokens), the three
speakers' centroids agree (max pairwise Δ ≈ 1 Bark), and pooling uses **median**
(outlier-robust). The only anomaly was Huberman's ɑː (F1=380), which the median
absorbed.

---

## Track B — GA cloning + identity/accent disentanglement

15-phrase short eval, num_beams=5. Accent measured vs RP/GenAm + rhoticity;
identity = mean ECAPA cosine to the BC and Huberman references.

| Run | RP dist | GenAm dist | rhoticity | id→BC | id→Huberman |
|---|---|---|---|---|---|
| gen_base (BC, anchor) | ~1.57 | ~1.75 | non_rhotic | high | — |
| **B1** clone, spk=Huberman | 1.625 | **1.582** | **rhotic 1893** | 0.029 | **0.839** |
| **B2i** spk=BC + emo=Huberman | **1.858** | 1.876 | non_rhotic 2345 | **0.763** | 0.192 |
| **B2ii** DSP own-voice→RP | — | — | — | **0.976 (self)** | — |

### B1 — IndexTTS-2 can clone GA ✅
A GA reference produces GenAm-closer, **rhotic** output (F3 1893) with Huberman
identity (0.839, BC 0.029). Cloning works — and this proves **accent is carried by
the speaker reference** (`spk_audio_prompt`).

### B2i — the emo prompt does NOT carry accent ❌
spk=BC + emo=Huberman keeps **BC identity (0.763 vs Huberman 0.192)** but the
accent **stays RP/non-rhotic** (2345). IndexTTS-2's emotion conditioning carries
prosody/affect, not pronunciation. (Emo even slightly *worsened* RP fidelity,
1.858 vs 1.57 — it's not a useful accent lever.)

### B2ii — DSP formant-shift separates identity from vowel-accent ✅ (vowels only)
Warping the owner's worst vowels (GOAT/BATH/GOOSE) toward RP centroids via the
WORLD vocoder (`accent_coach/dsp/formant_shift.py`) preserves identity almost
perfectly (**ECAPA 0.976**) and moves vowels toward RP (**+0.57 Bark**). Limit:
shifts F1/F2 of targeted vowels only — **not rhoticity, not prosody**.

---

## Verdict on the open question

**Can we divide voice identity from accent?** With the current stack:

- **In IndexTTS-2 zero-shot: no.** The single speaker reference fixes *both*
  identity and accent; the separate emo input does not move accent. So
  "favourite voice + arbitrary accent" is **not** achievable from the model alone.
  The owner's hunch — *"maybe the single reference is the limit"* — is **confirmed**.
- **Via DSP: partially, and well enough for coaching.** Formant-shifting separates
  identity from *vowel*-accent (id 0.976, vowels move). It does not touch
  rhoticity or prosody.

### The two features

| Feature | Verdict | Path | Limits |
|---|---|---|---|
| **F2 — own voice, target accent** | 🟢 viable now | DSP formant-shift (B2ii) | vowels only; no rhoticity/prosody. Enough for an accent-coach "hear the fix" aid. |
| **F1 — favourite voice, custom accent** (BC in GA) | 🟡 partial | DSP on the cloned voice (vowel-only) | zero-shot can't; full fidelity (rhoticity+prosody) needs an external voice-conversion model |

### Recommendation on the deferred "external VC" decision
Approach (iii) (cascade TTS→VC) is the **only** path to high-fidelity F1, but it
adds an out-of-repo model to the pinned stack. Since the accent-**coach** product
(measure-only, decided Phase 0.15) needs neither F1 nor F2, and F2 is already met
by in-repo DSP, **do not adopt an external VC model unless high-fidelity F1
becomes a product priority.** That remains the owner's call.

---

## Files

| Path | Contents |
|---|---|
| `scripts/accent_coach_build_genam.py` | GA lecture corpus builder (ECAPA dominant-speaker isolation) |
| `scripts/accent_coach_build_genam_norms.py` | derive modern GenAm centroids |
| `scripts/accent_coach_genam_loo.py` | leave-one-out GREEN gate |
| `scripts/accent_coach_build_genam_refs.py` | 14 s GA cloning refs |
| `scripts/accent_coach_phase0_16_score_b.py` | Track B accent + identity scorer |
| `scripts/accent_coach_phase0_16_dsp.py` | B2ii DSP own-voice→accent demo |
| `accent_coach/reference/genam_norms.py` | `_GENAM_MALE_MODERN` (Hillenbrand kept as legacy) |
| `accent_coach/diagnostics/coach_metrics.py` | broadened rhoticity (pre-/r/ contexts) |
| `tts_output/genam_lecture_corpus/` | 266-clip GA corpus + formants |
| `tts_output/accent_coach/phase0_16/` | clone / disentangle / DSP outputs |
