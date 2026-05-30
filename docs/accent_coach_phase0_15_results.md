# Phase 0.15 Results — Coach-Grade Measurement + Reliability Gate

**Date**: 2026-05-30
**Branch**: ivk-temp-branch
**Plan**: [accent_coach_phase0_15_plan.md](accent_coach_phase0_15_plan.md)

---

## Verdict: 🟡 YELLOW — RP measurement is reliable; GenAm is not yet

The reliability gate (native sources must score their own accent with CI
separation **and** the matching rhoticity verdict) returned **5/6**:

| Source | want | got | sep | |
|---|---|---|---|---|
| fry | RP / non_rhotic | RP / non_rhotic | y | ✅ |
| lindsey | RP / non_rhotic | RP / non_rhotic | y | ✅ |
| bbc | RP / non_rhotic | RP / non_rhotic | y | ✅ |
| gen_base | RP / non_rhotic | RP / non_rhotic | y | ✅ |
| real_bc | RP / non_rhotic | RP / non_rhotic | y | ✅ |
| **genam** | **GenAm / rhotic** | **RP / non_rhotic** | n | ❌ |

**The RP coach is ready.** All five RP-class sources score RP-closer with
non-overlapping CIs and read non-rhotic. **The GenAm coach is not** — and the
gate caught it before any GA feedback shipped. That is the gate working.

---

## What was built this phase

- **Coach-grade metric** ([coach_metrics.py](../accent_coach/diagnostics/coach_metrics.py)):
  per-token bootstrap CIs, within-category dispersion, Bhattacharyya minimal-pair
  overlap, F3 rhoticity. 26 unit tests.
- **Target-aware corrections**: `apply_accent_corrections(target=rp|genam|none)`
  — GenAm is identity (CMU is already American).
- **Owner re-extracted with F3** (was missing): NURSE F3 = **1888 Hz → rhotic**.
- **GenAm validation corpus**: Vsauce (Michael Stevens), 76 single-speaker clips,
  801 vowel tokens, extracted with `--target genam`. (A second source, Matt
  D'Avella, was **rejected** — multi-speaker, female voices, ECAPA min_pair −0.03.)
- **Dual-target scorer + gate** ([accent_coach_coach_eval.py](../scripts/accent_coach_coach_eval.py)).

---

## Root cause of the GenAm failure (diagnosed, not guessed)

### 1. Hillenbrand 1995 GenAm norms are citation-form + 30 years stale

Per-vowel, the GA speaker is scored against RP norms (our own connected-speech
corpus) vs GenAm norms (Hillenbrand 1995 /hVd/ citation words):

| Vowel | GA speaker (F1,F2) | RP norm | Hillenbrand GenAm | closer |
|---|---|---|---|---|
| æ TRAP | (678, 1846) | (545,1496) | (588,1952) | **GenAm** ✓ |
| ɑː BATH | (689, 1327) | (518,1215) | (768,1333) | **GenAm** ✓ |
| ɛ DRESS | (531, 1792) | (462,1571) | (580,1799) | **GenAm** ✓ |
| iː FLEECE | (371, 2215) | (348,1962) | (342,2322) | **GenAm** ✓ |
| uː GOOSE | (429, **1545**) | (352,1506) | (378, **997**) | RP ✗ |
| aɪ PRICE | (593, 1834) | (482,1599) | (727,1184) | RP ✗ |
| əʊ GOAT | (459, 1229) | (389,1353) | (497, 910) | RP ✗ |

The metric **correctly identifies GenAm** on the robust monophthong markers
(TRAP, BATH, DRESS, FLEECE — exactly the GA-vs-RP diagnostics). It loses only
where Hillenbrand is **wrong for modern connected GA**:

- **GOOSE-fronting**: Hillenbrand has back uː (F2=997); modern GA is fronted
  (F2≈1500), like our speaker (1545). The 30-year-old table misses a sound change.
- **Diphthong convention mismatch**: Hillenbrand tabulates PRICE/GOAT at the
  *onset nucleus*; our pipeline measures *steady-state*. Not comparable.
- **Citation vs connected**: /hVd/ lab vowels are more peripheral than running
  speech, so connected GA sits closer to our connected-speech RP norms.

This is **the same defect Phase 0.5 already fixed for RP** (rejected Deterding
1997 citation norms for a modern connected-speech corpus). GenAm needs the
identical treatment.

### 2. NURSE-only rhoticity is undersampled

NURSE is rare: GA n=7, owner n=10. The GA estimate (2499 Hz) has a 95% CI of
1987–2896 — it spans rhotic *and* non-rhotic, so the point classification is
noise. By contrast the RP sources (n=24–56) are tight and reliable. Rhoticity
works when n is adequate; NURSE alone is too sparse.

---

## Fix path → Phase 0.16 (well-scoped, cheap)

1. **Modern connected-speech GenAm norms** — derive monophthong + steady-state
   diphthong centroids from a GenAm corpus, replacing Hillenbrand (mirror the
   Phase 0.5 RP rebuild). Use the same measurement pipeline as RP so the two
   norm sets are commensurable.
2. **Held-out GA validation** — derive norms from one GA speaker, validate on a
   *different* clean GA male (avoid the Vsauce-on-Vsauce circularity). Needs one
   more clean single-speaker GA source (~30 min build).
3. **Broaden rhoticity** beyond NURSE to all r-colored contexts (START, NORTH,
   lettER/ɚ) to get ≥~20 tokens per speaker.

GREEN criterion unchanged: held-out GA speaker scores GenAm-closer (CI-separated)
and reads rhotic; RP speakers the reverse.

---

## Owner (learner) — usable RP feedback already

RP measurement is reliable, so owner→RP coaching is actionable now. Worst RP
vowels (rank, Bark + dispersion):

| Vowel | dist | dispersion | note |
|---|---|---|---|
| GOAT (əʊ) | 2.67 | 2.21 | worst; also most diffuse |
| THOUGHT (ɔː) | 2.11 | 2.25 | |
| BATH (ɑː) | 2.01 | 1.85 | BATH↔TRAP overlap 0.79 (L2 merger) |
| GOOSE (uː) | 1.82 | 2.08 | |

Plus a rhoticity flag: owner NURSE F3 = 1888 Hz (rhotic) vs RP ~2450
(non-rhotic) — owner uses an American r-colored /ɝ/. **Caveat**: owner NURSE
n=10, so treat the rhotic flag as provisional until rhoticity is broadened (fix 3).

---

## Files

| Path | Contents |
|---|---|
| `accent_coach/diagnostics/coach_metrics.py` | per-token metric (CIs, dispersion, overlap, rhoticity) |
| `scripts/accent_coach_coach_eval.py` | dual-target scorer + gate verdict |
| `configs/accent_coach_phase0_15/genam_urls.json` | GA source URLs |
| `tts_output/genam_corpus/formants_genam.csv` | Vsauce GA formants (target=genam) |
| `tts_output/owner_cal_50/formants.csv` | owner formants, now **with F3** |
| `tts_output/accent_coach/phase0_15/coach_eval.json` | full scored results |
