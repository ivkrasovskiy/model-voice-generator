# VOT (Voice Onset Time) — critical bug diagnosis

Symptom: VOT ≈ 0–3 ms for **every** speaker group on the consonant bench
(should be ~50–125 ms for aspirated English /p t k/). Persists with char-aligned
boundaries, so it is an **extractor-algorithm** problem, not only alignment.
File: [accent_coach/pipeline/vot.py](../accent_coach/pipeline/vot.py).
Findings below were both code-traced and reproduced numerically on synthetic
/pa/, /ka/ and connected-speech tokens. Date: 2026-06-07.

## Critical bugs

| line | bug | why VOT → ≈0 |
|---|---|---|
| `vot.py:67` | voicing loop `range(burst_frame, …)` starts **inclusive** and takes the first frame > threshold | if any periodicity at the burst frame, `voice_frame == burst_frame` → VOT = 0 |
| `vot.py:43-47,67-72` | 20 ms `_PRE_MS` + `start_time` jitter pulls the **tail of the preceding voiced sound** into the segment (English /p t k/ are mostly post-vocalic) | residual voicing read as the stop's voicing onset at frame 0 → VOT ≈ 0 (reproduced: ac@burst=0.58) |
| `vot.py:62 vs 72` | VOT reduces to `(i − burst_frame)·5 ms` — locked to the 5 ms **hop grid** with a hard 0 floor | no sub-frame resolution; same-frame detection = exactly 0 |
| `vot.py:53-58` | burst = first frame HF energy `> 3×median`, median taken over a **100 ms window containing the loud vowel** | threshold crossed at the vowel, not the burst → burst≈voicing → VOT≈0 (or None for weak bursts) |
| `vot.py:71` | voicing gate `autocorr > 0.35` far **too low** (modal voicing ~0.6–0.85) | formant ringing / residual voicing cross 0.35 early → onset pulled toward burst |
| `vot.py:38` | `min_lag` ≈ 1 ms → searches up to ~1000 Hz | formant-band periodicity gives a spurious "voicing" peak (should constrain to F0 75–400 Hz → lags ~40–213 samples @16 kHz) |

## Gaps vs validated methods (AutoVOT / Dr.VOT / Praat-Lisker&Abramson)
- **Closure-first anchoring** missing — no notion of "burst after silence"; should locate the low-energy closure, baseline the threshold from it (not a vowel-straddling median), then find the burst as the energy rise out of closure.
- **Burst via broadband spectral transient/derivative**, not a single 2–7.5 kHz energy-vs-median rule that fires on the vowel.
- **Voicing onset** via F0-constrained periodicity (parselmouth `To Pitch (ac)`, floor 75 / ceiling 400) strictly **after** the burst, persisting ≥20 ms; periodicity-alone at a low threshold is explicitly cautioned against in the literature.
- **No negative-VOT (prevoicing)** handling — code clamps `<0 → None`; voiced /b d g/ and many L2 productions have voicing before the burst.
- Greedy first-crossing (vs AutoVOT's joint optimisation) is what yields the degenerate same-frame solution.

## Recommended fix
Rewrite around a validated method rather than patching the greedy heuristic (the fixes interact):
- **Option A — adopt AutoVOT/Dr.VOT.** Export stop windows (we already have MMS forced alignment) as Praat TextGrids → run AutoVOT (field standard, validated vs human annotators).
- **Option B — correct parselmouth/Praat pipeline** (we already depend on parselmouth): closure → broadband-energy-derivative burst (sub-frame) → F0-constrained voicing onset strictly after burst, signed for prevoicing. `vot_ms = (voice_sample − burst_sample)/sr·1000`.

Two non-negotiables: **(a)** anchor the burst to the closure→release transient, not an HF median that includes the vowel; **(b)** detect voicing strictly after the burst with an F0-constrained periodicity test at a high threshold, never reading the preceding phone's voicing. Validate against a few hand-measured tokens; expect native > TTS > owner with aspirated /p t k/ in 50–125 ms.

## Sources
- AutoVOT: https://github.com/MLSpeech/AutoVOT — Sonderegger & Keshet (2012) JASA 132(6):3965-3979 https://pubs.aip.org/asa/jasa/article-abstract/132/6/3965/915621 (preprint http://people.linguistics.mcgill.ca/~morgan/interspeechVot.pdf)
- DeepVOT / Dr.VOT: https://github.com/adiyoss/DeepVOT — https://www.isca-archive.org/interspeech_2016/adi16_interspeech.html
- Lisker & Abramson (1964), *Word* 20:384-422 — https://www.haskinslaboratories.org/vot
- "VOT at 50" review (measurement pitfalls): https://pmc.ncbi.nlm.nih.gov/articles/PMC5665574/
- Reassignment-spectra burst onset: https://www.sciencedirect.com/science/article/abs/pii/S0167639309000892
- Praat VOT tutorial: https://pubs.aip.org/asa/jasa/article/147/2/852/994857
