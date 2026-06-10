# Consonant & Prosody Scoring — consolidated reference

Single source of truth for the consonant/prosody scoring pipeline: what's done,
what's open, the design principles, and the measurement references that the code
points at. Merges the former `consonant_scoring_audit.md` and `vot_bug_diagnosis.md`;
the blow-by-blow session history is summarised in [accent_coach_history.md](accent_coach_history.md).

## Status at a glance

| area | state |
|---|---|
| Silent fallbacks | **DONE** — removed everywhere (alignment, liquids, aggregator, prosody, compare); code fails explicitly instead of fabricating neutral/uniform values. |
| Audio comparability | **DONE** — one canonical `normalize_audio` (mono + 16 kHz + common bandwidth cap at 7.6 kHz) on the bench AND product path; vowels (formants ≤ ~4.5 kHz) unaffected. |
| Fricatives | **DONE** — bandwidth confound fixed; reference re-derived **in-domain** from native speakers; owner no longer scores above natives. |
| Rhotic /r/ | **DONE** — strong discriminator (native−owner gap ≈ +34). |
| Lateral /l/ | works; single-token variance (backlog). |
| VOT / aspiration | **FUNCTIONAL, not production-grade, gap status UNVERIFIED** — pitch-detection latency bias fixed (18 ms → ~5 ms error on synthetic). The "closure-gate fix restores discrimination (+8/+12.5/+8 for p/t/k)" claim from 2026-06-09 did **not** replicate in a 2026-06-10 re-run at the same gate (-18.1/+13.9/-3.5) — /p/ and /k/ sign-flip between runs at n=20-24, suggesting the gap is dominated by sampling noise at this n. A further `_MIN_CLOSURE_GATE_MS` 20→2ms change (2026-06-10) is logically sound and regression-free but, per independent review, affects only 2/15 real RP tokens and 0/9 owner /t/ tokens — it does not address the dominant zero-producing mechanism. See VOT section "2026-06-10" subsection for full doubts/assumptions. |
| /θ ð/ detection | **DONE** — phoneme-aware frication gate: /θ ð f v/ use HF>2 kHz at 0.04 threshold (was HF>3 kHz/0.20 for all fricatives). Real voiced /ð/ (strong F0 carrier, HF>3 kHz ratio ~0.05) now passes the gate. Batch yield: 12/12 diverse tokens scored (was ~0% for voiced dentals; unvoiced /θ/ already passed the old gate). CoG reference values (θ: 4500, ð: 4400) were derived under the old gate and may be biased; see potential_improvements for re-measurement. |

## Open items (priority order)

1. **Up-weight absent-in-target-accent markers** (/r/, dental /θ ð/, aspiration) vs
   phonemes shared across accents. Equal-per-phoneme weighting dilutes the few strong
   discriminators (/r/ +34, VOT gaps) among many tied phonemes — see Per-phoneme findings.
2. **In-domain VOT reference re-measurement** — `RP_VOT_MEAN_MS`/`SD` (and GenAm) are Lisker &
   Abramson (1964) literature values (67.5-87.5 ms); in-domain measured VOT through this
   pipeline is 0-22 ms even for natives, so stop scores compress to ~0-15/100 for everyone
   (gap ~+1 to +4/100). Re-measure mean/SD per phoneme from native corpora through this
   pipeline (mirroring `measure_fricative_cog.py`) — same fix pattern as the fricative
   bandwidth confound. Validate by native−owner gap, not absolute level.
3. **VOT real-speech absolute accuracy → AutoVOT / Dr.VOT** (trained models; refs below). Even
   after the closure-gate fix, native VOT measures 9.5-16 ms vs the 50-100 ms textbook range.
4. **Lateral single-token confidence weighting** — flag/down-weight sub-scores from < ~3 tokens.
5. **VOT extractor: `e_max`-windowing + `_onset_correction_s` clamp (new, 2026-06-10)** —
   independent review identified these as the likely DOMINANT zero-producing mechanisms,
   unaffected by the gate change: (a) `e_max = rms.max()` over the full 300 ms post-window
   lands 100-200 ms into the FOLLOWING VOWEL on real tokens, making the burst threshold
   vowel-dominated rather than burst-dominated; (b) `voice_t_local = max(burst_t_local,
   first_voiced - 1/_PITCH_FLOOR_HZ)` clamps VOT to 0 whenever `first_voiced` is within
   ~13.3 ms of the burst — confirmed on all 9 owner /t/ tokens. Before touching either,
   re-run the native−owner gap at larger n (≥40-50/group) to establish whether the gap is
   even stable enough to validate against — see subsection below.

*(Item removed: /θ ð/ gate bug — fixed 2026-06-09. Phoneme-aware gate; HF>2 kHz / 0.04
threshold for /θ ð f v/. CoG reference re-measurement deferred to potential_improvements.)*

*(Item removed: VOT collapse-to-0 on real speech + /k/ score inversion — fixed 2026-06-09/10.
Pitch-detection latency bias (+18ms) and burst-mislocation (`_MIN_CLOSURE_GATE_MS=20`) both
fixed; native−owner raw-VOT gap now +8/+12.5/+8 ms for /p t k/, was 0/0/0. See VOT section.)*

**CAVEAT (2026-06-10, see VOT section subsection below):** the "+8/+12.5/+8, was 0/0/0" claim
above did not replicate in a same-gate re-run (-18.1/+13.9/-3.5). The "was 0/0/0" framing is
also now suspect — re-opened informally pending item #2/#5 work; not re-added to the numbered
Open items list to avoid re-litigating, but should not be cited as settled.

## Design principles (these calibrate the whole pipeline — also CLAUDE.md rules)

- **Comparison over absolutes; calibrate from the native distribution.** Absolute reference
  tables cannot separate groups that share a measurement artefact; compare against the same
  measurement on the native/BC target through the identical pipeline.
- **Never tune a constant — or a test — to a score *level*.** Validate by the native−owner
  *gap*, never the absolute level. Canonical failure: widening the fricative CoG decay to land
  natives in a "believable band" inverted the owner from lowest to highest (every group shared
  the same alignment artefact). Reject any constant change that shrinks the gap.
- **No silent fallbacks.** "Couldn't measure" propagates as `None`/skip or raises — never a
  fabricated neutral (50/65), uniform split, or swallowed exception. The archetype: a silent
  uniform G2P split once made the whole consonant bench meaningless while looking plausible.

## Fricatives — bandwidth confound + in-domain reference

**Root cause of the early inversion (owner scored *above* natives):** an audio-bandwidth
confound, not articulation. Native corpora are 16 kHz at source (8 kHz Nyquist — /s/ frication
above 8 kHz is gone); the owner is 44.1 kHz studio. Spectral CoG therefore measured *recording
bandwidth*: Jongman's /s/ = 7000 Hz is a 22 kHz-band figure, uncapturable in our 16 kHz band, so
natives scored "too low" while the wider-band owner scored high.

**Fix (three parts):**
1. **Canonical normalization** (`accent_coach/pipeline/audio_io.py`) caps every source to a
   common bandwidth so no clip can win on bandwidth alone.
2. **Frication gate** — reject fricative windows that mis-aligned onto a vowel/closure
   (HF<3 kHz energy ratio), so they don't pollute the mean. (KNOWN BUG: too aggressive for
   *weak* fricatives — rejects ~100% of dental /θ ð/ for ALL groups, verified by an
   expected→aligned→measured trace, so /θ ð/ score nothing. Needs a phoneme-aware
   threshold; see Open items #1.)
3. **In-domain reference** — measured through THIS pipeline on native speakers
   (`scripts/tools/measure_fricative_cog.py`; RP = Fry/Lindsey/BBC/real-BC,
   GA = Huberman/Harris/Sapolsky/vsauce), pooled (CoG is place-driven, accent-neutral):

   | ph | n | CoG (Hz) | | ph | n | CoG (Hz) |
   |---|---|---|---|---|---|---|
   | s | 455 | 5200 | | f | 23 | 4700 |
   | z | 331 | 5300 | | v | 36 | 4850 |
   | ʃ | 41 | 3970 | | θ | (sparse) | 4500* |
   | ʒ | (sparse) | 3480 | | ð | (sparse) | 4400* |

   *θ ð measured under old gate (HF>3 kHz/0.20 → only HF-rich tokens kept) → CoG biased
   high; re-measure with corrected phoneme-aware gate (HF>2 kHz/0.04). See potential_improvements.

**Outcome:** natives lifted ~40 → ~77; composite ranks owner lowest. Fricative CoG then ties
across groups (~77–80) — which is *correct*: Russian /s z ʃ f/ ≈ English, so fricative CoG is
not an accent discriminator for this L1.

## VOT / aspiration — what was broken, what's fixed, path to production

**The old extractor collapsed to ≈0 ms for everyone** (greedy two-threshold heuristic): the
voicing loop read the *preceding* vowel's periodicity as the stop's onset; the burst threshold's
median included the following vowel; results were locked to a 5 ms grid with a hard 0 floor.

**Rewrite** (`accent_coach/pipeline/vot.py`, Lisker & Abramson via parselmouth, no new deps):
closure baseline → broadband burst transient searched *after* the closure minimum →
first F0-constrained voiced run (pitch 75–400 Hz) *strictly after* the burst. Returns `None`
(not a fabricated 0) when unmeasurable. TDD on hand-checkable synthetic ground truth
(`tests/accent_coach/test_vot.py`).

**Plus context filtering** (`filter_aspirating_stops`): English aspirates /p t k/ only when
stressed + prevocalic + NOT post-/s/. Measuring VOT in clusters (/str/, /pl/) or unreleased
finals dilutes the accent signal with tokens that are short for everyone. After filtering, every
*real* native scores above the owner on aspiration.

**2026-06-09 strict-test pass + closure-gate fix.** 25 ms test tolerance hid two real bugs:

1. **Pitch-detection latency bias (+18 ms on synthetic).** Praat's AC pitch analysis window is
   ≈`1/pitch_floor` (≈13 ms); the first frame reported as voiced is centred ~18 ms after the true
   onset. Fix: subtract `1/_PITCH_FLOOR_HZ` from the detected onset (clamped to not precede the
   burst). Synthetic error dropped from ~18 ms → ~5 ms across 15 cases (50–120 ms VOT, varying
   preceding-vowel/closure length). 17 new tests added (`tests/accent_coach/test_vot.py`,
   22 total): 15 parametrized aspiration cases + an aggregate ≥75%-within-12ms detection-rate
   test + a short-lag (20–30 ms) false-positive guard.

2. **Real-speech burst mislocation (collapsed everyone to 0 ms).** The burst search started
   immediately after the closure-energy minimum. In real speech that minimum often lands in the
   *preceding vowel's* context (not the actual closure silence), so the search fired on a noise
   transient only 8–24 ms later — then voicing followed almost immediately → VOT≈0 for
   **every** group, erasing all discrimination. Fix: `_MIN_CLOSURE_GATE_MS = 20` — the burst
   search now starts 20 ms after the closure minimum, skipping that artefact window.

**Real-speech result (n=24 clips/group, p/t/k pooled, native = rp_fry+rp_lindsey+real_bc+genam):**

| phoneme | native_pooled (n) | owner (n) | gap |
|---|---|---|---|
| /p/ | 9.8 ms (16) | 1.7 ms (8) | **+8.0** |
| /t/ | 16.0 ms (41) | 3.5 ms (13) | **+12.5** |
| /k/ | 9.5 ms (48) | 1.6 ms (8) | **+7.9** |

Before the closure-gate fix, all of the above were **0.0 ms for every group** (no discrimination
at all). Now native > owner on all three stops — correct direction, satisfying the
native > owner sanity rule. tts_bc (IndexTTS-2 BC clone) is mixed: /t/=0.5 ms (owner-like),
/k/=21.9 ms (above native pooled, n=5, one 106.7 ms outlier in "composure"); /p/ had zero
detections (n=0) in this sample.

**Current limit:** absolute values (1.6–22 ms) are still far below the textbook 50–100 ms
English aspirated range — directionally correct but not production-accurate. Sample sizes are
small (n=3-18 per group/phoneme) and noisy. Closure-gate at 20 ms is a heuristic, not derived
from a corpus distribution; AutoVOT remains the path to absolute accuracy.

**Stop SCORES still compressed near 0 (separate from the raw-VOT fix above).**
`scripts/tools/consonant_per_phoneme.py --n 20` after the closure-gate fix
(`tts_output/accent_coach/consonant_per_phoneme.csv`):

| ph | owner | rp_fry | rp_lindsey | real_bc | genam | tts_bc | nat_pool | gap |
|---|---|---|---|---|---|---|---|---|
| p | 1 (5) | 1 (3) | 1 (2) | 3 (7) | 2 (10) | — (0) | 2 | +1 |
| t | 0 (12) | 6 (14) | 15 (6) | 2 (12) | 1 (22) | 0 (5) | 4 | +4 |
| k | 0 (7) | 4 (10) | 0 (5) | 1 (16) | 0 (14) | 6 (5) | 1 | **+1 (was −1.3, INVERTED, before this fix)** |

`/k/` ordering inversion (owner > natives) is now fixed. But all scores sit in 0-15/100 because
`RP_VOT_MEAN_MS = {p:67.5, t:77.5, k:87.5}` (Lisker & Abramson 1964 literature, ~10 ms SD) is
compared against in-domain measured VOT of 0-22 ms — the same absolute-reference mismatch
pattern as the pre-fix fricative CoG bandwidth confound (see Fricatives section). Re-deriving
`RP_VOT_MEAN_MS`/`SD` and the GenAm equivalent **in-domain from the native corpora through this
pipeline** (mirroring `measure_fricative_cog.py`) would likely turn the current ~0 native−owner
score gap into something like the raw-VOT gap (+8/+12.5/+8 ms ⇒ large z-score separation once
the reference mean matches the measurable range). Deferred — see Open items.

## VOT — 2026-06-10: gate 20→2ms micro-fix, critical review, open doubts

**Change made**: `_MIN_CLOSURE_GATE_MS` 20 → 2 ms (`accent_coach/pipeline/vot.py`). Rationale:
the 2 ms gate is the logical minimum (1 frame) — `thresh > e_closure = rms[closure_idx]` by
construction, so the closure-minimum frame itself can never satisfy the burst threshold; no
gate smaller than 1 frame is possible, and nothing smaller than 1 frame is needed. New test
`test_short_closure_vot_not_collapsed_to_zero` (3 cases, `tests/accent_coach/test_vot.py`)
went RED→GREEN on a synthetic closure_ms×vot_ms sweep. Full suite after the change: 240
passed, 3 failed (pre-existing item #2 `test_in_domain_vot_gap_meaningful`), 10 skipped, ruff
clean (one pre-existing F841 in `test_consonants_quality.py:786`, predates this change,
verified via `git stash`).

**What this change does NOT do, per independent review (n=15 real RP tokens + all 9 owner
/t/ tokens, instrumented trace)**:
- The original diagnosis claimed `burst_idx == closure_idx + gate_frames` in 6/6 traced
  tokens (i.e. the gate itself was eating the burst). At n=15 this held for only **0/15**
  tokens at gate=2 and **5/15** at gate=20. The 6/6 trace was likely a non-representative
  sample (it sorted clips by duration descending and took the first 6, not a random sample).
- Changing gate 20→2 altered only **2/15** real RP tokens, both small (+3.64 ms, +4.67 ms).
- **Owner /t/ (9/9 tokens) is completely unaffected** — bit-identical VOT before/after. This
  is why the owner row in the re-measurement below didn't move; it is NOT a caching bug.

**DOUBT — the native−owner gap itself may not be stable at n=20-24.** Two measurements at the
SAME gate (20 ms) gave very different gaps:

| run | n/group | gap p | gap t | gap k |
|---|---|---|---|---|
| 2026-06-09 (status table, prior session) | 24 | **+8.0** | +12.5 | **+8.0** |
| 2026-06-10 (this session, re-run, gate=20) | 20 | **-18.1** | +13.9 | **-3.5** |
| 2026-06-10 (this session, gate=2) | 20 | -14.1 | +13.6 | -3.1 |

/t/ is consistent (+12.5 to +13.9) across all three. /p/ and /k/ **sign-flip** between the two
gate=20 runs (+8.0→-18.1 and +8.0→-3.5) — a swing far larger than the gate=2-vs-gate=20 delta
(-18.1→-14.1, -3.5→-3.1). This means: (1) the 2026-06-09 "+8/+12.5/+8, was 0/0/0" result for
/p/ and /k/ should be treated as **not reproduced**, not as a settled baseline; (2) at this
sample size the /p/ and /k/ gap sign is not trustworthy either way — neither "inverted" nor
"correct" should be asserted with confidence until n is much larger (item #5).

**Suspected real dominant mechanisms (untouched by this session, see Open item #5)**:
1. `e_max = rms.max()` over the full 300 ms post-window lands 100-200 ms into the following
   vowel for real tokens → `thresh = e_closure + 0.15*(e_max - e_closure)` is set by vowel
   energy, not burst energy — could push `burst_idx` early or late depending on the vowel's
   relative loudness.
2. `voice_t_local = max(burst_t_local, first_voiced - 1/_PITCH_FLOOR_HZ)` — the
   `1/_PITCH_FLOOR_HZ ≈ 13.3 ms` subtraction, when `first_voiced` is within ~13.3 ms of
   `burst_t_local`, clamps `voice_t_local == burst_t_local` → `vot_ms == 0.0` exactly. This
   reproduced on **all 9** owner /t/ tokens (matches `OWNER t n=9 mean=0.0 std=0.0 %=0:100%`
   in the table below) and is independent of `_MIN_CLOSURE_GATE_MS`.

**Decision for now**: keep `_MIN_CLOSURE_GATE_MS = 2.0` (no regression, simpler/more-correct
constant than the old fixed 20 ms heuristic) but do NOT cite it as having "restored
discrimination" — that framing in the 2026-06-09 entries above is now disputed. Item #2
(reference re-derivation) remains blocked on the same zero-heavy distribution as before; item
#5 (e_max windowing + onset-correction clamp + larger-n gap stability check) is the
recommended next investigation, NOT yet started.

**Re-measurement after gate=2 (`scripts/tools/measure_vot_reference.py --n 20`, seed=42)**:

```
RP pooled:    p n=14 mean=9.8  median=0.0 std=15.3 %=0:57% %>20:21%
              t n=40 mean=18.3 median=0.3 std=39.7 %=0:50% %>20:18%
              k n=48 mean=9.7  median=0.0 std=26.5 %=0:75% %>20:12%
GENAM pooled: p n=24 mean=7.5  median=0.0 %=0:62%
              t n=38 mean=8.7  median=0.0 %=0:66%
              k n=34 mean=8.4  median=0.0 %=0:71%
ALL NATIVES:  p n=38 mean=8.4  median=0.0 | t n=78 mean=13.6 median=0.0 | k n=82 mean=9.1 median=0.0
OWNER:        p n=6  mean=22.4 median=0.0 std=50.2 %=0:83%
              t n=9  mean=0.0  median=0.0 std=0.0  %=0:100%
              k n=6  mean=12.2 median=0.0 std=17.6 %=0:67%
GAP (mean):   p=-14.1 (inverted), t=+13.6 (correct), k=-3.1 (inverted)
```

**Path to production-grade VOT — trained models:**
- AutoVOT: https://github.com/MLSpeech/AutoVOT — Sonderegger & Keshet (2012) *JASA* 132(6):3965-3979
  (https://pubs.aip.org/asa/jasa/article-abstract/132/6/3965/915621)
- DeepVOT / Dr.VOT: https://github.com/adiyoss/DeepVOT
- Lisker & Abramson (1964) definition: https://www.haskinslaboratories.org/vot ;
  "VOT at 50" review (measurement pitfalls): https://pmc.ncbi.nlm.nih.gov/articles/PMC5665574/

## Per-phoneme discrimination (Phase 1)

`scripts/tools/consonant_per_phoneme.py` (CSV: `tts_output/accent_coach/consonant_per_phoneme.csv`).
Gap = native_pooled − owner:

```
TIE  (Russian-shared, frequent → dominate the class): s +2, z -3, ʃ -3, f -3, l -0
DISCRIMINATE:  r +34 (trill vs approximant),  p +9 (aspiration)
GATE-FIXED:    θ, ð — phoneme-aware gate (HF>2 kHz / 0.04) now measures them; CoG
               references still conservative (old gate biased the sample high → re-measure)
               v — labiodental, typically passes even old gate; low discrimination expected
```

Composites (both rank owner lowest): class-weighted owner 36.8 vs natives 40–56;
equal-per-phoneme owner 44.9 vs natives 51–57. **Equal-weighting gives a *smaller* gap** —
it dilutes the few discriminators (/r/, /p/) among many tied Russian-shared phonemes. So the
fix is to fix the /θ ð/ gate (Open items #1) and up-weight the absent-in-Russian markers, not
to flatten the weights. The tool keeps both composites for comparison.
