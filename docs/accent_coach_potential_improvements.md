# Accent Coach — Potential Improvements (non-critical)

Backlog of quality/robustness ideas that are **not** correctness-critical. Critical
fallback/measurement bugs are tracked in
[prosody_consonant_upgrade.md](prosody_consonant_upgrade.md); this file is the
"nice to have / revisit" list. Date: 2026-06-07.

## Scoring reliability

1. **Lateral /l/ — small-sample noise (NOT a detection bug).** Verified by an
   expected→aligned→measured trace: every spoken /l/ IS measured (15=15=15 for the
   owner; no gate loss, unlike /θ ð/). The problem is *count*: the short calibration
   sentences are /l/-light — mean ≈ 0.9–1.1 /l/ per clip, and ~6/16 owner clips have
   ZERO /l/ — so the group score rests on only ~15 tokens. The bench amplifies this by
   averaging per-CLIP means (a 1-/l/ clip is weighted the same as a 3-/l/ clip), which
   is what makes the group lateral score swing between random samples. Fixes (none
   involve tuning to noise): **pool /l/ tokens across clips** instead of averaging
   clip-means; use more/longer sentences; attach a token-count confidence and flag
   sub-scores from < ~3 tokens. Bonus cue: owner /l/ median 167 ms vs native 49 ms
   (3.4×) — a likely vocalized/syllabic or Russian-dark /l/, currently unused.

2. **Rhotic RP token scarcity.** The pre-vocalic /r/ gate (correct for
   non-rhotic RP) leaves only ~4–6 /r/ tokens per RP clip vs ~12–15 for GA, so
   RP rhotic means are higher-variance. Same confidence/flagging idea as (1);
   also log per-group rhotic token counts in the bench.

3. **#5b — prosody nPVI≈0 rate-estimate fallback.** `extract_syllable_durations_acoustic`
   still returns a uniform rate estimate (and `compute_npvi` returns 0.0) when
   < 3 acoustic nuclei are found. This is conservative (reads as "too even", not
   inflated) but is still a fabricated value. Cleaner: make "rhythm not
   measurable" explicit — `RhythmBreakdown.score = None` and redistribute — so
   nPVI is never computed on synthesized durations. Deferred because it needs a
   rhythm-pipeline refactor + ~4 test updates and has a legitimate
   false-positive-avoidance history.

## Bench / methodology

4. **Comparison mode as the bench default** once paired BC audio exists for each
   group's transcripts (see the audit doc and the "BC reference audio" note).
   Absolute references cannot rank groups that differ in register/material;
   comparison mode (`--target-manifest`) is the structural fix.

5. **Per-group observability in the bench.** Record and print per clip: which
   aligner ran, comparison-vs-absolute mode, and how many tokens of each
   sub-class were scored / dropped — so a silent degradation shows in the table.
   (Partial: stop-extraction errors are now surfaced.)

6. **Register-matched eval set.** The native corpora are conversational/lecture
   while owner/bc_cal are careful citation sentences. A controlled eval where
   natives, owner, and BC all read the SAME fixed sentence set would remove the
   register confound from the bench entirely.

## Skill-specific

7. **Intonation** currently uses absolute RP pitch templates in no-target mode —
   a weak signal. Prefer comparison mode (DTW vs the BC rendition of the same
   sentence) when a target exists.

8. **Aspiration / VOT double-counting.** `aggregator.py` notes stop VOT is
   counted both in the consonant composite (`stops` weight) and as the separate
   `aspiration` skill in `scoring.py`. Resolve once VOT is functional (see the
   VOT bug investigation) so the same signal is not weighted twice.

9. **VOT RMS normalisation.** If/when the VOT extractor is fixed, RMS-normalise
   the filtered signal before the burst threshold so detection is gain-invariant
   (run the same clip at two gains → same VOT).
