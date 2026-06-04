from __future__ import annotations

import math
from itertools import groupby

import numpy as np

from accent_coach.models import PhonemeInstance, RhythmBreakdown, SentenceAnalysis
from accent_coach.pipeline.prosody import compute_npvi
from accent_coach.reference.rp_norms import RP_NPVI_MAX, RP_NPVI_MIN

_NPVI_REF = (RP_NPVI_MIN + RP_NPVI_MAX) / 2  # 51.0  (was 65 with old Grabe & Low norms)
_DECAY = 30.0  # nPVI units for e^{-1} decay (was 20; gentler so native p25/p75 score ~75)

FUNCTION_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "to", "of", "in", "on", "at", "by", "for",
    "and", "but", "or", "so", "yet", "nor",
    "he", "she", "it", "we", "they", "him", "her", "us", "them",
    "his", "its", "our", "their",
    "was", "were", "is", "are", "am", "be", "been", "being",
    "do", "does", "did", "have", "has", "had",
    "will", "would", "could", "should", "might", "may", "can", "shall", "must",
    "that", "which", "who", "whom", "this", "there", "here",
})


def _syllable_pattern_score(
    user_durs: list[float],
    target_durs: list[float],
) -> float:
    """Pearson r of mean-normalised duration vectors → 0–100.

    Normalising by each speaker's own mean removes speech-rate differences —
    only the rhythmic pattern (which syllables are long vs short) is compared.

    Returns 50.0 (neutral) when counts differ by > 30%: truncating to the
    shorter vector would correlate different phonological positions (e.g. if
    the user's detector misses a function word at the start, every subsequent
    index is off by one).
    """
    n = min(len(user_durs), len(target_durs))
    if n < 3:
        return 50.0
    longer = max(len(user_durs), len(target_durs))
    if (longer - n) / longer > 0.30:
        return 50.0
    u = np.array(user_durs[:n], dtype=float)
    t = np.array(target_durs[:n], dtype=float)
    u_mean, t_mean = u.mean(), t.mean()
    if u_mean > 0:
        u = u / u_mean
    if t_mean > 0:
        t = t / t_mean
    if np.std(u) < 1e-9 or np.std(t) < 1e-9:
        return 50.0
    corr = float(np.corrcoef(u, t)[0, 1])
    return float(np.clip((corr + 1) / 2 * 100, 0.0, 100.0))


def _word_mean_durations(phonemes: list[PhonemeInstance]) -> dict[str, float]:
    """Return word → mean duration in ms, averaged across all occurrences."""
    durations: dict[str, list[float]] = {}
    for word_key, group in groupby(phonemes, key=lambda p: p.word.lower().strip(".,!?;:")):
        phs = list(group)
        dur_ms = (phs[-1].end_time - phs[0].start_time) * 1000
        durations.setdefault(word_key, []).append(dur_ms)
    return {w: float(np.mean(durs)) for w, durs in durations.items() if w}


def _per_syllable_outliers(
    user_durs: list[float],
    target_durs: list[float],
    threshold: float = 0.5,
) -> list[tuple[int, str]]:
    """Return (position_0based, 'long'|'short') for syllables that deviate > threshold.

    Both duration vectors are normalised by their own mean before comparing, so
    only the rhythmic shape (not the absolute speaking rate) is considered.
    Returns at most the 4 worst outliers sorted by magnitude descending.
    """
    n = min(len(user_durs), len(target_durs))
    if n < 3:
        return []
    u = np.array(user_durs[:n], dtype=float)
    t = np.array(target_durs[:n], dtype=float)
    u_mean, t_mean = u.mean(), t.mean()
    if u_mean <= 0 or t_mean <= 0:
        return []
    diff = u / u_mean - t / t_mean
    outliers = [
        (i, "long" if diff[i] > 0 else "short")
        for i in range(n) if abs(diff[i]) > threshold
    ]
    outliers.sort(key=lambda x: abs(diff[x[0]]), reverse=True)
    return outliers[:4]


def _function_word_analysis(
    user_phonemes: list[PhonemeInstance],
    target_phonemes: list[PhonemeInstance],
) -> tuple[float | None, list[str]]:
    """Returns (score 0-100 or None, list of inflated word strings).

    Returns (None, []) when no function words are matched — caller redistributes
    the FW weight to nPVI+pattern rather than applying a fictional perfect score.
    Inflated = user duration > 1.7× target duration for that function word.
    """
    user_durs = _word_mean_durations(user_phonemes)
    target_durs = _word_mean_durations(target_phonemes)

    inflated: list[str] = []
    matched = 0
    for word, u_ms in user_durs.items():
        if word not in FUNCTION_WORDS:
            continue
        t_ms = target_durs.get(word)
        if t_ms is None or t_ms <= 0:
            continue
        matched += 1
        if u_ms / t_ms > 1.7:
            inflated.append(word)

    if matched == 0:
        return None, []
    score = 100.0 * (1.0 - len(inflated) / matched)
    return score, inflated


def score_rhythm(
    user: SentenceAnalysis,
    target: SentenceAnalysis | None = None,
) -> RhythmBreakdown:
    user_npvi = compute_npvi(user.syllable_durations)

    if target is not None:
        ref_npvi = compute_npvi(target.syllable_durations)
        ref_min = ref_npvi - 10.0
        ref_max = ref_npvi + 10.0
    else:
        ref_npvi = _NPVI_REF
        ref_min = RP_NPVI_MIN
        ref_max = RP_NPVI_MAX

    delta = abs(user_npvi - ref_npvi)
    npvi_score = 100.0 * math.exp(-delta / _DECAY)

    # Pattern correlation — only meaningful with a target
    pattern_correlation: float | None = None
    pattern_score = 50.0
    if target is not None:
        pattern_correlation = _syllable_pattern_score(
            user.syllable_durations, target.syllable_durations
        )
        pattern_score = pattern_correlation

    # Function word inflation — requires phoneme data in both
    function_word_score: float | None = None
    inflated: list[str] = []
    if target is not None and user.phonemes and target.phonemes:
        function_word_score, inflated = _function_word_analysis(
            user.phonemes, target.phonemes
        )

    # Composite weights.
    # With target + phoneme data: 3-signal blend (40/40/20).
    # With target but no phoneme data: FW analysis unavailable — redistribute its
    #   weight to avoid inflating the composite with a fictional perfect score.
    # Without target: nPVI alone (absolute mode).
    if target is not None:
        if function_word_score is not None:
            composite = 0.4 * npvi_score + 0.4 * pattern_score + 0.2 * function_word_score
        else:
            composite = 0.5 * npvi_score + 0.5 * pattern_score
    else:
        composite = npvi_score

    # Per-syllable outliers — only when target is available
    outlier_syllables: list[tuple[int, str]] = []
    if target is not None:
        outlier_syllables = _per_syllable_outliers(
            user.syllable_durations, target.syllable_durations
        )

    diagnostics: list[str] = []
    if user_npvi < ref_npvi - 10:
        diagnostics.append(
            f"Your rhythm is too even (nPVI={user_npvi:.0f} vs target {ref_npvi:.0f}). "
            "Stressed syllables should be much longer than unstressed ones."
        )
    elif user_npvi > ref_npvi + 10:
        diagnostics.append(
            f"Your speech sounds over-stressed (nPVI={user_npvi:.0f} vs target {ref_npvi:.0f}). "
            "Try to speak more naturally — each stressed syllable is too exaggerated."
        )
    if inflated:
        words_str = ", ".join(f"'{w}'" for w in inflated[:4])
        diagnostics.append(
            f"Function words {words_str} are too long. "
            "These should be short and compressed — ~25–40 ms in natural speech."
        )
    if pattern_correlation is not None and pattern_correlation < 60:
        diagnostics.append(
            "The overall timing pattern differs from the target. "
            "Listen to which syllables the target rushes through vs. dwells on."
        )
    if outlier_syllables:
        long_pos = [str(i + 1) for i, d in outlier_syllables if d == "long"]
        short_pos = [str(i + 1) for i, d in outlier_syllables if d == "short"]
        parts = []
        if long_pos:
            parts.append(f"syllable(s) {', '.join(long_pos)} are too long")
        if short_pos:
            parts.append(f"syllable(s) {', '.join(short_pos)} are too short")
        diagnostics.append("Timing: " + "; ".join(parts) + " relative to target.")

    return RhythmBreakdown(
        npvi=user_npvi,
        reference_npvi_min=ref_min,
        reference_npvi_max=ref_max,
        score=float(np.clip(composite, 0.0, 100.0)),
        pattern_correlation=pattern_correlation,
        inflated_function_words=inflated,
        function_word_score=function_word_score,
        outlier_syllables=outlier_syllables,
        diagnostics=diagnostics,
    )
