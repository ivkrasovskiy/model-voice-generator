"""VOT extraction — business-logic tests on hand-checkable ground truth.

We synthesise stops with a KNOWN voice onset time (closure → burst → aspiration
→ voicing), so the expected VOT is exact by construction. Real English /p t k/
are post-vocalic, so the realistic case has a PRECEDING vowel — which is exactly
where the old greedy extractor collapses to ~0 (it reads the preceding/following
vowel's periodicity as the stop's voicing onset).

Several tests here are EXPECTED TO FAIL against the current extractor and pass
after the Lisker-&-Abramson rewrite. See docs/prosody_consonant_upgrade.md.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.signal import lfilter

from accent_coach.models import PhonemeInstance
from accent_coach.pipeline.audio_io import normalize_audio
from accent_coach.pipeline.vot import extract_vot

SR = 16_000


def _voiced(dur_s: float, f0: float = 120.0,
            formants=((700, 90), (1200, 90), (2500, 150))) -> np.ndarray:
    n = int(SR * dur_s)
    pulse = np.zeros(n)
    pulse[:: int(SR / f0)] = 1.0
    out = pulse
    for f, bw in formants:
        r = math.exp(-math.pi * bw / SR)
        out = lfilter([1.0], [1.0, -2 * r * math.cos(2 * math.pi * f / SR), r**2], out)
    return out / (np.abs(out).max() + 1e-9) * 0.6


def _synth_stop(vot_ms: float, pre_vowel_ms: float = 90.0, closure_ms: float = 60.0,
                seed: int = 0) -> tuple[np.ndarray, int, PhonemeInstance, float]:
    """Return (normalized audio, sr, stop PhonemeInstance, true_VOT_ms).

    Layout: [preceding vowel] closure(silence) burst(5 ms) aspiration(VOT) [following vowel].
    The stop's start_time is the closure onset (where char-alignment lands it).
    """
    rng = np.random.default_rng(seed)
    parts = []
    if pre_vowel_ms > 0:
        parts.append(_voiced(pre_vowel_ms / 1000))
    parts.append(np.zeros(int(closure_ms / 1000 * SR)))               # closure
    parts.append(rng.standard_normal(int(0.005 * SR)) * 0.7)          # 5 ms broadband burst
    parts.append(rng.standard_normal(int(vot_ms / 1000 * SR)) * 0.25)  # aspiration during VOT
    parts.append(_voiced(0.18))                                       # following vowel
    audio = np.concatenate(parts).astype(np.float32)
    audio, sr = normalize_audio(audio, SR)
    t_closure = pre_vowel_ms / 1000
    stop = PhonemeInstance(phoneme="p", arpabet="P", start_time=t_closure,
                           end_time=t_closure + 0.05, sentence_id=1, word="pa", is_stressed=True)
    return audio, sr, stop, vot_ms


def test_vot_within_tolerance(synthetic_burst_voice_audio):
    """Isolated burst+voice (no preceding vowel) — the clean case."""
    audio, sr = synthetic_burst_voice_audio
    stop = PhonemeInstance(phoneme="t", arpabet="T", start_time=0.02, end_time=0.07,
                           sentence_id=1, word="top", is_stressed=True)
    vot = extract_vot(audio, sr, stop)
    assert vot is not None
    assert abs(vot - 70) < 20, f"VOT={vot:.1f} ms too far from 70 ms"


def test_connected_speech_vot_not_zero():
    """A post-vocalic /p/ with a true 70 ms VOT must NOT read as ~0.

    Core failure: with a preceding vowel the old extractor reads the vowel's
    voicing as the stop's onset → VOT≈0. Must land in the aspirated range.
    """
    audio, sr, stop, true_vot = _synth_stop(vot_ms=70, pre_vowel_ms=90)
    vot = extract_vot(audio, sr, stop)
    assert vot is not None, "VOT must be measurable on a clear post-vocalic stop."
    assert abs(vot - true_vot) < 25, (
        f"VOT={vot:.1f} ms vs true {true_vot:.0f} ms — preceding vowel must not "
        "collapse the measurement to ~0."
    )


def test_vot_distinguishes_short_vs_long():
    """Short-lag (30 ms) vs long-lag aspirated (90 ms) post-vocalic stops must be
    distinguishable by ≥ 30 ms — the whole point of measuring VOT."""
    a_s, sr, stop_s, _ = _synth_stop(vot_ms=30, pre_vowel_ms=90, seed=1)
    a_l, _, stop_l, _ = _synth_stop(vot_ms=90, pre_vowel_ms=90, seed=2)
    vot_s = extract_vot(a_s, sr, stop_s)
    vot_l = extract_vot(a_l, sr, stop_l)
    assert vot_s is not None and vot_l is not None, "Both VOTs must be measurable."
    assert vot_l - vot_s >= 30, (
        f"long={vot_l:.1f} short={vot_s:.1f} ms — must separate by ≥30 ms."
    )


def test_aspirated_vot_in_english_range():
    """A native-like aspirated /p/ (true 80 ms) must measure in the 50–125 ms
    English voiceless-aspirated range, not near zero."""
    audio, sr, stop, _ = _synth_stop(vot_ms=80, pre_vowel_ms=90, seed=3)
    vot = extract_vot(audio, sr, stop)
    assert vot is not None and 50 <= vot <= 125, (
        f"Aspirated /p/ VOT={vot} ms — expected 50–125 ms."
    )


def _seq(*specs):
    """Build a phoneme sequence from (ipa, stressed) tuples at 0.1 s spacing."""
    out = []
    for i, (ipa, st) in enumerate(specs):
        out.append(PhonemeInstance(phoneme=ipa, arpabet=ipa.upper(), start_time=0.1 * i,
                                   end_time=0.1 * i + 0.05, sentence_id=1, word="w", is_stressed=st))
    return out


def test_aspirating_filter_keeps_only_prevocalic_non_s_cluster_stops():
    """VOT must be measured only on aspirating-context stops (the accent signal).

    Kept: stressed /p/ before a vowel. Dropped: /t/ after /s/ (st cluster,
    unaspirated), /k/ before a consonant (not prevocalic), unstressed /p/.
    """
    from accent_coach.pipeline.alignment import filter_aspirating_stops

    phons = _seq(
        ("p", True), ("ɑː", True),   # /pɑ/  → KEEP (stressed, prevocalic)
        ("s", True), ("t", True), ("ɪ", True),  # /stɪ/ → /t/ post-/s/ → DROP
        ("k", True), ("l", True),    # /kl/  → /k/ before consonant → DROP
        ("p", False), ("ə", True),   # unstressed /p/ → DROP
    )
    kept = filter_aspirating_stops(phons)
    assert [p.phoneme for p in kept] == ["p"], (
        f"expected only the prevocalic stressed /p/, got {[p.phoneme for p in kept]}"
    )
    assert kept[0].is_stressed and abs(kept[0].start_time - 0.0) < 1e-6


# ── Aspiration detection coverage (15 diverse cases) ──────────────────────────
#
# Each case: (case_id, phoneme, vot_ms, pre_vowel_ms, closure_ms, seed)
# Ground truth VOT is exact by construction; we measure what % the extractor finds
# within ±25 ms.  The individual parametrized tests document which cases pass;
# test_aspiration_detection_rate guards the aggregate against regression.

_ASPIRATION_TOLERANCE_MS = 12.0  # tighter than the ~18 ms pitch-detection bias we need to fix
_MIN_DETECTION_RATE = 0.75   # ≥ 75 % of the 15 aspirated cases must be detected

_ASPIRATION_CASES: list[tuple[str, str, float, float, float, int]] = [
    # Canonical English aspirated stops — native-range VOT
    ("p_60ms",   "p",  60,  90,  60, 10),
    ("p_80ms",   "p",  80,  90,  60, 11),
    ("p_100ms",  "p", 100,  90,  60, 12),
    ("t_70ms",   "t",  70,  90,  60, 13),
    ("t_90ms",   "t",  90,  90,  60, 14),
    ("k_85ms",   "k",  85,  90,  60, 15),
    ("k_110ms",  "k", 110,  90,  60, 16),
    # Varying preceding-vowel duration (tests closure-anchoring logic)
    ("p_pre50",  "p",  75,  50,  60, 17),
    ("p_pre150", "p",  75, 150,  60, 18),
    ("t_pre200", "t",  80, 200,  60, 19),
    # Varying closure duration
    ("p_cl30",   "p",  70,  90,  30, 20),
    ("p_cl100",  "p",  70,  90, 100, 21),
    ("k_cl80",   "k",  90,  90,  80, 22),
    # Edge of English aspirated range
    ("p_50ms",   "p",  50,  90,  60, 23),
    ("k_120ms",  "k", 120,  90,  60, 24),
]


def _synth_stop_ph(
    phoneme: str,
    vot_ms: float,
    pre_vowel_ms: float = 90.0,
    closure_ms: float = 60.0,
    seed: int = 0,
) -> tuple[np.ndarray, int, PhonemeInstance, float]:
    """Like _synth_stop but accepts any stop phoneme label."""
    audio, sr, stop, true_vot = _synth_stop(vot_ms, pre_vowel_ms, closure_ms, seed)
    stop = PhonemeInstance(
        phoneme=phoneme, arpabet=phoneme.upper(),
        start_time=stop.start_time, end_time=stop.end_time,
        sentence_id=stop.sentence_id, word=stop.word, is_stressed=stop.is_stressed,
    )
    return audio, sr, stop, true_vot


@pytest.mark.parametrize("case_id,phoneme,vot_ms,pre_ms,clos_ms,seed", _ASPIRATION_CASES)
def test_aspiration_detection_individual(case_id, phoneme, vot_ms, pre_ms, clos_ms, seed):
    """Each aspirated stop must be detected within ±25 ms of its true VOT."""
    audio, sr, stop, true_vot = _synth_stop_ph(phoneme, vot_ms, pre_ms, clos_ms, seed)
    vot = extract_vot(audio, sr, stop)
    assert vot is not None, f"[{case_id}] VOT not detected (expected {true_vot:.0f} ms)"
    assert abs(vot - true_vot) <= _ASPIRATION_TOLERANCE_MS, (
        f"[{case_id}] VOT={vot:.1f} ms, true={true_vot:.0f} ms, "
        f"error={abs(vot - true_vot):.1f} ms > tolerance {_ASPIRATION_TOLERANCE_MS:.0f} ms"
    )


def test_aspiration_detection_rate():
    """Overall detection rate across 15 diverse aspirated-stop cases must reach ≥ 75 %.

    Run all cases and report the rate; use individual parametrized tests above
    to see exactly which cases fail.
    """
    hits = 0
    misses: list[str] = []
    for case_id, phoneme, vot_ms, pre_ms, clos_ms, seed in _ASPIRATION_CASES:
        audio, sr, stop, true_vot = _synth_stop_ph(phoneme, vot_ms, pre_ms, clos_ms, seed)
        vot = extract_vot(audio, sr, stop)
        if vot is not None and abs(vot - true_vot) <= _ASPIRATION_TOLERANCE_MS:
            hits += 1
        else:
            got = f"{vot:.1f} ms" if vot is not None else "None"
            misses.append(f"{case_id}(true={true_vot:.0f}ms got={got})")

    total = len(_ASPIRATION_CASES)
    rate = hits / total
    assert rate >= _MIN_DETECTION_RATE, (
        f"Detection rate {hits}/{total} ({rate:.0%}) < {_MIN_DETECTION_RATE:.0%}. "
        f"Missed: {misses}"
    )


def test_short_lag_not_misclassified_as_aspirated():
    """Short-lag stops (Russian-like 20–30 ms) must not measure as aspirated.

    If the extractor returns a value it must be short (< 45 ms), not in the
    English aspirated range — a false positive there would inflate accent scores.
    """
    for vot_ms, seed in [(20, 25), (30, 26)]:
        audio, sr, stop, true_vot = _synth_stop(vot_ms, pre_vowel_ms=90, seed=seed)
        vot = extract_vot(audio, sr, stop)
        if vot is not None:
            assert vot < 45, (
                f"Short-lag stop (true={true_vot:.0f} ms) returned {vot:.1f} ms — "
                "would be misclassified as aspirated (false positive)."
            )


# ── Short-closure VOT collapse (prosody_consonant_upgrade.md Open item #2/#3) ─
#
# Real in-context /p t k/ (measure_vot_reference.py, n=20/source) measure
# 0-22 ms even for natives, with VOT == 0.0 EXACTLY for 44-100 % of tokens in
# EVERY corpus and group. An instrumented trace (docs/prosody_consonant_upgrade.md)
# showed `burst_idx == closure_idx + gate_frames` in 6/6 real tokens (i.e. the
# burst-rise THRESHOLD is already exceeded at the very first frame the gate
# allows), and `first_voiced_after_burst` within 0-2 ms of that same frame — so
# the `-1/_PITCH_FLOOR_HZ` onset-correction clamp floors VOT to 0.
#
# Root cause, confirmed by a closure_ms x vot_ms sweep on _synth_stop (clean,
# correctly-located closure_idx by construction): `_MIN_CLOSURE_GATE_MS = 20`
# is a FIXED offset added to closure_idx before the burst search starts. When
# the true closure is SHORTER than 20 ms (plausible given in-domain VOT itself
# is 0-22 ms — connected-speech closures are short), the gate pushes the search
# start PAST the true burst and INTO/AT voicing onset, eating
# `(20 - closure_ms)` ms of the true VOT:
#   closure=10ms: true=10 -> measured=0.0   true=20 -> measured=5.2
#   closure=15ms: true=10 -> measured=0.7   true=20 -> measured=8.7
#   closure=20ms: true=5  -> measured=0.7   (gate == closure, borderline)
# whereas closure>=25ms tracks true_vot to within ~5 ms (the existing
# _ASPIRATION_CASES all use closure_ms 30-100, so they never exercised this).


@pytest.mark.parametrize(
    "vot_ms,closure_ms,seed",
    [(15, 10, 40), (20, 10, 41), (25, 10, 42)],
)
def test_short_closure_vot_not_collapsed_to_zero(vot_ms, closure_ms, seed):
    """A real (10-15 ms) closure with a clear VOT must not collapse to ~0.

    `_MIN_CLOSURE_GATE_MS = 20` is a fixed post-closure offset. When the true
    closure is shorter than the gate, the gate overshoots into/past voicing
    onset, eating `(20 - closure_ms)` ms of the true VOT -- exactly the
    pattern seen on real /p t k/ tokens (median VOT == 0 across every corpus).
    """
    audio, sr, stop, true_vot = _synth_stop(vot_ms, pre_vowel_ms=90, closure_ms=closure_ms, seed=seed)
    vot = extract_vot(audio, sr, stop)
    assert vot is not None, f"VOT not detected (true={true_vot:.0f}ms)"
    assert abs(vot - true_vot) <= _ASPIRATION_TOLERANCE_MS, (
        f"VOT={vot:.1f}ms, true={true_vot:.0f}ms (closure={closure_ms:.0f}ms) — "
        f"short closure must not collapse VOT toward 0 "
        f"(error={abs(vot - true_vot):.1f}ms > {_ASPIRATION_TOLERANCE_MS}ms)"
    )
