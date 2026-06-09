"""VOT extraction — business-logic tests on hand-checkable ground truth.

We synthesise stops with a KNOWN voice onset time (closure → burst → aspiration
→ voicing), so the expected VOT is exact by construction. Real English /p t k/
are post-vocalic, so the realistic case has a PRECEDING vowel — which is exactly
where the old greedy extractor collapses to ~0 (it reads the preceding/following
vowel's periodicity as the stop's voicing onset).

Several tests here are EXPECTED TO FAIL against the current extractor and pass
after the Lisker-&-Abramson rewrite. See docs/vot_bug_diagnosis.md.
"""
from __future__ import annotations

import math

import numpy as np
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
