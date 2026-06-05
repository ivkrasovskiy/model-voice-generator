"""Alignment tests use a pre-built phoneme list to avoid model downloads in CI."""
from __future__ import annotations

from accent_coach.models import PhonemeInstance
from accent_coach.pipeline.alignment import (
    IPA_VOWELS,
    _BATH_WORDS,
    _LOT_WORDS,
    _word_to_phoneme_instances,
    filter_stops,
    filter_vowels,
)


def _phoneme(p: str, t0: float, t1: float, stressed: bool = False) -> PhonemeInstance:
    return PhonemeInstance(
        phoneme=p, arpabet="XX", start_time=t0, end_time=t1,
        sentence_id=1, word="test", is_stressed=stressed,
    )


def test_timestamps_monotonic():
    phonemes = [
        _phoneme("t", 0.0, 0.05, stressed=True),
        _phoneme("iː", 0.05, 0.15),
        _phoneme("s", 0.15, 0.25),
        _phoneme("t", 0.25, 0.30),
    ]
    for prev, nxt in zip(phonemes, phonemes[1:], strict=False):
        assert prev.end_time <= nxt.start_time


def test_filter_vowels():
    phonemes = [
        _phoneme("t", 0.0, 0.05),
        _phoneme("iː", 0.05, 0.15),
        _phoneme("æ", 0.15, 0.25),
    ]
    vowels = filter_vowels(phonemes)
    assert all(v.phoneme in IPA_VOWELS for v in vowels)
    assert len(vowels) == 2


def test_filter_stops_stressed_only():
    phonemes = [
        _phoneme("p", 0.0, 0.05, stressed=True),
        _phoneme("t", 0.05, 0.10, stressed=False),
        _phoneme("k", 0.10, 0.15, stressed=True),
    ]
    stops = filter_stops(phonemes, stressed_only=True)
    assert all(s.is_stressed for s in stops)
    assert len(stops) == 2


# ---------------------------------------------------------------------------
# TRAP/BATH split
# ---------------------------------------------------------------------------

def test_ipa_vowels_no_dead_entries():
    """ɐ removed from IPA_VOWELS — it was unreachable (not in ARPABET_TO_IPA)."""
    assert "ɐ" not in IPA_VOWELS


def test_bath_word_set_contains_expected():
    for w in ("bath", "path", "class", "last", "dance", "can't", "after", "half", "ask"):
        assert w in _BATH_WORDS, f"'{w}' should be in _BATH_WORDS"


def test_trap_word_not_in_bath_set():
    for w in ("cat", "hat", "man", "bag", "tap", "fan"):
        assert w not in _BATH_WORDS, f"'{w}' should NOT be in _BATH_WORDS"


def test_bath_word_mapped_to_long_a():
    """'bath', 'path', 'last' etc. must produce ɑː, not æ."""
    for word in ("bath", "path", "last", "class", "dance", "after", "half"):
        instances = _word_to_phoneme_instances(word, 0.0, 0.3, sentence_id=0)
        vowels = [p.phoneme for p in instances if p.phoneme in IPA_VOWELS]
        assert vowels, f"no vowels found for '{word}'"
        assert "ɑː" in vowels, (
            f"'{word}' should have ɑː (BATH); got {vowels}"
        )
        assert "æ" not in vowels, (
            f"'{word}' should NOT have æ (TRAP); got {vowels}"
        )


def test_trap_word_mapped_to_short_a():
    """TRAP words ('cat', 'hat') must keep æ — not affected by BATH override."""
    for word in ("cat", "hat", "man", "bag"):
        instances = _word_to_phoneme_instances(word, 0.0, 0.2, sentence_id=0)
        vowels = [p.phoneme for p in instances if p.phoneme in IPA_VOWELS]
        assert "æ" in vowels, f"'{word}' should have æ (TRAP); got {vowels}"
        assert "ɑː" not in vowels, f"'{word}' should NOT have ɑː; got {vowels}"


# ---------------------------------------------------------------------------
# LOT/THOUGHT split
# ---------------------------------------------------------------------------

def test_lot_word_set_contains_expected():
    for w in ("lot", "not", "hot", "got", "stop", "box", "clock", "bottle"):
        assert w in _LOT_WORDS, f"'{w}' should be in _LOT_WORDS"


def test_lot_word_mapped_to_short_o():
    """LOT words ('lot', 'not', 'stop') must produce ɒ, not ɑː (PALM/START)."""
    for word in ("lot", "not", "hot", "stop", "box"):
        instances = _word_to_phoneme_instances(word, 0.0, 0.2, sentence_id=0)
        vowels = [p.phoneme for p in instances if p.phoneme in IPA_VOWELS]
        assert vowels, f"no vowels for '{word}'"
        assert "ɒ" in vowels, f"'{word}' should have ɒ (LOT); got {vowels}"


def test_thought_word_keeps_long_o():
    """THOUGHT words go through AO1 → ɔː — NOT remapped to ɒ.
    Note: 'caught' uses AA1 in this cmudict (caught-cot merger) so excluded here.
    """
    for word in ("thought", "law", "saw", "taught"):
        instances = _word_to_phoneme_instances(word, 0.0, 0.2, sentence_id=0)
        vowels = [p.phoneme for p in instances if p.phoneme in IPA_VOWELS]
        assert vowels, f"no vowels for '{word}'"
        assert "ɔː" in vowels, f"'{word}' should have ɔː (THOUGHT/AO1); got {vowels}"
        assert "ɒ" not in vowels, f"'{word}' should NOT have ɒ; got {vowels}"


# ---------------------------------------------------------------------------
# _char_timestamps_to_phoneme_instances — MMS char-level to phoneme mapping
#
# The MMS forced aligner (torchaudio.functional.forced_align) produces
# character-level timestamps.  These must drive phoneme boundaries instead of
# the uniform word-duration/n_phonemes split.
#
# FAIL reason for all tests below: _char_timestamps_to_phoneme_instances does
# not yet exist in alignment.py — ImportError expected until implementation.
# ---------------------------------------------------------------------------


def test_char_timestamps_not_uniform():
    """Phoneme times must reflect char durations, not equal-width uniform splits.

    'stop' = 4 chars (s,t,o,p) → 4 phonemes (S, T, AA1/ɒ, P) — 1:1 mapping.
    Char durations: 0.02, 0.02, 0.32, 0.04 s (strongly unequal).
    Uniform split gives each phoneme 0.10 s → stdev = 0.
    Char-based mapping gives [0.02, 0.02, 0.32, 0.04] → stdev ≈ 0.15.

    FAIL reason: _char_timestamps_to_phoneme_instances not yet implemented.
    """
    import statistics

    from accent_coach.pipeline.alignment import _char_timestamps_to_phoneme_instances

    char_ts = [(0.0, 0.02), (0.02, 0.04), (0.04, 0.36), (0.36, 0.40)]
    instances = _char_timestamps_to_phoneme_instances("stop", char_ts, sentence_id=1)
    assert len(instances) == 4, f"Expected 4 phonemes for 'stop', got {len(instances)}"

    durations = sorted([inst.end_time - inst.start_time for inst in instances])
    stdev = statistics.stdev(durations)
    assert stdev > 0.08, (
        f"Phoneme duration stdev={stdev:.3f}. "
        "Uniform split gives stdev=0 (all 0.10 s). "
        "Char-based mapping must reflect the strongly unequal char durations "
        "[0.02, 0.02, 0.32, 0.04] → expected stdev > 0.08."
    )


def test_char_timestamps_exact_boundaries_for_equal_count():
    """1:1 case (n_chars == n_phones): each char boundary becomes a phoneme boundary exactly.

    'stop' = S(s) T(t) AA1→ɒ(o) P(p): 4:4 mapping.
    Phoneme start times must match char start times directly.
    Uniform split for 0.40 s word gives T starting at 0.10, ɒ at 0.20.
    Char-based gives T starting at 0.05, ɒ at 0.12 (the actual char boundaries).

    FAIL reason: _char_timestamps_to_phoneme_instances not yet implemented.
    """
    from accent_coach.pipeline.alignment import _char_timestamps_to_phoneme_instances

    # 't' starts at 0.05, 'o' starts at 0.12, 'p' starts at 0.31
    char_ts = [(0.0, 0.05), (0.05, 0.12), (0.12, 0.31), (0.31, 0.40)]
    instances = _char_timestamps_to_phoneme_instances("stop", char_ts, sentence_id=1)
    sorted_inst = sorted(instances, key=lambda x: x.start_time)
    assert len(sorted_inst) == 4

    # Uniform split (0.40/4 = 0.10 each): T starts at 0.10, vowel at 0.20, P at 0.30.
    # Char-based: T starts at 0.05, vowel at 0.12, P at 0.31.
    t_start = sorted_inst[1].start_time
    assert abs(t_start - 0.05) < 0.005, (
        f"/t/ start_time={t_start:.4f}. Expected 0.05 (char 't' boundary). "
        "Uniform split gives 0.10 — this must be char-based."
    )
    vowel_start = sorted_inst[2].start_time
    assert abs(vowel_start - 0.12) < 0.005, (
        f"Vowel start_time={vowel_start:.4f}. Expected 0.12 (char 'o' boundary). "
        "Uniform split gives 0.20."
    )


def test_char_timestamps_proportional_for_more_chars_than_phones():
    """When n_chars > n_phones, proportional interpolation maps each phoneme to a time slice.

    'rain' = R EY1 N (3 phones), chars r,a,i,n (4 chars).
    Uniform split (0.35/3 ≈ 0.117): EY1 starts at 0.117, N at 0.233.
    Char-based interpolation: EY1 starts at ~0.083, N starts at ~0.183.
    The second phoneme must start earlier than 0.10 (not the uniform 0.117).

    FAIL reason: _char_timestamps_to_phoneme_instances not yet implemented.
    """
    from accent_coach.pipeline.alignment import _char_timestamps_to_phoneme_instances

    # r=0.05s, a=0.10s, i=0.05s, n=0.15s (total 0.35s)
    char_ts = [(0.0, 0.05), (0.05, 0.15), (0.15, 0.20), (0.20, 0.35)]
    instances = _char_timestamps_to_phoneme_instances("rain", char_ts, sentence_id=1)
    sorted_inst = sorted(instances, key=lambda x: x.start_time)
    assert len(sorted_inst) == 3, f"Expected 3 phonemes for 'rain', got {len(sorted_inst)}"

    # /r/ must start at 0.0
    assert abs(sorted_inst[0].start_time - 0.0) < 0.005, (
        f"/r/ start={sorted_inst[0].start_time:.4f}, expected 0.0"
    )
    # Second phoneme (/eɪ/) must start before the uniform-split position (0.117)
    uniform_second = char_ts[0][0] + (char_ts[-1][1] - char_ts[0][0]) / 3  # ≈ 0.117
    eiy_start = sorted_inst[1].start_time
    assert eiy_start < uniform_second - 0.01, (
        f"/eɪ/ start={eiy_start:.4f}. Uniform split gives {uniform_second:.4f}. "
        "Char-based interpolation must place it earlier (~0.083) "
        "because 'r' is short and 'a+i' comes early in the word."
    )


def test_char_timestamps_bath_override_preserved():
    """BATH/LOT accent overrides must still apply when using char-based timestamps.

    'bath' in RP mode must produce ɑː (BATH override), not æ (GenAm).
    The char-based function shares the same override logic as _word_to_phoneme_instances.

    FAIL reason: _char_timestamps_to_phoneme_instances not yet implemented.
    """
    from accent_coach.pipeline.alignment import _char_timestamps_to_phoneme_instances

    # 'bath' = B AE1 TH → 5 chars (b,a,t,h → 4 actually: b,a,t,h), 3 phones
    # Use simple equal-duration chars to test override logic, not timing
    char_ts = [(0.0, 0.05), (0.05, 0.15), (0.15, 0.20), (0.20, 0.30)]
    instances = _char_timestamps_to_phoneme_instances("bath", char_ts, sentence_id=1,
                                                      accent_target="rp")
    vowels = [p.phoneme for p in instances if p.phoneme in IPA_VOWELS]
    assert "ɑː" in vowels, (
        f"'bath' in RP must produce ɑː (BATH override); got {vowels}"
    )
    assert "æ" not in vowels, (
        f"'bath' in RP must NOT produce æ; got {vowels}"
    )
