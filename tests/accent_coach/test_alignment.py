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
