"""Alignment tests use a pre-built phoneme list to avoid model downloads in CI."""
from __future__ import annotations

from accent_coach.models import PhonemeInstance
from accent_coach.pipeline.alignment import IPA_VOWELS, filter_stops, filter_vowels


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
