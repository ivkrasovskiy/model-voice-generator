"""Unit tests for the consonant bench's comparison-mode plumbing (§1D).

Pure-function coverage of transcript normalization + target matching, so the
comparison-mode code path is exercised without loading audio or alignment models.
"""
from __future__ import annotations

from scripts.bench.accent_coach_consonant_bench import _normalize_transcript


def test_normalize_transcript_ignores_case_and_punctuation():
    a = _normalize_transcript("Please, leave the keys on the table.")
    b = _normalize_transcript("please leave the keys on the table")
    assert a == b == "please leave the keys on the table"


def test_normalize_transcript_collapses_whitespace():
    assert _normalize_transcript("  hello   world\n") == "hello world"


def test_normalize_distinguishes_different_phrases():
    assert _normalize_transcript("the cat sat") != _normalize_transcript("the dog sat")
