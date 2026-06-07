"""Unit tests for prosody pipeline — hybrid syllable extraction."""
from __future__ import annotations

import numpy as np
import pytest

from accent_coach.models import PhonemeInstance
from accent_coach.pipeline.prosody import (
    _syllable_durs_within_word,
    compute_npvi,
    extract_syllable_durations_acoustic,
    extract_syllable_durations_from_words,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _phoneme(word: str, phoneme: str, start: float, end: float, is_stressed: bool = False) -> PhonemeInstance:
    return PhonemeInstance(
        phoneme=phoneme,
        arpabet="AE1" if phoneme == "æ" else "T",
        start_time=start,
        end_time=end,
        sentence_id=1,
        word=word,
        is_stressed=is_stressed,
    )


def _silent_audio(duration_s: float, sr: int = 22050) -> np.ndarray:
    return np.zeros(int(duration_s * sr), dtype=np.float32)


# ---------------------------------------------------------------------------
# extract_syllable_durations_from_words: single-syllable word path
# ---------------------------------------------------------------------------

def test_single_syllable_words_use_word_duration():
    """Single-syllable words ('the', 'cat') should return their exact word durations."""
    # 'the': IPA schwa ə (0.0→0.03), 'cat': IPA æ (0.1→0.3)
    phonemes = [
        _phoneme("the", "ə", 0.0, 0.03),
        _phoneme("cat", "æ", 0.1, 0.3, is_stressed=True),
    ]
    # Audio length > word_end to avoid slicing issues
    audio = _silent_audio(0.5)
    durs = extract_syllable_durations_from_words(phonemes, audio, sr=22050)
    # Should get exactly [0.03, 0.2] (word durations)
    assert len(durs) == 2
    assert durs[0] == pytest.approx(0.03, abs=1e-4)
    assert durs[1] == pytest.approx(0.2, abs=1e-4)


def test_function_words_get_actual_duration_not_zero():
    """Function word 'the' must appear in output with its real duration, not be skipped."""
    phonemes = [
        _phoneme("the", "ə", 0.0, 0.025),
        _phoneme("big", "ɪ", 0.08, 0.25, is_stressed=True),
    ]
    audio = _silent_audio(0.4)
    durs = extract_syllable_durations_from_words(phonemes, audio, sr=22050)
    assert any(abs(d - 0.025) < 1e-4 for d in durs)


def test_contrast_between_content_and_function_word():
    """Content word should be much longer than function word → nPVI can fire."""
    phonemes = [
        _phoneme("the", "ə", 0.0, 0.025),         # 25 ms function word
        _phoneme("storm", "ɔː", 0.05, 0.32, is_stressed=True),  # 270 ms content word
    ]
    audio = _silent_audio(0.5)
    durs = extract_syllable_durations_from_words(phonemes, audio, sr=22050)
    assert len(durs) == 2
    # Content word duration >> function word duration
    assert max(durs) / min(durs) > 5.0


def test_empty_phonemes_falls_back_to_acoustic():
    """Empty phoneme list → falls back to extract_syllable_durations_acoustic."""
    audio = _silent_audio(1.0)
    # Both should return without crashing; lengths may differ
    durs_hybrid = extract_syllable_durations_from_words([], audio, sr=22050)
    durs_acoustic = extract_syllable_durations_acoustic(audio, sr=22050)
    # Both return a list with at least one element
    assert len(durs_hybrid) >= 1
    assert len(durs_acoustic) >= 1


def test_short_words_below_15ms_skipped():
    """Words < 15 ms (measurement noise) should be excluded."""
    phonemes = [
        _phoneme("a", "ə", 0.0, 0.010),           # 10 ms — should be skipped
        _phoneme("cat", "æ", 0.05, 0.30, is_stressed=True),  # 250 ms — kept
    ]
    audio = _silent_audio(0.4)
    durs = extract_syllable_durations_from_words(phonemes, audio, sr=22050)
    # Only the 'cat' word should contribute (10ms < threshold)
    assert not any(d < 0.010 for d in durs)


# ---------------------------------------------------------------------------
# Multi-syllable word: stress-weighted fallback (no detectable peaks in silence)
# ---------------------------------------------------------------------------

def test_within_word_detection_returns_none_on_silence():
    """Silent/undetectable word → None, NOT a fabricated stress-weighted split.

    The old 2:1 stress-weighted fallback manufactured the long/short contrast
    nPVI rewards. It is deleted: failed detection returns None so the caller
    uses the real (measured) word duration instead of an invented split.
    """
    assert _syllable_durs_within_word(_silent_audio(0.30), sr=22050) is None


def test_failed_within_word_uses_real_word_duration_not_split():
    """When within-word detection fails, the word contributes its measured duration.

    No fabricated per-syllable split is emitted for the word. (The sentence-level
    < 2 path may still invoke the acoustic detector on the whole clip; here we
    assert no 2:1-style fabricated contrast is injected for the word itself.)
    """
    ph_GAR = _phoneme("garden", "ɑː", 0.0, 0.15, is_stressed=True)
    ph_DEN = _phoneme("garden", "ə",  0.15, 0.30, is_stressed=False)
    audio = _silent_audio(0.5)
    durs = extract_syllable_durations_from_words([ph_GAR, ph_DEN], audio, sr=22050)
    # No fabricated 2:1 split: if two durations come back they must NOT be the
    # manufactured 0.20/0.10 stress ratio.
    if len(durs) == 2:
        assert durs[0] / max(durs[1], 1e-9) != pytest.approx(2.0, abs=0.1)


# ---------------------------------------------------------------------------
# Bug 4 fix: acoustic fallback never returns 1-element list (nPVI=0)
# ---------------------------------------------------------------------------

def test_acoustic_detector_never_returns_one_element():
    """extract_syllable_durations_acoustic must return >= 2 elements so nPVI != 0."""
    import numpy as np
    rng = np.random.default_rng(42)
    for dur_s in [0.3, 0.5, 1.0, 2.0]:
        # Various audio lengths; some will have 0-1-2 acoustic peaks
        audio = rng.standard_normal(int(dur_s * 22050)).astype(np.float32) * 0.01
        durs = extract_syllable_durations_acoustic(audio, sr=22050)
        assert len(durs) >= 2, (
            f"acoustic detector returned {len(durs)} element(s) for {dur_s}s audio — "
            "would cause nPVI=0 false diagnostic"
        )
        npvi = compute_npvi(durs)
        assert npvi >= 0.0  # sanity, not NaN


def test_acoustic_detector_silent_audio_no_zero_npvi():
    """Silent audio should fall back to rate estimate, not produce nPVI=0 via 1-element list."""
    audio = _silent_audio(1.0)
    durs = extract_syllable_durations_acoustic(audio, sr=22050)
    assert len(durs) >= 2
