from __future__ import annotations

from accent_coach.models import PhonemeInstance
from accent_coach.pipeline.formants import extract_vowel_features


def _make_phoneme(start: float, end: float) -> PhonemeInstance:
    return PhonemeInstance(
        phoneme="æ",
        arpabet="AE",
        start_time=start,
        end_time=end,
        sentence_id=1,
        word="cat",
        is_stressed=True,
    )


def test_vowel_features_within_tolerance(synthetic_vowel_audio):
    audio, sr = synthetic_vowel_audio
    phonemes = [_make_phoneme(0.05, 0.25)]
    results = extract_vowel_features(audio, sr, phonemes)
    assert len(results) == 1
    vf = results[0]
    # Synthetic additive signal won't perfectly match Praat's LPC; allow ±300 Hz
    assert abs(vf.f1 - 748) < 300, f"F1={vf.f1:.0f} too far from 748"
    assert abs(vf.f2 - 1710) < 300, f"F2={vf.f2:.0f} too far from 1710"


def test_short_vowel_rejected(synthetic_vowel_audio):
    audio, sr = synthetic_vowel_audio
    # < 30 ms — should be rejected
    phonemes = [_make_phoneme(0.0, 0.02)]
    results = extract_vowel_features(audio, sr, phonemes)
    assert results == []
