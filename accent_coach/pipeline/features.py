from __future__ import annotations

from pathlib import Path

import librosa

from accent_coach.calibration.sentences import Sentence
from accent_coach.models import SentenceAnalysis
from accent_coach.pipeline.alignment import align_audio
from accent_coach.pipeline.audio_io import load_standard_audio
from accent_coach.pipeline.formants import extract_vowel_features
from accent_coach.pipeline.prosody import (
    extract_pitch_contour,
    extract_stress_pattern,
    extract_syllable_durations_from_words,
)
from accent_coach.pipeline.vot import extract_stop_features


def analyse_audio(
    audio_path: Path,
    transcript: str,
    sentence_meta: Sentence,
) -> SentenceAnalysis:
    # Canonical normalization (mono + 16 kHz + common bandwidth cap) so every
    # user/corpus recording is measured on identical footing — no source
    # sample-rate/bandwidth confound. Fail loudly on an unreadable clip.
    loaded = load_standard_audio(audio_path)
    if loaded is None:
        raise ValueError(f"Could not read audio for analysis: {audio_path}")
    audio, sr = loaded

    phonemes = align_audio(audio_path, transcript, sentence_meta.id)

    vowels = extract_vowel_features(audio, sr, phonemes)
    stops = extract_stop_features(audio, sr, phonemes)
    pitch_contour = extract_pitch_contour(audio, sr)
    # Hybrid: word boundaries from WhisperX alignment (reliable for function words)
    # + per-word acoustic nucleus detection for within-word stress contrast.
    syl_durs = extract_syllable_durations_from_words(phonemes, audio, sr)
    stress = extract_stress_pattern(phonemes, audio, sr)

    duration_s = librosa.get_duration(y=audio, sr=sr)

    return SentenceAnalysis(
        sentence_id=sentence_meta.id,
        sentence_type=sentence_meta.sentence_type,  # type: ignore[arg-type]
        duration_s=duration_s,
        syllable_durations=syl_durs,
        pitch_contour=pitch_contour,
        stress_pattern=stress,
        vowels=vowels,
        stops=stops,
        phonemes=phonemes,
    )
