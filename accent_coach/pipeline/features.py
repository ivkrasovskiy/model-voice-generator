from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from accent_coach.calibration.sentences import Sentence
from accent_coach.models import SentenceAnalysis
from accent_coach.pipeline.alignment import align_audio
from accent_coach.pipeline.formants import extract_vowel_features
from accent_coach.pipeline.prosody import (
    compute_npvi,
    extract_pitch_contour,
    extract_stress_pattern,
    extract_syllable_durations,
)
from accent_coach.pipeline.vot import extract_stop_features


def analyse_audio(
    audio_path: Path,
    transcript: str,
    sentence_meta: Sentence,
) -> SentenceAnalysis:
    audio, sr = sf.read(str(audio_path), always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    phonemes = align_audio(audio_path, transcript, sentence_meta.id)

    vowels = extract_vowel_features(audio, sr, phonemes)
    stops = extract_stop_features(audio, sr, phonemes)
    pitch_contour = extract_pitch_contour(audio, sr)
    syl_durs = extract_syllable_durations(phonemes, len(audio) / sr)
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
    )
