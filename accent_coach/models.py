from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class PhonemeInstance(BaseModel):
    phoneme: str
    arpabet: str
    start_time: float
    end_time: float
    sentence_id: int
    word: str
    is_stressed: bool


class VowelFeatures(BaseModel):
    phoneme: PhonemeInstance
    f1: float
    f2: float
    f3: float | None = None
    duration_ms: float
    pitch_mean: float


class StopFeatures(BaseModel):
    phoneme: PhonemeInstance
    vot_ms: float
    burst_energy: float


class SentenceAnalysis(BaseModel):
    sentence_id: int
    sentence_type: Literal[
        "statement", "yes_no_question", "wh_question", "complex", "list", "exclamation"
    ]
    duration_s: float
    syllable_durations: list[float]
    pitch_contour: list[float]
    stress_pattern: list[bool]
    vowels: list[VowelFeatures]
    stops: list[StopFeatures]
    phonemes: list[PhonemeInstance] = []


class VowelDiagnostic(BaseModel):
    phoneme: str
    target_f1: float
    target_f2: float
    user_f1: float
    user_f2: float
    deviation_magnitude: float
    articulatory_advice: str


class RhythmBreakdown(BaseModel):
    npvi: float
    reference_npvi_min: float
    reference_npvi_max: float
    score: float
    pattern_correlation: float | None = None
    inflated_function_words: list[str] = []
    function_word_score: float | None = None
    diagnostics: list[str] = []


class AspirationBreakdown(BaseModel):
    per_stop: dict[str, float]
    score: float


class IntonationBreakdown(BaseModel):
    per_sentence_dtw: list[float]
    score: float


class ComparisonResult(BaseModel):
    skill_scores: dict[str, float]
    composite_score: float
    vowel_diagnostics: list[VowelDiagnostic]
    rhythm_breakdown: RhythmBreakdown
    aspiration_breakdown: AspirationBreakdown
    intonation_breakdown: IntonationBreakdown
