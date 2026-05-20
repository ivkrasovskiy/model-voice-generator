from __future__ import annotations

from pathlib import Path

import numpy as np

from accent_coach.models import PhonemeInstance

# ARPABET → IPA mapping (subset covering English phoneme inventory)
ARPABET_TO_IPA: dict[str, str] = {
    "AA": "ɑː", "AE": "æ", "AH": "ʌ", "AO": "ɔː", "AW": "aʊ",
    "AY": "aɪ", "B": "b", "CH": "tʃ", "D": "d", "DH": "ð",
    "EH": "ɛ", "ER": "ɜː", "EY": "eɪ", "F": "f", "G": "ɡ",
    "HH": "h", "IH": "ɪ", "IY": "iː", "JH": "dʒ", "K": "k",
    "L": "l", "M": "m", "N": "n", "NG": "ŋ", "OW": "əʊ",
    "OY": "ɔɪ", "P": "p", "R": "r", "S": "s", "SH": "ʃ",
    "T": "t", "TH": "θ", "UH": "ʊ", "UW": "uː", "V": "v",
    "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
    # Reduced/schwa
    "AX": "ə", "IX": "ɪ",
}

IPA_VOWELS = frozenset(
    "æ ɑː ɒ ɔː ʊ uː ɪ iː ɛ ʌ ɜː eɪ aɪ ɔɪ aʊ əʊ ə ɐ".split()
)

# Minimal CMU stress dict for calibration sentence vocabulary.
# Only words whose stress differs from first-syllable default are listed.
# Format: word (lowercase) → set of stressed syllable indices (0-based).
_STRESS_EXCEPTIONS: dict[str, set[int]] = {
    "because": {1}, "about": {1}, "above": {1}, "across": {1},
    "again": {1}, "against": {1}, "ahead": {1}, "already": {1},
    "although": {1}, "among": {1}, "around": {1}, "arrived": {1},
    "away": {1}, "before": {1}, "belong": {1}, "below": {1},
    "beside": {1}, "between": {1}, "beyond": {1}, "begin": {1},
    "behind": {1}, "believe": {1}, "below": {1}, "beneath": {1},
    "beside": {1}, "between": {1}, "beyond": {1},
    "photography": {1}, "photographer": {1}, "photographic": {2},
    "economy": {1}, "economic": {2}, "economics": {2},
    "democracy": {1}, "democratic": {2},
    "original": {1}, "originality": {4},
}


def _is_word_stressed(word: str, syllable_idx: int) -> bool:
    key = word.lower().rstrip(".,!?;:")
    if key in _STRESS_EXCEPTIONS:
        return syllable_idx in _STRESS_EXCEPTIONS[key]
    return syllable_idx == 0


def _whisperx_align(
    audio_path: Path, transcript: str, sentence_id: int
) -> list[PhonemeInstance]:
    import whisperx  # type: ignore[import-untyped]

    device = "cpu"
    model, metadata = whisperx.load_align_model(language_code="en", device=device)
    audio = whisperx.load_audio(str(audio_path))
    segments = [{"text": transcript, "start": 0.0, "end": float(len(audio)) / 16000}]
    result = whisperx.align(segments, model, metadata, audio, device)

    instances: list[PhonemeInstance] = []
    for word_seg in result.get("word_segments", []):
        word = word_seg.get("word", "").strip()
        for char_seg in word_seg.get("chars", []):
            arpabet = char_seg.get("char", "").upper()
            ipa = ARPABET_TO_IPA.get(arpabet, arpabet)
            start = char_seg.get("start", 0.0)
            end = char_seg.get("end", 0.0)
            if end - start <= 0:
                continue
            instances.append(
                PhonemeInstance(
                    phoneme=ipa,
                    arpabet=arpabet,
                    start_time=start,
                    end_time=end,
                    sentence_id=sentence_id,
                    word=word,
                    is_stressed=_is_word_stressed(word, 0),
                )
            )
    return instances


def _mms_align(
    audio_path: Path, transcript: str, sentence_id: int
) -> list[PhonemeInstance]:
    import torch
    import torchaudio

    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model()
    waveform, sample_rate = torchaudio.load(str(audio_path))
    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

    with torch.inference_mode():
        emission, _ = model(waveform)

    tokens = [c for c in transcript.lower() if c in bundle.get_labels()]
    alignment, _ = torchaudio.functional.forced_align(
        emission, torch.tensor([bundle.get_labels().index(t) for t in tokens])
    )

    duration = waveform.shape[-1] / bundle.sample_rate
    n = len(alignment)
    instances: list[PhonemeInstance] = []
    for i, (token_idx, _score) in enumerate(zip(alignment, alignment)):
        arpabet = tokens[i].upper()
        ipa = ARPABET_TO_IPA.get(arpabet, arpabet)
        start = float(i) / n * duration
        end = float(i + 1) / n * duration
        instances.append(
            PhonemeInstance(
                phoneme=ipa,
                arpabet=arpabet,
                start_time=start,
                end_time=end,
                sentence_id=sentence_id,
                word="",
                is_stressed=False,
            )
        )
    return instances


def align_audio(
    audio_path: Path, transcript: str, sentence_id: int = 0
) -> list[PhonemeInstance]:
    try:
        instances = _whisperx_align(audio_path, transcript, sentence_id)
        if instances:
            return instances
    except Exception:  # noqa: BLE001 — intentional fallback
        pass
    return _mms_align(audio_path, transcript, sentence_id)


def filter_vowels(phonemes: list[PhonemeInstance]) -> list[PhonemeInstance]:
    return [p for p in phonemes if p.phoneme in IPA_VOWELS]


def filter_stops(
    phonemes: list[PhonemeInstance], stressed_only: bool = True
) -> list[PhonemeInstance]:
    stop_ipas = {"p", "t", "k"}
    result = [p for p in phonemes if p.phoneme in stop_ipas]
    if stressed_only:
        result = [p for p in result if p.is_stressed]
    return result
