from __future__ import annotations

import re
from pathlib import Path

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
    ["æ", "ɑː", "ɒ", "ɔː", "ʊ", "uː", "ɪ", "iː", "ɛ", "ʌ", "ɜː", "eɪ", "aɪ", "ɔɪ", "aʊ", "əʊ", "ə", "ɐ"]
)

# ARPABET vowel tokens (with stress digits stripped) for stress detection
_ARPABET_VOWELS = frozenset(ARPABET_TO_IPA.keys()) & {
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
    "IH", "IY", "OW", "OY", "UH", "UW", "AX", "IX",
}

# Minimal word-level stress exceptions (secondary syllable is primary stress)
_STRESS_EXCEPTIONS: dict[str, set[int]] = {
    "because": {1}, "about": {1}, "above": {1}, "across": {1},
    "again": {1}, "against": {1}, "ahead": {1}, "already": {1},
    "although": {1}, "among": {1}, "around": {1}, "arrived": {1},
    "away": {1}, "before": {1}, "belong": {1}, "below": {1},
    "beside": {1}, "between": {1}, "beyond": {1}, "begin": {1},
    "behind": {1}, "believe": {1}, "beneath": {1},
    "photography": {1}, "photographer": {1}, "photographic": {2},
    "economy": {1}, "economic": {2}, "economics": {2},
    "democracy": {1}, "democratic": {2},
    "original": {1}, "originality": {4},
}

# Lazy-loaded cmudict
_cmu_dict: dict[str, list[str]] | None = None


def _get_cmu_dict() -> dict[str, list[str]]:
    global _cmu_dict
    if _cmu_dict is None:
        import nltk
        try:
            from nltk.corpus import cmudict
            _cmu_dict = cmudict.dict()
        except LookupError:
            nltk.download("cmudict", quiet=True)
            from nltk.corpus import cmudict
            _cmu_dict = cmudict.dict()
    return _cmu_dict


def _g2p(word: str) -> list[str]:
    """Return ARPABET phoneme list (no stress digits) for a word via CMU dict."""
    key = re.sub(r"[^a-z']", "", word.lower())
    pronunciations = _get_cmu_dict().get(key)
    if not pronunciations:
        return []
    # Strip stress digits from vowels; keep consonant labels as-is
    return [re.sub(r"\d", "", p) for p in pronunciations[0]]


def _syllable_index(phonemes: list[str]) -> list[int]:
    """Return the syllable index (0-based) of each phoneme in the word."""
    syl = 0
    result = []
    for p in phonemes:
        result.append(syl)
        if p in _ARPABET_VOWELS:
            syl += 1
    return result


def _is_word_stressed(word: str, syllable_idx: int) -> bool:
    key = word.lower().rstrip(".,!?;:")
    if key in _STRESS_EXCEPTIONS:
        return syllable_idx in _STRESS_EXCEPTIONS[key]
    return syllable_idx == 0


def _word_to_phoneme_instances(
    word: str,
    word_start: float,
    word_end: float,
    sentence_id: int,
) -> list[PhonemeInstance]:
    """Convert one word's time span into per-phoneme PhonemeInstance list."""
    arpabet_seq = _g2p(word)
    if not arpabet_seq:
        return []

    syl_indices = _syllable_index(arpabet_seq)
    dur_per_ph = (word_end - word_start) / len(arpabet_seq)
    instances: list[PhonemeInstance] = []
    for i, (arpabet, syl_idx) in enumerate(zip(arpabet_seq, syl_indices, strict=True)):
        ipa = ARPABET_TO_IPA.get(arpabet, arpabet)
        instances.append(
            PhonemeInstance(
                phoneme=ipa,
                arpabet=arpabet,
                start_time=word_start + i * dur_per_ph,
                end_time=word_start + (i + 1) * dur_per_ph,
                sentence_id=sentence_id,
                word=word,
                is_stressed=_is_word_stressed(word, syl_idx),
            )
        )
    return instances


def _whisperx_align(
    audio_path: Path, transcript: str, sentence_id: int
) -> list[PhonemeInstance]:
    """WhisperX word-level alignment + cmudict G2P → phoneme instances.

    WhisperX gives accurate word timestamps; cmudict maps each word to its
    ARPABET sequence; duration is split uniformly across phonemes within a word.
    """
    import whisperx  # type: ignore[import-untyped]

    device = "cpu"
    model, metadata = whisperx.load_align_model(language_code="en", device=device)
    audio = whisperx.load_audio(str(audio_path))
    # End time estimate based on 16 kHz samples (WhisperX loads at 16 kHz)
    approx_end = float(len(audio)) / 16000
    segments = [{"text": transcript, "start": 0.0, "end": approx_end}]
    result = whisperx.align(segments, model, metadata, audio, device)

    instances: list[PhonemeInstance] = []
    for word_seg in result.get("word_segments", []):
        word = word_seg.get("word", "").strip()
        start = word_seg.get("start")
        end = word_seg.get("end")
        if start is None or end is None or end <= start:
            continue
        instances.extend(_word_to_phoneme_instances(word, start, end, sentence_id))
    return instances


def _mms_align(
    audio_path: Path, transcript: str, sentence_id: int
) -> list[PhonemeInstance]:
    """MMS forced alignment fallback. Uses word-level + G2P like WhisperX path."""
    import torch
    import torchaudio

    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model()
    waveform, sample_rate = torchaudio.load(str(audio_path))
    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

    with torch.inference_mode():
        emission, _ = model(waveform)

    labels = bundle.get_labels()
    words = transcript.split()
    # Build char-level token sequence (MMS_FA is char-level)
    char_tokens: list[str] = []
    word_boundaries: list[tuple[int, int, str]] = []  # (token_start, token_end, word)
    for word in words:
        chars = [c for c in word.lower() if c in labels]
        if not chars:
            continue
        t_start = len(char_tokens)
        char_tokens.extend(chars)
        word_boundaries.append((t_start, len(char_tokens), word))

    if not char_tokens:
        return []

    targets = torch.tensor([[labels.index(c) for c in char_tokens]])  # [1, N]
    with torch.inference_mode():
        alignment, _ = torchaudio.functional.forced_align(emission, targets)

    # alignment shape: [1, time] — frame index of active token
    frame_tokens = alignment[0].tolist()
    duration = waveform.shape[-1] / bundle.sample_rate
    n_frames = len(frame_tokens)

    # Map frame ranges back to words, then use G2P for phonemes
    instances: list[PhonemeInstance] = []
    for t_start, t_end, word in word_boundaries:
        # Find frames belonging to this word's token range
        word_frames = [f for f, tok in enumerate(frame_tokens) if t_start <= tok < t_end]
        if not word_frames:
            continue
        w_start = word_frames[0] / n_frames * duration
        w_end = (word_frames[-1] + 1) / n_frames * duration
        instances.extend(_word_to_phoneme_instances(word, w_start, w_end, sentence_id))
    return instances


def align_audio(
    audio_path: Path, transcript: str, sentence_id: int = 0
) -> list[PhonemeInstance]:
    try:
        instances = _whisperx_align(audio_path, transcript, sentence_id)
        if instances:
            return instances
    except Exception:  # noqa: BLE001 — intentional fallback to MMS
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
