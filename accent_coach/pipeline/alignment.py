from __future__ import annotations

import re
from pathlib import Path

from accent_coach.models import PhonemeInstance
from accent_coach.reference.bath_words import BATH_WORDS as _BATH_WORDS
from accent_coach.reference.bath_words import LOT_WORDS as _LOT_WORDS

# ARPABET → IPA mapping (stress-aware for vowels that reduce in unstressed position).
# Why stress digits matter: CMU dict uses AH0 for schwa and AH1/AH2 for STRUT (/ʌ/).
# Stripping digits before mapping turns every "the/a/of" into /ʌ/ and contaminates scoring.
ARPABET_TO_IPA: dict[str, str] = {
    # Vowels — stressed forms
    "AA1": "ɑː", "AA2": "ɑː",
    "AE1": "æ",  "AE2": "æ",
    "AH1": "ʌ",  "AH2": "ʌ",   # stressed STRUT
    "AH0": "ə",                  # unstressed schwa
    "AO1": "ɔː", "AO2": "ɔː",
    "AW1": "aʊ", "AW2": "aʊ",
    "AY1": "aɪ", "AY2": "aɪ",
    "EH1": "ɛ",  "EH2": "ɛ",
    "ER1": "ɜː", "ER2": "ɜː",   "ER0": "ə",
    "EY1": "eɪ", "EY2": "eɪ",
    "IH1": "ɪ",  "IH2": "ɪ",   "IH0": "ɪ",
    "IY1": "iː", "IY2": "iː",   "IY0": "ɪ",
    "OW1": "əʊ", "OW2": "əʊ",   "OW0": "ə",
    "OY1": "ɔɪ", "OY2": "ɔɪ",
    "UH1": "ʊ",  "UH2": "ʊ",    "UH0": "ə",
    "UW1": "uː", "UW2": "uː",   "UW0": "ʊ",
    # Consonants (no stress digits in CMU dict)
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "F": "f",
    "G": "ɡ", "HH": "h",  "JH": "dʒ", "K": "k", "L": "l",
    "M": "m", "N": "n",  "NG": "ŋ",  "P": "p",  "R": "r",
    "S": "s", "SH": "ʃ", "T": "t",  "TH": "θ", "V": "v",
    "W": "w", "Y": "j",  "Z": "z",  "ZH": "ʒ",
    # Legacy bare vowel codes (fallback for non-cmudict paths)
    "AA": "ɑː", "AE": "æ", "AH": "ʌ", "AO": "ɔː", "AW": "aʊ",
    "AY": "aɪ", "EH": "ɛ", "ER": "ɜː", "EY": "eɪ", "IH": "ɪ",
    "IY": "iː", "OW": "əʊ", "OY": "ɔɪ", "UH": "ʊ", "UW": "uː",
    "AX": "ə", "IX": "ɪ",
}

IPA_VOWELS = frozenset(
    # ɒ reachable via LOT override (AA1 → ɒ for LOT_WORDS)
    # ɐ removed — genuinely unreachable (not in ARPABET_TO_IPA, no override path)
    ["æ", "ɑː", "ɒ", "ɔː", "ʊ", "uː", "ɪ", "iː", "ɛ", "ʌ", "ɜː", "eɪ", "aɪ", "ɔɪ", "aʊ", "əʊ", "ə"]
)

# All ARPABET tokens that are vowels (stress-digit forms + legacy bare forms)
_ARPABET_VOWEL_BASES = {"AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW","AX","IX"}
_ARPABET_VOWELS = frozenset(
    k for k in ARPABET_TO_IPA if re.sub(r"\d","",k) in _ARPABET_VOWEL_BASES
)

# ---------------------------------------------------------------------------
# RP phoneme overrides — TRAP/BATH split and LOT/THOUGHT split
#
# cmudict follows GenAm phonology.  Two RP-specific remappings needed:
#
#   BATH words (Wells 1982 §2.2): cmudict AE1/AE2 → RP /ɑː/
#     e.g. "bath", "path", "class", "last", "dance", "can't", "after"
#     (In GenAm these stay /æ/ — override is RP-mode only)
#
#   LOT words: cmudict AA1/AA2 → RP /ɒ/ (short open back rounded)
#     e.g. "lot", "not", "hot", "stop", "box", "clock"
#     (LOT–THOUGHT merger in GenAm collapses both to /ɑ/; RP keeps them distinct)
#
# Source: accent_coach/reference/bath_words.py (Wells 1982 + corpus extensions)
# ---------------------------------------------------------------------------

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
    """Return ARPABET phoneme list (stress digits preserved on vowels) for a word."""
    key = re.sub(r"[^a-z']", "", word.lower())
    pronunciations = _get_cmu_dict().get(key)
    if not pronunciations:
        return []
    # Keep stress digits: AH0=schwa, AH1=STRUT — stripping them collapses that distinction.
    return list(pronunciations[0])


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
    accent_target: str = "rp",
) -> list[PhonemeInstance]:
    """Convert one word's time span into per-phoneme PhonemeInstance list."""
    arpabet_seq = _g2p(word)
    if not arpabet_seq:
        return []

    word_key = re.sub(r"[^a-z']", "", word.lower())
    is_bath = word_key in _BATH_WORDS

    syl_indices = _syllable_index(arpabet_seq)
    dur_per_ph = (word_end - word_start) / len(arpabet_seq)
    instances: list[PhonemeInstance] = []
    for i, (arpabet, syl_idx) in enumerate(zip(arpabet_seq, syl_indices, strict=True)):
        # RP-specific phoneme overrides — gated on accent_target to avoid
        # corrupting GenAm scoring (GenAm has no TRAP/BATH split, no /ɒ/).
        if accent_target == "rp" and is_bath and arpabet in ("AE1", "AE2"):
            ipa = "ɑː"
        elif accent_target == "rp" and word_key in _LOT_WORDS and arpabet in ("AA1", "AA2"):
            ipa = "ɒ"
        else:
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
    audio_path: Path, transcript: str, sentence_id: int, accent_target: str = "rp"
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
        instances.extend(_word_to_phoneme_instances(word, start, end, sentence_id, accent_target))
    return instances


def _char_timestamps_to_phoneme_instances(
    word: str,
    char_timestamps: list[tuple[float, float]],
    sentence_id: int,
    accent_target: str = "rp",
) -> list[PhonemeInstance]:
    """Map MMS character-level timestamps to phoneme instances.

    Uses CMU dict for the phoneme sequence.  Each phoneme i is assigned the
    time slice [i/n_phones, (i+1)/n_phones] of the character sequence via
    linear interpolation over actual char boundaries — so phoneme durations
    reflect real character durations, not uniform word/n_phones splits.

    For n_chars == n_phones the mapping is exact (char boundary == phoneme
    boundary).  For n_chars != n_phones the interpolation distributes the
    phonemes proportionally across the character timeline.

    Replaces the uniform-split _word_to_phoneme_instances call in _mms_align.
    """
    arpabet_seq = _g2p(word)
    if not arpabet_seq or not char_timestamps:
        return []

    word_key = re.sub(r"[^a-z']", "", word.lower())
    is_bath = word_key in _BATH_WORDS
    syl_indices = _syllable_index(arpabet_seq)
    n_phones = len(arpabet_seq)
    n_chars = len(char_timestamps)

    def _time_at(pos: float) -> float:
        """Return the audio timestamp at fractional char-sequence position pos."""
        if pos <= 0.0:
            return char_timestamps[0][0]
        if pos >= n_chars:
            return char_timestamps[-1][1]
        idx = int(pos)
        frac = pos - idx
        if idx >= n_chars:
            return char_timestamps[-1][1]
        c_start, c_end = char_timestamps[idx]
        return c_start + frac * (c_end - c_start)

    instances: list[PhonemeInstance] = []
    for i, (arpabet, syl_idx) in enumerate(zip(arpabet_seq, syl_indices, strict=True)):
        start_time = _time_at(i * n_chars / n_phones)
        end_time = _time_at((i + 1) * n_chars / n_phones)

        if accent_target == "rp" and is_bath and arpabet in ("AE1", "AE2"):
            ipa = "ɑː"
        elif accent_target == "rp" and word_key in _LOT_WORDS and arpabet in ("AA1", "AA2"):
            ipa = "ɒ"
        else:
            ipa = ARPABET_TO_IPA.get(arpabet, arpabet)

        instances.append(
            PhonemeInstance(
                phoneme=ipa,
                arpabet=arpabet,
                start_time=start_time,
                end_time=end_time,
                sentence_id=sentence_id,
                word=word,
                is_stressed=_is_word_stressed(word, syl_idx),
            )
        )
    return instances


def _mms_align(
    audio_path: Path, transcript: str, sentence_id: int, accent_target: str = "rp"
) -> list[PhonemeInstance]:
    """MMS forced alignment. Uses char-level timestamps for phoneme boundaries.

    torchaudio.functional.forced_align gives per-character acoustic boundaries.
    These are passed to _char_timestamps_to_phoneme_instances so phoneme start
    times reflect actual character durations rather than uniform word/n_phones splits.
    """
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

    targets = torch.tensor([[labels.index(c) for c in char_tokens]])
    with torch.inference_mode():
        alignment, _ = torchaudio.functional.forced_align(emission, targets)

    frame_tokens = alignment[0].tolist()
    duration = waveform.shape[-1] / bundle.sample_rate
    n_frames = len(frame_tokens)

    instances: list[PhonemeInstance] = []
    for t_start, t_end, word in word_boundaries:
        # Extract per-character timestamps for this word from the frame alignment
        char_ts: list[tuple[float, float]] = []
        for char_idx in range(t_start, t_end):
            frames = [f for f, tok in enumerate(frame_tokens) if tok == char_idx]
            if not frames:
                continue
            char_ts.append((
                frames[0] / n_frames * duration,
                (frames[-1] + 1) / n_frames * duration,
            ))
        if not char_ts:
            continue
        instances.extend(
            _char_timestamps_to_phoneme_instances(word, char_ts, sentence_id, accent_target)
        )
    return instances


def align_audio(
    audio_path: Path, transcript: str, sentence_id: int = 0, accent_target: str = "rp"
) -> list[PhonemeInstance]:
    try:
        instances = _whisperx_align(audio_path, transcript, sentence_id, accent_target)
        if instances:
            return instances
    except Exception:  # noqa: BLE001 — intentional fallback to MMS
        pass
    return _mms_align(audio_path, transcript, sentence_id, accent_target)


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
