from __future__ import annotations

import re
from pathlib import Path

from accent_coach.models import PhonemeInstance
from accent_coach.reference.bath_words import BATH_WORDS as _BATH_WORDS
from accent_coach.reference.bath_words import LOT_WORDS as _LOT_WORDS


class AlignmentError(RuntimeError):
    """Raised when acoustic alignment cannot produce real phoneme boundaries.

    We deliberately do NOT fall back to a uniform word/n_phones split: uniform
    boundaries are fabricated rhythm that silently corrupts every downstream
    score (the consonant bench once ran entirely on uniform splits and ranked
    the owner above natives without erroring).  Fail loudly instead.
    """

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


_CHAR_WORD_SEPARATORS = frozenset({" ", "|", ""})


def _chars_to_ts(chars: list[dict] | None) -> list[tuple[float, float]]:
    """Filter a list of WhisperX char dicts to usable (start, end) pairs.

    Characters WhisperX could not place have ``start``/``end`` == None; drop
    them.  Returns ``[]`` when no usable timing remains.
    """
    if not chars:
        return []
    out: list[tuple[float, float]] = []
    for c in chars:
        cs, ce = c.get("start"), c.get("end")
        if cs is None or ce is None or ce < cs:
            continue
        out.append((float(cs), float(ce)))
    return out


def _segment_char_groups(seg_chars: list[dict]) -> list[list[dict]]:
    """Split a flat segment-level WhisperX char list into per-word groups.

    This WhisperX build stores char alignments at the SEGMENT level
    (``segments[].chars``) as one flat sequence for the whole utterance, with
    word boundaries marked by space characters — NOT nested per word.  Splitting
    on the separators recovers one char group per spoken word, in order.
    """
    groups: list[list[dict]] = []
    cur: list[dict] = []
    for c in seg_chars:
        if c.get("char", "") in _CHAR_WORD_SEPARATORS:
            if cur:
                groups.append(cur)
                cur = []
            continue
        cur.append(c)
    if cur:
        groups.append(cur)
    return groups


def _word_char_ts_by_index(seg: dict) -> list[list[tuple[float, float]]] | None:
    """Per-word char timing groups for a segment, aligned to ``seg['words']`` order.

    Handles both WhisperX shapes: per-word ``words[].chars`` (older) and flat
    segment-level ``segments[].chars`` split on spaces (this build).  Returns
    None when neither carries usable char timing (caller uses uniform split).
    """
    words = seg.get("words", [])
    # Shape A: chars nested under each word.
    if any(w.get("chars") for w in words):
        return [_chars_to_ts(w.get("chars")) for w in words]
    # Shape B: flat segment-level char list, split on spaces into word groups.
    seg_chars = seg.get("chars")
    if seg_chars:
        groups = _segment_char_groups(seg_chars)
        if len(groups) == len(words):
            return [_chars_to_ts(g) for g in groups]
    return None


def _whisperx_result_to_instances(
    result: dict, sentence_id: int, accent_target: str
) -> list[PhonemeInstance]:
    """Convert a WhisperX align() result into phoneme instances.

    Pure function (no model) so it is unit-testable.  Uses acoustic char-level
    boundaries ONLY (§2A).  There is no uniform fallback: a word with no usable
    char timing is DROPPED, and if words exist but none yield char-based
    instances the alignment is treated as failed (raises AlignmentError).
    Uniform word/n_phones splits are fabricated rhythm that silently corrupt
    every downstream score, so we refuse to emit them.
    """
    instances: list[PhonemeInstance] = []
    words_seen = 0
    for seg in result.get("segments", []):
        words = seg.get("words", [])
        if not words:
            continue
        char_ts_by_word = _word_char_ts_by_index(seg)
        for i, word_seg in enumerate(words):
            word = word_seg.get("word", "").strip()
            if not word:
                continue
            words_seen += 1
            char_ts = char_ts_by_word[i] if char_ts_by_word else []
            if not char_ts:
                # No acoustic char timing for this word — drop it; never fabricate.
                continue
            instances.extend(
                _char_timestamps_to_phoneme_instances(word, char_ts, sentence_id, accent_target)
            )

    if words_seen and not instances:
        raise AlignmentError(
            "WhisperX returned words but no usable character timing "
            "(return_char_alignments not honoured or unexpected 'chars' shape); "
            "refusing to fabricate uniform phoneme boundaries."
        )
    return instances


def _whisperx_align(
    audio_path: Path, transcript: str, sentence_id: int, accent_target: str = "rp"
) -> list[PhonemeInstance]:
    """WhisperX word + char alignment + cmudict G2P → phoneme instances.

    WhisperX gives accurate word AND character timestamps (with
    ``return_char_alignments=True``); cmudict maps each word to its ARPABET
    sequence; phoneme boundaries are interpolated over the acoustic character
    timeline (§2A) rather than split uniformly.  Words with no usable char
    timing fall back to the uniform split.
    """
    import whisperx  # type: ignore[import-untyped]

    device = "cpu"
    model, metadata = whisperx.load_align_model(language_code="en", device=device)
    audio = whisperx.load_audio(str(audio_path))
    # End time estimate based on 16 kHz samples (WhisperX loads at 16 kHz)
    approx_end = float(len(audio)) / 16000
    segments = [{"text": transcript, "start": 0.0, "end": approx_end}]
    result = whisperx.align(segments, model, metadata, audio, device, return_char_alignments=True)

    return _whisperx_result_to_instances(result, sentence_id, accent_target)


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
    errors: list[str] = []
    for name, fn in (("whisperx", _whisperx_align), ("mms", _mms_align)):
        try:
            instances = fn(audio_path, transcript, sentence_id, accent_target)
        except Exception as e:  # noqa: BLE001 — try the next real aligner, then raise
            errors.append(f"{name}: {type(e).__name__}: {e}")
            continue
        if instances:
            return instances
        errors.append(f"{name}: produced no phoneme instances")
    # No silent empty/uniform result: a clip we cannot align must fail loudly so
    # the caller drops it instead of scoring fabricated boundaries.
    raise AlignmentError(
        f"Alignment failed for {audio_path}: no aligner produced real boundaries. "
        + " | ".join(errors)
    )


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
