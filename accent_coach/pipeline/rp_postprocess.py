"""Post-processing of CMU-aligned phoneme sequences for non-rhotic RP evaluation.

Three corrections applied in order:
  1. strip_coda_r  — removes non-rhotic coda /r/ (silent before consonant or pause;
                     kept as linking R before a vowel).
  2. relabel_bath  — changes æ → ɑː for BATH-class words (CMU uses AE; RP uses ɑː).
  3. relabel_lot   — changes ɑː → ɒ for LOT-class words (CMU AA → RP ɒ, not ɑː).

Usage:
    from accent_coach.pipeline.rp_postprocess import apply_rp_corrections
    phonemes = apply_rp_corrections(phonemes)
"""
from __future__ import annotations

from accent_coach.models import PhonemeInstance
from accent_coach.pipeline.alignment import IPA_VOWELS
from accent_coach.reference.bath_words import BATH_WORDS, LOT_WORDS


def strip_coda_r(phonemes: list[PhonemeInstance]) -> list[PhonemeInstance]:
    """Remove non-rhotic coda /r/ phonemes, extending the preceding vowel's window.

    Rules:
      - /r/ that is preceded by a vowel AND NOT followed by a vowel → silent (drop).
      - /r/ that is preceded by a vowel AND followed by a vowel → linking R (keep).
      - /r/ that is preceded by a consonant → onset R (keep, e.g. 'spring', 'great').
      - ER/ɜː phonemes (ARPABET ER1/ER2) are a single unit with no separate R → untouched.
    """
    result: list[PhonemeInstance] = []
    for i, ph in enumerate(phonemes):
        if ph.phoneme != "r":
            result.append(ph)
            continue

        prev_is_vowel = result and result[-1].phoneme in IPA_VOWELS
        next_is_vowel = (i + 1 < len(phonemes)) and phonemes[i + 1].phoneme in IPA_VOWELS

        if not prev_is_vowel:
            # Onset R (e.g. 'spring', 'great') — keep unchanged
            result.append(ph)
        elif next_is_vowel:
            # Linking R (e.g. 'car alarm', 'here it is') — keep
            result.append(ph)
        else:
            # Coda R before consonant or end of utterance — silent in RP.
            # Extend preceding vowel's end_time to absorb the R's window so
            # formant extraction samples the full vowel duration.
            prev = result[-1]
            result[-1] = prev.model_copy(update={"end_time": ph.end_time})

    return result


def relabel_bath(phonemes: list[PhonemeInstance]) -> list[PhonemeInstance]:
    """Change æ → ɑː for BATH-class words (CMU AE; RP ɑː)."""
    result: list[PhonemeInstance] = []
    for ph in phonemes:
        if ph.phoneme == "æ" and ph.word.lower().rstrip("'") in BATH_WORDS:
            result.append(ph.model_copy(update={"phoneme": "ɑː", "arpabet": "AA1"}))
        else:
            result.append(ph)
    return result


def relabel_lot(phonemes: list[PhonemeInstance]) -> list[PhonemeInstance]:
    """Change ɑː → ɒ for LOT-class words (CMU AA; RP ɒ)."""
    result: list[PhonemeInstance] = []
    for ph in phonemes:
        if ph.phoneme == "ɑː" and ph.word.lower() in LOT_WORDS:
            result.append(ph.model_copy(update={"phoneme": "ɒ", "arpabet": "AA1"}))
        else:
            result.append(ph)
    return result


def apply_rp_corrections(phonemes: list[PhonemeInstance]) -> list[PhonemeInstance]:
    """Apply all three RP corrections in the correct order."""
    phonemes = strip_coda_r(phonemes)
    phonemes = relabel_bath(phonemes)
    phonemes = relabel_lot(phonemes)
    return phonemes
