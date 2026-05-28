"""Unit tests for accent_coach.pipeline.rp_postprocess.

Tests:
  strip_coda_r:
    - coda R before consonant is dropped, preceding vowel extended
    - coda R at end of utterance is dropped
    - linking R before vowel is kept
    - onset R after consonant is kept
    - ER (NURSE unit) is untouched — no separate R to strip
  relabel_bath:
    - BATH-class word with æ → ɑː
    - non-BATH word with æ unchanged
    - case-insensitive word lookup
    - contraction "can't" relabeled
  relabel_lot:
    - LOT-class word with ɑː → ɒ
    - non-LOT word with ɑː unchanged
  apply_rp_corrections:
    - end-to-end: "after dark" gets BATH relabel + coda-R strip
"""
from __future__ import annotations

import pytest

from accent_coach.models import PhonemeInstance
from accent_coach.pipeline.rp_postprocess import (
    apply_rp_corrections,
    relabel_bath,
    relabel_lot,
    strip_coda_r,
)


def _ph(phoneme: str, arpabet: str = "", word: str = "x",
        start: float = 0.0, end: float = 0.1) -> PhonemeInstance:
    return PhonemeInstance(
        phoneme=phoneme, arpabet=arpabet or phoneme.upper(),
        word=word, start_time=start, end_time=end,
        sentence_id=0, is_stressed=False,
    )


# ---------------------------------------------------------------------------
# strip_coda_r
# ---------------------------------------------------------------------------

class TestStripCodaR:

    def test_coda_r_before_consonant_is_dropped(self):
        # "car park" → [k, ɑː(0.0-0.1), r(0.1-0.15), p, ɑː, k]
        # R is coda (preceded by vowel, followed by consonant) → dropped, vowel extended to 0.15
        phs = [
            _ph("k",  start=0.0, end=0.05),
            _ph("ɑː", start=0.05, end=0.10),
            _ph("r",  start=0.10, end=0.15),
            _ph("p",  start=0.15, end=0.20),
        ]
        result = strip_coda_r(phs)
        assert len(result) == 3
        assert result[1].phoneme == "ɑː"
        assert result[1].end_time == pytest.approx(0.15)   # extended to cover R
        assert result[2].phoneme == "p"

    def test_coda_r_at_end_of_utterance_is_dropped(self):
        # "car" in isolation: [k, ɑː, r]
        phs = [
            _ph("k",  start=0.0, end=0.05),
            _ph("ɑː", start=0.05, end=0.10),
            _ph("r",  start=0.10, end=0.15),
        ]
        result = strip_coda_r(phs)
        assert len(result) == 2
        assert result[1].phoneme == "ɑː"
        assert result[1].end_time == pytest.approx(0.15)

    def test_linking_r_before_vowel_is_kept(self):
        # "car alarm" → [k, ɑː, r, æ, l, ɑː, m]
        # R followed by æ (vowel) → linking R, kept
        phs = [
            _ph("k",  start=0.0,  end=0.05),
            _ph("ɑː", start=0.05, end=0.10),
            _ph("r",  start=0.10, end=0.15),
            _ph("æ",  start=0.15, end=0.22),
        ]
        result = strip_coda_r(phs)
        assert len(result) == 4
        assert result[2].phoneme == "r"
        assert result[1].end_time == pytest.approx(0.10)  # vowel NOT extended

    def test_onset_r_after_consonant_is_kept(self):
        # "spring" → [s, p, r, ɪ, ŋ]
        # R is preceded by p (consonant) → onset R, kept
        phs = [
            _ph("s", start=0.0,  end=0.04),
            _ph("p", start=0.04, end=0.07),
            _ph("r", start=0.07, end=0.10),
            _ph("ɪ", start=0.10, end=0.18),
            _ph("ŋ", start=0.18, end=0.23),
        ]
        result = strip_coda_r(phs)
        assert len(result) == 5
        assert result[2].phoneme == "r"

    def test_nurse_vowel_er_unit_untouched(self):
        # "word" → [w, ɜː, d]  (ER is a single IPA unit, no separate r)
        phs = [
            _ph("w",  start=0.0,  end=0.05),
            _ph("ɜː", start=0.05, end=0.15),
            _ph("d",  start=0.15, end=0.20),
        ]
        result = strip_coda_r(phs)
        assert result == phs   # unchanged

    def test_here_it_is_linking_r(self):
        # "here it" → [h, ɪ, r, ɪ, t]  — R followed by vowel ɪ → linking, keep
        phs = [
            _ph("h", start=0.0,  end=0.04),
            _ph("ɪ", start=0.04, end=0.10),
            _ph("r", start=0.10, end=0.13),
            _ph("ɪ", start=0.13, end=0.18),
            _ph("t", start=0.18, end=0.22),
        ]
        result = strip_coda_r(phs)
        assert len(result) == 5
        assert result[2].phoneme == "r"

    def test_multiple_coda_rs_in_sequence(self):
        # "father and" → [f, ɑː, ð, ə, r, æ, n, d]
        # R in "father": preceded by ə (vowel), followed by æ (vowel) → linking → keep
        # (In fast speech "father and" has linking R)
        phs = [
            _ph("f",  start=0.0,  end=0.05),
            _ph("ɑː", start=0.05, end=0.13),
            _ph("ð",  start=0.13, end=0.17),
            _ph("ə",  start=0.17, end=0.22),
            _ph("r",  start=0.22, end=0.25),
            _ph("æ",  start=0.25, end=0.32),
        ]
        result = strip_coda_r(phs)
        assert len(result) == 6
        assert result[4].phoneme == "r"   # linking R preserved


# ---------------------------------------------------------------------------
# relabel_bath
# ---------------------------------------------------------------------------

class TestRelabelBath:

    def test_bath_word_ae_becomes_ɑː(self):
        phs = [_ph("æ", arpabet="AE1", word="path")]
        result = relabel_bath(phs)
        assert result[0].phoneme == "ɑː"
        assert result[0].arpabet == "AA1"

    def test_non_bath_word_ae_unchanged(self):
        phs = [_ph("æ", arpabet="AE1", word="cat")]
        result = relabel_bath(phs)
        assert result[0].phoneme == "æ"

    def test_case_insensitive(self):
        phs = [_ph("æ", arpabet="AE1", word="After")]
        result = relabel_bath(phs)
        assert result[0].phoneme == "ɑː"

    def test_contraction_cant(self):
        phs = [_ph("æ", arpabet="AE1", word="can't")]
        result = relabel_bath(phs)
        assert result[0].phoneme == "ɑː"

    def test_non_ae_phoneme_in_bath_word_unchanged(self):
        # consonants in BATH word should not be touched
        phs = [_ph("p", arpabet="P", word="path")]
        result = relabel_bath(phs)
        assert result[0].phoneme == "p"


# ---------------------------------------------------------------------------
# relabel_lot
# ---------------------------------------------------------------------------

class TestRelabelLot:

    def test_lot_word_ɑː_becomes_ɒ(self):
        phs = [_ph("ɑː", arpabet="AA1", word="not")]
        result = relabel_lot(phs)
        assert result[0].phoneme == "ɒ"

    def test_non_lot_word_ɑː_unchanged(self):
        # "car" is START class — ɑː should stay
        phs = [_ph("ɑː", arpabet="AA1", word="car")]
        result = relabel_lot(phs)
        assert result[0].phoneme == "ɑː"

    def test_lot_word_non_ɑː_phoneme_unchanged(self):
        phs = [_ph("n", arpabet="N", word="not")]
        result = relabel_lot(phs)
        assert result[0].phoneme == "n"


# ---------------------------------------------------------------------------
# apply_rp_corrections end-to-end
# ---------------------------------------------------------------------------

class TestApplyRpCorrections:

    def test_after_dark(self):
        # "after dark" — both BATH relabel and coda-R strip apply
        # CMU alignment would give: æ (after-vowel), f, t, ə, r(coda), d, ɑː, r(coda)
        phs = [
            _ph("æ",  arpabet="AE1", word="after",  start=0.00, end=0.10),
            _ph("f",  arpabet="F",   word="after",  start=0.10, end=0.14),
            _ph("t",  arpabet="T",   word="after",  start=0.14, end=0.17),
            _ph("ə",  arpabet="AH0", word="after",  start=0.17, end=0.22),
            _ph("r",  arpabet="R",   word="after",  start=0.22, end=0.25),
            _ph("d",  arpabet="D",   word="dark",   start=0.25, end=0.29),
            _ph("ɑː", arpabet="AA1", word="dark",   start=0.29, end=0.38),
            _ph("r",  arpabet="R",   word="dark",   start=0.38, end=0.42),
            _ph("k",  arpabet="K",   word="dark",   start=0.42, end=0.46),
        ]
        result = apply_rp_corrections(phs)

        # Step 1: strip_coda_r
        #   "after" coda R (after ə, before d) → dropped, ə extended to 0.25
        #   "dark" coda R (after ɑː, before k) → dropped, ɑː extended to 0.42
        # Step 2: relabel_bath
        #   æ in "after" → ɑː
        # Step 3: relabel_lot — nothing to change

        phonemes_out = [p.phoneme for p in result]
        assert "r" not in phonemes_out, "All coda Rs should be stripped"
        assert result[0].phoneme == "ɑː", "'after' vowel should be relabeled to ɑː"

        # ə in "after" should have been extended to cover the R window
        schwa = next(p for p in result if p.word == "after" and p.phoneme == "ə")
        assert schwa.end_time == pytest.approx(0.25)

        # ɑː in "dark" should have been extended to cover the R window
        dark_vowel = next(p for p in result if p.word == "dark" and p.phoneme == "ɑː")
        assert dark_vowel.end_time == pytest.approx(0.42)

    def test_lot_and_bath_together(self):
        # "not after" — LOT relabel + BATH relabel, no coda R
        phs = [
            _ph("n",  arpabet="N",   word="not",   start=0.0, end=0.05),
            _ph("ɑː", arpabet="AA1", word="not",   start=0.05, end=0.12),
            _ph("t",  arpabet="T",   word="not",   start=0.12, end=0.16),
            _ph("æ",  arpabet="AE1", word="after", start=0.16, end=0.26),
        ]
        result = apply_rp_corrections(phs)
        assert result[1].phoneme == "ɒ"   # LOT: "not" ɑː → ɒ
        assert result[3].phoneme == "ɑː"  # BATH: "after" æ → ɑː
