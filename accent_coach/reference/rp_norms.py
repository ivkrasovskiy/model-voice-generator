"""Canonical RP / SSBE acoustic norms.

All numeric constants carry citation comments. Any value that could not be
sourced from a published table is marked # TODO(cite).
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Vowel formants (Hz) — adult male SSBE — MODERN RP (Fry + Lindsey + BBC-male)
# Pooled mean F1/F2 from tts_output/modern_rp_corpus/formants.csv filtered to
# duration >= 50 ms.  Sources: modern_rp_fry (n=9109 tokens), modern_rp_lindsey
# (n=1910), modern_rp_bbc_male (n=2473; Phase E filter_male decision).
# Supersedes RP_VOWEL_F1_F2_MALE_LEGACY (Deterding 1997) — see
# docs/accent_coach_history.md#phase-05 for the supersession rationale.
# ---------------------------------------------------------------------------
RP_VOWEL_F1_F2_MALE_MODERN: dict[str, tuple[float, float]] = {
    "iː": (348, 1962),   # FLEECE   — n=749; modern_rp_fry+lindsey+bbc_male; see docs/accent_coach_history.md#phase-07
    "ɪ":  (386, 1773),   # KIT      — n=2440; modern_rp_fry+lindsey+bbc_male; see docs/accent_coach_history.md#phase-07
    "ɛ":  (462, 1571),   # DRESS    — n=1113; modern_rp_fry+lindsey+bbc_male; see docs/accent_coach_history.md#phase-07
    "æ":  (545, 1496),   # TRAP     — n=718; modern_rp_fry+lindsey+bbc_male; see docs/accent_coach_history.md#phase-07
    "ɑː": (518, 1215),   # BATH/PALM — n=564; modern_rp_fry+lindsey+bbc_male; see docs/accent_coach_history.md#phase-07
    "ɒ":  (532, 1114),   # LOT      — n=153; modern_rp_corpus (lot_word_clips_manifest); measured with LOT override active
    "ɔː": (459, 1138),   # THOUGHT  — n=647; modern_rp_fry+lindsey+bbc_male; see docs/accent_coach_history.md#phase-07
    "ʊ":  (386, 1427),   # FOOT     — n=174; modern_rp_fry+lindsey+bbc_male; see docs/accent_coach_history.md#phase-07
    "uː": (352, 1506),   # GOOSE    — n=597; modern_rp_fry+lindsey+bbc_male; see docs/accent_coach_history.md#phase-07
    "ʌ":  (472, 1339),   # STRUT    — n=497; modern_rp_fry+lindsey+bbc_male; see docs/accent_coach_history.md#phase-07
    "ɜː": (482, 1440),   # NURSE    — n=310; modern_rp_fry+lindsey+bbc_male; see docs/accent_coach_history.md#phase-07
    "ə":  (412, 1550),   # SCHWA    — n=3146; modern_rp_fry+lindsey+bbc_male; see docs/accent_coach_history.md#phase-07
    # Diphthongs: onglide values (initial position)
    "eɪ": (383, 1873),   # FACE onglide  — n=817; modern_rp_fry+lindsey+bbc_male; see docs/accent_coach_history.md#phase-07
    "aɪ": (482, 1599),   # PRICE onglide — n=824; modern_rp_fry+lindsey+bbc_male; see docs/accent_coach_history.md#phase-07
    "ɔɪ": (429, 1560),   # CHOICE onglide — n=56; modern_rp_fry+lindsey+bbc_male; see docs/accent_coach_history.md#phase-07
    "əʊ": (389, 1353),   # GOAT onglide  — n=539; modern_rp_fry+lindsey+bbc_male; see docs/accent_coach_history.md#phase-07
    "aʊ": (536, 1304),   # MOUTH onglide — n=301; modern_rp_fry+lindsey+bbc_male; see docs/accent_coach_history.md#phase-07
}

# ---------------------------------------------------------------------------
# LEGACY — Deterding 1997 (kept for historical comparison, not used by product)
# Deterding 1997, Table 1, "The formants of monophthong vowels in Standard
# Southern British English pronunciation", JIPA 27(1-2), pp. 47-55.
# Superseded by RP_VOWEL_F1_F2_MALE_MODERN — see docs/accent_coach_history.md#phase-05
# ---------------------------------------------------------------------------
RP_VOWEL_F1_F2_MALE_LEGACY: dict[str, tuple[float, float]] = {
    "iː": (280, 2249),   # FLEECE   — Deterding 1997, Table 1
    "ɪ":  (367, 1757),   # KIT      — Deterding 1997, Table 1
    "ɛ":  (580, 1799),   # DRESS    — Deterding 1997, Table 1
    "æ":  (748, 1710),   # TRAP     — Deterding 1997, Table 1
    "ɑː": (680, 1100),   # BATH/PALM — Deterding 1997, Table 1
    "ɒ":  (600, 900),    # LOT      — Deterding 1997, Table 1; approx
    "ɔː": (430, 700),    # THOUGHT  — Deterding 1997, Table 1
    "ʊ":  (378, 950),    # FOOT     — Deterding 1997, Table 1
    "uː": (310, 1156),   # GOOSE    — Deterding 1997, Table 1
    "ʌ":  (623, 1224),   # STRUT    — Deterding 1997, Table 1
    "ɜː": (490, 1570),   # NURSE    — Deterding 1997, Table 1
    "ə":  (490, 1350),   # SCHWA    — TODO(cite): typical estimate
    "eɪ": (530, 1680),   # FACE onglide — Cruttenden 2014, Table 3.4 approx
    "aɪ": (730, 1100),   # PRICE onglide — Cruttenden 2014, Table 3.4 approx
    "ɔɪ": (430, 700),    # CHOICE onglide — TODO(cite)
    "əʊ": (490, 1000),   # GOAT onglide  — Cruttenden 2014, Table 3.4 approx
    "aʊ": (730, 1100),   # MOUTH onglide — Cruttenden 2014, Table 3.4 approx
}

# Convenience alias — now points at modern corpus-derived norms (Fry+Lindsey+BBC-male)
RP_VOWEL_F1_F2_MALE = RP_VOWEL_F1_F2_MALE_MODERN

# ---------------------------------------------------------------------------
# Vowel formants (Hz) — adult female SSBE
# Deterding 1997, Table 1 (female speaker data).
# ---------------------------------------------------------------------------
RP_VOWEL_F1_F2_FEMALE: dict[str, tuple[float, float]] = {
    "iː": (310, 2793),   # FLEECE   — Deterding 1997, Table 1
    "ɪ":  (427, 2199),   # KIT      — Deterding 1997, Table 1
    "ɛ":  (703, 2273),   # DRESS    — Deterding 1997, Table 1
    "æ":  (880, 1952),   # TRAP     — Deterding 1997, Table 1
    "ɑː": (800, 1197),   # BATH/PALM — Deterding 1997, Table 1
    "ɒ":  (680, 1010),   # LOT      — Deterding 1997, Table 1; approx
    "ɔː": (500, 820),    # THOUGHT  — Deterding 1997, Table 1
    "ʊ":  (434, 1100),   # FOOT     — Deterding 1997, Table 1
    "uː": (370, 1530),   # GOOSE    — Deterding 1997, Table 1
    "ʌ":  (753, 1426),   # STRUT    — Deterding 1997, Table 1
    "ɜː": (540, 1700),   # NURSE    — Deterding 1997, Table 1
    "ə":  (540, 1450),   # SCHWA    — TODO(cite): typical estimate
    "eɪ": (620, 1980),   # FACE onglide  — TODO(cite)
    "aɪ": (830, 1280),   # PRICE onglide — TODO(cite)
    "ɔɪ": (500, 820),    # CHOICE onglide — TODO(cite)
    "əʊ": (540, 1150),   # GOAT onglide   — TODO(cite)
    "aʊ": (830, 1280),   # MOUTH onglide  — TODO(cite)
}

# ---------------------------------------------------------------------------
# VOT ranges (ms) — word-initial stressed /p t k/ in RP-adjacent English
# Lisker & Abramson 1964, "A Cross-Language Study of Voicing in Initial
# Stops", Word 20(3), pp. 384-422 (English column).
# Docherty 1992, "The Timing of Voicing in British English Obstruents",
# Foris, The Hague.
# ---------------------------------------------------------------------------
RP_VOT_RANGE_MS: dict[str, tuple[float, float]] = {
    "p": (55.0, 80.0),   # Lisker & Abramson 1964 / Docherty 1992
    "t": (65.0, 90.0),   # Lisker & Abramson 1964 / Docherty 1992
    "k": (75.0, 100.0),  # Lisker & Abramson 1964 / Docherty 1992
}

# VOT mean and SD per stop (ms) for z-score scoring in aspiration.py
# TODO(cite): mean = midpoint of range; SD estimated as 10 ms
RP_VOT_MEAN_MS: dict[str, float] = {
    "p": 67.5,
    "t": 77.5,
    "k": 87.5,
}
RP_VOT_SD_MS: dict[str, float] = {
    "p": 10.0,
    "t": 10.0,
    "k": 10.0,
}

# ---------------------------------------------------------------------------
# nPVI (normalised Pairwise Variability Index) range for English
#
# Corpus-derived from accent_coach modern_rp_corpus (Fry n=120, BBC n=122,
# Lindsey n=121) + genam_corpus (n=98), measured with hybrid word-alignment
# detector (extract_syllable_durations_from_words).
#
# Acoustic-only measurement: all-native mean=41.0, p25=29.3, p75=50.7
# Hybrid correction (WhisperX word boundaries vs acoustic-only): +11 nPVI
# Hybrid-adjusted estimates: mean≈52, p25≈40, p75≈62
#
# Supersedes Grabe & Low 2002 (55–75) which was a controlled lab read-aloud
# task — ~14 nPVI points above natural conversational speech.
# Syllable-timed languages (Spanish, Japanese) remain ~25–35, clearly below.
# ---------------------------------------------------------------------------
RP_NPVI_MIN: float = 40.0   # corpus p25, hybrid-adjusted (was 55 Grabe & Low 2002)
RP_NPVI_MAX: float = 62.0   # corpus p75, hybrid-adjusted (was 75 Grabe & Low 2002)

# ---------------------------------------------------------------------------
# Pitch-contour templates per sentence type (normalised, 50 points, 0–1 range)
# Derived from Cruttenden 2014, "Gimson's Pronunciation of English", 8th ed.,
# chapter 11, description of British English intonation patterns.
# These are idealised approximations — exact shape is language-universal to
# within ± one tone unit. TODO(cite): numeric shapes are author constructions
# based on Cruttenden's qualitative descriptions.
# ---------------------------------------------------------------------------

def _make_falling(n: int = 50) -> list[float]:
    """High-plateau then step-fall (declarative / wh-question)."""
    xs = np.linspace(0, 1, n)
    return np.where(xs < 0.6, 1.0 - 0.2 * xs, 1.0 - 0.2 * 0.6 - 3.0 * (xs - 0.6)).clip(0, 1).tolist()


def _make_rising(n: int = 50) -> list[float]:
    """Low then rising at end (yes/no question)."""
    xs = np.linspace(0, 1, n)
    return np.where(xs < 0.7, 0.3 + 0.1 * xs, 0.3 + 0.1 * 0.7 + 2.5 * (xs - 0.7)).clip(0, 1).tolist()


def _make_plateau_fall(n: int = 50) -> list[float]:
    """Mid-plateau then fall (declarative complex)."""
    xs = np.linspace(0, 1, n)
    return np.where(xs < 0.75, 0.7, 0.7 - 2.8 * (xs - 0.75)).clip(0, 1).tolist()


RP_PITCH_TEMPLATES: dict[str, list[float]] = {
    "statement":       _make_falling(),
    "wh_question":     _make_falling(),
    "yes_no_question": _make_rising(),
    "complex":         _make_plateau_fall(),
    "list":            _make_plateau_fall(),
    "exclamation":     _make_falling(),
}


# ---------------------------------------------------------------------------
# Fricative spectral centroid (CoG) reference — RP/SSBE adult male
# Jongman et al. (2000), "Acoustic characteristics of English fricatives",
# JASA 108(3), 1252-1263, Table 2 (male speakers, connected speech values).
# /θ ð/ CoG is highly variable; values below are corpus means.
# /f v/ are broadband; centre is the spectral centre of gravity.
# ---------------------------------------------------------------------------
RP_FRICATIVE_COG_HZ: dict[str, float] = {
    "s":  7000.0,   # Jongman 2000, Table 2 (male English)
    "z":  6500.0,   # Jongman 2000, Table 2
    "ʃ":  3800.0,   # Jongman 2000, Table 2 (postalveolar)
    "ʒ":  3300.0,   # Jongman 2000, Table 2
    "θ":  4500.0,   # Jongman 2000, Table 2 (dental, high variance ±2000 Hz)
    "ð":  3800.0,   # Jongman 2000, Table 2 (dental voiced, high variance)
    "f":  5500.0,   # Jongman 2000, Table 2 (labiodental, broadband)
    "v":  5000.0,   # Jongman 2000, Table 2 (labiodental voiced)
}

# CoG decay constant (Hz) for exponential scoring: score = exp(-|cog - ref| / decay)
RP_FRICATIVE_COG_DECAY_HZ: float = 2000.0

# Threshold above which a /θ/ token sounds like /s/ substitution — TH-fronting marker
# Spec: if CoG > 5500 Hz for /θ/ → /s/ substitution → tongue-placement instruction
RP_TH_S_SUBSTITUTION_THRESHOLD_HZ: float = 5500.0

# ---------------------------------------------------------------------------
# Rhotic /r/ — F3 depression norms (Ladefoged & Johnson 2011, "A Course in
# Phonetics", 7th ed., Ch. 9; Stevens 2000, "Acoustic Phonetics", Ch. 8)
# English retroflex/bunched /r/: F3 depressed to 1800-2200 Hz.
# Non-English /r/ (tap, trill, uvular): F3 remains at 2400-2800 Hz.
# ---------------------------------------------------------------------------
RP_RHOTIC_F3_RHOTIC_MAX_HZ: float = 2200.0    # F3 ≤ this → English /r/
RP_RHOTIC_F3_NONNATIVE_MIN_HZ: float = 2500.0  # F3 ≥ this → non-rhotic substitution
RP_RHOTIC_F3_TARGET_HZ: float = 1950.0         # centre of English /r/ F3 range
RP_RHOTIC_F3_DECAY_HZ: float = 350.0           # Hz decay for exponential scoring

# ---------------------------------------------------------------------------
# Lateral /l/ — F2 norms for dark vs clear allophone
# Wells (1982), "Accents of English", Vol. 1, p. 259;
# Recasens (1993), "Fonètica i Fonologia", Enciclopèdia Catalana.
# Dark /l/ (syllable-final in RP): F2 ≈ 800-1300 Hz (velarisation lowers F2)
# Clear /l/ (syllable-initial in RP): F2 ≈ 1400-1800 Hz
# Note: RP has categorical dark /l/ in final position only.
# ---------------------------------------------------------------------------
RP_LATERAL_DARK_F2_TARGET_HZ: float = 1050.0   # centre of dark /l/ F2 range
RP_LATERAL_CLEAR_F2_THRESHOLD_HZ: float = 1350.0  # F2 above this in final pos = clear /l/ error
RP_LATERAL_F2_DECAY_HZ: float = 300.0


def get_rp_norms(mean_f0: float) -> dict[str, tuple[float, float]]:
    """Return sex-appropriate F1/F2 table based on estimated speaker f0."""
    # Female modern norms not yet derived (Phase E did not yield usable female corpus).
    # Male path uses modern RP (Fry + Lindsey + BBC-male); female path keeps Deterding
    # until a future pass with female-annotated data.  # TODO(cite): female modern norms
    return RP_VOWEL_F1_F2_MALE_MODERN if mean_f0 < 165 else RP_VOWEL_F1_F2_FEMALE
