"""General American (GenAm) acoustic norms.

All numeric constants carry citation comments.  Any value that could not be
sourced from a published table is marked # TODO(cite).

Primary source:
  Hillenbrand et al. (1995), "Acoustic characteristics of American English
  vowels", JASA 97(5), 3099-3111, Table II (men) and Table III (women).
  Values are mean F1/F2 at the steady-state measurement point (t=50% for
  monophthongs; onset nucleus for diphthongs EY, OW, AY, AW, OY).

GenAm mapping notes:
  /ɒ/ does not exist in GenAm (LOT–THOUGHT merger: LOT → /ɑ/); mapped to
  the AA (ɑː) values.
  /ɜː/ in RP corresponds to the rhotic /ɝ/ (ER) in GenAm; ER values used.
  /ə/ (schwa) is not separately tabulated by Hillenbrand 1995; approximated
  from the AH (ʌ) column — typical unstressed schwa is centralized relative
  to stressed ʌ.  # TODO(cite): no direct Hillenbrand table entry.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Men — Hillenbrand et al. 1995, Table II (mean F1/F2, Hz)
# ---------------------------------------------------------------------------
_GENAM_MALE: dict[str, tuple[float, float]] = {
    "iː": (342, 2322),   # FLEECE  / IY — Hillenbrand 1995 Table II
    "ɪ":  (427, 2034),   # KIT     / IH — Hillenbrand 1995 Table II
    "ɛ":  (580, 1799),   # DRESS   / EH — Hillenbrand 1995 Table II
    "æ":  (588, 1952),   # TRAP    / AE — Hillenbrand 1995 Table II
    "ɑː": (768, 1333),   # BATH/LOT/PALM / AA — Hillenbrand 1995 Table II
    "ɒ":  (768, 1333),   # LOT→AA (LOT–THOUGHT merger in GenAm; same as ɑː)
    "ɔː": (652,  997),   # THOUGHT / AO — Hillenbrand 1995 Table II (pre-merger speakers)
    "ʊ":  (469, 1122),   # FOOT    / UH — Hillenbrand 1995 Table II
    "uː": (378,  997),   # GOOSE   / UW — Hillenbrand 1995 Table II
    "ʌ":  (623, 1200),   # STRUT   / AH — Hillenbrand 1995 Table II
    "ɜː": (474, 1379),   # NURSE→ɝ / ER (rhotic) — Hillenbrand 1995 Table II
    "ə":  (623, 1200),   # SCHWA — approximated from AH; # TODO(cite): no direct entry in Hillenbrand 1995
    # Diphthongs: onglide (onset) values
    "eɪ": (476, 2089),   # FACE   / EY onset — Hillenbrand 1995 Table II
    "aɪ": (727, 1184),   # PRICE  / AY onset — Hillenbrand 1995 Table II
    "ɔɪ": (652,  997),   # CHOICE / OY onset — approximated from AO; # TODO(cite)
    "əʊ": (497,  910),   # GOAT   / OW onset — Hillenbrand 1995 Table II
    "aʊ": (762, 1186),   # MOUTH  / AW onset — Hillenbrand 1995 Table II
}

# ---------------------------------------------------------------------------
# Women — Hillenbrand et al. 1995, Table III (mean F1/F2, Hz)
# ---------------------------------------------------------------------------
_GENAM_FEMALE: dict[str, tuple[float, float]] = {
    "iː": (437, 2761),   # FLEECE  / IY — Hillenbrand 1995 Table III
    "ɪ":  (483, 2365),   # KIT     / IH — Hillenbrand 1995 Table III
    "ɛ":  (731, 2058),   # DRESS   / EH — Hillenbrand 1995 Table III
    "æ":  (669, 2349),   # TRAP    / AE — Hillenbrand 1995 Table III
    "ɑː": (936, 1551),   # BATH/LOT/PALM / AA — Hillenbrand 1995 Table III
    "ɒ":  (936, 1551),   # LOT→AA (LOT–THOUGHT merger; same as ɑː)
    "ɔː": (781, 1136),   # THOUGHT / AO — Hillenbrand 1995 Table III
    "ʊ":  (519, 1225),   # FOOT    / UH — Hillenbrand 1995 Table III
    "uː": (459, 1105),   # GOOSE   / UW — Hillenbrand 1995 Table III
    "ʌ":  (753, 1426),   # STRUT   / AH — Hillenbrand 1995 Table III
    "ɜː": (523, 1588),   # NURSE→ɝ / ER (rhotic) — Hillenbrand 1995 Table III
    "ə":  (753, 1426),   # SCHWA — approximated from AH; # TODO(cite): no direct entry in Hillenbrand 1995
    # Diphthongs: onglide (onset) values
    "eɪ": (536, 2530),   # FACE   / EY onset — Hillenbrand 1995 Table III
    "aɪ": (860, 1551),   # PRICE  / AY onset — Hillenbrand 1995 Table III
    "ɔɪ": (781, 1136),   # CHOICE / OY onset — approximated from AO; # TODO(cite)
    "əʊ": (555, 1035),   # GOAT   / OW onset — Hillenbrand 1995 Table III
    "aʊ": (860, 1551),   # MOUTH  / AW onset — Hillenbrand 1995 Table III
}


# ---------------------------------------------------------------------------
# Men — MODERN connected-speech GenAm (Phase 0.16)
# Pooled MEDIAN F1/F2 from tts_output/genam_lecture_corpus (Huberman + Harris +
# Sapolsky, 3 modern male GA speakers, dur>=50ms), measured by the SAME pipeline
# and measurement point as RP_VOWEL_F1_F2_MALE_MODERN — steady-state for
# diphthongs (NOT Hillenbrand's onset nucleus), connected speech (not /hVd/).
# Supersedes _GENAM_MALE (Hillenbrand 1995) which is citation-form + 30 yr stale
# (archaic un-fronted GOOSE, onset-convention diphthongs). See
# docs/accent_coach_phase0_16_results.md.  Derived by
# scripts/accent_coach_build_genam_norms.py.
# ---------------------------------------------------------------------------
_GENAM_MALE_MODERN: dict[str, tuple[float, float]] = {
    "iː": (285, 2105),   # FLEECE — n=162
    "ɪ":  (369, 1718),   # KIT — n=505
    "ɛ":  (497, 1534),   # DRESS — n=211
    "æ":  (645, 1561),   # TRAP — n=135
    "ɑː": (556, 1142),   # BATH/PALM/LOT — n=134
    "ɒ":  (556, 1142),   # LOT→ɑː (GenAm LOT–PALM merger; same as ɑː)
    "ɔː": (550, 1009),   # THOUGHT — n=113
    "ʊ":  (399, 1325),   # FOOT — n=31
    "uː": (316, 1301),   # GOOSE — n=165 (FRONTED — modern GA, vs Hillenbrand 997)
    "ʌ":  (505, 1234),   # STRUT — n=105
    "ɜː": (422, 1255),   # NURSE/ɝ — n=42
    "ə":  (393, 1471),   # SCHWA — n=646
    "eɪ": (385, 1965),   # FACE onset — n=176 (steady-state)
    "aɪ": (545, 1550),   # PRICE — n=190 (steady-state, vs Hillenbrand onset)
    "ɔɪ": (550, 1009),   # CHOICE — approx from THOUGHT (sparse); # TODO(cite)
    "əʊ": (429, 1095),   # GOAT — n=116 (steady-state, vs Hillenbrand onset)
    "aʊ": (628, 1191),   # MOUTH — n=63
}


# ---------------------------------------------------------------------------
# GenAm fricative CoG norms — adult male connected speech
# Jongman et al. (2000) Table 2 (American English male column).
# GenAm and RP CoG values differ minimally for /s ʃ/; dentals and labio-
# dentals are near-identical across dialects (place-of-articulation is the
# primary determinant, not dialect).  Values below follow RP where identical.
# ---------------------------------------------------------------------------
GA_FRICATIVE_COG_HZ: dict[str, float] = {
    "s":  7000.0,   # Jongman 2000, Table 2 (American English male)
    "z":  6400.0,   # Jongman 2000, Table 2
    "ʃ":  3700.0,   # Jongman 2000, Table 2 (postalveolar)
    "ʒ":  3200.0,   # Jongman 2000, Table 2
    "θ":  4500.0,   # Jongman 2000, Table 2 (same as RP; place-of-art driven)
    "ð":  3800.0,   # Jongman 2000, Table 2
    "f":  5500.0,   # Jongman 2000, Table 2 (labiodental)
    "v":  5000.0,   # Jongman 2000, Table 2
}

GA_FRICATIVE_COG_DECAY_HZ: float = 2000.0
GA_TH_S_SUBSTITUTION_THRESHOLD_HZ: float = 5500.0

# ---------------------------------------------------------------------------
# GenAm rhotic /r/ — F3 depression (same physics as RP; American English is
# rhotic, so F3 ≤ 2200 Hz in connected speech)
# Ladefoged & Johnson 2011, Ch. 9.
# ---------------------------------------------------------------------------
GA_RHOTIC_F3_RHOTIC_MAX_HZ: float = 2200.0
GA_RHOTIC_F3_NONNATIVE_MIN_HZ: float = 2500.0
GA_RHOTIC_F3_TARGET_HZ: float = 1900.0   # GenAm /r/ slightly more depressed (Hagiwara 1995)
GA_RHOTIC_F3_DECAY_HZ: float = 350.0

# ---------------------------------------------------------------------------
# GenAm lateral /l/ — dark /l/ is more prevalent throughout the syllable in
# GenAm than in RP; F2 of dark /l/ is similar or slightly lower than RP.
# Wells 1982, Vol. 3, p. 489; Sproat & Fujimura 1993.
# ---------------------------------------------------------------------------
GA_LATERAL_DARK_F2_TARGET_HZ: float = 1000.0    # slightly darker than RP (Wells 1982)
GA_LATERAL_CLEAR_F2_THRESHOLD_HZ: float = 1350.0
GA_LATERAL_CLEAR_F2_TARGET_HZ: float = 1500.0  # GA clear /l/ slightly darker than RP (Wells 1982)
GA_LATERAL_F2_DECAY_HZ: float = 300.0

# ---------------------------------------------------------------------------
# GenAm VOT — similar to RP; Cho & Ladefoged (1999) "Variations and
# universals in VOT: evidence from 18 languages", J. Phonetics 27(2).
# English /p t k/ initial VOT slightly shorter than RP by ~5 ms.
# ---------------------------------------------------------------------------
GA_VOT_MEAN_MS: dict[str, float] = {
    "p": 60.0,   # Cho & Ladefoged 1999 (English); RP = 67.5
    "t": 70.0,   # Cho & Ladefoged 1999; RP = 77.5
    "k": 80.0,   # Cho & Ladefoged 1999; RP = 87.5
}
GA_VOT_SD_MS: dict[str, float] = {
    "p": 10.0,
    "t": 10.0,
    "k": 10.0,
}


def get_genam_norms(mean_f0: float) -> dict[str, tuple[float, float]]:
    """Return sex-appropriate GenAm F1/F2 table based on estimated speaker f0.

    Male path uses MODERN connected-speech norms (Phase 0.16); female path keeps
    Hillenbrand 1995 until a female GA corpus exists (mirrors get_rp_norms).
    Mirrors the signature of get_rp_norms in rp_norms.py.
    """
    return _GENAM_MALE_MODERN if mean_f0 < 165 else _GENAM_FEMALE
