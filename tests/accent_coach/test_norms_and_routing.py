"""
Contract tests for RP/GenAm norm tables and accent-routing plumbing.

Tests marked 'BUG:' are expected to FAIL until the named bug is fixed.
Tests without a BUG tag are passing baselines / regression guards.
"""
from __future__ import annotations

import inspect

import pytest

from accent_coach.reference.genam_norms import get_genam_norms
from accent_coach.reference.rp_norms import (
    RP_VOWEL_F1_F2_MALE,
    RP_VOWEL_F1_F2_MALE_LEGACY,
    RP_VOWEL_F1_F2_MALE_MODERN,
)

import re as _re

from accent_coach.pipeline.alignment import _char_timestamps_to_phoneme_instances


def _word_phonemes(word, **kwargs):
    """Word -> phoneme instances via the char-based path (equal char timing).

    The uniform splitter _word_to_phoneme_instances was deleted (it fabricated
    rhythm). Accent routing (BATH/LOT overrides) lives in the char path; these
    routing tests only check phoneme identity, so equal char timing is fine.
    """
    chars = _re.sub(r"[^a-z\']", "", word.lower())
    n = max(1, len(chars))
    char_ts = [(i / n * 0.3, (i + 1) / n * 0.3) for i in range(n)]
    accent = kwargs.get("accent_target") or kwargs.get("accent") or kwargs.get("dialect") or "rp"
    return _char_timestamps_to_phoneme_instances(word, char_ts, 0, accent)


# ---------------------------------------------------------------------------
# RP norm table content
# ---------------------------------------------------------------------------


def test_rp_alias_points_to_modern():
    """RP_VOWEL_F1_F2_MALE must be the modern corpus-derived norms, not Deterding 1997.

    BUG: alias currently points to RP_VOWEL_F1_F2_MALE_LEGACY (Deterding 1997).
    Any code that imports RP_VOWEL_F1_F2_MALE silently uses 30-year-old data.
    Fix: change the alias line to RP_VOWEL_F1_F2_MALE = RP_VOWEL_F1_F2_MALE_MODERN.
    """
    assert RP_VOWEL_F1_F2_MALE is RP_VOWEL_F1_F2_MALE_MODERN, (
        f"RP_VOWEL_F1_F2_MALE points to LEGACY norms "
        f"(TRAP F1={RP_VOWEL_F1_F2_MALE['æ'][0]:.0f} Hz, expected ~545 Hz from modern corpus). "
        "Fix: RP_VOWEL_F1_F2_MALE = RP_VOWEL_F1_F2_MALE_MODERN"
    )


def test_modern_rp_key_vowels_not_deterding():
    """Modern RP norms must differ from Deterding 1997 by the known shifts.

    Phase 0.7 documented: TRAP F1 748→545, FOOT F2 950→1427, THOUGHT F2 700→1138.
    If the modern dict still contains Deterding values, the supersession failed.
    """
    m = RP_VOWEL_F1_F2_MALE_MODERN
    assert m["æ"][0] < 620, f"TRAP F1={m['æ'][0]:.0f} looks like Deterding (748 Hz)"
    assert m["ʊ"][1] > 1300, f"FOOT F2={m['ʊ'][1]:.0f} looks like Deterding (950 Hz)"
    assert m["ɔː"][1] > 1000, f"THOUGHT F2={m['ɔː'][1]:.0f} looks like Deterding (700 Hz)"


def test_rp_lot_centroid_is_not_deterding_stub():
    """RP LOT /ɒ/ must be corpus-measured, not the Deterding 1997 stub (600, 900).

    BUG: RP_VOWEL_F1_F2_MALE_MODERN still has /ɒ/ = (600, 900).
    The LOT override went live in Phase 0.17 but the centroid was never re-measured.
    Fix: run formant extraction on modern_rp_corpus with LOT override active and
    replace (600, 900) with the corpus median.
    """
    f1, f2 = RP_VOWEL_F1_F2_MALE_MODERN["ɒ"]
    assert (f1, f2) != (600.0, 900.0), (
        "LOT /ɒ/ centroid is still the Deterding 1997 stub (600, 900). "
        "Re-measure from modern_rp_corpus now that the LOT override is live in the aligner."
    )


def test_rp_modern_covers_all_vowels():
    """Modern RP norms must cover all IPA vowels reachable through the aligner."""
    from accent_coach.pipeline.alignment import IPA_VOWELS
    # ə (schwa) is legitimately low-weight but must be present
    missing = IPA_VOWELS - set(RP_VOWEL_F1_F2_MALE_MODERN.keys())
    assert not missing, f"Modern RP norms missing phonemes: {missing}"


# ---------------------------------------------------------------------------
# GenAm norm table content
# ---------------------------------------------------------------------------


def test_genam_modern_goose_is_fronted():
    """Modern GenAm GOOSE F2 must be > 1200 Hz (connected-speech fronting, Phase 0.16).

    Hillenbrand 1995 had GOOSE F2=997 Hz — 30-year-old citation-form pronunciation.
    Phase 0.16 corpus measured GOOSE F2≈1301 Hz in modern connected speech.
    """
    ga = get_genam_norms(120.0)
    _, goose_f2 = ga["uː"]
    assert goose_f2 > 1200, (
        f"GOOSE F2={goose_f2:.0f} Hz looks like old Hillenbrand (997 Hz). "
        "Phase 0.16 modern GenAm should be > 1200 Hz."
    )


def test_genam_and_rp_differ_for_diagnostic_vowels():
    """GenAm and RP norms must be substantially different for accent-diagnostic vowels.

    If the norms are too similar the scorer cannot distinguish a GenAm from an RP speaker.
    Key discriminators established in the phase history:
      TRAP F1: GenAm higher (no BATH split) — expected gap > 50 Hz
      GOOSE F2: GenAm lower than RP (different merger history) — expected gap > 100 Hz
    """
    rp = RP_VOWEL_F1_F2_MALE_MODERN
    ga = get_genam_norms(120.0)

    trap_gap = ga["æ"][0] - rp["æ"][0]
    assert trap_gap > 50, (
        f"TRAP F1 gap: GA={ga['æ'][0]:.0f} − RP={rp['æ'][0]:.0f} = {trap_gap:.0f} Hz. "
        "Expected > 50 Hz to discriminate TRAP/BATH split."
    )
    goose_gap = abs(ga["uː"][1] - rp["uː"][1])
    assert goose_gap > 100, (
        f"GOOSE F2 gap: |GA={ga['uː'][1]:.0f} − RP={rp['uː'][1]:.0f}| = {goose_gap:.0f} Hz. "
        "Expected > 100 Hz — GOOSE fronting is a key RP vs GenAm discriminator."
    )


def test_genam_covers_all_vowels():
    """Modern GenAm norms must cover all IPA vowels reachable through the aligner."""
    from accent_coach.pipeline.alignment import IPA_VOWELS
    ga = get_genam_norms(120.0)
    missing = IPA_VOWELS - set(ga.keys())
    assert not missing, f"GenAm norms missing phonemes: {missing}"


# ---------------------------------------------------------------------------
# Accent-target routing through the aligner
# ---------------------------------------------------------------------------


def test_word_to_phoneme_accepts_accent_target():
    """_word_to_phoneme_instances() must accept an accent_target param.

    BUG: no such param — BATH/LOT RP overrides always fire regardless of target dialect.
    Consequence: when evaluating GenAm speech, 'dance'/'last'/'class' get /ɑː/ instead
    of /æ/, adding ~130 Hz F2 error on every BATH word.
    Fix: add accent_target='rp'|'genam' and gate overrides on target == 'rp'.
    """
    from accent_coach.pipeline.alignment import _char_timestamps_to_phoneme_instances

    sig = inspect.signature(_char_timestamps_to_phoneme_instances)
    has_target = any(n in sig.parameters for n in ("accent_target", "accent", "dialect"))
    assert has_target, (
        "_word_to_phoneme_instances() has no accent routing param. "
        "BATH/LOT overrides always fire — GenAm speakers get wrong phoneme labels, "
        "breaking vowel scoring for any non-RP evaluation."
    )


def test_bath_words_use_ae_for_genam_target():
    """BATH words ('dance', 'last', 'class', …) must emit /æ/ for GenAm accent target.

    GenAm has no TRAP/BATH split — both classes use /æ/.
    BUG: currently always emits /ɑː/ for BATH words regardless of target dialect.
    Fix: gate the BATH override on accent_target == 'rp'.
    """
    from accent_coach.pipeline.alignment import IPA_VOWELS, _char_timestamps_to_phoneme_instances  # noqa: F401

    sig = inspect.signature(_char_timestamps_to_phoneme_instances)
    param = next((n for n in ("accent_target", "accent", "dialect") if n in sig.parameters), None)
    if param is None:
        pytest.fail(
            "_word_to_phoneme_instances() has no accent_target param. "
            "Cannot route BATH words to /æ/ for GenAm. Fix Bug 1."
        )
    for word in ("dance", "last", "class", "bath", "after", "path"):
        instances = _word_phonemes(word, **{param: "genam"})
        vowels = [p.phoneme for p in instances if p.phoneme in IPA_VOWELS]
        assert "æ" in vowels, (
            f"'{word}' with GenAm target: expected /æ/, got {vowels}"
        )
        assert "ɑː" not in vowels, (
            f"'{word}' with GenAm target: /ɑː/ must not appear, got {vowels}"
        )


def test_bath_words_still_use_aː_for_rp_target():
    """BATH override must still fire for RP target after accent routing is added.

    Regression guard: adding the GenAm path must not break the RP behaviour.
    """
    from accent_coach.pipeline.alignment import IPA_VOWELS, _char_timestamps_to_phoneme_instances  # noqa: F401

    sig = inspect.signature(_char_timestamps_to_phoneme_instances)
    param = next((n for n in ("accent_target", "accent", "dialect") if n in sig.parameters), None)
    kwargs = {param: "rp"} if param else {}

    for word in ("dance", "last", "class", "bath"):
        instances = _word_phonemes(word, **kwargs)
        vowels = [p.phoneme for p in instances if p.phoneme in IPA_VOWELS]
        assert "ɑː" in vowels, f"'{word}' with RP target: expected /ɑː/, got {vowels}"
        assert "æ" not in vowels, f"'{word}' with RP target: /æ/ must not appear, got {vowels}"


def test_lot_words_use_aː_for_genam_target():
    """LOT words ('lot', 'not', 'hot') must emit /ɑː/ (not /ɒ/) for GenAm target.

    /ɒ/ does not exist in GenAm — the LOT–THOUGHT merger means LOT uses /ɑ/.
    BUG: currently always emits /ɒ/ for LOT words regardless of target dialect.
    Fix: gate the LOT override on accent_target == 'rp'.
    """
    from accent_coach.pipeline.alignment import IPA_VOWELS, _char_timestamps_to_phoneme_instances  # noqa: F401

    sig = inspect.signature(_char_timestamps_to_phoneme_instances)
    param = next((n for n in ("accent_target", "accent", "dialect") if n in sig.parameters), None)
    if param is None:
        pytest.fail(
            "_word_to_phoneme_instances() has no accent_target param. "
            "Cannot route LOT words for GenAm. Fix Bug 1."
        )
    for word in ("lot", "not", "hot", "stop", "box"):
        instances = _word_phonemes(word, **{param: "genam"})
        vowels = [p.phoneme for p in instances if p.phoneme in IPA_VOWELS]
        assert "ɒ" not in vowels, (
            f"'{word}' with GenAm target: /ɒ/ must not appear, got {vowels}"
        )
        assert "ɑː" in vowels, (
            f"'{word}' with GenAm target: expected /ɑː/, got {vowels}"
        )


# ---------------------------------------------------------------------------
# Accent-target routing through the scoring layer
# ---------------------------------------------------------------------------


def test_compare_has_accent_target_param():
    """compare() must accept an accent_target parameter to auto-select RP vs GenAm norms.

    BUG: compare() always calls get_rp_norms() when reference_norms is None.
    Any GenAm speaker evaluated through compare() without pre-computed norms is silently
    scored against RP — the score has no diagnostic meaning.
    Fix: add accent_target='rp'|'genam' and branch accordingly.
    """
    from accent_coach.comparison.scoring import compare

    sig = inspect.signature(compare)
    has_target = any(
        n in sig.parameters
        for n in ("accent_target", "accent", "dialect", "target_accent")
    )
    assert has_target, (
        "compare() has no accent routing param. "
        "GenAm speakers are always scored against RP norms when reference_norms is None. "
        "Fix: add accent_target='rp'|'genam' and route norm selection accordingly."
    )
