"""Unit tests for accent_coach.diagnostics.coach_metrics (Phase 0.15)."""
from __future__ import annotations

import numpy as np
import pytest

from accent_coach.diagnostics.coach_metrics import (
    Token,
    bhattacharyya_overlap,
    bootstrap_ci,
    ci_separated,
    dispersion,
    get_norms,
    per_token_distances,
    rhoticity,
    score_source,
)


def _tok(ph: str, f1: float, f2: float, f3: float | None = None, nxt: str = "") -> Token:
    return Token(phoneme=ph, next_phoneme=nxt, f1=f1, f2=f2, f3=f3)


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    def test_constant_array_zero_width(self):
        mean, lo, hi = bootstrap_ci(np.full(50, 3.0))
        assert mean == pytest.approx(3.0)
        assert lo == pytest.approx(3.0)
        assert hi == pytest.approx(3.0)

    def test_single_value(self):
        assert bootstrap_ci(np.array([2.5])) == (2.5, 2.5, 2.5)

    def test_empty(self):
        mean, lo, hi = bootstrap_ci(np.array([]))
        assert np.isnan(mean) and np.isnan(lo) and np.isnan(hi)

    def test_ci_brackets_mean_and_is_ordered(self):
        rng = np.random.default_rng(1)
        mean, lo, hi = bootstrap_ci(rng.normal(10, 2, 500))
        assert lo < mean < hi
        assert mean == pytest.approx(10, abs=0.3)


# ---------------------------------------------------------------------------
# per_token_distances / dispersion
# ---------------------------------------------------------------------------

class TestDistances:
    def test_token_on_target_is_zero(self):
        d = per_token_distances([_tok("æ", 545, 1496)], (545, 1496))
        assert d[0] == pytest.approx(0.0, abs=1e-9)

    def test_empty_tokens(self):
        assert per_token_distances([], (500, 1500)).size == 0

    def test_dispersion_zero_for_identical(self):
        toks = [_tok("æ", 500, 1500) for _ in range(5)]
        assert dispersion(toks) == pytest.approx(0.0, abs=1e-9)

    def test_dispersion_positive_for_spread(self):
        toks = [_tok("æ", 500, 1500), _tok("æ", 700, 1900)]
        assert dispersion(toks) > 0

    def test_dispersion_nan_single_token(self):
        assert np.isnan(dispersion([_tok("æ", 500, 1500)]))


# ---------------------------------------------------------------------------
# bhattacharyya_overlap
# ---------------------------------------------------------------------------

class TestOverlap:
    def test_identical_distributions_overlap_near_one(self):
        rng = np.random.default_rng(0)
        cloud = [_tok("a", 500 + rng.normal(0, 20), 1500 + rng.normal(0, 40))
                 for _ in range(40)]
        cloud_b = [_tok("b", t["f1"], t["f2"]) for t in cloud]
        assert bhattacharyya_overlap(cloud, cloud_b) == pytest.approx(1.0, abs=0.05)

    def test_far_apart_distributions_low_overlap(self):
        rng = np.random.default_rng(2)
        a = [_tok("a", 500 + rng.normal(0, 15), 1500 + rng.normal(0, 15)) for _ in range(40)]
        b = [_tok("b", 800 + rng.normal(0, 15), 1100 + rng.normal(0, 15)) for _ in range(40)]
        assert bhattacharyya_overlap(a, b) < 0.1

    def test_too_few_tokens_nan(self):
        assert np.isnan(bhattacharyya_overlap([_tok("a", 500, 1500)], [_tok("b", 800, 1100)]))


# ---------------------------------------------------------------------------
# rhoticity (the RP↔GenAm separator)
# ---------------------------------------------------------------------------

class TestRhoticity:
    def test_non_rhotic_high_f3(self):
        toks = [_tok("ɜː", 480, 1440, 2450) for _ in range(8)]
        r = rhoticity(toks)
        assert r["verdict"] == "non_rhotic"
        assert r["f3_hz"] == pytest.approx(2450, abs=1)

    def test_rhotic_low_f3(self):
        toks = [_tok("ɜː", 474, 1379, 1700) for _ in range(8)]
        assert rhoticity(toks)["verdict"] == "rhotic"

    def test_none_when_no_f3(self):
        assert rhoticity([_tok("ɜː", 480, 1440, None)]) is None

    def test_none_when_no_nurse(self):
        # plain TRAP with no following /r/ is not a rhotic context
        assert rhoticity([_tok("æ", 545, 1496, 2500)]) is None

    def test_pre_r_vowels_pooled(self):
        # START (ɑ before r) with low r-coloured F3 counts as rhotic context,
        # even with zero NURSE tokens (Phase 0.16 broadening)
        toks = [_tok("ɑː", 700, 1100, 1650, nxt="r") for _ in range(6)]
        r = rhoticity(toks)
        assert r is not None and r["n"] == 6
        assert r["verdict"] == "rhotic"

    def test_single_phoneme_override(self):
        # explicit phoneme= keeps the old NURSE-only behaviour
        toks = [_tok("ɑː", 700, 1100, 1650, nxt="r"), _tok("ɜː", 480, 1440, 2450)]
        assert rhoticity(toks, phoneme="ɜː")["n"] == 1


# ---------------------------------------------------------------------------
# get_norms / score_source / ci_separated
# ---------------------------------------------------------------------------

class TestNormsAndScoring:
    def test_rp_and_genam_norms_differ(self):
        rp = get_norms("rp", 110.0)
        ga = get_norms("genam", 110.0)
        # GenAm BATH (ɑː) is much more open/back than RP BATH
        assert rp["ɑː"] != ga["ɑː"]

    def test_unknown_target_raises(self):
        with pytest.raises(ValueError):
            get_norms("scottish", 110.0)

    def test_score_source_structure(self):
        toks = ([_tok("ɑː", 518, 1215) for _ in range(10)]
                + [_tok("æ", 545, 1496) for _ in range(10)]
                + [_tok("ɜː", 482, 1440, 2450) for _ in range(6)])
        out = score_source(toks, "rp", 110.0)
        assert out["target"] == "rp"
        assert set(out["overall"]) == {"mean", "lo", "hi"}
        assert "ɑː" in out["per_vowel"]
        assert out["per_vowel"]["ɑː"]["mean"] == pytest.approx(0.0, abs=0.05)
        assert out["rhoticity"]["verdict"] == "non_rhotic"

    def test_ci_separated(self):
        a = {"mean": 1.0, "lo": 0.8, "hi": 1.2}
        b = {"mean": 2.0, "lo": 1.8, "hi": 2.2}
        c = {"mean": 1.3, "lo": 1.1, "hi": 1.5}
        assert ci_separated(a, b) is True
        assert ci_separated(a, c) is False
