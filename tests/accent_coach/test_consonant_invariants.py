"""Top-level business-logic invariants for consonant scoring (corpus-driven).

These are the guardrails mandated by CLAUDE.md "Accent coach quality rules":

  * one **owner-is-strictly-lowest** test per sub-score (N sub-scores → N tests),
  * a composite-ordering test, and
  * a native > TTS > owner direction test.

Unlike ``test_consonants_quality.py`` (synthetic, per-item — necessary but unable
to catch a bench-level ordering inversion), these run the *real corpus* through
the same path as ``scripts/bench/accent_coach_consonant_bench.py`` and assert the
ordering that actually matters for coaching.

They are **heavy** (WhisperX / MMS forced alignment per clip), so they are gated
behind ``RUN_CONSONANT_BENCH=1`` and ``skip`` — never silently pass — when either
the flag is unset or the corpus audio is absent.  A change that inverts any
ordering must fail here; that is the whole point.

    RUN_CONSONANT_BENCH=1 uv run pytest tests/accent_coach/test_consonant_invariants.py -q
    RUN_CONSONANT_BENCH=1 CONSONANT_BENCH_N=8 uv run pytest ... # more clips/group

Currently EXPECTED TO FAIL on ``fricative`` until comparison mode (§1D) or
char-aligned boundaries (§2A) land — the absolute fricative path cannot separate
owner from natives because they share the same G2P alignment artefact.  That red
is the documented state, not a flaky test.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# The bench module is the single source of truth for how a clip is scored.
from scripts.bench.accent_coach_consonant_bench import (  # noqa: E402
    GROUPS,
    bench_group,
)
from scripts.lib.manifest import load_manifest  # noqa: E402

_ENABLED = os.environ.get("RUN_CONSONANT_BENCH") == "1"
_N = int(os.environ.get("CONSONANT_BENCH_N", "6"))
_ACCENT = os.environ.get("CONSONANT_BENCH_ACCENT", "rp")

# Sub-scores that must obey owner-is-lowest, one parametrised test each.
_SUBSCORES = ["fricative", "stop_vot", "rhotic", "lateral"]

# Native groups that must outscore the owner (the core coaching invariant).
_NATIVE_LABELS = {"rp_fry", "rp_lindsey", "genam_harris"}


def _available_groups() -> list[tuple[str, Path, str]]:
    return [(label, path, fam) for label, path, fam in GROUPS if path.exists()]


_SKIP_REASON: str | None = None
if not _ENABLED:
    _SKIP_REASON = "set RUN_CONSONANT_BENCH=1 to run corpus invariant tests (heavy: forced alignment)"
elif not _available_groups():
    _SKIP_REASON = "corpus audio/manifests absent"
elif "owner" not in {label for label, _, _ in _available_groups()}:
    _SKIP_REASON = "owner corpus absent — cannot assert owner-is-lowest"

pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")


# Score every group once; share across all tests in this module (alignment is slow).
_SCORES_CACHE: dict[str, dict[str, float]] | None = None


def _group_scores() -> dict[str, dict[str, float]]:
    """{label: {subscore: mean, ...}} for every available group. Cached."""
    global _SCORES_CACHE
    if _SCORES_CACHE is not None:
        return _SCORES_CACHE

    random.seed(42)
    out: dict[str, dict[str, float]] = {}
    for label, manifest_path, _family in _available_groups():
        entries = load_manifest(manifest_path)
        agg = bench_group(
            label, entries, manifest_path,
            n=_N, verbose=False, accent_target=_ACCENT,
        )
        if agg:
            out[label] = agg
    _SCORES_CACHE = out
    return out


def _owner_and_others(subscore: str) -> tuple[float, dict[str, float]]:
    """Return (owner_value, {label: value}) for a sub-score, skipping NaNs."""
    scores = _group_scores()
    import math

    vals = {
        label: agg[subscore]
        for label, agg in scores.items()
        if subscore in agg and not math.isnan(agg[subscore])
    }
    if "owner" not in vals:
        pytest.skip(f"owner has no scorable '{subscore}' tokens in this sample")
    owner = vals.pop("owner")
    if not vals:
        pytest.skip(f"no comparison groups have '{subscore}' tokens in this sample")
    return owner, vals


@pytest.mark.parametrize("subscore", _SUBSCORES)
def test_owner_is_strictly_lowest_per_subscore(subscore: str) -> None:
    """Owner must score strictly below every other group on each sub-score.

    One test per sub-score (CLAUDE.md: N sub-scores → N invariant tests).
    A metric that grades the owner at or above any native/TTS group is broken.
    """
    owner, others = _owner_and_others(subscore)
    offenders = {label: v for label, v in others.items() if v <= owner}
    assert not offenders, (
        f"[{subscore}] owner={owner:.1f} is NOT strictly lowest. "
        f"Groups at/below owner: { {k: round(v, 1) for k, v in offenders.items()} }. "
        f"All groups: { {k: round(v, 1) for k, v in {**others, 'owner': owner}.items()} }"
    )


@pytest.mark.parametrize("subscore", _SUBSCORES)
def test_natives_beat_owner_per_subscore(subscore: str) -> None:
    """Every native group present must outscore the owner on each sub-score."""
    owner, others = _owner_and_others(subscore)
    natives = {k: v for k, v in others.items() if k in _NATIVE_LABELS}
    if not natives:
        pytest.skip(f"no native group has scorable '{subscore}' tokens in this sample")
    failing = {k: v for k, v in natives.items() if v <= owner}
    assert not failing, (
        f"[{subscore}] natives must beat owner={owner:.1f}; "
        f"these did not: { {k: round(v, 1) for k, v in failing.items()} }"
    )


def test_composite_owner_is_lowest() -> None:
    """Composite: owner must be the single lowest group (inviolable per CLAUDE.md)."""
    owner, others = _owner_and_others("composite")
    offenders = {label: v for label, v in others.items() if v <= owner}
    assert not offenders, (
        f"[composite] owner={owner:.1f} is not lowest. "
        f"At/below owner: { {k: round(v, 1) for k, v in offenders.items()} }"
    )


def test_composite_native_beats_tts_beats_owner() -> None:
    """Direction invariant: mean(native composite) > TTS composite > owner composite."""
    scores = _group_scores()
    if "tts_bc" not in scores or "owner" not in scores:
        pytest.skip("need both tts_bc and owner groups for the direction invariant")
    natives = [scores[lbl]["composite"] for lbl in _NATIVE_LABELS if lbl in scores]
    if not natives:
        pytest.skip("no native group present for the direction invariant")
    native_mean = sum(natives) / len(natives)
    tts = scores["tts_bc"]["composite"]
    owner = scores["owner"]["composite"]
    assert native_mean > tts > owner, (
        f"Expected native({native_mean:.1f}) > TTS({tts:.1f}) > owner({owner:.1f})."
    )
