"""Unit tests for centroid scoring (accent_coach.pipeline.experiment).

Focus: defensive checks for the bugs we've actually hit (None handling on
sparse phonemes) and edge cases that would silently produce garbage scores.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from accent_coach.pipeline.experiment import (
    build_centroids_from_formants,
    load_baseline_centroids,
    score_against,
)


# ---------------------------------------------------------------------------
# build_centroids_from_formants
# ---------------------------------------------------------------------------

def _write_formants_csv(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "formants.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip_id", "source_label", "phoneme",
                                          "F1", "F2", "voiced_fraction",
                                          "duration_s", "transcript"])
        w.writeheader()
        w.writerows(rows)
    return p


def test_build_centroids_basic(tmp_path):
    csv_path = _write_formants_csv(tmp_path, [
        {"clip_id": "c1", "source_label": "x", "phoneme": "iː",
         "F1": 280, "F2": 2200, "voiced_fraction": "", "duration_s": 0.1, "transcript": ""},
        {"clip_id": "c2", "source_label": "x", "phoneme": "iː",
         "F1": 320, "F2": 2300, "voiced_fraction": "", "duration_s": 0.1, "transcript": ""},
        {"clip_id": "c3", "source_label": "x", "phoneme": "æ",
         "F1": 750, "F2": 1700, "voiced_fraction": "", "duration_s": 0.1, "transcript": ""},
    ])
    centroids = build_centroids_from_formants(csv_path)
    assert "iː" in centroids and "æ" in centroids
    assert centroids["iː"]["f1"] == 300.0
    assert centroids["iː"]["f2"] == 2250.0
    assert centroids["iː"]["n"] == 2
    assert centroids["æ"]["n"] == 1


def test_build_centroids_filters_short_durations(tmp_path):
    """Rows with duration_s < 50 ms must be excluded."""
    csv_path = _write_formants_csv(tmp_path, [
        {"clip_id": "c1", "source_label": "x", "phoneme": "iː",
         "F1": 280, "F2": 2200, "voiced_fraction": "", "duration_s": 0.030, "transcript": ""},
        {"clip_id": "c2", "source_label": "x", "phoneme": "iː",
         "F1": 320, "F2": 2300, "voiced_fraction": "", "duration_s": 0.060, "transcript": ""},
    ])
    centroids = build_centroids_from_formants(csv_path)
    assert centroids["iː"]["n"] == 1
    assert centroids["iː"]["f1"] == 320.0


def test_build_centroids_handles_invalid_rows(tmp_path):
    """Non-numeric F1/F2 or missing duration → row skipped, no crash."""
    csv_path = _write_formants_csv(tmp_path, [
        {"clip_id": "c1", "source_label": "x", "phoneme": "iː",
         "F1": "not-a-number", "F2": 2200, "voiced_fraction": "", "duration_s": 0.1, "transcript": ""},
        {"clip_id": "c2", "source_label": "x", "phoneme": "iː",
         "F1": 320, "F2": 2300, "voiced_fraction": "", "duration_s": 0.1, "transcript": ""},
    ])
    centroids = build_centroids_from_formants(csv_path)
    assert centroids["iː"]["n"] == 1


def test_build_centroids_empty_csv(tmp_path):
    csv_path = _write_formants_csv(tmp_path, [])
    centroids = build_centroids_from_formants(csv_path)
    assert centroids == {}


# ---------------------------------------------------------------------------
# load_baseline_centroids
# ---------------------------------------------------------------------------

def test_load_baseline_returns_nonempty():
    b = load_baseline_centroids()
    assert isinstance(b, dict) and len(b) > 0
    # Every centroid must be a dict of speaker → {f1, f2} or None (sparse phoneme)
    for phoneme, spks in b.items():
        assert isinstance(spks, dict), f"phoneme {phoneme}: spks is not a dict"
        for spk_name, vals in spks.items():
            if vals is None:
                continue  # Sparse-phoneme None is allowed (e.g. /ɒ/)
            assert isinstance(vals, dict), f"{phoneme}.{spk_name}: not a dict, got {type(vals)}"
            if "f1" in vals:
                assert isinstance(vals["f1"], (int, float))
                assert isinstance(vals["f2"], (int, float))


def test_load_baseline_env_override_to_original(tmp_path, monkeypatch):
    """ACCENT_COACH_CENTROIDS_PATH override should force the named file."""
    from accent_coach.pipeline.experiment import BASELINE_CENTROIDS_PATH
    monkeypatch.setenv("ACCENT_COACH_CENTROIDS_PATH", str(BASELINE_CENTROIDS_PATH))
    b = load_baseline_centroids()
    assert isinstance(b, dict) and len(b) > 0


# ---------------------------------------------------------------------------
# score_against — the bug we hit (None handling) + edge cases
# ---------------------------------------------------------------------------

def test_score_against_basic():
    """Identical synth and target should produce near-perfect score."""
    base = load_baseline_centroids()
    synth = {ph: spks["modern_rp"] for ph, spks in base.items()
             if spks.get("modern_rp") is not None}
    r = score_against(synth, "modern_rp", baseline_centroids=base)
    assert "composite" in r and "per_phoneme" in r
    assert r["composite"] >= 95.0, f"identity scoring should be ~100, got {r['composite']}"


def test_score_against_none_target_phoneme():
    """When a phoneme has None for the target speaker, scoring must not crash."""
    base = load_baseline_centroids()
    # /ɒ/ has all-None speakers in baseline — make sure scoring still works
    synth = {ph: spks["modern_rp"] for ph, spks in base.items()
             if spks.get("modern_rp") is not None}
    # Add /ɒ/ to synth so score sees the None target side
    synth["ɒ"] = {"f1": 550.0, "f2": 1100.0, "n": 1}
    r = score_against(synth, "modern_rp", baseline_centroids=base)
    assert r["composite"] >= 0  # didn't crash


def test_score_against_subset_phonemes():
    """Synth with only a few phonemes — should still produce a valid composite."""
    base = load_baseline_centroids()
    synth = {
        "iː": base["iː"]["modern_rp"],
        "æ":  base["æ"]["modern_rp"],
    }
    r = score_against(synth, "modern_rp", baseline_centroids=base)
    assert 0 <= r["composite"] <= 100
    # Phonemes present in synth must score ~perfectly (identity vs target)
    assert r["per_phoneme"].get("iː", 0) >= 95
    assert r["per_phoneme"].get("æ",  0) >= 95


def test_score_against_missing_target_speaker():
    """If target speaker is missing entirely, behavior should not silently return junk."""
    base = load_baseline_centroids()
    synth = {ph: spks["modern_rp"] for ph, spks in base.items()
             if spks.get("modern_rp") is not None}
    r = score_against(synth, "this_speaker_does_not_exist", baseline_centroids=base)
    # No target data → no per-phoneme scores → composite should be 0 or NaN-handled, not crash
    assert "composite" in r
    assert r["per_phoneme"] == {} or all(s == 0 for s in r["per_phoneme"].values())


def test_score_against_phoneme_not_in_synth():
    """Phonemes present in target but not synth: per_phoneme reports the target's
    inventory; absent-synth phonemes get a penalty score (not crash)."""
    base = load_baseline_centroids()
    synth = {"iː": base["iː"]["modern_rp"]}
    r = score_against(synth, "modern_rp", baseline_centroids=base)
    # /iː/ is in synth → should score near-perfect
    assert r["per_phoneme"].get("iː", 0) >= 95
    # All per-phoneme scores in valid range, no crash on absent phonemes
    for ph, s in r["per_phoneme"].items():
        assert 0 <= s <= 100, f"phoneme {ph} score {s} out of range"


def test_score_against_does_not_mutate_baseline():
    """score_against should not mutate the caller's baseline_centroids."""
    base = load_baseline_centroids()
    snapshot = json.dumps(base, sort_keys=True, default=str)
    synth = {ph: spks["modern_rp"] for ph, spks in base.items()
             if spks.get("modern_rp") is not None}
    score_against(synth, "modern_rp", baseline_centroids=base)
    assert json.dumps(base, sort_keys=True, default=str) == snapshot, "baseline was mutated"


def test_score_against_score_range():
    """Composite must be in [0, 100], per-phoneme scores in [0, 100]."""
    base = load_baseline_centroids()
    synth = {ph: spks["modern_rp"] for ph, spks in base.items()
             if spks.get("modern_rp") is not None}
    r = score_against(synth, "fry", baseline_centroids=base)
    assert 0 <= r["composite"] <= 100
    for ph, s in r["per_phoneme"].items():
        assert 0 <= s <= 100, f"phoneme {ph} score {s} out of [0, 100]"


def test_score_against_distance_monotonicity():
    """Larger F1/F2 deviation from target should produce lower scores."""
    base = load_baseline_centroids()
    target_vals = base["iː"]["modern_rp"]
    # 'Perfect' synth — same as target
    synth_perfect = {"iː": dict(target_vals)}
    # 'Far' synth — push F1 by +300 Hz, F2 by -400 Hz
    synth_far = {"iː": {"f1": target_vals["f1"] + 300,
                        "f2": target_vals["f2"] - 400,
                        "n": 1}}
    r_perfect = score_against(synth_perfect, "modern_rp", baseline_centroids=base)
    r_far     = score_against(synth_far,     "modern_rp", baseline_centroids=base)
    assert r_perfect["per_phoneme"]["iː"] > r_far["per_phoneme"]["iː"], \
        "perfect synth should outscore far synth"


# ---------------------------------------------------------------------------
# Regression — the exact /ɒ/ bug from Phase 0.12
# ---------------------------------------------------------------------------

def test_regression_aggregate_modern_rp_with_none_phoneme():
    """Phase 0.12 bug: aggregate_modern_rp crashed on phonemes where one source
    speaker had None. Verify the fix handles None gracefully."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "rebuild",
        Path(__file__).parent.parent.parent / "scripts/accent_coach_phase0_12_rebuild.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    lindsey = {"iː": {"f1": 280, "f2": 2200, "n": 10}}
    fry     = {"iː": {"f1": 320, "f2": 2300, "n": 20}}
    bbc     = {"iː": {"f1": 300, "f2": 2250, "n": 5},
               "ɒ":  None}  # the bug case
    out = mod.aggregate_modern_rp(lindsey, fry, bbc)
    assert "iː" in out
    assert out["iː"]["f1"] == round((280 + 320 + 300) / 3, 1)
    # /ɒ/ should be excluded (only 1 valid speaker, fewer than threshold of 2)
    assert "ɒ" not in out


def test_regression_aggregate_with_only_one_valid_speaker():
    """If only 1 speaker has data for a phoneme, aggregate must skip it (needs ≥2)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "rebuild",
        Path(__file__).parent.parent.parent / "scripts/accent_coach_phase0_12_rebuild.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    lindsey = {"æ": {"f1": 750, "f2": 1700, "n": 5}}
    fry     = {}  # no data
    bbc     = {}  # no data
    out = mod.aggregate_modern_rp(lindsey, fry, bbc)
    assert "æ" not in out, "phoneme with only 1 valid speaker should be excluded"


@pytest.mark.parametrize("missing_value", [None, {}, {"f1": 100}])  # None | empty | malformed
def test_regression_aggregate_robust_to_malformed(missing_value):
    """Aggregate must not crash on None / empty / partial-dict speaker data."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "rebuild",
        Path(__file__).parent.parent.parent / "scripts/accent_coach_phase0_12_rebuild.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    lindsey = {"iː": {"f1": 280, "f2": 2200, "n": 10}}
    fry     = {"iː": {"f1": 320, "f2": 2300, "n": 20}}
    bbc     = {"iː": missing_value}
    out = mod.aggregate_modern_rp(lindsey, fry, bbc)
    # Should not crash; iː should still aggregate from the 2 valid speakers
    assert "iː" in out
