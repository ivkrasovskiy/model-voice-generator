"""Centroid building, loading, and scoring."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

from accent_coach.comparison.vowels import score_vowels_piecewise
from accent_coach.diagnostics.bark_distance import bark_transform
from accent_coach.diagnostics.cluster_eval import per_phoneme_sigma_rp
from accent_coach.reference.rp_norms import RP_VOWEL_F1_F2_MALE_LEGACY

PROJECT_ROOT = Path(__file__).parent.parent.parent
BASELINE_CENTROIDS_PATH = (
    PROJECT_ROOT / "tts_output/accent_coach/bench/phase0_7/speaker_centroids.json"
)
CLEANED_CENTROIDS_PATH = (
    PROJECT_ROOT / "tts_output/accent_coach/cleaned_corpus/speaker_centroids_cleaned.json"
)


def build_centroids_from_formants(
    formants_csv: Path,
    min_duration_s: float = 0.050,
) -> dict[str, dict]:
    """Per-phoneme mean F1/F2 from formant CSV (duration >= min_duration_s filter)."""
    rows: list[dict] = []
    with Path(formants_csv).open() as f:
        for row in csv.DictReader(f):
            try:
                if float(row["duration_s"]) >= min_duration_s:
                    rows.append(row)
            except (ValueError, KeyError):
                continue

    by_phoneme: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        ph = row["phoneme"]
        try:
            f1, f2 = float(row["F1"]), float(row["F2"])
        except (ValueError, KeyError):
            continue
        by_phoneme.setdefault(ph, []).append((f1, f2))

    centroids: dict[str, dict] = {}
    for ph, pairs in by_phoneme.items():
        centroids[ph] = {
            "f1": round(float(np.mean([p[0] for p in pairs])), 1),
            "f2": round(float(np.mean([p[1] for p in pairs])), 1),
            "n":  len(pairs),
        }
    return centroids


def load_baseline_centroids() -> dict:
    """Load speaker centroids + add deterding pseudo-speaker.

    Precedence: Phase 0.12 cleaned overlay if present, else Phase 0.7 frozen
    baseline. Override via env var ACCENT_COACH_CENTROIDS_PATH (absolute path).
    """
    override = os.environ.get("ACCENT_COACH_CENTROIDS_PATH")
    if override:
        path = Path(override)
    elif CLEANED_CENTROIDS_PATH.exists():
        path = CLEANED_CENTROIDS_PATH
    else:
        path = BASELINE_CENTROIDS_PATH
    with path.open() as f:
        raw = json.load(f)
    for phoneme, f1f2 in RP_VOWEL_F1_F2_MALE_LEGACY.items():
        if phoneme not in raw:
            raw[phoneme] = {}
        raw[phoneme]["deterding"] = {"f1": f1f2[0], "f2": f1f2[1], "n": 0}
    return raw


def aggregate_modern_rp(lindsey_cent: dict, fry_cent: dict, bbc_cent: dict) -> dict:
    """Mean of lindsey + fry + bbc per phoneme (≥2 speakers required)."""
    all_phonemes = set(lindsey_cent) | set(fry_cent) | set(bbc_cent)
    aggregated = {}
    for ph in all_phonemes:
        f1s, f2s, ns = [], [], 0
        for cent in [lindsey_cent, fry_cent, bbc_cent]:
            v = cent.get(ph)
            if v is None or "f1" not in v or "f2" not in v:
                continue
            f1s.append(v["f1"])
            f2s.append(v["f2"])
            ns += v.get("n", 0)
        if len(f1s) >= 2:
            aggregated[ph] = {
                "f1": round(sum(f1s) / len(f1s), 1),
                "f2": round(sum(f2s) / len(f2s), 1),
                "n":  ns,
            }
    return aggregated


def score_against(
    synth_centroids: dict[str, dict],
    target: str,
    baseline_centroids: dict | None = None,
) -> dict:
    """Apply bark_transform, compute sigma_rp, score synth_BC vs target.

    sigma_rp is always derived from {fry, lindsey, bbc_male} regardless of target.
    Returns {'composite': float, 'per_phoneme': {phoneme: score}}.
    """
    if baseline_centroids is None:
        baseline_centroids = load_baseline_centroids()

    centroids: dict[str, dict] = {ph: dict(spks) for ph, spks in baseline_centroids.items()}
    for phoneme, vals in synth_centroids.items():
        if phoneme not in centroids:
            centroids[phoneme] = {}
        centroids[phoneme]["synth_BC"] = vals

    bark_centroids = bark_transform(centroids)
    sigma_rp = per_phoneme_sigma_rp(bark_centroids)

    target_f1f2 = {
        ph: (spks[target]["f1"], spks[target]["f2"])
        for ph, spks in bark_centroids.items()
        if spks.get(target)
    }
    synth_f1f2 = {
        ph: (spks["synth_BC"]["f1"], spks["synth_BC"]["f2"])
        for ph, spks in bark_centroids.items()
        if spks.get("synth_BC")
    }

    scores = score_vowels_piecewise(synth_f1f2, target_f1f2, sigma_rp)
    return {"composite": scores["composite"], "per_phoneme": scores["per_phoneme"]}
