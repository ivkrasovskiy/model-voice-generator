"""Coach-grade reliability metrics on per-token vowel formants (Phase 0.15).

Unlike the centroid-collapse metric in scripts/score_rp_all.py, every function
here operates on *per-token* formant rows, so the coach can report what actually
matters for a learner:

  - bootstrap CIs on distance-to-target  → kills sub-noise comparisons
  - within-category dispersion (Bark)    → L2 vowels are diffuse, RP tight
  - minimal-pair overlap (Bhattacharyya) → the real learner confusion (BATH↔TRAP)
  - F3 rhoticity (NURSE F3 + Z3−Z2)      → the ONLY axis separating RP from GenAm

Targets: "rp" (rp_norms.get_rp_norms) or "genam" (genam_norms.get_genam_norms).
F1/F2 distances are computed in Bark (Zwicker hz_to_bark from bark_distance.py).
Rhoticity uses the Traunmüller Bark in syrdal_gopal (f3_normalization.py).

This module is dependency-light by design: only numpy + the bark/F3 helpers +
the norm tables. No pipeline or alignment imports — it is pure and testable.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from accent_coach.diagnostics.bark_distance import hz_to_bark
from accent_coach.diagnostics.f3_normalization import syrdal_gopal
from accent_coach.reference.genam_norms import get_genam_norms
from accent_coach.reference.rp_norms import get_rp_norms

# Vowels the coach reports on, with human-readable lexical-set names.
FOCUS_VOWELS: list[tuple[str, str]] = [
    ("ɑː", "BATH/PALM"), ("æ", "TRAP"),   ("ɛ", "DRESS"),
    ("əʊ", "GOAT"),       ("ə", "SCHWA"),  ("ɔː", "THOUGHT"),
    ("ɪ",  "KIT"),        ("ɜː", "NURSE"), ("aɪ", "PRICE"),
    ("uː", "GOOSE"),
]

# Minimal pairs whose *overlap* is a learner error (not a centroid offset).
# BATH↔TRAP is the canonical RP marker; an L2 speaker merges them.
MINIMAL_PAIRS: list[tuple[str, str, str]] = [
    ("ɑː", "æ", "BATH↔TRAP"),
    ("ɒ", "ɔː", "LOT↔THOUGHT"),
]

# NURSE F3 (Hz) heuristic thresholds for the rhoticity verdict. Rhotic /ɝ/
# (GenAm) has a dramatically lowered F3; non-rhotic /ɜː/ (RP) keeps it high.
RHOTIC_F3_MAX = 2000.0   # below → rhotic (GenAm-like)
NONRHOTIC_F3_MIN = 2200.0  # above → non-rhotic (RP-like)


# ---------------------------------------------------------------------------
# Token loading
# ---------------------------------------------------------------------------

class Token(dict):
    """A formant token row. Keys: phoneme, next_phoneme, f1, f2, f3 (None if absent)."""


def load_tokens(formants_csv: str | Path, min_duration_s: float = 0.050) -> list[Token]:
    """Read per-token formant rows from a formants CSV (duration-filtered).

    ``next_phoneme`` (added Phase 0.16) is "" for older CSVs — rhoticity then
    falls back to NURSE-only.
    """
    tokens: list[Token] = []
    with Path(formants_csv).open() as fh:
        for row in csv.DictReader(fh):
            try:
                if float(row.get("duration_s", 0.0)) < min_duration_s:
                    continue
                f1, f2 = float(row["F1"]), float(row["F2"])
            except (ValueError, KeyError):
                continue
            f3_raw = row.get("F3", "")
            try:
                f3 = float(f3_raw) if f3_raw not in ("", "nan") else None
            except ValueError:
                f3 = None
            tokens.append(Token(phoneme=row["phoneme"], next_phoneme=row.get("next_phoneme", ""),
                                f1=f1, f2=f2, f3=f3))
    return tokens


def _by_phoneme(tokens: list[Token]) -> dict[str, list[Token]]:
    out: dict[str, list[Token]] = {}
    for t in tokens:
        out.setdefault(t["phoneme"], []).append(t)
    return out


def get_norms(target: str, mean_f0: float) -> dict[str, tuple[float, float]]:
    """Sex-appropriate F1/F2 target table for the chosen accent."""
    if target == "rp":
        return get_rp_norms(mean_f0)
    if target == "genam":
        return get_genam_norms(mean_f0)
    raise ValueError(f"unknown target {target!r} (expected rp|genam)")


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------

def _bark_xy(tokens: list[Token]) -> np.ndarray:
    """N×2 array of (F1, F2) tokens in Bark."""
    return np.array([[hz_to_bark(t["f1"]), hz_to_bark(t["f2"])] for t in tokens])


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0,
) -> tuple[float, float, float]:
    """Return (mean, lo, hi) percentile bootstrap CI of the mean.

    Degenerate inputs (0 or 1 value) return the point estimate for all three.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(values.mean())
    if values.size == 1:
        return (mean, mean, mean)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boot_means = values[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (mean, lo, hi)


def per_token_distances(
    tokens: list[Token], target_f1f2: tuple[float, float],
) -> np.ndarray:
    """Per-token Bark distance from each token to a single target point."""
    if not tokens:
        return np.array([])
    tf1, tf2 = hz_to_bark(target_f1f2[0]), hz_to_bark(target_f1f2[1])
    xy = _bark_xy(tokens)
    return np.sqrt((xy[:, 0] - tf1) ** 2 + (xy[:, 1] - tf2) ** 2)


def dispersion(tokens: list[Token]) -> float:
    """Within-category spread (Bark): RMS distance of tokens from their own mean.

    A tight cluster (native-like) → small; a diffuse cluster (L2) → large.
    """
    if len(tokens) < 2:
        return float("nan")
    xy = _bark_xy(tokens)
    centre = xy.mean(axis=0)
    return float(np.sqrt(((xy - centre) ** 2).sum(axis=1).mean()))


def bhattacharyya_overlap(tokens_a: list[Token], tokens_b: list[Token]) -> float:
    """Overlap of two vowels' bivariate Bark distributions, in [0, 1].

    1.0 = identical distributions (full confusion); ~0 = cleanly separated.
    Computed as exp(-BC) where BC is the Bhattacharyya distance of two 2-D
    Gaussians fit to the token clouds. Needs ≥2 tokens per vowel.
    """
    if len(tokens_a) < 2 or len(tokens_b) < 2:
        return float("nan")
    xa, xb = _bark_xy(tokens_a), _bark_xy(tokens_b)
    mu_a, mu_b = xa.mean(axis=0), xb.mean(axis=0)
    cov_a = np.cov(xa, rowvar=False) + np.eye(2) * 1e-6
    cov_b = np.cov(xb, rowvar=False) + np.eye(2) * 1e-6
    cov = 0.5 * (cov_a + cov_b)
    diff = mu_a - mu_b
    try:
        inv = np.linalg.inv(cov)
        term1 = 0.125 * float(diff @ inv @ diff)
        term2 = 0.5 * np.log(
            np.linalg.det(cov) / np.sqrt(np.linalg.det(cov_a) * np.linalg.det(cov_b))
        )
    except np.linalg.LinAlgError:
        return float("nan")
    bc = term1 + term2
    return float(np.exp(-max(bc, 0.0)))


def is_rhotic_context(t: Token) -> bool:
    """True if the token is an r-coloured context: NURSE (ɜː/ER) or any vowel
    immediately followed by /r/ (START, NORTH, NEAR, lettER…). The pre-/r/ vowels
    only survive when extracted with target≠rp (rp strips coda-R), which is exactly
    when we want to *measure* rhoticity (genam / owner-raw)."""
    return t["phoneme"] == "ɜː" or t.get("next_phoneme") == "r"


def rhoticity(tokens: list[Token], phoneme: str | None = None) -> dict | None:
    """F3-based rhoticity — the RP↔GenAm separator.

    Pools all r-coloured contexts (NURSE + pre-/r/ vowels; Phase 0.16 broadening,
    was NURSE-only and too sparse at n≈7). r-colouring lowers F3, so a low pooled
    F3 ⇒ rhotic (GenAm), high ⇒ non-rhotic (RP). Returns mean F3 (Hz) with CI, mean
    Syrdal-Gopal Z3−Z2 (Bark) with CI, token count, verdict. None if no F3 data.

    ``phoneme`` overrides the selection to a single phoneme (kept for tests/compat).
    """
    if phoneme is not None:
        sel = [t for t in tokens if t["phoneme"] == phoneme and t["f3"] is not None]
    else:
        sel = [t for t in tokens if is_rhotic_context(t) and t["f3"] is not None]
    if not sel:
        return None
    f3 = np.array([t["f3"] for t in sel])
    z3_z2 = np.array([syrdal_gopal(t["f1"], t["f2"], t["f3"])[1] for t in sel])
    f3_mean, f3_lo, f3_hi = bootstrap_ci(f3)
    z_mean, z_lo, z_hi = bootstrap_ci(z3_z2)
    if f3_mean < RHOTIC_F3_MAX:
        verdict = "rhotic"
    elif f3_mean > NONRHOTIC_F3_MIN:
        verdict = "non_rhotic"
    else:
        verdict = "ambiguous"
    return {
        "context": phoneme or "NURSE+pre-r", "n": len(sel),
        "f3_hz": round(f3_mean, 1), "f3_ci": (round(f3_lo, 1), round(f3_hi, 1)),
        "z3_z2": round(z_mean, 3), "z3_z2_ci": (round(z_lo, 3), round(z_hi, 3)),
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Source-level assembly
# ---------------------------------------------------------------------------

def score_source(
    tokens: list[Token], target: str, mean_f0: float,
    focus: list[tuple[str, str]] | None = None,
    pairs: list[tuple[str, str, str]] | None = None,
) -> dict:
    """Full coach report for one source against one accent target.

    Returns:
      {
        "target": str,
        "overall": {"mean","lo","hi"},          # mean of per-vowel mean dists, bootstrapped
        "per_vowel": {ph: {name,n,mean,lo,hi,dispersion}},
        "pairs": {label: {overlap, a, b}},
        "rhoticity": {...} | None,
      }
    """
    focus = focus or FOCUS_VOWELS
    pairs = pairs or MINIMAL_PAIRS
    norms = get_norms(target, mean_f0)
    grouped = _by_phoneme(tokens)

    per_vowel: dict[str, dict] = {}
    vowel_dist_arrays: list[np.ndarray] = []
    for ph, name in focus:
        if ph not in norms or ph not in grouped:
            continue
        dists = per_token_distances(grouped[ph], norms[ph])
        if dists.size == 0:
            continue
        mean, lo, hi = bootstrap_ci(dists)
        per_vowel[ph] = {
            "name": name, "n": int(dists.size),
            "mean": round(mean, 3), "lo": round(lo, 3), "hi": round(hi, 3),
            "dispersion": round(dispersion(grouped[ph]), 3),
        }
        vowel_dist_arrays.append(dists)

    overall = _bootstrap_overall(vowel_dist_arrays)

    pair_out: dict[str, dict] = {}
    for a, b, label in pairs:
        if a in grouped and b in grouped:
            ov = bhattacharyya_overlap(grouped[a], grouped[b])
            pair_out[label] = {"overlap": round(ov, 3), "a": a, "b": b}

    return {
        "target": target,
        "overall": overall,
        "per_vowel": per_vowel,
        "pairs": pair_out,
        "rhoticity": rhoticity(tokens),
    }


def _bootstrap_overall(
    vowel_dist_arrays: list[np.ndarray], n_boot: int = 2000, seed: int = 0,
) -> dict:
    """CI of the unweighted mean-of-vowel-means.

    Each bootstrap iteration resamples tokens within each vowel, averages each
    vowel, then averages across vowels — so the CI reflects both within-vowel
    sampling noise and the small number of vowels.
    """
    if not vowel_dist_arrays:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    point = float(np.mean([a.mean() for a in vowel_dist_arrays]))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        vmeans = [a[rng.integers(0, a.size, a.size)].mean() for a in vowel_dist_arrays]
        boot[b] = np.mean(vmeans)
    return {
        "mean": round(point, 3),
        "lo": round(float(np.percentile(boot, 2.5)), 3),
        "hi": round(float(np.percentile(boot, 97.5)), 3),
    }


def ci_separated(a: dict, b: dict) -> bool:
    """True if two {'mean','lo','hi'} estimates have non-overlapping 95% CIs."""
    return a["hi"] < b["lo"] or b["hi"] < a["lo"]
