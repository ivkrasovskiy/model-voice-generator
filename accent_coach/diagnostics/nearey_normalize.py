"""H2 — Nearey log-mean normalisation using corner vowels (Phase 0.8).

For each speaker, centres log(F1) and log(F2) on the mean log value
across {iː, ɑː, uː}.  Removes overall vocal-tract scale while
preserving the relative shape and (crucially) vowel-space compression.
"""
from __future__ import annotations

import math

Centroids = dict[str, dict[str, dict[str, float] | None]]

CORNER_VOWELS = ("iː", "ɑː", "uː")


def nearey_corner_normalize(centroids: Centroids) -> Centroids:
    """Return Nearey-normalised centroids.

    For each speaker:
        μ_F1 = mean( log(F1_iː), log(F1_ɑː), log(F1_uː) )
        μ_F2 = mean( log(F2_iː), log(F2_ɑː), log(F2_uː) )
        F1'_p = log(F1_p) − μ_F1
        F2'_p = log(F2_p) − μ_F2
    """
    all_speakers: set[str] = set()
    for phon_dict in centroids.values():
        all_speakers.update(phon_dict.keys())

    speaker_mu: dict[str, dict[str, float]] = {}
    for spk in all_speakers:
        log_f1s = []
        log_f2s = []
        for cv in CORNER_VOWELS:
            vals = centroids.get(cv, {}).get(spk)
            if vals and vals["f1"] > 0 and vals["f2"] > 0:
                log_f1s.append(math.log(vals["f1"]))
                log_f2s.append(math.log(vals["f2"]))
        if len(log_f1s) < 2:
            speaker_mu[spk] = {"mu_f1": 0.0, "mu_f2": 0.0}
        else:
            speaker_mu[spk] = {
                "mu_f1": sum(log_f1s) / len(log_f1s),
                "mu_f2": sum(log_f2s) / len(log_f2s),
            }

    out: Centroids = {}
    for phoneme, speakers in centroids.items():
        out[phoneme] = {}
        for spk, vals in speakers.items():
            if vals is None or vals["f1"] <= 0 or vals["f2"] <= 0:
                out[phoneme][spk] = None
                continue
            mu = speaker_mu.get(spk, {"mu_f1": 0.0, "mu_f2": 0.0})
            out[phoneme][spk] = {
                "f1": math.log(vals["f1"]) - mu["mu_f1"],
                "f2": math.log(vals["f2"]) - mu["mu_f2"],
                "n": vals.get("n", 0),
            }
    return out
