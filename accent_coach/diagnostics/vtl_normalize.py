"""H1 — F0-based VTL scaling normalisation (Phase 0.8).

Scales each speaker's formants by F0_ref / F0_speaker, where
F0_ref = mean F0 across the three modern-RP training sources
(fry, lindsey, bbc_male).  Higher-F0 speakers have shorter vocal
tracts → higher formants; scaling removes that bias.

Deterding (a literature table, not a real speaker) is assigned F0_ref
so no correction is applied to it.
"""
from __future__ import annotations

Centroids = dict[str, dict[str, dict[str, float] | None]]


def vtl_normalize(
    centroids: Centroids,
    f0_per_speaker: dict[str, float],
) -> Centroids:
    """Return a deep copy of `centroids` with formants VTL-scaled.

    Args:
        centroids: {phoneme: {speaker: {f1, f2, n}}}
        f0_per_speaker: {speaker: mean_f0_hz} — must cover all speakers
            present in centroids that should be scaled.  Missing speakers
            are left unmodified.

    Returns:
        New centroids dict with f1/f2 scaled by (F0_ref / F0_speaker).
        F0_ref = mean of f0_per_speaker values for {fry, lindsey, bbc_male}.
    """
    modern_rp_sources = ["fry", "lindsey", "bbc_male"]
    f0_ref_values = [f0_per_speaker[s] for s in modern_rp_sources if s in f0_per_speaker]
    if not f0_ref_values:
        raise ValueError("f0_per_speaker must include at least one of fry/lindsey/bbc_male")
    f0_ref = sum(f0_ref_values) / len(f0_ref_values)

    out: Centroids = {}
    for phoneme, speakers in centroids.items():
        out[phoneme] = {}
        for spk, vals in speakers.items():
            if vals is None:
                out[phoneme][spk] = None
                continue
            f0_spk = f0_per_speaker.get(spk, f0_ref)
            scale = f0_ref / f0_spk
            out[phoneme][spk] = {
                "f1": vals["f1"] * scale,
                "f2": vals["f2"] * scale,
                "n": vals.get("n", 0),
            }
    return out
