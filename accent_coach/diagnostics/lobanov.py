"""Lobanov (per-speaker z-score) normalisation for vowel formants."""

import pandas as pd


def lobanov_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Per-source z-scoring of F1 and F2 across phonemes.

    For each source, compute mu/sigma over its 9 vowels independently, then
    z = (x - mu) / sigma. Strips out vowel-space scale; preserves shape.

    Args:
        df: long-format with columns [source, phoneme, F1_mean, F2_mean]
    Returns:
        Same df with added [F1_lobanov, F2_lobanov]
    """
    out = df.copy()
    out["F1_lobanov"] = float("nan")
    out["F2_lobanov"] = float("nan")

    for _src, grp in df.groupby("source"):
        f1_mu, f1_sig = grp["F1_mean"].mean(), grp["F1_mean"].std(ddof=0)
        f2_mu, f2_sig = grp["F2_mean"].mean(), grp["F2_mean"].std(ddof=0)
        idx = grp.index
        out.loc[idx, "F1_lobanov"] = (grp["F1_mean"] - f1_mu) / f1_sig if f1_sig > 0 else 0.0
        out.loc[idx, "F2_lobanov"] = (grp["F2_mean"] - f2_mu) / f2_sig if f2_sig > 0 else 0.0

    return out
