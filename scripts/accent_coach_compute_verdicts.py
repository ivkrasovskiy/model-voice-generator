"""Compute H1–H4 numerical verdicts from the comparison table + refclip summary.

Usage:
    .venv/bin/python scripts/accent_coach_compute_verdicts.py
    .venv/bin/python scripts/accent_coach_compute_verdicts.py \\
        --table docs/accent_coach_phase0_5_table.csv \\
        --refclip-summary tts_output/refclip_robustness/summary.csv \\
        --out-json docs/accent_coach_phase0_5_verdicts.json \\
        --out-md   docs/accent_coach_phase0_5_verdicts.md
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

DEFAULT_TABLE = "docs/accent_coach_phase0_5_table.csv"
DEFAULT_REFCLIP = "tts_output/refclip_robustness/summary.csv"
DEFAULT_OUT_JSON = "docs/accent_coach_phase0_5_verdicts.json"
DEFAULT_OUT_MD = "docs/accent_coach_phase0_5_verdicts.md"

PROBLEM_PHONEMES = ["æ", "ɛ", "ʌ"]
THRESHOLD_HZ = 75.0


def _median_deviation(
    table: pd.DataFrame,
    source_a: str,
    source_b: str,
    phonemes: list[str],
) -> tuple[float, float]:
    """Median |A - B| over F1 and F2 for listed phonemes."""
    a = table[table.source == source_a].set_index("phoneme")
    b = table[table.source == source_b].set_index("phoneme")
    present = [p for p in phonemes if p in a.index and p in b.index]
    if not present:
        return float("nan"), float("nan")
    dF1 = np.median([abs(a.loc[p].F1_mean - b.loc[p].F1_mean) for p in present])
    dF2 = np.median([abs(a.loc[p].F2_mean - b.loc[p].F2_mean) for p in present])
    return float(dF1), float(dF2)


def verdict_H1(table: pd.DataFrame) -> dict:
    """H1: Is Deterding 1997 stale vs modern RP?"""
    dF1, dF2 = _median_deviation(table, "deterding_rp", "modern_rp", PROBLEM_PHONEMES)
    stale = max(dF1, dF2) > THRESHOLD_HZ if not math.isnan(dF1) else None
    return {
        "median_dF1": round(dF1, 1) if not math.isnan(dF1) else None,
        "median_dF2": round(dF2, 1) if not math.isnan(dF2) else None,
        "threshold_hz": THRESHOLD_HZ,
        "verdict": ("stale" if stale else "fresh") if stale is not None else "no_data",
    }


def verdict_H2(table: pd.DataFrame) -> dict:
    """H2: Does real BC deviate from modern RP on /æ ɛ ʌ/?"""
    dF1, dF2 = _median_deviation(table, "real_bc", "modern_rp", PROBLEM_PHONEMES)
    deviates = max(dF1, dF2) > THRESHOLD_HZ if not math.isnan(dF1) else None
    return {
        "median_dF1": round(dF1, 1) if not math.isnan(dF1) else None,
        "median_dF2": round(dF2, 1) if not math.isnan(dF2) else None,
        "threshold_hz": THRESHOLD_HZ,
        "verdict": ("bc_deviates" if deviates else "bc_close_to_rp") if deviates is not None else "no_data",
    }


def verdict_H3(table: pd.DataFrame, robustness: pd.DataFrame | None) -> dict:
    """H3: Does IndexTTS distort BC's vowel space vs real BC?"""
    dF1, dF2 = _median_deviation(table, "synth_bc", "real_bc", PROBLEM_PHONEMES)
    shifts = max(dF1, dF2) > THRESHOLD_HZ if not math.isnan(dF1) else None

    var_F1 = var_F2 = float("nan")
    if robustness is not None and not robustness.empty:
        rob_prob = robustness[robustness.phoneme.isin(PROBLEM_PHONEMES)]
        if not rob_prob.empty:
            var_F1 = float(rob_prob.groupby("phoneme").F1_mean.std().mean())
            var_F2 = float(rob_prob.groupby("phoneme").F2_mean.std().mean())

    consistent = max(var_F1, var_F2) < 50.0 if not math.isnan(var_F1) else None

    if shifts is None:
        verdict = "no_data"
    elif shifts and consistent:
        verdict = "architecture"
    elif shifts and not consistent:
        verdict = "ref_dependent"
    else:
        verdict = "no_shift"

    return {
        "median_dF1": round(dF1, 1) if not math.isnan(dF1) else None,
        "median_dF2": round(dF2, 1) if not math.isnan(dF2) else None,
        "cross_ref_var_F1": round(var_F1, 1) if not math.isnan(var_F1) else None,
        "cross_ref_var_F2": round(var_F2, 1) if not math.isnan(var_F2) else None,
        "threshold_shift_hz": THRESHOLD_HZ,
        "threshold_var_hz": 50.0,
        "verdict": verdict,
    }


def verdict_H4(table: pd.DataFrame) -> dict:
    """H4: Do synth-BC /æ ɛ/ collide with where Slavic L1 maps /ɛ/-substituted-/æ/?"""
    out: dict[str, dict] = {}
    for ph in ["æ", "ɛ"]:
        synth_row = table[(table.source == "synth_bc") & (table.phoneme == ph)]
        real_row  = table[(table.source == "real_bc")  & (table.phoneme == ph)]
        owner_row = table[(table.source == "owner")    & (table.phoneme == ph)]

        if synth_row.empty or real_row.empty or owner_row.empty:
            out[ph] = {"verdict": "no_data"}
            continue

        s = synth_row.iloc[0]
        r = real_row.iloc[0]
        o = owner_row.iloc[0]

        d_so = math.hypot(s.F1_mean - o.F1_mean, s.F2_mean - o.F2_mean)
        d_sr = math.hypot(s.F1_mean - r.F1_mean, s.F2_mean - r.F2_mean)
        d_or = math.hypot(o.F1_mean - r.F1_mean, o.F2_mean - r.F2_mean)

        out[ph] = {
            "d_synth_owner": round(d_so, 1),
            "d_synth_real": round(d_sr, 1),
            "d_owner_real": round(d_or, 1),
            # H4 confirmed if synth is CLOSER to owner than to real BC
            "confirmed": bool(d_so < d_sr),
        }
    return out


def _md_row(label: str, val) -> str:
    return f"| {label} | {val} |"


def format_markdown(verdicts: dict) -> str:
    lines = ["## Phase 0.5 Verdict Summary", ""]

    h1 = verdicts["H1"]
    lines += [
        "### H1 — Is Deterding 1997 stale?",
        f"Median |Deterding − Modern RP|: F1={h1['median_dF1']} Hz, F2={h1['median_dF2']} Hz",
        f"**Verdict: {h1['verdict'].upper()}**",
        f"*(threshold {THRESHOLD_HZ} Hz)*",
        "",
    ]

    h2 = verdicts["H2"]
    lines += [
        "### H2 — Does real BC deviate from modern RP on /æ ɛ ʌ/?",
        f"Median |Real BC − Modern RP|: F1={h2['median_dF1']} Hz, F2={h2['median_dF2']} Hz",
        f"**Verdict: {h2['verdict'].upper()}**",
        "",
    ]

    h3 = verdicts["H3"]
    lines += [
        "### H3 — Does IndexTTS distort BC's vowel space?",
        f"Median |Synth BC − Real BC|: F1={h3['median_dF1']} Hz, F2={h3['median_dF2']} Hz",
        f"Cross-ref variance: F1={h3['cross_ref_var_F1']} Hz, F2={h3['cross_ref_var_F2']} Hz",
        f"**Verdict: {h3['verdict'].upper()}**",
        f"*(shift threshold {THRESHOLD_HZ} Hz, variance threshold 50 Hz)*",
        "",
    ]

    h4 = verdicts["H4"]
    lines += ["### H4 — Do synth-BC /æ ɛ/ collide with owner?", ""]
    lines += ["| Phoneme | d(synth, owner) | d(synth, real) | d(owner, real) | Confirmed |"]
    lines += ["|---------|----------------|----------------|----------------|-----------|"]
    for ph, v in h4.items():
        if isinstance(v, dict) and "d_synth_owner" in v:
            lines.append(
                f"| /{ph}/ | {v['d_synth_owner']} Hz | {v['d_synth_real']} Hz "
                f"| {v['d_owner_real']} Hz | {'YES' if v['confirmed'] else 'no'} |"
            )
        else:
            lines.append(f"| /{ph}/ | — | — | — | no_data |")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute H1–H4 verdicts")
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--refclip-summary", type=Path, default=DEFAULT_REFCLIP)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    def _resolve(p: Path) -> Path:
        return p if p.is_absolute() else PROJECT_ROOT / p

    table_path = _resolve(args.table)
    refclip_path = _resolve(args.refclip_summary)
    out_json = _resolve(args.out_json)
    out_md = _resolve(args.out_md)

    if not table_path.exists():
        print(f"ERROR: table not found: {table_path}", file=sys.stderr)
        return 1

    table = pd.read_csv(table_path)
    robustness = pd.read_csv(refclip_path) if refclip_path.exists() else None

    verdicts = {
        "H1": verdict_H1(table),
        "H2": verdict_H2(table),
        "H3": verdict_H3(table, robustness),
        "H4": verdict_H4(table),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(verdicts, indent=2, ensure_ascii=False))
    print(f"Verdicts JSON: {out_json}")

    md = format_markdown(verdicts)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md)
    print(f"Verdicts MD:   {out_md}")

    print("\n--- VERDICT SUMMARY ---")
    for h, v in verdicts.items():
        print(f"  {h}: {v.get('verdict', v)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
