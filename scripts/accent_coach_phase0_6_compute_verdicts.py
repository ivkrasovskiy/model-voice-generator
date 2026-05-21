"""Phase C: Compute verdicts from Phase A and Phase B outputs."""

import json
import pathlib
import textwrap
import pandas as pd
import numpy as np

REPO = pathlib.Path(__file__).parent.parent
CSV_GEO = REPO / "docs" / "accent_coach_phase0_6_geometry.csv"
CSV_DIST = REPO / "docs" / "accent_coach_phase0_6_lobanov_distances.csv"
OUT_JSON = REPO / "docs" / "accent_coach_phase0_6_verdicts.json"
OUT_MD = REPO / "docs" / "accent_coach_phase0_6_verdicts.md"

NATIVE_SOURCES = ["modern_rp_fry", "modern_rp_lindsey", "real_bc", "synth_bc"]
BASELINES = ["modern_rp_full", "modern_rp_no_bbc", "native_wide"]


def main():
    geo = pd.read_csv(CSV_GEO).set_index("source")
    dist_df = pd.read_csv(CSV_DIST, index_col=0)

    # V1: owner vowel space compressed
    owner_ai = float(geo.loc["owner", "articulation_idx"])
    v1 = {
        "value": round(owner_ai, 4),
        "threshold": 0.7,
        "pass": owner_ai <= 0.7,
    }

    # V2: owner clusters separately from natives in Ward dendrogram
    # Proxy: owner's min distance to any native > median pairwise among natives
    owner_to_native = [float(dist_df.loc["owner", s]) for s in NATIVE_SOURCES]
    owner_nn = min(owner_to_native)

    native_pairs = []
    for i, s1 in enumerate(NATIVE_SOURCES):
        for s2 in NATIVE_SOURCES[i + 1:]:
            native_pairs.append(float(dist_df.loc[s1, s2]))
    native_median = float(np.median(native_pairs))

    v2 = {
        "owner_nn_distance": round(owner_nn, 4),
        "native_median_distance": round(native_median, 4),
        "pass": owner_nn > native_median,
    }

    # V3: distance reversal per baseline
    v3 = {}
    passes = 0
    for bl in BASELINES:
        d_owner = float(dist_df.loc["owner", bl])
        d_det = float(dist_df.loc["deterding_rp", bl])
        p = d_owner > d_det
        if p:
            passes += 1
        v3[bl] = {
            "d_owner": round(d_owner, 4),
            "d_deterding": round(d_det, 4),
            "pass": p,
        }

    # V4: cross-baseline robustness
    v4 = {
        "baselines_passing": passes,
        "pass": passes >= 2,
        "phase_e_required": passes < 2,
    }

    verdicts = {"V1": v1, "V2": v2, "V3": v3, "V4": v4}

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(verdicts, f, indent=2)
    print(f"Written: {OUT_JSON}")

    # Markdown table
    lines = ["# Phase 0.6 Verdicts\n"]
    lines.append(f"| Verdict | Label | Value | Pass |")
    lines.append(f"|---------|-------|-------|------|")
    lines.append(f"| V1 | Owner vowel space compressed | articulation_idx = {v1['value']} (threshold ≤ {v1['threshold']}) | {'✓' if v1['pass'] else '✗'} |")
    lines.append(f"| V2 | Owner clusters separately | owner_nn={v2['owner_nn_distance']}, native_median={v2['native_median_distance']} | {'✓' if v2['pass'] else '✗'} |")
    for bl, r in v3.items():
        lines.append(f"| V3/{bl} | Distance reversal | d_owner={r['d_owner']}, d_deterding={r['d_deterding']} | {'✓' if r['pass'] else '✗'} |")
    lines.append(f"| V4 | Cross-baseline robustness | {passes}/3 baselines pass | {'✓' if v4['pass'] else '✗'} |")

    phase_e = "**Phase E required.**" if v4["phase_e_required"] else "Phase E not required."
    lines.append(f"\n{phase_e}")

    md_text = "\n".join(lines)
    with open(OUT_MD, "w") as f:
        f.write(md_text)
    print(f"Written: {OUT_MD}")
    print()
    print(md_text)


if __name__ == "__main__":
    main()
