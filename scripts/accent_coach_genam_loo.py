"""Leave-one-out validation of the modern GenAm norms (Phase 0.16, A4).

For each of the 3 GA lecture speakers: build GenAm norms from the OTHER two,
then score the held-out speaker's tokens against those norms (GenAm) and against
RP norms. A speaker the norms have never seen must score GenAm-closer and read
rhotic — that is the non-circular GREEN test the sparse Vsauce sample can't give.

  .venv/bin/python scripts/accent_coach_genam_loo.py
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from accent_coach.diagnostics.coach_metrics import (  # noqa: E402
    FOCUS_VOWELS,
    Token,
    _bootstrap_overall,
    ci_separated,
    per_token_distances,
    rhoticity,
)
from accent_coach.reference.rp_norms import get_rp_norms  # noqa: E402

LECTURE_CSV = PROJECT_ROOT / "tts_output/genam_lecture_corpus/formants_genam_lecture.csv"
MEAN_F0 = 110.0
RP = get_rp_norms(MEAN_F0)
MIN_N = 20  # min tokens to form a norm for a vowel


def _load_by_speaker(csv_path: Path) -> dict[str, list[Token]]:
    out: dict[str, list[Token]] = defaultdict(list)
    with csv_path.open() as fh:
        for r in csv.DictReader(fh):
            try:
                if float(r.get("duration_s", 0)) < 0.05:
                    continue
                f1, f2 = float(r["F1"]), float(r["F2"])
            except (ValueError, KeyError):
                continue
            f3 = float(r["F3"]) if r.get("F3", "") not in ("", "nan") else None
            spk = r["clip_id"].split("_")[0]
            out[spk].append(Token(phoneme=r["phoneme"], next_phoneme=r.get("next_phoneme", ""),
                                  f1=f1, f2=f2, f3=f3))
    return out


def _norms_from(tokens: list[Token]) -> dict[str, tuple[float, float]]:
    by_ph: dict[str, list[Token]] = defaultdict(list)
    for t in tokens:
        by_ph[t["phoneme"]].append(t)
    return {ph: (float(np.median([t["f1"] for t in ts])), float(np.median([t["f2"] for t in ts])))
            for ph, ts in by_ph.items() if len(ts) >= MIN_N}


def _overall(tokens: list[Token], norms: dict) -> dict:
    by_ph: dict[str, list[Token]] = defaultdict(list)
    for t in tokens:
        by_ph[t["phoneme"]].append(t)
    arrs = [per_token_distances(by_ph[ph], norms[ph])
            for ph, _ in FOCUS_VOWELS if ph in norms and ph in by_ph and by_ph[ph]]
    arrs = [a for a in arrs if a.size]
    return _bootstrap_overall(arrs)


def main() -> int:
    by_spk = _load_by_speaker(LECTURE_CSV)
    speakers = sorted(by_spk)
    print(f"Leave-one-out over {len(speakers)} GA lecture speakers: {speakers}\n")
    print(f"{'held-out':12}{'vs RP':22}{'vs GenAm(LOO)':22}{'closer':8}{'sep':5}{'rhotic':10}")
    print("-" * 85)
    results = []
    for held in speakers:
        train = [t for s in speakers if s != held for t in by_spk[s]]
        ga_norms = _norms_from(train)
        ho = by_spk[held]
        rp_o = _overall(ho, RP)
        ga_o = _overall(ho, ga_norms)
        closer = "GenAm" if ga_o["mean"] < rp_o["mean"] else "RP"
        sep = ci_separated(rp_o, ga_o)
        rho = rhoticity(ho)
        verdict = rho["verdict"] if rho else "no_F3"
        ok = (closer == "GenAm") and (verdict == "rhotic")
        results.append(ok)
        rp_s = "{:.3f} ({:.2f}-{:.2f})".format(rp_o["mean"], rp_o["lo"], rp_o["hi"])
        ga_s = "{:.3f} ({:.2f}-{:.2f})".format(ga_o["mean"], ga_o["lo"], ga_o["hi"])
        rho_s = verdict + (" {:.0f}".format(rho["f3_hz"]) if rho else "")
        flag = "✅" if ok else "❌"
        print(f"{held:12}{rp_s:22}{ga_s:22}{closer:8}{'y' if sep else 'n':5}{rho_s:10}  {flag}")
    n_ok = sum(results)
    print("-" * 85)
    verdict = "🟢 GREEN" if n_ok == len(results) else ("🟡 YELLOW" if n_ok else "🔴 RED")
    print(f"LOO: {n_ok}/{len(results)} held-out GA speakers correctly GenAm + rhotic → {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
