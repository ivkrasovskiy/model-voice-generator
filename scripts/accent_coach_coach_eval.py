"""Coach-grade evaluation: score every source against RP *and* GenAm.

Reports, per source:
  - overall distance-to-target with bootstrap 95% CI (RP and GenAm)
  - NURSE-F3 rhoticity verdict (the RP↔GenAm separator)
  - BATH↔TRAP minimal-pair overlap (the canonical L2 confusion)
And a per-vowel breakdown for the owner (the learner) against both targets.

Operates on the per-token formants_*.csv files already on disk — no audio,
no generation. Pure numpy; runs in seconds.

  .venv/bin/python scripts/accent_coach_coach_eval.py

NOTE on GenAm columns: the corpus/generated CSVs were extracted with RP
corrections baked in (coda-R stripped, BATH→ɑː). Their GenAm-target scores are
therefore APPROXIMATE — a clean GenAm score requires re-extraction with
--target genam (Phase D). Rhoticity is exact regardless: F3 is untouched by the
RP label corrections.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from accent_coach.diagnostics.coach_metrics import (  # noqa: E402
    FOCUS_VOWELS,
    ci_separated,
    load_tokens,
    score_source,
)

MEAN_F0 = 110.0  # all sources here are male → selects male norm tables
OUT_DIR = PROJECT_ROOT / "tts_output/accent_coach/phase0_15"

# (label, formants_csv, rp_corrections_baked_in)
SOURCES: list[tuple[str, Path, bool]] = [
    ("fry",           PROJECT_ROOT / "tts_output/modern_rp_corpus/formants_fry.csv", True),
    ("lindsey",       PROJECT_ROOT / "tts_output/modern_rp_corpus/formants_lindsey.csv", True),
    ("bbc",           PROJECT_ROOT / "tts_output/modern_rp_corpus/formants_bbc.csv", True),
    ("gen_base",      PROJECT_ROOT / "tts_output/cross_eval_50/gen_base/formants_gen_base.csv", True),
    ("gen_lora_best", PROJECT_ROOT / "tts_output/cross_eval_50/gen_lora_best/formants_gen_lora_best.csv", True),
    ("real_bc",       PROJECT_ROOT / "tts_output/real_bc_corpus/formants_real_bc.csv", True),
    ("genam",         PROJECT_ROOT / "tts_output/genam_corpus/formants_genam.csv", False),
    ("owner",         PROJECT_ROOT / "tts_output/owner_cal_50/formants.csv", False),
]

# Known-answer expectations for the reliability gate. Each native source must
# score closer to its own accent (with CI separation) AND show the matching
# rhoticity verdict (RP = non_rhotic, GenAm = rhotic). "owner" is the learner —
# no fixed expectation, reported only.
EXPECT: dict[str, dict[str, str]] = {
    "fry":           {"target": "RP", "rhotic": "non_rhotic"},
    "lindsey":       {"target": "RP", "rhotic": "non_rhotic"},
    "bbc":           {"target": "RP", "rhotic": "non_rhotic"},
    "gen_base":      {"target": "RP", "rhotic": "non_rhotic"},
    "real_bc":       {"target": "RP", "rhotic": "non_rhotic"},
    "genam":         {"target": "GenAm", "rhotic": "rhotic"},
}


def _fmt_ci(d: dict) -> str:
    return f"{d['mean']:.3f} ({d['lo']:.3f}-{d['hi']:.3f})"


def main() -> int:
    results: dict[str, dict] = {}
    for label, csv_path, _ in SOURCES:
        if not csv_path.exists():
            print(f"[SKIP] {label}: {csv_path} not found")
            continue
        tokens = load_tokens(csv_path)
        results[label] = {
            "n_tokens": len(tokens),
            "rp": score_source(tokens, "rp", MEAN_F0),
            "genam": score_source(tokens, "genam", MEAN_F0),
        }

    if not results:
        print("No sources scored.")
        return 1

    # ---- Overall summary table -------------------------------------------
    print("\n" + "=" * 100)
    print("OVERALL distance-to-target (Bark, mean + 95% CI)   [lower = closer]")
    print("=" * 100)
    print(f"{'source':<16}{'vs RP':<26}{'vs GenAm (approx)':<26}{'NURSE F3':<14}{'rhotic?':<12}")
    print("-" * 100)
    for label, r in results.items():
        rp, ga = r["rp"]["overall"], r["genam"]["overall"]
        rho = r["rp"]["rhoticity"]
        f3 = f"{rho['f3_hz']:.0f}Hz" if rho else "no F3"
        verdict = rho["verdict"] if rho else "-"
        print(f"{label:<16}{_fmt_ci(rp):<26}{_fmt_ci(ga):<26}{f3:<14}{verdict:<12}")

    # ---- RP vs GenAm separation (the reliability signal) -----------------
    print("\n" + "=" * 100)
    print("RP↔GenAm SEPARATION  (does the metric tell the two targets apart?)")
    print("=" * 100)
    print(f"{'source':<16}{'closer target':<16}{'CI-separated?':<16}{'BATH↔TRAP overlap':<20}")
    print("-" * 100)
    for label, r in results.items():
        rp, ga = r["rp"]["overall"], r["genam"]["overall"]
        closer = "RP" if rp["mean"] < ga["mean"] else "GenAm"
        sep = "yes" if ci_separated(rp, ga) else "no (overlap)"
        bt = r["rp"]["pairs"].get("BATH↔TRAP", {})
        bt_s = f"{bt['overlap']:.3f}" if bt and bt.get("overlap") == bt.get("overlap") else "n/a"
        print(f"{label:<16}{closer:<16}{sep:<16}{bt_s:<20}")

    # ---- Owner per-vowel breakdown (the learner) -------------------------
    if "owner" in results:
        print("\n" + "=" * 100)
        print("OWNER per-vowel distance + dispersion (the coaching targets)")
        print("=" * 100)
        print(f"{'vowel':<16}{'vs RP (mean,disp)':<28}{'vs GenAm (mean,disp)':<28}")
        print("-" * 100)
        owner_rp = results["owner"]["rp"]["per_vowel"]
        owner_ga = results["owner"]["genam"]["per_vowel"]
        # rank RP vowels worst-first
        ranked = sorted(owner_rp.items(), key=lambda kv: kv[1]["mean"], reverse=True)
        for ph, v in ranked:
            ga = owner_ga.get(ph, {})
            rp_s = f"{v['mean']:.3f}  disp={v['dispersion']:.3f}  n={v['n']}"
            ga_s = (f"{ga['mean']:.3f}  disp={ga['dispersion']:.3f}"
                    if ga else "-")
            print(f"{v['name']+' ('+ph+')':<16}{rp_s:<28}{ga_s:<28}")
        # name vowels with no F3 → rhoticity blind spot
        if results["owner"]["rp"]["rhoticity"] is None:
            print("\n  ⚠ owner CSV has no F3 — rhoticity not measurable; re-extract "
                  "with scripts/accent_coach_extract_formants.py --target none")

    # ---- Reliability gate verdict ----------------------------------------
    print("\n" + "=" * 100)
    print("RELIABILITY GATE  (native sources must score their own accent + matching rhoticity)")
    print("=" * 100)
    gate_rows: list[tuple[str, bool]] = []
    for label, exp in EXPECT.items():
        if label not in results:
            print(f"{label:<16} [SKIP] no formants yet")
            continue
        r = results[label]
        closer = "RP" if r["rp"]["overall"]["mean"] < r["genam"]["overall"]["mean"] else "GenAm"
        sep = ci_separated(r["rp"]["overall"], r["genam"]["overall"])
        rho = r["rp"]["rhoticity"]
        verdict = rho["verdict"] if rho else "no_F3"
        target_ok = (closer == exp["target"]) and sep
        rhotic_ok = (verdict == exp["rhotic"])
        ok = target_ok and rhotic_ok
        gate_rows.append((label, ok))
        flag = "✅" if ok else "❌"
        print(f"{label:<16} want[{exp['target']:<5} {exp['rhotic']:<10}]  "
              f"got[{closer:<5} {verdict:<10}] sep={'y' if sep else 'n'}  {flag}")
    if gate_rows:
        n_ok = sum(1 for _, ok in gate_rows if ok)
        passed = n_ok == len(gate_rows)
        verdict = "🟢 GREEN — gate PASSED" if passed else (
            "🟡 YELLOW — partial" if n_ok >= len(gate_rows) - 1 else "🔴 RED — gate FAILED")
        print("-" * 100)
        print(f"GATE: {n_ok}/{len(gate_rows)} native sources correct  →  {verdict}")
        if "owner" in results:
            orho = results["owner"]["rp"]["rhoticity"]
            ov = orho["verdict"] if orho else "no_F3"
            print(f"owner (learner): rhoticity = {ov}"
                  + (f" (NURSE F3={orho['f3_hz']:.0f}Hz)" if orho else ""))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "coach_eval.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nWrote {out_json.relative_to(PROJECT_ROOT)}")
    print(f"Focus vowels: {', '.join(n for _, n in FOCUS_VOWELS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
