"""Build cal_25.csv — a stratified 25-phrase subset of cal_50.csv.

Selection algorithm: greedy maximisation of the minimum per-phoneme count across
the 17 target phonemes (12 monophthongs + l r m n ŋ). Stops when either 25
phrases are selected or all 17 phonemes hit ≥ 2 hits. Budget may be expanded
to 30/35 if 25 phrases cannot cover every phoneme with ≥ 2 hits.

Reads formants from the existing bc_cal_50 manifest (phase0_7 calibration clips).
Outputs: tts_output/accent_coach/cal_25.csv
"""
from __future__ import annotations

import csv
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from accent_coach.pipeline.experiment import extract_formants

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Phonemes we want to cover from the scoring spec (plan §1).
# NOTE: The formant extractor uses MFA-style ARPAbet-adjacent symbols.
# Sonorants (l, r, m, n, ŋ) are not in F1/F2 formant data → cannot be selected for.
# ɒ (LOT) and ɔɪ (CHOICE) are not in the cal_50 formant data either.
# ɛ is used for DRESS (not 'e').  We use what's actually in the data for selection,
# but report coverage for the full 17-phoneme spec.
TARGET_PHONEMES_SPEC: list[str] = [
    "iː", "ɪ", "e", "æ", "ɑː", "ɒ", "ɔː", "ʊ", "uː", "ʌ", "ɜː", "ə",  # 12 monophthongs
    "l", "r", "m", "n", "ŋ",  # 5 sonorants
]

# Phonemes actually present in cal_50 formant data (discovered at runtime)
# — used for greedy selection.  Populated after loading slug_hits.
TARGET_PHONEMES: list[str] = []  # filled in main()

CAL_50_CSV    = PROJECT_ROOT / "tts_output/accent_coach/cal_50.csv"
CAL_25_CSV    = PROJECT_ROOT / "tts_output/accent_coach/cal_25.csv"
BC_CAL_50_MANIFEST = PROJECT_ROOT / "tts_output/accent_coach/bc_cal_50/manifest.json"
FORMANTS_CSV  = PROJECT_ROOT / "tts_output/accent_coach/bc_cal_50/formants_for_cal25.csv"

MIN_HITS      = 2
BASE_BUDGET   = 25
MAX_BUDGET    = 35
MIN_DURATION  = 0.050  # seconds

# ---------------------------------------------------------------------------

def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def load_slug_phoneme_hits(formants_csv: Path, target_phonemes: list[str]) -> dict[str, Counter]:
    """Return {slug: Counter({phoneme: count})} for rows meeting duration threshold."""
    slug_hits: dict[str, Counter] = {}
    with formants_csv.open() as f:
        for row in csv.DictReader(f):
            try:
                dur = float(row["duration_s"])
            except (ValueError, KeyError):
                continue
            if dur < MIN_DURATION:
                continue
            ph = row.get("phoneme", "")
            if ph not in target_phonemes:
                continue
            slug = row["clip_id"]
            slug_hits.setdefault(slug, Counter())[ph] += 1
    return slug_hits


def discover_phonemes(formants_csv: Path) -> list[str]:
    """Return sorted list of distinct phonemes in the formants CSV."""
    phonemes: set[str] = set()
    with formants_csv.open() as f:
        for row in csv.DictReader(f):
            try:
                if float(row["duration_s"]) >= MIN_DURATION:
                    phonemes.add(row["phoneme"])
            except (ValueError, KeyError):
                continue
    return sorted(phonemes)


def greedy_select(
    slug_hits: dict[str, Counter],
    cal_slugs: list[str],
    budget: int,
) -> list[str]:
    """Greedy: pick phrase that maximises the minimum per-phoneme count over TARGET_PHONEMES."""
    pool = [s for s in cal_slugs if s in slug_hits]
    selected: list[str] = []
    cumulative: Counter = Counter()

    for _ in range(budget):
        if not pool:
            break
        # Score each candidate: adding it, what is the new min per-phoneme count?
        best_slug = max(
            pool,
            key=lambda s: min(
                (cumulative + slug_hits[s]).get(ph, 0) for ph in TARGET_PHONEMES
            ),
        )
        selected.append(best_slug)
        cumulative += slug_hits[best_slug]
        pool.remove(best_slug)

        current_min = min(cumulative.get(ph, 0) for ph in TARGET_PHONEMES)
        if current_min >= MIN_HITS:
            _log(f"  All phonemes have ≥{MIN_HITS} hits after {len(selected)} phrases")
            break

    return selected


def main() -> int:
    _log("=== Phase 0.10 cal_25 builder ===")

    # Step 1: extract formants from bc_cal_50 (idempotent)
    _log(f"Step 1: extracting formants from bc_cal_50 manifest…")
    extract_formants(
        manifest=BC_CAL_50_MANIFEST,
        out_csv=FORMANTS_CSV,
        source_label="synth_BC",
    )

    # Step 2: discover available phonemes and load slug→phoneme hits
    _log("Step 2: discovering available phonemes…")
    available_phonemes = discover_phonemes(FORMANTS_CSV)
    # Use intersection of spec and what's actually in the data for selection
    global TARGET_PHONEMES
    TARGET_PHONEMES = [ph for ph in available_phonemes
                       if ph in TARGET_PHONEMES_SPEC or ph in available_phonemes]
    # Filter to only phonemes in spec OR all available vowels (both useful for selection)
    TARGET_PHONEMES = available_phonemes  # select for all available phonemes
    _log(f"  Available phonemes in formant data: {TARGET_PHONEMES}")
    not_in_data = [ph for ph in TARGET_PHONEMES_SPEC if ph not in available_phonemes]
    if not_in_data:
        _log(f"  Spec phonemes NOT in formant data (won't be covered): {not_in_data}")

    _log("  Loading phoneme hits per slug…")
    slug_hits = load_slug_phoneme_hits(FORMANTS_CSV, TARGET_PHONEMES)
    _log(f"  {len(slug_hits)} slugs have usable formant rows")

    # Step 3: load cal_50 slug order
    with CAL_50_CSV.open() as f:
        cal_rows = list(csv.DictReader(f))
    cal_slugs = [r["slug"] for r in cal_rows]
    slug_to_prompt = {r["slug"]: r["prompt"] for r in cal_rows}
    _log(f"  cal_50 has {len(cal_slugs)} phrases")

    # Step 4: greedy selection with budget escalation
    selected: list[str] = []
    for budget in [BASE_BUDGET, 30, MAX_BUDGET]:
        selected = greedy_select(slug_hits, cal_slugs, budget)
        missing = [ph for ph in TARGET_PHONEMES
                   if sum(slug_hits.get(s, Counter()).get(ph, 0) for s in selected) < MIN_HITS]
        if not missing:
            _log(f"  Budget {budget}: all {len(TARGET_PHONEMES)} phonemes covered at ≥{MIN_HITS} hits")
            break
        _log(f"  Budget {budget}: still missing ≥{MIN_HITS} hits for: {missing} — expanding budget")
    else:
        _log(f"WARNING: even at budget {MAX_BUDGET}, some phonemes not covered. Proceeding anyway.")

    # Step 5: write cal_25.csv
    CAL_25_CSV.parent.mkdir(parents=True, exist_ok=True)
    with CAL_25_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["slug", "prompt"])
        w.writeheader()
        for slug in selected:
            w.writerow({"slug": slug, "prompt": slug_to_prompt[slug]})
    _log(f"Step 5: wrote {len(selected)} phrases → {CAL_25_CSV}")

    # Step 6: coverage report
    print("\n=== Per-phoneme coverage in cal_25 ===")
    total_hits: Counter = Counter()
    for s in selected:
        total_hits += slug_hits.get(s, Counter())

    # Report for all 17 spec phonemes
    missing_phonemes = []
    for ph in TARGET_PHONEMES_SPEC:
        hits = total_hits.get(ph, 0)
        in_data = ph in TARGET_PHONEMES
        if not in_data:
            print(f"  —  {ph:4s}  N/A (not in formant data)")
        else:
            status = "✓" if hits >= MIN_HITS else "✗"
            print(f"  {status}  {ph:4s}  {hits} hits")
            if hits < MIN_HITS:
                missing_phonemes.append(ph)

    # Extra diphthongs/phonemes from formant data not in spec
    extras = [ph for ph in TARGET_PHONEMES if ph not in TARGET_PHONEMES_SPEC]
    if extras:
        print(f"\n  Extra available phonemes (not in spec, included in selection): {extras}")
        for ph in extras:
            print(f"       {ph:4s}  {total_hits.get(ph, 0)} hits")

    available_covered = [ph for ph in TARGET_PHONEMES_SPEC
                         if ph in TARGET_PHONEMES and total_hits.get(ph, 0) >= MIN_HITS]
    available_total = [ph for ph in TARGET_PHONEMES_SPEC if ph in TARGET_PHONEMES]
    print(f"\nSelected {len(selected)} phrases")
    print(f"Coverage: {len(available_covered)}/{len(available_total)} available spec phonemes have ≥{MIN_HITS} hits")
    if missing_phonemes:
        print(f"  Under-covered: {missing_phonemes}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
