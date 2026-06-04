"""Derive modern connected-speech GenAm vowel norms from the GA corpus (Phase 0.16).

Replaces the stale Hillenbrand-1995 citation norms (archaic back GOOSE, onset-
nucleus diphthongs) with centroids measured by OUR pipeline — so they are
commensurable with the modern RP norms (same measurement point, connected speech,
steady-state diphthongs).

Prints a ready-to-paste Python dict for genam_norms.py plus a per-speaker table
(for leave-one-out sanity). Does NOT auto-edit the norms module.

  .venv/bin/python scripts/accent_coach_build_genam_norms.py \\
      --formants tts_output/genam_lecture_corpus/formants_genam_lecture.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

# Lexical-set names for readable comments, in a stable order.
VOWEL_NAMES = {
    "iː": "FLEECE", "ɪ": "KIT", "ɛ": "DRESS", "æ": "TRAP", "ɑː": "BATH/PALM/LOT",
    "ɒ": "LOT", "ɔː": "THOUGHT", "ʊ": "FOOT", "uː": "GOOSE", "ʌ": "STRUT",
    "ɜː": "NURSE", "ə": "SCHWA", "eɪ": "FACE", "aɪ": "PRICE", "ɔɪ": "CHOICE",
    "əʊ": "GOAT", "aʊ": "MOUTH",
}


def _load_rows(csv_path: Path, min_dur: float = 0.050) -> list[dict]:
    rows = []
    with csv_path.open() as fh:
        for r in csv.DictReader(fh):
            try:
                if float(r.get("duration_s", 0)) < min_dur:
                    continue
                r["F1"], r["F2"] = float(r["F1"]), float(r["F2"])
            except (ValueError, KeyError):
                continue
            r["speaker"] = r.get("clip_id", "?").split("_")[0]
            rows.append(r)
    return rows


def _centroids(rows: list[dict], min_n: int) -> dict[str, tuple[float, float, int]]:
    by_ph: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        by_ph.setdefault(r["phoneme"], []).append((r["F1"], r["F2"]))
    out = {}
    for ph, pairs in by_ph.items():
        if len(pairs) < min_n:
            continue
        f1 = float(np.median([p[0] for p in pairs]))  # median: robust to tracker outliers
        f2 = float(np.median([p[1] for p in pairs]))
        out[ph] = (round(f1), round(f2), len(pairs))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--formants", default="tts_output/genam_lecture_corpus/formants_genam_lecture.csv")
    ap.add_argument("--min-n", type=int, default=15, help="Min tokens to emit a vowel norm")
    args = ap.parse_args()

    csv_path = PROJECT_ROOT / args.formants if not Path(args.formants).is_absolute() else Path(args.formants)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found", file=sys.stderr)
        return 1
    rows = _load_rows(csv_path)
    speakers = sorted(set(r["speaker"] for r in rows))
    print(f"Loaded {len(rows)} tokens from {len(speakers)} speakers: {speakers}\n")

    # Per-speaker sanity (helps spot a contaminated source before it pollutes norms)
    print("Per-speaker token counts:")
    for spk in speakers:
        n = sum(1 for r in rows if r["speaker"] == spk)
        print(f"  {spk:12} {n}")

    pooled = _centroids(rows, args.min_n)

    print("\n# --- paste into accent_coach/reference/genam_norms.py ---")
    print("# Modern connected-speech GenAm — pooled MEDIAN F1/F2 (Hz), adult male,")
    print(f"# from {csv_path.relative_to(PROJECT_ROOT)} ({', '.join(speakers)}), dur>=50ms,")
    print("# same pipeline/measurement point as RP_VOWEL_F1_F2_MALE_MODERN (steady-state).")
    print("_GENAM_MALE_MODERN: dict[str, tuple[float, float]] = {")
    for ph in VOWEL_NAMES:
        if ph in pooled:
            f1, f2, n = pooled[ph]
            print(f'    "{ph}": ({f1}, {f2}),  # {VOWEL_NAMES[ph]} — n={n}')
        else:
            print(f'    # "{ph}": MISSING ({VOWEL_NAMES[ph]}) — <{args.min_n} tokens')
    print("}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
