"""Run rhythm bench on conversational reference groups.

Tests hypothesis: conversational speech (panel shows, talk show interviews)
scores >= 70 on the rhythm bench, matching or exceeding TTS (generated_bc=70.1).

Sources:
  rp_conv_mitchell  — David Mitchell, Would I Lie to You? (sF_tPdV1Jjc)
  rp_conv_mack      — Lee Mack, Would I Lie to You? (QPRDPIrwmL8)
  genam_conv_carell — Steve Carell, Late Night with Conan O'Brien (9RViEPCoEec)
  genam_conv_jlaw   — Jennifer Lawrence, Late Night with Seth Meyers (KZDYjcE9mIc)
  genam_conv_freshair — Fresh Air NPR interview (XogN47yBM_M)

Usage:
    PYTHONPATH=. uv run python scripts/bench/accent_coach_conv_bench.py           # aligned (default)
    PYTHONPATH=. uv run python scripts/bench/accent_coach_conv_bench.py --fast    # fast/approximate
"""
from __future__ import annotations

import argparse
import random
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent

from scripts.bench.accent_coach_rhythm_bench import (  # noqa: E402
    bench_group,
    load_manifest,
    print_table,
)

GROUPS = [
    ("rp_conv_mitchell", REPO_ROOT / "tts_output/conversational_refs/rp_conv/manifest.json", "RP", "mitchell"),
    ("rp_conv_mack",     REPO_ROOT / "tts_output/conversational_refs/rp_conv/manifest.json", "RP", "mack"),
    ("genam_conv_carell",  REPO_ROOT / "tts_output/conversational_refs/genam_conv/manifest.json", "GenAm", "carell"),
    ("genam_conv_jlaw",    REPO_ROOT / "tts_output/conversational_refs/genam_conv/manifest.json", "GenAm", "jlaw"),
    ("genam_conv_freshair",REPO_ROOT / "tts_output/conversational_refs/genam_conv/manifest.json", "GenAm", "freshair"),
]

# Reference from existing bench (fast mode)
REFERENCE_TABLE = """
Reference (from existing bench, fast mode):
  rp_fry          RP         nPVI=48.5  score=78.3
  rp_bbc          RP         nPVI=35.1  score=60.8
  rp_lindsey      RP         nPVI=36.5  score=64.5
  genam_harris    GenAm      nPVI=43.4  score=69.6
  genam_sapolsky  GenAm      nPVI=49.1  score=70.7
  generated_bc    Generated  nPVI=52.7  score=70.1
"""

def main() -> None:
    parser = argparse.ArgumentParser(description="Conversational rhythm bench")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    mode = "fast" if args.fast else "aligned"
    print(f"\nConversational Rhythm Bench — mode: {mode}, N={args.n}, seed={args.seed}")
    if args.fast:
        print("NOTE: fast mode underestimates nPVI by ~11 pts vs hybrid — use for relative ranking only")

    rows = []
    for label, manifest_path, accent, speaker_filter in GROUPS:
        if not manifest_path.exists():
            print(f"  SKIP {label}: manifest not found at {manifest_path}")
            continue
        entries = load_manifest(manifest_path)
        print(f"  Processing {label} ({speaker_filter})...", flush=True)
        npvis, scores = bench_group(
            label, entries, manifest_path, args.n,
            args.verbose, args.fast, speaker_filter=speaker_filter,
        )
        rows.append((label, accent, npvis, scores))

    print_table(rows)

    print(REFERENCE_TABLE)
    print("Hypothesis check: conversational speakers should score >= 70 (generated_bc baseline)")
    print()
    for label, _accent, npvis, scores in rows:
        if not scores:
            continue
        mean_score = statistics.mean(scores)
        mean_npvi = statistics.mean(npvis)
        verdict = "PASS" if mean_score >= 70 else "BELOW"
        print(f"  {label:<28} score={mean_score:5.1f}  nPVI={mean_npvi:5.1f}  [{verdict}]")


if __name__ == "__main__":
    main()
