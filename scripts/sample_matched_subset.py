"""
Sample a random subset from one dataset whose duration distribution matches another.

Used for matched-size comparison: e.g. produce 379 Casanova clips whose duration
histogram matches the 379 Sherlock narrator clips, so the only training-data variable
is content/register, not clip-length distribution.

Algorithm: bin both datasets' durations into the same buckets, then sample from the
source with per-bin probabilities equal to the target's bin frequencies (rejection
sampling). Deterministic via seed.

Usage:
  .venv/bin/python scripts/sample_matched_subset.py \
      --source cumberbatch_casanova_clean \
      --target cumberbatch_sherlock \
      --n 379 \
      --out-name casanova_matched_379 \
      --seed 42
"""

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lib.dataset import load_metadata

PROJECT_ROOT = Path(__file__).parent.parent


def load_entries(name: str) -> list[dict]:
    return load_metadata(name, project_root=PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Source dataset name (e.g. cumberbatch_casanova_clean)")
    parser.add_argument("--target", required=True, help="Target dataset whose duration histogram to match")
    parser.add_argument("--n", type=int, required=True, help="Number of clips to sample")
    parser.add_argument("--out-name", required=True, help="Output dataset name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bins", type=int, default=20,
                        help="Number of duration bins for matching (default 20)")
    args = parser.parse_args()

    source = load_entries(args.source)
    target = load_entries(args.target)
    print(f"Source ({args.source}): {len(source)} clips")
    print(f"Target ({args.target}): {len(target)} clips")
    print(f"Sampling {args.n} clips from source matched to target's duration distribution")

    # Build bins from both pooled (for fair boundaries)
    src_durs = [e["duration"] for e in source]
    tgt_durs = [e["duration"] for e in target]
    lo = min(min(src_durs), min(tgt_durs))
    hi = max(max(src_durs), max(tgt_durs))
    n_bins = args.n_bins
    bin_edges = [lo + (hi - lo) * i / n_bins for i in range(n_bins + 1)]

    def bin_of(d: float) -> int:
        if d >= hi:
            return n_bins - 1
        return min(n_bins - 1, int((d - lo) / (hi - lo) * n_bins))

    # Target histogram (target distribution → desired counts in subset)
    tgt_hist = [0] * n_bins
    for d in tgt_durs:
        tgt_hist[bin_of(d)] += 1

    # Convert to fractional sample counts for the requested N
    tgt_total = sum(tgt_hist)
    target_per_bin_float = [args.n * h / tgt_total for h in tgt_hist]
    # Round, then adjust to exactly N
    target_per_bin = [round(x) for x in target_per_bin_float]
    diff = args.n - sum(target_per_bin)
    if diff != 0:
        # distribute the rounding error to the bins with largest fractional remainders
        rems = [(i, target_per_bin_float[i] - int(target_per_bin_float[i])) for i in range(n_bins)]
        rems.sort(key=lambda x: -x[1] if diff > 0 else x[1])
        for i, _ in rems[:abs(diff)]:
            target_per_bin[i] += 1 if diff > 0 else -1
    assert sum(target_per_bin) == args.n

    # Index source clips by bin
    source_by_bin = {i: [] for i in range(n_bins)}
    for e in source:
        source_by_bin[bin_of(e["duration"])].append(e)

    # Check feasibility (every requested bin has enough source clips)
    rng = random.Random(args.seed)
    picked = []
    for b, want in enumerate(target_per_bin):
        avail = source_by_bin[b]
        if want > len(avail):
            print(f"  WARN: bin {b} (dur ~{bin_edges[b]:.1f}-{bin_edges[b+1]:.1f}s): "
                  f"need {want} but source has only {len(avail)} — taking all and rebalancing")
            picked.extend(avail)
        else:
            picked.extend(rng.sample(avail, want))

    # If we under-filled, randomly top up from remaining source clips
    if len(picked) < args.n:
        rest = [e for e in source if e not in picked]
        rng.shuffle(rest)
        picked.extend(rest[: args.n - len(picked)])

    print(f"\nPicked {len(picked)} clips, total duration {sum(e['duration'] for e in picked)/60:.1f} min")
    print(f"  source dur p10/p50/p90: "
          f"{sorted(src_durs)[len(src_durs)//10]:.1f}/"
          f"{sorted(src_durs)[len(src_durs)//2]:.1f}/"
          f"{sorted(src_durs)[len(src_durs)*9//10]:.1f}")
    print(f"  target dur p10/p50/p90: "
          f"{sorted(tgt_durs)[len(tgt_durs)//10]:.1f}/"
          f"{sorted(tgt_durs)[len(tgt_durs)//2]:.1f}/"
          f"{sorted(tgt_durs)[len(tgt_durs)*9//10]:.1f}")
    picked_durs = sorted(e["duration"] for e in picked)
    print(f"  picked dur p10/p50/p90: "
          f"{picked_durs[len(picked_durs)//10]:.1f}/"
          f"{picked_durs[len(picked_durs)//2]:.1f}/"
          f"{picked_durs[len(picked_durs)*9//10]:.1f}")

    # Write output dataset (copies WAVs, deterministic ordering by source filename)
    out_dir = PROJECT_ROOT / "data" / f"cumberbatch_{args.out_name}"
    (out_dir / "wavs").mkdir(parents=True, exist_ok=True)
    picked.sort(key=lambda e: e["audio_file"])  # deterministic order
    metadata_out = out_dir / "metadata.csv"
    with metadata_out.open("w", newline="") as f:
        w = csv.writer(f, delimiter="|")
        w.writerow(["audio_file", "text", "duration"])
        for i, e in enumerate(picked):
            new_id = f"seg_{i+1:05d}"
            dest = out_dir / "wavs" / f"{new_id}.wav"
            if not dest.exists():
                shutil.copy(str(e["wav"]), str(dest))
            w.writerow([new_id, e["text"], f"{e['duration']:.3f}"])
    print(f"\n✓ Wrote {out_dir}")


if __name__ == "__main__":
    main()
