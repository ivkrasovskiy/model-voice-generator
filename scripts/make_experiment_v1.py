"""
Set up the data-driven weighting experiment.

Builds three training datasets and one cross-eval set from the cleaned sources:

  Input datasets (must exist first):
    data/cumberbatch_casanova_clean       — 1735 clips (sim ≥ 0.65 to own centroid)
    data/cumberbatch_sherlock_narrator    —  379 clips (Sherlock narrator cluster)

  Output:
    data/cumberbatch_sherlock_train_354   — 354 Sherlock narrator clips (val removed)
    data/cumberbatch_casanova_train_354   — 354 Casanova clips, duration-matched
                                            to Sherlock train, val excluded
    tts_output/cross_eval_50/eval_50.csv  — 50 phrases: 25 Casanova + 25 Sherlock,
                                            tagged with source label
"""

import csv
import random
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SEED = 42
N_VAL = 25


def load_entries(name: str) -> list[dict]:
    src_dir = PROJECT_ROOT / "data" / name
    metadata = src_dir / "metadata.csv"
    entries = []
    with metadata.open() as f:
        for row in csv.DictReader(f, delimiter="|"):
            wav = src_dir / "wavs" / f"{row['audio_file']}.wav"
            if wav.exists():
                entries.append({
                    "audio_file": row["audio_file"],
                    "text": row["text"],
                    "duration": float(row["duration"]),
                    "wav": wav,
                    "source_dataset": name,
                })
    return entries


def deterministic_split(entries: list[dict], n_val: int, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = list(entries)
    rng.shuffle(shuffled)
    val = shuffled[:n_val]
    train = shuffled[n_val:]
    return train, val


def write_dataset(name: str, entries: list[dict]):
    out_dir = PROJECT_ROOT / "data" / name
    (out_dir / "wavs").mkdir(parents=True, exist_ok=True)
    sorted_entries = sorted(entries, key=lambda e: e["audio_file"])  # stable order
    with (out_dir / "metadata.csv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="|")
        w.writerow(["audio_file", "text", "duration"])
        for i, e in enumerate(sorted_entries):
            new_id = f"seg_{i+1:05d}"
            dest = out_dir / "wavs" / f"{new_id}.wav"
            if not dest.exists():
                shutil.copy(str(e["wav"]), str(dest))
            w.writerow([new_id, e["text"], f"{e['duration']:.3f}"])
    return out_dir


def stratified_match(source: list[dict], target_durs: list[float], n: int,
                     seed: int, n_bins: int = 20) -> list[dict]:
    """Pick n entries from source whose duration histogram matches target_durs."""
    src_durs = [e["duration"] for e in source]
    lo = min(min(src_durs), min(target_durs))
    hi = max(max(src_durs), max(target_durs))

    def bin_of(d: float) -> int:
        if d >= hi:
            return n_bins - 1
        return min(n_bins - 1, int((d - lo) / (hi - lo) * n_bins))

    tgt_hist = [0] * n_bins
    for d in target_durs:
        tgt_hist[bin_of(d)] += 1
    tgt_total = sum(tgt_hist)
    per_bin_float = [n * h / tgt_total for h in tgt_hist]
    per_bin = [round(x) for x in per_bin_float]
    diff = n - sum(per_bin)
    if diff != 0:
        rems = [(i, per_bin_float[i] - int(per_bin_float[i])) for i in range(n_bins)]
        rems.sort(key=lambda x: -x[1] if diff > 0 else x[1])
        for i, _ in rems[:abs(diff)]:
            per_bin[i] += 1 if diff > 0 else -1
    rng = random.Random(seed)
    by_bin = {i: [] for i in range(n_bins)}
    for e in source:
        by_bin[bin_of(e["duration"])].append(e)
    picked = []
    for b, want in enumerate(per_bin):
        avail = by_bin[b]
        if want >= len(avail):
            picked.extend(avail)
        else:
            picked.extend(rng.sample(avail, want))
    if len(picked) < n:
        rest = [e for e in source if e not in picked]
        rng.shuffle(rest)
        picked.extend(rest[: n - len(picked)])
    return picked[:n]


def main():
    print("=== Loading cleaned sources ===")
    cas = load_entries("cumberbatch_casanova_clean")
    sher = load_entries("cumberbatch_sherlock_narrator")
    print(f"  casanova_clean: {len(cas)} clips")
    print(f"  sherlock_narrator: {len(sher)} clips")

    print(f"\n=== Holding out {N_VAL} val clips per source (seed={SEED}) ===")
    cas_train_pool, cas_val = deterministic_split(cas, N_VAL, SEED)
    sher_train_pool, sher_val = deterministic_split(sher, N_VAL, SEED)
    print(f"  casanova: {len(cas_train_pool)} train pool / {len(cas_val)} val")
    print(f"  sherlock: {len(sher_train_pool)} train pool / {len(sher_val)} val")

    print("\n=== Building cross-eval set (eval_50.csv) ===")
    eval_dir = PROJECT_ROOT / "tts_output" / "cross_eval_50"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_csv = eval_dir / "eval_50.csv"
    with eval_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slug", "prompt", "source", "ref_audio_path", "ref_duration"])
        for i, e in enumerate(cas_val):
            w.writerow([f"cas_{i+1:02d}", e["text"], "casanova",
                        str(e["wav"].relative_to(PROJECT_ROOT)), f"{e['duration']:.3f}"])
        for i, e in enumerate(sher_val):
            w.writerow([f"sher_{i+1:02d}", e["text"], "sherlock",
                        str(e["wav"].relative_to(PROJECT_ROOT)), f"{e['duration']:.3f}"])
    print(f"  → {eval_csv} ({len(cas_val) + len(sher_val)} clips)")

    print("\n=== Writing Sherlock training set (354 clips) ===")
    sher_train_dir = write_dataset("cumberbatch_sherlock_train_354", sher_train_pool)
    sher_total_dur = sum(e["duration"] for e in sher_train_pool)
    print(f"  → {sher_train_dir} ({len(sher_train_pool)} clips, {sher_total_dur/60:.1f} min)")

    print("\n=== Sampling matched Casanova training set (354 clips) ===")
    n_match = len(sher_train_pool)
    sher_train_durs = [e["duration"] for e in sher_train_pool]
    cas_matched = stratified_match(cas_train_pool, sher_train_durs, n_match, SEED)
    cas_train_dir = write_dataset("cumberbatch_casanova_train_354", cas_matched)
    cas_total_dur = sum(e["duration"] for e in cas_matched)
    print(f"  → {cas_train_dir} ({len(cas_matched)} clips, {cas_total_dur/60:.1f} min)")

    # Sanity: duration distribution comparison
    def p(durs, q):
        s = sorted(durs)
        return s[int(len(s) * q)]
    cas_match_durs = [e["duration"] for e in cas_matched]
    print("\nDuration distribution check:")
    print(f"  {'set':<25}  {'p10':>5}  {'p50':>5}  {'p90':>5}")
    for label, ds in [
        ("sherlock_train (target)", sher_train_durs),
        ("casanova_train (matched)", cas_match_durs),
        ("casanova_train_pool (all)", [e["duration"] for e in cas_train_pool]),
    ]:
        print(f"  {label:<25}  {p(ds, 0.10):>5.1f}  {p(ds, 0.50):>5.1f}  {p(ds, 0.90):>5.1f}")

    print("\n✓ Experiment setup complete. Run training with:")
    print("  finetune_f5.py --dataset cumberbatch_sherlock_train_354 ...")
    print("  finetune_f5.py --dataset cumberbatch_casanova_train_354 ...")


if __name__ == "__main__":
    main()
