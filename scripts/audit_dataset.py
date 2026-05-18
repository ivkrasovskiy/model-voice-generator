"""
Audit a Cumberbatch dataset by ECAPA-similarity to the dataset's own centroid.

For each clip, compute cosine similarity to the *robust* centroid (mean of all
embeddings, with outliers excluded iteratively). Outliers fall into 3 buckets:

  - very-low (sim < 0.50)  — likely different speaker, music, ad, or pure noise
  - low (sim < 0.65)       — possibly character voice or noisy clip
  - normal (sim >= 0.65)   — assumed narrator

Output:
  data/<dataset>/audit.csv    — per-clip sim, duration, suggested action
  data/<dataset>/centroid.npy — saved centroid embedding for downstream filtering
  printed summary             — histogram, suspect clips, suggested drops

Usage:
  .venv/bin/python scripts/audit_dataset.py --dataset cumberbatch_casanova
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _dotenv_init import init_env_then_reexec
init_env_then_reexec(__file__)

import warnings
warnings.filterwarnings("ignore")

import numpy as np

from lib.dataset import load_metadata
from lib.identity import load_ecapa, embed_file, cosine, robust_centroid

PROJECT_ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cumberbatch_casanova")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only embed first N clips (for testing)")
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / "data" / args.dataset
    entries = load_metadata(args.dataset, project_root=PROJECT_ROOT)
    if args.limit:
        entries = entries[:args.limit]
    print(f"Auditing {len(entries)} clips from {args.dataset}")

    ecapa = load_ecapa(device="cpu")
    print("Embedding all clips (this takes a few minutes)...")

    embs = []
    valid_entries = []
    for i, e in enumerate(entries):
        emb = embed_file(e["wav"], ecapa)
        if emb is None:
            print(f"  failed: {e['wav'].name}")
            continue
        embs.append(emb)
        valid_entries.append(e)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(entries)}")
    embs = np.stack(embs)
    print(f"Embedded {len(embs)} clips, dim={embs.shape[1]}")

    print("\nComputing robust centroid...")
    centroid = robust_centroid(embs)
    print(f"  centroid norm={np.linalg.norm(centroid):.3f}")

    sims = np.array([cosine(centroid, e) for e in embs])
    print(f"\nSimilarity distribution:")
    print(f"  min:    {sims.min():.4f}")
    print(f"  p10:    {np.quantile(sims, 0.10):.4f}")
    print(f"  p25:    {np.quantile(sims, 0.25):.4f}")
    print(f"  median: {np.median(sims):.4f}")
    print(f"  p75:    {np.quantile(sims, 0.75):.4f}")
    print(f"  p90:    {np.quantile(sims, 0.90):.4f}")
    print(f"  max:    {sims.max():.4f}")

    very_low = sims < 0.50
    low = (sims >= 0.50) & (sims < 0.65)
    normal = sims >= 0.65
    print(f"\nBins (threshold suggestions):")
    print(f"  very-low (<0.50): {very_low.sum():4d} clips ({very_low.mean()*100:.1f}%) — likely DROP")
    print(f"  low (0.50-0.65):  {low.sum():4d} clips ({low.mean()*100:.1f}%) — REVIEW (character voice/noise)")
    print(f"  normal (>=0.65):  {normal.sum():4d} clips ({normal.mean()*100:.1f}%) — keep")

    centroid_path = data_dir / "centroid.npy"
    np.save(str(centroid_path), centroid)
    print(f"\n✓ centroid saved → {centroid_path}")

    audit_csv = data_dir / "audit.csv"
    order = np.argsort(sims)
    with audit_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["audio_file", "duration", "ecapa_sim", "bin", "text"])
        for i in order:
            e = valid_entries[i]
            sim = sims[i]
            bin_name = ("very-low" if sim < 0.50 else
                        "low" if sim < 0.65 else "normal")
            w.writerow([e["audio_file"], f"{e['duration']:.2f}",
                        f"{sim:.4f}", bin_name, e["text"][:120]])
    print(f"✓ audit CSV → {audit_csv}")

    print(f"\n=== Worst 15 clips (lowest ECAPA sim to dataset centroid) ===")
    print(f"{'idx':>5}  {'sim':>7}  {'dur':>5}  {'file':>14}  text...")
    for i in order[:15]:
        e = valid_entries[i]
        print(f"{i:>5}  {sims[i]:>7.4f}  {e['duration']:>5.1f}  {e['audio_file']:>14}  "
              f"{e['text'][:80]}")


if __name__ == "__main__":
    main()
