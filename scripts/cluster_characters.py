"""
Cluster VAD utterances into character/speaker groups (k-means on ECAPA embeddings),
then identify the cluster whose mean voice is closest to a target narrator centroid.

Used after process_audiobook.py to refine the dataset: instead of a simple
"keep clips with sim >= threshold" rule (which can mix multiple characters that all
happen to be close to target), we identify the natural voice clusters and pick
the most-narrator-like one.

Usage:
  .venv/bin/python scripts/cluster_characters.py \
      --utts-dir dataset/audiobook_work/sherlock/utts \
      --narrator-centroid data/cumberbatch_casanova/centroid.npy \
      --work-dir dataset/audiobook_work/sherlock \
      --k-list 3 5 7

Output:
  <work-dir>/cluster_labels.csv     — per-utterance cluster assignment for best k
  <work-dir>/cluster_summary.json   — cluster stats (size, sim, sample texts)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _dotenv_init import init_env_then_reexec

init_env_then_reexec(__file__)

import warnings

warnings.filterwarnings("ignore")

import numpy as np
from lib.identity import cosine, embed_file, load_ecapa

PROJECT_ROOT = Path(__file__).parent.parent


def embed_utts(utts: list[Path], ecapa) -> tuple[np.ndarray, list[Path]]:
    """Embed every utterance; drop unreadable ones. Returns (embs[N, D], valid_paths[N])."""
    embs = []
    valid = []
    for i, p in enumerate(utts):
        emb = embed_file(p, ecapa)
        if emb is None:
            print(f"  failed {p.name}")
            continue
        embs.append(emb)
        valid.append(p)
        if (i + 1) % 200 == 0:
            print(f"  embedded {i+1}/{len(utts)}")
    return np.stack(embs), valid


def cluster_eval(embs: np.ndarray, k: int, centroid: np.ndarray, n_iter: int = 20):
    """Run k-means and report per-cluster stats."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=42, n_init=n_iter).fit(embs)
    labels = km.labels_
    stats = []
    for c in range(k):
        mask = labels == c
        if mask.sum() == 0:
            continue
        cluster_mean = embs[mask].mean(axis=0)
        sim_to_centroid = cosine(cluster_mean, centroid)
        within_sims = [cosine(cluster_mean, e) for e in embs[mask]]
        stats.append({
            "cluster": int(c),
            "size": int(mask.sum()),
            "sim_to_target": float(sim_to_centroid),
            "within_mean_sim": float(np.mean(within_sims)),
            "within_min_sim": float(np.min(within_sims)),
        })
    stats.sort(key=lambda s: -s["sim_to_target"])
    return labels, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--utts-dir", required=True)
    parser.add_argument("--narrator-centroid", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--k-list", nargs="+", type=int, default=[3, 5, 7],
                        help="Try k-means with these k values")
    args = parser.parse_args()

    utts_dir = Path(args.utts_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    centroid = np.load(args.narrator_centroid)
    print(f"Target centroid loaded: dim={centroid.shape}, norm={np.linalg.norm(centroid):.3f}")

    utts = sorted(utts_dir.glob("*.wav"))
    print(f"Found {len(utts)} utterances in {utts_dir}")

    ecapa = load_ecapa(device="cpu")

    embs_path = work_dir / "embs.npy"
    valid_path = work_dir / "embs_valid_files.json"
    if embs_path.exists() and valid_path.exists():
        embs = np.load(str(embs_path))
        valid = [Path(p) for p in json.loads(valid_path.read_text())]
        print(f"Loaded cached embeddings: {embs.shape}")
    else:
        print("Embedding all utterances (this takes a few minutes)...")
        embs, valid = embed_utts(utts, ecapa)
        np.save(str(embs_path), embs)
        valid_path.write_text(json.dumps([str(p) for p in valid]))
        print(f"Saved embeddings to {embs_path}")

    print(f"\nTrying k = {args.k_list}")
    all_results = {}
    for k in args.k_list:
        print(f"\n=== k={k} ===")
        labels, stats = cluster_eval(embs, k, centroid)
        all_results[k] = {"labels": labels.tolist(), "stats": stats}
        print(f"{'cluster':>8}  {'size':>5}  {'sim→cas':>8}  {'within':>7}")
        for s in stats:
            star = "  ★" if s == stats[0] else ""
            print(f"{s['cluster']:>8}  {s['size']:>5}  {s['sim_to_target']:>8.4f}  "
                  f"{s['within_mean_sim']:>7.4f}{star}")
        top = stats[0]
        print(f"  → best: cluster {top['cluster']}  size={top['size']}  "
              f"sim_to_casanova={top['sim_to_target']:.4f}")

    # Pick the best k automatically: top cluster's sim weighted by separation from next
    best_k = None
    best_score = -1.0
    for k in args.k_list:
        stats = all_results[k]["stats"]
        if len(stats) < 2:
            continue
        top, second = stats[0], stats[1]
        if top["size"] < 0.05 * len(embs):
            continue
        sep = top["sim_to_target"] - second["sim_to_target"]
        score = top["sim_to_target"] + 0.5 * sep
        if score > best_score:
            best_score = score
            best_k = k
    if best_k is None:
        best_k = args.k_list[0]
    print(f"\n*** Auto-selected k={best_k} ***")

    best_labels = np.array(all_results[best_k]["labels"])
    best_stats = all_results[best_k]["stats"]
    best_cluster_id = best_stats[0]["cluster"]
    best_mask = best_labels == best_cluster_id
    best_files = [valid[i] for i, m in enumerate(best_mask) if m]
    print(f"Best (narrator) cluster: id={best_cluster_id}, size={best_mask.sum()}")

    labels_csv = work_dir / "cluster_labels.csv"
    with labels_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["utt_file", "cluster", "is_narrator_cluster"])
        for path, lab in zip(valid, best_labels, strict=True):
            w.writerow([path.name, int(lab), int(lab == best_cluster_id)])
    print(f"✓ labels → {labels_csv}")

    summary = {
        "best_k": best_k,
        "narrator_cluster_id": best_cluster_id,
        "narrator_cluster_size": int(best_mask.sum()),
        "all_k_results": {str(k): all_results[k]["stats"] for k in args.k_list},
        "narrator_utt_files": [str(p.relative_to(PROJECT_ROOT)) for p in best_files],
    }
    summary_path = work_dir / "cluster_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"✓ summary → {summary_path}")
    print(f"\nNarrator cluster has {len(best_files)} clips.")


if __name__ == "__main__":
    main()
