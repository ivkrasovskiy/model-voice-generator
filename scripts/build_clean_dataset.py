"""
Build a combined / cleaned dataset by merging filtered subsets of one or more
source audiobook datasets.

Inputs are described by a sources list where each entry says:
  - source dataset name (under data/)
  - filter rule: "audit_min_sim X", "cluster <cluster_csv> <utts_dir>", or "metadata"

Output: data/cumberbatch_<name>/  with renumbered seg_NNNNN.wav + metadata.csv
       plus a manifest.json describing the provenance of each clip.

Usage examples:
  # Clean Casanova: drop clips with sim < 0.50 from its own centroid
  .venv/bin/python scripts/build_clean_dataset.py \
      --out-name casanova_clean \
      --source cumberbatch_casanova audit_min_sim 0.50

  # Build combined: pre-filtered Casanova + pre-filtered Sherlock narrator
  .venv/bin/python scripts/build_clean_dataset.py \
      --out-name combined_cas_sher \
      --source cumberbatch_casanova_clean metadata \
      --source cumberbatch_sherlock_narrator metadata
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def filter_audit(source_dir: Path, min_sim: float) -> set[str]:
    """Read audit.csv from source dataset, return audio_file names to KEEP (sim >= min_sim)."""
    audit_csv = source_dir / "audit.csv"
    if not audit_csv.exists():
        raise FileNotFoundError(f"No audit.csv at {audit_csv}. Run audit_dataset.py first.")
    keep = set()
    with audit_csv.open() as f:
        for row in csv.DictReader(f):
            sim = float(row["ecapa_sim"])
            if sim >= min_sim:
                keep.add(row["audio_file"])
    print(f"  audit filter (sim>={min_sim}): keeping {len(keep)} clips")
    return keep


def filter_cluster(cluster_csv: Path) -> set[str]:
    """Read cluster_labels.csv, return utt files in the narrator cluster."""
    keep = set()
    with cluster_csv.open() as f:
        for row in csv.DictReader(f):
            if int(row["is_narrator_cluster"]):
                keep.add(row["utt_file"])
    print(f"  cluster filter: keeping {len(keep)} clips in narrator cluster")
    return keep


def add_source_audit(source_name: str, min_sim: float, all_rows: list, all_wavs: dict):
    """Add clips from an audit-filtered dataset (audio_file = canonical name)."""
    src_dir = PROJECT_ROOT / "data" / source_name
    keep = filter_audit(src_dir, min_sim)
    metadata_csv = src_dir / "metadata.csv"
    n_added = 0
    with metadata_csv.open() as f:
        for row in csv.DictReader(f, delimiter="|"):
            if row["audio_file"] in keep:
                wav_path = src_dir / "wavs" / f"{row['audio_file']}.wav"
                if not wav_path.exists():
                    continue
                key = f"{source_name}__{row['audio_file']}"
                all_wavs[key] = wav_path
                all_rows.append({
                    "key": key, "source": source_name,
                    "original_id": row["audio_file"],
                    "text": row["text"], "duration": float(row["duration"]),
                })
                n_added += 1
    print(f"  → added {n_added} clips from {source_name}")


def add_source_metadata(source_name: str, all_rows: list, all_wavs: dict):
    """Add all clips from a pre-filtered dataset's metadata.csv without re-filtering."""
    src_dir = PROJECT_ROOT / "data" / source_name
    metadata_csv = src_dir / "metadata.csv"
    if not metadata_csv.exists():
        raise FileNotFoundError(f"No metadata.csv at {metadata_csv}")
    n_added = 0
    with metadata_csv.open() as f:
        for row in csv.DictReader(f, delimiter="|"):
            wav_path = src_dir / "wavs" / f"{row['audio_file']}.wav"
            if not wav_path.exists():
                continue
            key = f"{source_name}__{row['audio_file']}"
            all_wavs[key] = wav_path
            all_rows.append({
                "key": key, "source": source_name,
                "original_id": row["audio_file"],
                "text": row["text"], "duration": float(row["duration"]),
            })
            n_added += 1
    print(f"  → added {n_added} clips from {source_name}")


def add_source_cluster(source_name: str, cluster_csv: Path, utts_dir: Path,
                       all_rows: list, all_wavs: dict):
    """Add clips from a clustered audiobook work dir.

    Need to also load transcripts from work_dir/transcripts.json.
    """
    work_dir = cluster_csv.parent
    transcripts_path = work_dir / "transcripts.json"
    if not transcripts_path.exists():
        raise FileNotFoundError(transcripts_path)
    transcripts = json.loads(transcripts_path.read_text())

    keep = filter_cluster(cluster_csv)
    n_added = 0
    n_no_text = 0
    import soundfile as sf
    for utt_file in sorted(keep):
        utt_path = utts_dir / utt_file
        if not utt_path.exists():
            continue
        text = transcripts.get(utt_file, "").strip()
        if not text:
            n_no_text += 1
            continue
        info = sf.info(str(utt_path))
        key = f"{source_name}__{utt_path.stem}"
        all_wavs[key] = utt_path
        all_rows.append({
            "key": key, "source": source_name,
            "original_id": utt_path.stem,
            "text": text, "duration": float(info.duration),
        })
        n_added += 1
    print(f"  → added {n_added} clips from {source_name} (skipped {n_no_text} without transcript)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-name", required=True)
    parser.add_argument("--source", action="append", nargs="+", required=True,
                        help="Source spec: '<name> audit_min_sim <X>' or "
                             "'<name> cluster <cluster_csv> <utts_dir>'")
    parser.add_argument("--shuffle", action="store_true", default=True,
                        help="Shuffle clip order in output metadata (default True)")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / "data" / f"cumberbatch_{args.out_name}"
    out_wavs = out_dir / "wavs"
    out_wavs.mkdir(parents=True, exist_ok=True)

    print(f"Output: {out_dir}\n")

    all_rows: list[dict] = []
    all_wavs: dict[str, Path] = {}

    for spec in args.source:
        if len(spec) < 2:
            print(f"Bad source spec: {spec}")
            continue
        source_name, rule = spec[0], spec[1]
        print(f"Source: {source_name}  rule: {rule}")
        if rule == "audit_min_sim":
            if len(spec) != 3:
                raise ValueError(f"audit_min_sim needs 1 arg, got {spec}")
            add_source_audit(source_name, float(spec[2]), all_rows, all_wavs)
        elif rule == "cluster":
            if len(spec) != 4:
                raise ValueError(f"cluster needs <cluster_csv> <utts_dir>, got {spec}")
            add_source_cluster(source_name, Path(spec[2]).resolve(),
                               Path(spec[3]).resolve(), all_rows, all_wavs)
        elif rule == "metadata":
            if len(spec) != 2:
                raise ValueError(f"metadata rule takes no extra args, got {spec}")
            add_source_metadata(source_name, all_rows, all_wavs)
        else:
            raise ValueError(f"Unknown rule: {rule}")
        print()

    # Shuffle (deterministic) so source mixing happens at training time
    if args.shuffle:
        import random
        rng = random.Random(42)
        rng.shuffle(all_rows)

    # Write metadata + copy WAVs
    metadata_csv = out_dir / "metadata.csv"
    manifest_json = out_dir / "manifest.json"
    manifest = []
    with metadata_csv.open("w", newline="") as f:
        w = csv.writer(f, delimiter="|")
        w.writerow(["audio_file", "text", "duration"])
        for i, row in enumerate(all_rows):
            new_id = f"seg_{i+1:05d}"
            dest = out_wavs / f"{new_id}.wav"
            if not dest.exists():
                shutil.copy(str(all_wavs[row["key"]]), str(dest))
            w.writerow([new_id, row["text"], f"{row['duration']:.3f}"])
            manifest.append({
                "new_id": new_id, "source": row["source"],
                "original_id": row["original_id"],
                "duration": row["duration"], "text": row["text"][:80],
            })

    manifest_json.write_text(json.dumps(manifest, indent=2))
    total_sec = sum(r["duration"] for r in all_rows)
    print(f"\n✓ Output: {len(all_rows)} clips, {total_sec/60:.1f} min ({total_sec/3600:.2f} h)")
    print(f"  metadata → {metadata_csv}")
    print(f"  manifest → {manifest_json}")

    # By-source breakdown
    by_source = {}
    for r in all_rows:
        by_source.setdefault(r["source"], []).append(r)
    print("\nBy source:")
    for src, rows in sorted(by_source.items()):
        sec = sum(r["duration"] for r in rows)
        print(f"  {src:<25}  {len(rows):5d} clips  {sec/60:>5.1f} min")


if __name__ == "__main__":
    main()
