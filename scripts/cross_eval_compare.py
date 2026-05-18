"""
Compare multiple model checkpoints on the same eval set, breaking down by source.

Reads scores_detail.csv from N posthoc_eval runs (each with --phrases-csv pointing
at the same source-tagged eval set) and produces a markdown report showing the
4-cell breakdown: {model A, B, ...} × {source casanova, sherlock}.

Usage:
  .venv/bin/python scripts/cross_eval_compare.py \
      tts_output/cross_eval_baseline/scores_detail.csv \
      tts_output/cross_eval_sherlock/scores_detail.csv \
      tts_output/cross_eval_casanova/scores_detail.csv \
      --labels baseline sherlock_ft casanova_ft \
      --out tts_output/cross_eval_report.md
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def load_detail(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            # Coerce numeric fields
            try:
                row["wer"] = float(row["wer"])
                row["ecapa_sim"] = float(row["ecapa_sim"])
                row["ecapa_centroid_sim"] = float(row["ecapa_centroid_sim"]) if row["ecapa_centroid_sim"] else float("nan")
                row["dnsmos_ovr"] = float(row["dnsmos_ovr"]) if row["dnsmos_ovr"] else float("nan")
            except (ValueError, TypeError):
                continue
            rows.append(row)
    return rows


def aggregate(rows: list[dict], filter_fn=None) -> dict:
    if filter_fn:
        rows = [r for r in rows if filter_fn(r)]
    def mean_std(key):
        vals = [r[key] for r in rows if not np.isnan(r.get(key, float("nan")))]
        if not vals:
            return float("nan"), float("nan"), 0
        return float(np.mean(vals)), float(np.std(vals)), len(vals)
    wer = mean_std("wer")
    ecapa = mean_std("ecapa_sim")
    cent = mean_std("ecapa_centroid_sim")
    ovr = mean_std("dnsmos_ovr")
    return {
        "n": wer[2],
        "wer": wer[0], "wer_std": wer[1],
        "ecapa": ecapa[0], "ecapa_std": ecapa[1],
        "cent": cent[0], "cent_std": cent[1],
        "ovr": ovr[0],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("detail_csvs", nargs="+", help="Per-model scores_detail.csv paths")
    parser.add_argument("--labels", nargs="+", required=True, help="Label per CSV")
    parser.add_argument("--out", required=True, help="Output markdown report")
    args = parser.parse_args()
    if len(args.labels) != len(args.detail_csvs):
        sys.exit("Need same number of --labels as detail_csvs")

    models = []
    for label, path in zip(args.labels, args.detail_csvs):
        rows = load_detail(Path(path))
        if not rows:
            sys.exit(f"No rows loaded from {path}")
        models.append((label, rows))

    sources = sorted({r["source"] for _, rows in models for r in rows if r.get("source")})
    print(f"Models: {[m[0] for m in models]}")
    print(f"Sources detected: {sources}")

    lines = ["# Cross-eval comparison\n"]
    lines.append("Eval set: 50 clips total. Metrics: WER ↓, ECAPA / CENT / DNSMOS-OVR ↑.\n")

    # Overall table (across all sources)
    lines.append("\n## Overall (all sources combined)\n")
    lines.append("| Model | n | WER | ECAPA | CENT | DNSMOS-OVR |")
    lines.append("|---|---|---|---|---|---|")
    for label, rows in models:
        a = aggregate(rows)
        lines.append(f"| `{label}` | {a['n']} | {a['wer']:.3f} ± {a['wer_std']:.2f} | "
                     f"{a['ecapa']:.4f} ± {a['ecapa_std']:.3f} | "
                     f"{a['cent']:.4f} ± {a['cent_std']:.3f} | {a['ovr']:.2f} |")

    # Per-source breakdown (the 4-cell story)
    for src in sources:
        lines.append(f"\n## Source: `{src}` only\n")
        lines.append("| Model | n | WER | ECAPA | CENT | DNSMOS-OVR |")
        lines.append("|---|---|---|---|---|---|")
        for label, rows in models:
            a = aggregate(rows, filter_fn=lambda r: r.get("source") == src)
            lines.append(f"| `{label}` | {a['n']} | {a['wer']:.3f} ± {a['wer_std']:.2f} | "
                         f"{a['ecapa']:.4f} ± {a['ecapa_std']:.3f} | "
                         f"{a['cent']:.4f} ± {a['cent_std']:.3f} | {a['ovr']:.2f} |")

    # Δ-vs-baseline per model per source (signal-to-noise check)
    base_label = args.labels[0]
    lines.append(f"\n## Δ vs `{base_label}` (positive = better; sign-flipped for WER)\n")
    lines.append("|Model × Source | ΔWER | ΔECAPA | ΔCENT |")
    lines.append("|---|---|---|---|")
    base_rows = models[0][1]
    for label, rows in models[1:]:
        for src in sources:
            base_a = aggregate(base_rows, filter_fn=lambda r: r.get("source") == src)
            a = aggregate(rows, filter_fn=lambda r: r.get("source") == src)
            d_wer = base_a["wer"] - a["wer"]  # lower is better → positive delta = improvement
            d_ecapa = a["ecapa"] - base_a["ecapa"]
            d_cent = a["cent"] - base_a["cent"]
            lines.append(f"| `{label}` × `{src}` | {d_wer:+.3f} | {d_ecapa:+.4f} | {d_cent:+.4f} |")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"\n✓ {out}")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
