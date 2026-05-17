"""
Side-by-side comparison of posthoc_eval.py runs.

Reads scores.csv + scores_detail.csv from each given eval dir, produces a
Markdown report: aggregate table, per-phrase ECAPA, per-phrase WER, and
deltas vs the first (baseline) dir.

Usage:
    python scripts/eval_compare.py \
        tts_output/posthoc_6phrase \
        tts_output/sweep_cfg30 \
        tts_output/sweep_cfg40 \
        [--baseline tts_output/posthoc_6phrase] \
        [--out report.md]

Defaults:
    --baseline = first dir
    --out      = stdout
"""

import argparse
import csv
import sys
from pathlib import Path


def read_summary(eval_dir: Path) -> list[dict]:
    csv_path = eval_dir / "scores.csv"
    if not csv_path.exists():
        return []
    with csv_path.open() as f:
        return [r for r in csv.DictReader(f)]


def read_detail(eval_dir: Path) -> list[dict]:
    csv_path = eval_dir / "scores_detail.csv"
    if not csv_path.exists():
        return []
    with csv_path.open() as f:
        return [r for r in csv.DictReader(f)]


def fnum(s: str, fmt: str = "{:.3f}") -> str:
    try:
        v = float(s)
        if v != v:  # NaN
            return "n/a"
        return fmt.format(v)
    except (ValueError, TypeError):
        return "n/a"


def delta(a: str, b: str, fmt: str = "{:+.3f}") -> str:
    try:
        d = float(a) - float(b)
        return fmt.format(d)
    except (ValueError, TypeError):
        return ""


def emit(lines: list[str], out: Path | None):
    text = "\n".join(lines) + "\n"
    if out:
        out.write_text(text)
        print(f"→ {out}", file=sys.stderr)
    else:
        print(text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("eval_dirs", nargs="+", type=Path)
    p.add_argument("--baseline", type=Path, default=None,
                   help="Eval dir whose scores are subtracted from all others. Default: first dir.")
    p.add_argument("--out", type=Path, default=None,
                   help="Write Markdown report to this file. Default: stdout.")
    p.add_argument("--config-label", action="append", default=[],
                   help="Override dir-name labels: --config-label posthoc_6phrase=cfg2.0  (repeatable)")
    args = p.parse_args()

    label_overrides = {}
    for kv in args.config_label:
        k, _, v = kv.partition("=")
        label_overrides[k] = v

    # Build list of (dir_label, dir_path, [summary rows]).
    # Each summary row is one "config" (baseline or one checkpoint).
    configs = []  # list of (config_label, dir_path, summary_row)
    for d in args.eval_dirs:
        dir_label = label_overrides.get(d.name, d.name)
        rows = read_summary(d)
        if not rows:
            print(f"WARN: {d}/scores.csv missing or empty — skipping", file=sys.stderr)
            continue
        for r in rows:
            row_label = f"{dir_label}/{r['label']}" if len(rows) > 1 else dir_label
            configs.append((row_label, d, r))

    if not configs:
        print("No configs found.", file=sys.stderr)
        sys.exit(1)

    baseline_dir = args.baseline or args.eval_dirs[0]
    baseline_label = label_overrides.get(baseline_dir.name, baseline_dir.name)
    baseline_rows = read_summary(baseline_dir)
    if not baseline_rows:
        print(f"Baseline {baseline_dir} has no scores.csv — aborting", file=sys.stderr)
        sys.exit(1)
    baseline_row = baseline_rows[0]  # use first row of baseline dir as the reference

    lines = []
    lines.append(f"# Eval comparison — baseline: `{baseline_label}/{baseline_row['label']}`")
    lines.append("")
    lines.append(f"Metrics: WER ↓ (lower better), ECAPA / CENT ↑, DNSMOS ↑")
    lines.append("- **ECAPA** = cosine sim of gen vs single 12s ref clip (`tts_output/ref_narrator.wav`)")
    lines.append("- **CENT** = cosine sim of gen vs *centroid* of N Cumberbatch dataset clips (more robust identity measure)")
    lines.append("")

    # Aggregate table
    lines.append("## Aggregate (mean across phrases)")
    lines.append("")
    lines.append("| Config | n | WER | ECAPA | CENT | DNSMOS OVR | ΔECAPA | ΔWER |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for (label, d, r) in configs:
        d_ecapa = delta(r["ecapa_sim_mean"], baseline_row["ecapa_sim_mean"], "{:+.3f}")
        d_wer = delta(r["wer_mean"], baseline_row["wer_mean"], "{:+.3f}")
        lines.append(f"| `{label}` | {r['n_scored']} | "
                     f"{fnum(r['wer_mean'])} | {fnum(r['ecapa_sim_mean'], '{:.4f}')} | "
                     f"{fnum(r.get('ecapa_centroid_mean', 'nan'), '{:.4f}')} | "
                     f"{fnum(r['dnsmos_ovr_mean'], '{:.2f}')} | "
                     f"{d_ecapa} | {d_wer} |")
    lines.append("")

    # Per-phrase ECAPA table
    detail_by_dir = {d: read_detail(d) for (_, d, _) in configs}
    phrases = sorted({row["slug"] for rows in detail_by_dir.values() for row in rows})

    def per_phrase_table(metric_key: str, title: str, fmt: str = "{:.4f}"):
        lines.append(f"## Per-phrase {title}")
        lines.append("")
        header_cells = ["Phrase"] + [f"`{lbl}`" for (lbl, _, _) in configs]
        lines.append("| " + " | ".join(header_cells) + " |")
        lines.append("|" + "---|" * (len(configs) + 1))
        for slug in phrases:
            row = [slug]
            for (_, d, r) in configs:
                # Find the matching detail row by (label, slug)
                cell = "n/a"
                for det in detail_by_dir.get(d, []):
                    if det["slug"] == slug and det["label"] == r["label"]:
                        cell = fnum(det[metric_key], fmt)
                        break
                row.append(cell)
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    per_phrase_table("ecapa_sim", "ECAPA (single-ref)", "{:.4f}")
    per_phrase_table("ecapa_centroid_sim", "CENT (centroid)", "{:.4f}")
    per_phrase_table("wer", "WER", "{:.3f}")

    # Quick verdict
    lines.append("## Verdict")
    lines.append("")
    best_ecapa = max(configs, key=lambda c: float(c[2]["ecapa_sim_mean"] or "nan"))
    best_wer = min(configs, key=lambda c: float(c[2]["wer_mean"] or "inf"))
    lines.append(f"- Best **ECAPA**: `{best_ecapa[0]}` ({fnum(best_ecapa[2]['ecapa_sim_mean'], '{:.4f}')})")
    lines.append(f"- Best **WER**:   `{best_wer[0]}` ({fnum(best_wer[2]['wer_mean'])})")

    emit(lines, args.out)


if __name__ == "__main__":
    main()
