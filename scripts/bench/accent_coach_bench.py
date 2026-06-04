"""Validation bench — runs Experiments A–D and emits a Markdown report."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def _load_manifest(csv_path: Path) -> list[dict]:
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def _run_experiment(
    rows: list[dict],
    label: str,
) -> dict[str, float]:
    """Run analysis on all rows and return mean skill scores + composite."""
    import soundfile as sf

    from accent_coach.calibration.sentences import get_by_id
    from accent_coach.comparison.scoring import compare
    from accent_coach.pipeline.features import analyse_audio
    from accent_coach.reference.rp_norms import get_rp_norms

    skill_totals: dict[str, list[float]] = {}
    composites: list[float] = []

    for row in rows:
        user_wav = Path(row["user_wav"])
        transcript = row["transcript"]
        sentence_id = int(row.get("sentence_id", 0))
        target_wav = Path(row["target_wav"]) if row.get("target_wav") else None

        if not user_wav.exists():
            print(f"  SKIP (missing): {user_wav}")
            continue

        try:
            if sentence_id:
                meta = get_by_id(sentence_id)
            else:
                from accent_coach.calibration.sentences import Sentence
                meta = Sentence(
                    id=0, text=transcript, sentence_type="statement", targets=[]
                )

            user_analysis = analyse_audio(user_wav, transcript, meta)
            audio, sr = sf.read(str(user_wav), always_2d=False)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)

            target_analysis = None
            if target_wav and target_wav.exists():
                target_analysis = analyse_audio(target_wav, transcript, meta)

            mean_f0 = float(
                np.mean(
                    [v.pitch_mean for v in user_analysis.vowels if v.pitch_mean > 70] or [120.0]
                )
            )
            rp_norms = get_rp_norms(mean_f0)
            result = compare(user_analysis, audio.astype(np.float32), sr,
                             target=target_analysis, reference_norms=rp_norms)

            composites.append(result.composite_score)
            for skill, score in result.skill_scores.items():
                skill_totals.setdefault(skill, []).append(score)
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR on {user_wav}: {e}")

    summary: dict[str, float] = {
        "composite": float(np.mean(composites)) if composites else 0.0,
    }
    for skill, vals in skill_totals.items():
        summary[skill] = float(np.mean(vals))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Accent Coach bench — Experiments A–D")
    parser.add_argument(
        "--manifest", required=True, type=Path,
        help="CSV with columns: experiment,speaker,label,user_wav,target_wav,transcript,sentence_id"
    )
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("tts_output/accent_coach/bench") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_manifest(args.manifest)
    experiments = sorted({r["experiment"] for r in rows})

    report_lines = [
        f"# Accent Coach Phase 0 Bench — {run_id}\n",
        f"Manifest: `{args.manifest}`\n\n",
        "| Experiment | Composite | Vowels | Consonants | Aspiration | Rhythm | Stress | Intonation |",
        "|---|---|---|---|---|---|---|---|",
    ]

    all_results: dict[str, dict] = {}
    for exp in experiments:
        exp_rows = [r for r in rows if r["experiment"] == exp]
        print(f"\n=== Experiment {exp} ({len(exp_rows)} rows) ===")
        summary = _run_experiment(exp_rows, exp)
        all_results[exp] = summary
        row_str = (
            f"| {exp} "
            f"| {summary.get('composite', 0):.1f} "
            f"| {summary.get('vowels', 0):.1f} "
            f"| {summary.get('consonants', 0):.1f} "
            f"| {summary.get('aspiration', 0):.1f} "
            f"| {summary.get('rhythm', 0):.1f} "
            f"| {summary.get('stress', 0):.1f} "
            f"| {summary.get('intonation', 0):.1f} |"
        )
        report_lines.append(row_str)

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"\nReport → {report_path}")

    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(all_results, indent=2))
    print(f"JSON   → {json_path}")


if __name__ == "__main__":
    main()
