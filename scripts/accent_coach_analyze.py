"""Analyze a single (user.wav, optional target.wav, transcript) triplet."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), sr


def main() -> None:
    parser = argparse.ArgumentParser(description="Accent Coach — single-sentence analysis")
    parser.add_argument("--user-wav", required=True, type=Path)
    parser.add_argument("--target-wav", type=Path, default=None)
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--sentence-id", type=int, default=0)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    from accent_coach.calibration.sentences import get_by_id
    from accent_coach.comparison.scoring import compare
    from accent_coach.pipeline.features import analyse_audio
    from accent_coach.reference.rp_norms import get_rp_norms

    if args.sentence_id:
        sentence_meta = get_by_id(args.sentence_id)
    else:
        from accent_coach.calibration.sentences import Sentence
        sentence_meta = Sentence(
            id=0, text=args.transcript, sentence_type="statement", targets=[]
        )

    user_analysis = analyse_audio(args.user_wav, args.transcript, sentence_meta)
    user_audio, user_sr = _load_mono(args.user_wav)

    target_analysis = None
    if args.target_wav:
        target_analysis = analyse_audio(args.target_wav, args.transcript, sentence_meta)

    mean_f0 = float(
        np.mean([v.pitch_mean for v in user_analysis.vowels if v.pitch_mean > 70] or [120.0])
    )
    rp_norms = get_rp_norms(mean_f0)

    result = compare(
        user_analysis,
        user_audio,
        user_sr,
        target=target_analysis,
        reference_norms=rp_norms,
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(result.model_dump_json(indent=2))
    print(f"Written → {args.out_json}")
    print(f"Composite score: {result.composite_score:.1f}/100")
    for skill, score in result.skill_scores.items():
        print(f"  {skill:15s}: {score:.1f}")


if __name__ == "__main__":
    main()
