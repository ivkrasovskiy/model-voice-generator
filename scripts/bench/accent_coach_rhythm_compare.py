"""Rhythm comparison: owner vs TTS target vs native speakers.

Runs score_rhythm() in three modes:
  1. Comparison mode (user=owner, target=TTS) — uses hybrid syllable detection
     when WhisperX alignment is available; acoustic fallback otherwise.
  2. TTS self-score (absolute, no target) — shows what "ideal" looks like.
  3. Native spot-check (absolute, no target) — context for native rhythm.

Usage:
    PYTHONPATH=. uv run python scripts/accent_coach_rhythm_compare.py
    PYTHONPATH=. uv run python scripts/accent_coach_rhythm_compare.py --align   # WhisperX alignment
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import soundfile as sf

from accent_coach.comparison.rhythm import score_rhythm
from accent_coach.models import SentenceAnalysis
from accent_coach.pipeline.prosody import extract_syllable_durations_acoustic
from scripts.lib.manifest import load_manifest as _load_manifest_json

REPO_ROOT = Path(__file__).parent.parent.parent

# Matched pairs: (slug, transcript, owner_path, tts_path)
MATCHED_PAIRS = [
    (
        "rb_01",
        "The ship hit a big cliff in the mist.",
        REPO_ROOT / "tts_output/accent_coach/users/owner/002_the_ship_hit_a_big_cliff_in_th.wav",
        REPO_ROOT / "tts_output/rhythm_bench_tts/indextts_rb_01.wav",
    ),
    (
        "rb_02",
        "He left the red pen on the desk.",
        REPO_ROOT / "tts_output/accent_coach/users/owner/003_he_left_the_red_pen_on_the_des.wav",
        REPO_ROOT / "tts_output/rhythm_bench_tts/indextts_rb_02.wav",
    ),
    (
        "rb_03",
        "The thought of walking to the court was daunting.",
        REPO_ROOT / "tts_output/accent_coach/users/owner/007_the_thought_of_walking_to_the.wav",
        REPO_ROOT / "tts_output/rhythm_bench_tts/indextts_rb_03.wav",
    ),
]


def _load(path: Path) -> tuple[np.ndarray, int] | None:
    try:
        audio, sr = sf.read(str(path), always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32), sr
    except Exception:  # noqa: BLE001
        return None


def _align_and_build(audio: np.ndarray, sr: int, transcript: str, audio_path: Path):
    """Full pipeline: WhisperX alignment → hybrid syllable durations."""
    from accent_coach.pipeline.alignment import align_audio
    from accent_coach.pipeline.prosody import extract_syllable_durations_from_words

    phonemes = align_audio(audio_path, transcript, sentence_id=0)
    syl_durs = extract_syllable_durations_from_words(phonemes, audio, sr)
    return syl_durs, phonemes


def _acoustic_build(audio: np.ndarray, sr: int):
    from accent_coach.pipeline.prosody import extract_syllable_durations_acoustic
    return extract_syllable_durations_acoustic(audio, sr), []


def _make_sentence(syl_durs: list[float], phonemes: list):
    from accent_coach.models import SentenceAnalysis
    return SentenceAnalysis(
        sentence_id=0,
        sentence_type="statement",
        duration_s=sum(syl_durs),
        syllable_durations=syl_durs,
        pitch_contour=[],
        stress_pattern=[],
        vowels=[],
        stops=[],
        phonemes=phonemes,
    )


def run_comparison(use_alignment: bool) -> None:
    from accent_coach.comparison.rhythm import score_rhythm

    print(f"\n{'='*68}")
    print(f"RHYTHM COMPARISON — mode: {'aligned (hybrid)' if use_alignment else 'acoustic-only'}")
    print(f"{'='*68}\n")

    print("── Owner vs TTS target (matched sentences) ──\n")
    fmt = f"{'Sentence':<48} {'nPVI user':>9} {'nPVI tgt':>9} {'Score':>6}  Inflated FW"
    print(fmt)
    print("-" * 85)

    for slug, transcript, owner_path, tts_path in MATCHED_PAIRS:
        owner_data = _load(owner_path)
        tts_data = _load(tts_path)
        if owner_data is None or tts_data is None:
            print(f"  {slug}: SKIP (file not found)")
            continue

        owner_audio, owner_sr = owner_data
        tts_audio, tts_sr = tts_data

        if use_alignment:
            user_durs, user_ph = _align_and_build(owner_audio, owner_sr, transcript, owner_path)
            tgt_durs, tgt_ph = _align_and_build(tts_audio, tts_sr, transcript, tts_path)
        else:
            user_durs, user_ph = _acoustic_build(owner_audio, owner_sr)
            tgt_durs, tgt_ph = _acoustic_build(tts_audio, tts_sr)

        user_sa = _make_sentence(user_durs, user_ph)
        tgt_sa = _make_sentence(tgt_durs, tgt_ph)
        result = score_rhythm(user_sa, target=tgt_sa)

        inflated_str = ", ".join(result.inflated_function_words) if result.inflated_function_words else "—"
        label = transcript[:47]
        print(f"  {label:<48} {result.npvi:>9.1f} {result.reference_npvi_min+10:>9.1f} {result.score:>6.1f}  {inflated_str}")
        if result.diagnostics:
            for d in result.diagnostics:
                print(f"    ⚑ {d}")
        print()

    # TTS self-scores (absolute — no target, shows what "ideal" looks like)
    print("\n── TTS absolute scores (no target — ideal reference) ──\n")
    tts_manifest = _load_manifest_json(REPO_ROOT / "tts_output/rhythm_bench_tts/manifest.json")
    print(f"  {'Sentence':<50} {'nPVI':>6}  {'Score':>6}")
    print("  " + "-" * 66)
    for entry in tts_manifest:
        wav = REPO_ROOT / entry["wav_path"]
        data = _load(wav)
        if data is None:
            continue
        audio, sr = data
        durs = extract_syllable_durations_acoustic(audio, sr)
        sa = SentenceAnalysis(
            sentence_id=0, sentence_type="statement", duration_s=sum(durs),
            syllable_durations=durs, pitch_contour=[], stress_pattern=[], vowels=[], stops=[],
        )
        r = score_rhythm(sa)
        print(f"  {entry['prompt'][:50]:<50} {r.npvi:>6.1f}  {r.score:>6.1f}")

    # Native spot-check
    print("\n── Native speaker spot-check (RP + GenAm, 5 clips each) ──\n")
    _spot_check_corpus(REPO_ROOT / "tts_output/modern_rp_corpus/fry_manifest.json", "RP/Fry", 5)
    _spot_check_corpus(REPO_ROOT / "tts_output/genam_corpus/manifest.json", "GenAm", 5)


def _spot_check_corpus(manifest_path: Path, label: str, n: int) -> None:
    if not manifest_path.exists():
        print(f"  {label}: manifest not found")
        return

    entries = _load_manifest_json(manifest_path)
    sample = random.sample(entries, min(n, len(entries)))
    npvis, scores = [], []
    for e in sample:
        wav_raw = e.get("wav_path") or e.get("path", "")
        for candidate in [Path(wav_raw), REPO_ROOT / wav_raw, manifest_path.parent / Path(wav_raw).name]:
            if candidate.exists():
                data = _load(candidate)
                if data:
                    audio, sr = data
                    durs = extract_syllable_durations_acoustic(audio, sr)
                    if len(durs) >= 2:
                        sa = SentenceAnalysis(
                            sentence_id=0, sentence_type="statement", duration_s=sum(durs),
                            syllable_durations=durs, pitch_contour=[], stress_pattern=[],
                            vowels=[], stops=[],
                        )
                        r = score_rhythm(sa)
                        npvis.append(r.npvi)
                        scores.append(r.score)
                break

    if npvis:
        import statistics
        print(f"  {label:<12}  n={len(npvis)}  "
              f"nPVI={statistics.mean(npvis):.1f}±{statistics.stdev(npvis) if len(npvis)>1 else 0:.1f}  "
              f"score={statistics.mean(scores):.1f}±{statistics.stdev(scores) if len(scores)>1 else 0:.1f}")
    else:
        print(f"  {label}: no valid clips")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--align", action="store_true",
                        help="Use WhisperX alignment + hybrid syllable detection (slow, more accurate)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import random
    random.seed(args.seed)
    run_comparison(use_alignment=args.align)


if __name__ == "__main__":
    main()
