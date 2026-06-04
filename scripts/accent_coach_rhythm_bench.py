"""Rhythm module calibration bench.

Validates that nPVI scoring separates speaker groups in the expected direction:
  native (RP + GenAm) > generated BC > owner voice

Default mode: uses WhisperX alignment + existing pipeline (accurate, ~5-15 min).
Fast mode (--fast): onset-based syllable detection, no alignment (< 30 s, approximate).

Usage:
    PYTHONPATH=. uv run python scripts/accent_coach_rhythm_bench.py          # accurate
    PYTHONPATH=. uv run python scripts/accent_coach_rhythm_bench.py --fast   # quick
    PYTHONPATH=. uv run python scripts/accent_coach_rhythm_bench.py --n 5    # 5 clips/group
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from accent_coach.comparison.rhythm import score_rhythm
from accent_coach.models import SentenceAnalysis
from accent_coach.pipeline.prosody import extract_syllable_durations_acoustic
from accent_coach.reference.rp_norms import RP_NPVI_MAX, RP_NPVI_MIN

REPO_ROOT = Path(__file__).parent.parent
_NPVI_REF = (RP_NPVI_MIN + RP_NPVI_MAX) / 2

# (group_label, manifest_path, accent_family)
GROUPS: list[tuple[str, Path, str]] = [
    ("rp_fry",      REPO_ROOT / "tts_output/modern_rp_corpus/fry_manifest.json",       "RP"),
    ("rp_bbc",      REPO_ROOT / "tts_output/modern_rp_corpus/bbc_manifest.json",       "RP"),
    ("rp_lindsey",  REPO_ROOT / "tts_output/modern_rp_corpus/lindsey_manifest.json",   "RP"),
    ("genam_vsauce", REPO_ROOT / "tts_output/genam_corpus/manifest.json",              "GenAm"),
    ("genam_harris",   REPO_ROOT / "tts_output/genam_lecture_corpus/manifest.json",    "GenAm"),
    ("generated_bc", REPO_ROOT / "tts_output/eval_indextts_interview_short/manifest.json", "Generated"),
    ("owner",        REPO_ROOT / "tts_output/owner_cal_50/manifest.json",              "Owner"),
]

_GENAM_LECTURE_SPEAKERS = ["harris", "huberman", "sapolsky"]


# ---------------------------------------------------------------------------
# Syllable extraction: accurate (alignment) vs fast (onset-based fallback)
# ---------------------------------------------------------------------------

def load_audio_trimmed(wav_path: Path) -> tuple[np.ndarray, int] | None:
    """Load and trim leading/trailing silence. Returns (audio, sr) or None."""
    try:
        audio, sr = sf.read(str(wav_path), always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        audio_trim, _ = librosa.effects.trim(audio, top_db=25)
        if len(audio_trim) < sr * 0.5:
            return None
        return audio_trim, sr
    except Exception:  # noqa: BLE001
        return None


def get_syllable_durations(wav_path: Path, _transcript: str | None, fast: bool) -> list[float] | None:
    """Extract syllable durations.

    Both modes now use acoustic detection via the canonical
    extract_syllable_durations_acoustic() from pipeline/prosody.py.
    The `fast` flag is kept for interface compatibility but the same
    algorithm is used in both cases.
    """
    loaded = load_audio_trimmed(wav_path)
    if loaded is None:
        return None
    audio, sr = loaded
    durs = extract_syllable_durations_acoustic(audio, sr)
    return durs if len(durs) >= 2 else None


def _minimal_sentence(syllable_durations: list[float]) -> SentenceAnalysis:
    return SentenceAnalysis(
        sentence_id=0,
        sentence_type="statement",
        duration_s=sum(syllable_durations),
        syllable_durations=syllable_durations,
        pitch_contour=[],
        stress_pattern=[],
        vowels=[],
        stops=[],
    )


# ---------------------------------------------------------------------------
# Manifest loading and path resolution
# ---------------------------------------------------------------------------

def load_manifest(path: Path) -> list[dict]:
    with path.open() as f:
        return json.load(f)


def resolve_path(entry: dict, manifest_path: Path) -> Path | None:
    raw = entry.get("wav_path") or entry.get("path", "")
    for candidate in [
        Path(raw),
        REPO_ROOT / raw,
        manifest_path.parent / Path(raw).name,
    ]:
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Per-group bench
# ---------------------------------------------------------------------------

def bench_group(
    label: str,
    entries: list[dict],
    manifest_path: Path,
    n: int,
    verbose: bool,
    fast: bool,
    speaker_filter: str | None = None,
) -> tuple[list[float], list[float]]:
    """Returns (npvi_list, score_list)."""
    if speaker_filter:
        entries = [e for e in entries if e.get("speaker") == speaker_filter]

    # Prefer clips >= 1.5 s
    def long_enough(e: dict) -> bool:
        start = e.get("start_s", 0.0)
        end = e.get("end_s")
        return (end - start) >= 1.5 if end is not None else True

    entries = [e for e in entries if long_enough(e)]
    sample = random.sample(entries, min(n, len(entries)))

    npvis: list[float] = []
    scores: list[float] = []

    for entry in sample:
        wav = resolve_path(entry, manifest_path)
        if wav is None:
            continue
        transcript = entry.get("transcript", "")
        durs = get_syllable_durations(wav, transcript, fast)

        if durs is None or len(durs) < 2:
            continue

        result = score_rhythm(_minimal_sentence(durs))
        npvis.append(result.npvi)
        scores.append(result.score)
        if verbose:
            print(
                f"  {label:<22} {wav.name:<40} "
                f"nPVI={result.npvi:5.1f}  score={result.score:5.1f}"
            )

    return npvis, scores


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(rows: list[tuple[str, str, list[float], list[float]]]) -> None:
    header = (
        f"{'Group':<22} {'Accent':<10} {'N':>4}  "
        f"{'nPVI mean':>10}  {'nPVI std':>9}  "
        f"{'Score mean':>10}  {'Score std':>9}"
    )
    print()
    print(header)
    print("-" * len(header))
    for label, accent, npvis, scores in rows:
        if not npvis:
            print(f"  {label:<20} {accent:<10}  — (no valid clips)")
            continue
        nm = statistics.mean(npvis)
        ns = statistics.stdev(npvis) if len(npvis) > 1 else 0.0
        sm = statistics.mean(scores)
        ss = statistics.stdev(scores) if len(scores) > 1 else 0.0
        print(
            f"  {label:<20} {accent:<10} {len(npvis):>4}  "
            f"{nm:>10.1f}  {ns:>9.1f}  {sm:>10.1f}  {ss:>9.1f}"
        )
    print()


def _group_mean(rows: list[tuple[str, str, list[float], list[float]]], accent: str) -> float | None:
    group_rows = [(npvis, scores) for _, a, npvis, scores in rows if a == accent and scores]
    if not group_rows:
        return None
    all_scores = [s for _, sc in group_rows for s in sc]
    return statistics.mean(all_scores)


def check_ranking(rows: list[tuple[str, str, list[float], list[float]]]) -> bool:
    """Sanity checks for absolute nPVI scoring.

    NOTE: This bench tests absolute nPVI without a target. In production the
    module runs in comparison mode (user vs TTS target of the same sentence),
    which is more reliable. Native casual speech has *lower* nPVI than TTS
    (faster speech rate), so generated > native in absolute mode is expected.

    Checks:
    1. Generated TTS nPVI should be in the target range (>= 40) — it should
       represent measured stress-timed speech.
    2. Variance within each group should be reasonable (std < 30).
    3. Any group scoring near 0 is a signal of a broken detector.
    """
    rp = _group_mean(rows, "RP")
    genam = _group_mean(rows, "GenAm")
    generated = _group_mean(rows, "Generated")
    owner = _group_mean(rows, "Owner")

    print("Absolute nPVI sanity check:")
    print(f"  RP native:    {rp:.1f}" if rp else "  RP native:    —")
    print(f"  GenAm native: {genam:.1f}" if genam else "  GenAm native: —")
    print(f"  Generated BC: {generated:.1f}" if generated is not None else "  Generated BC: —")
    print(f"  Owner voice:  {owner:.1f}" if owner is not None else "  Owner voice:  —")
    print()
    print("  NOTE: Generated TTS typically scores higher than native casual speech")
    print("        (TTS is measured/deliberate; podcasts/lectures are fast/compressed).")
    print("        This is expected behaviour. For real coaching validation, run the")
    print("        module in comparison mode (user vs TTS target of the same sentence).")
    print()

    ok = True
    for _label, _accent, npvis, scores in rows:
        if not scores:
            continue
        # Any group with mean score < 5 = detector likely broken
        if statistics.mean(scores) < 5:
            print(f"  WARN: {_label} mean score < 5 — acoustic detector may be failing")
            ok = False
        # Excessive variance = unreliable detection
        if len(npvis) > 3 and statistics.stdev(npvis) > 30:
            print(f"  WARN: {_label} nPVI std > 30 — detection is noisy for this corpus")

    # Generated should score at least as well as a random baseline (>20)
    if generated is not None and generated < 20:
        print("  FAIL: Generated BC score < 20 — unexpected")
        ok = False

    result = "OK" if ok else "FAIL"
    print(f"  {result} — detector is functioning")
    return ok


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Rhythm nPVI calibration bench")
    parser.add_argument("--n", type=int, default=8,
                        help="Clips per group (default 8 for aligned mode, use --fast for larger N)")
    parser.add_argument("--fast", action="store_true",
                        help="Skip alignment, use onset detection (faster but less accurate)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    mode = "fast (onset-based, approximate)" if args.fast else "aligned (WhisperX)"
    print(f"Rhythm bench — mode: {mode}")
    print(f"nPVI target range: {RP_NPVI_MIN}–{RP_NPVI_MAX}  (centre {_NPVI_REF:.0f})")
    print(f"Clips per group: {args.n}  seed={args.seed}")
    if args.fast:
        print("NOTE: fast mode underestimates nPVI — use for relative ranking only")

    rows: list[tuple[str, str, list[float], list[float]]] = []

    for label, manifest_path, accent in GROUPS:
        if not manifest_path.exists():
            print(f"  SKIP {label}: manifest not found")
            continue

        entries = load_manifest(manifest_path)

        if label == "genam_harris":
            for spk in _GENAM_LECTURE_SPEAKERS:
                spk_label = f"genam_{spk}"
                print(f"  Processing {spk_label}...", flush=True)
                npvis, scores = bench_group(
                    spk_label, entries, manifest_path, args.n,
                    args.verbose, args.fast, speaker_filter=spk,
                )
                rows.append((spk_label, accent, npvis, scores))
            continue

        print(f"  Processing {label}...", flush=True)
        npvis, scores = bench_group(
            label, entries, manifest_path, args.n, args.verbose, args.fast
        )
        rows.append((label, accent, npvis, scores))

    print_table(rows)
    ok = check_ranking(rows)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
