"""Consonant quality bench — per-speaker-group scores.

Validates that the consonant module separates speaker groups in the expected direction:
  native RP (Fry, Lindsey) ≈ native GA > TTS BC ≥ real BC > owner

Runs WhisperX forced alignment + G2P phoneme extraction for each clip then
calls score_consonants() with the resulting features.

Usage:
    uv run python scripts/bench/accent_coach_consonant_bench.py          # all groups
    uv run python scripts/bench/accent_coach_consonant_bench.py --n 8    # 8 clips/group
    uv run python scripts/bench/accent_coach_consonant_bench.py --accent genam
    uv run python scripts/bench/accent_coach_consonant_bench.py --group rp_fry,owner
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from accent_coach.comparison.consonants import score_consonants
from accent_coach.models import SentenceAnalysis
from accent_coach.pipeline.alignment import align_audio
from accent_coach.pipeline.audio_io import load_standard_audio
from accent_coach.pipeline.vot import extract_stop_features
from scripts.lib.manifest import load_manifest, resolve_path

GROUPS: list[tuple[str, Path, str]] = [
    ("rp_fry",      REPO_ROOT / "tts_output/modern_rp_corpus/fry_manifest.json",              "RP"),
    ("rp_lindsey",  REPO_ROOT / "tts_output/modern_rp_corpus/lindsey_manifest.json",          "RP"),
    ("real_bc",     REPO_ROOT / "tts_output/real_bc_corpus/manifest.json",                    "Real-BC"),
    ("tts_bc",      REPO_ROOT / "tts_output/eval_indextts_interview_short/manifest.json",      "TTS"),
    ("owner",       REPO_ROOT / "tts_output/owner_cal_50/manifest.json",                      "Owner"),
    ("genam_harris",   REPO_ROOT / "tts_output/genam_lecture_corpus/manifest.json",           "GenAm"),
]

# Sub-score column widths
_COLS = ("composite", "fricative", "stop_vot", "rhotic", "lateral")


def _normalize_transcript(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace — for target matching.

    Comparison mode (§1D) compares a clip against a BC target *saying the same
    words*.  Two transcripts are 'the same phrase' iff they normalise equal.
    """
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", text.lower())).strip()


def _build_target_index(manifest_path: Path | None) -> dict[str, tuple[Path, str]]:
    """Map normalized transcript → (wav path, raw transcript) for a BC target manifest.

    Empty when no target manifest is given, so comparison mode is simply off and
    the bench falls back to absolute references (today's behaviour).
    """
    if manifest_path is None or not manifest_path.exists():
        return {}
    index: dict[str, tuple[Path, str]] = {}
    for entry in load_manifest(manifest_path):
        transcript = entry.get("transcript") or entry.get("prompt", "")
        wav = resolve_path(entry, manifest_path, REPO_ROOT)
        if not transcript or wav is None:
            continue
        index.setdefault(_normalize_transcript(transcript), (wav, transcript))
    return index


def _load_audio_16k(wav: Path) -> tuple[np.ndarray, int] | None:
    # Canonical normalization: mono + resample to 16 kHz + common bandwidth cap,
    # so source sample-rate/bandwidth differences cannot bias the scores
    # (the fricative-inversion root cause). Single path for corpus and user audio.
    return load_standard_audio(wav)


def _build_sentence_analysis(
    audio: np.ndarray, sr: int, phonemes: list, stops: list, sentence_id: int
) -> SentenceAnalysis:
    return SentenceAnalysis(
        sentence_id=sentence_id,
        sentence_type="statement",
        duration_s=len(audio) / sr,
        syllable_durations=[0.2],
        pitch_contour=[0.5] * 50,
        stress_pattern=[True],
        vowels=[],
        stops=stops,
        phonemes=phonemes,
    )


def _score_clip(
    wav: Path,
    transcript: str,
    sentence_id: int,
    accent_target: str,
    target_index: dict[str, tuple[Path, str]] | None = None,
) -> dict | None:
    loaded = _load_audio_16k(wav)
    if loaded is None:
        return None
    audio, sr = loaded
    if len(audio) < sr * 0.8:
        return None

    try:
        phonemes = align_audio(wav, transcript, sentence_id, accent_target)
    except Exception as e:  # noqa: BLE001
        print(f"    [align err] {wav.name}: {e}", file=sys.stderr)
        return None

    if not phonemes:
        return None

    try:
        stops = extract_stop_features(audio, sr, phonemes)
    except Exception as e:  # noqa: BLE001 — record, don't silently drop the VOT sub-class
        print(f"    [stop-extract err] {wav.name}: {type(e).__name__}: {e}", file=sys.stderr)
        stops = []

    sa = _build_sentence_analysis(audio, sr, phonemes, stops, sentence_id)

    # Comparison mode (§1D): when a BC target clip exists for the SAME phrase,
    # score against it through the identical pipeline so the shared alignment
    # artefact cancels.  Falls back to absolute references when no match.
    t_audio = t_sr = t_sa = None
    if target_index:
        match = target_index.get(_normalize_transcript(transcript))
        if match is not None:
            t_loaded = _load_audio_16k(match[0])
            if t_loaded is not None:
                t_audio, t_sr = t_loaded
                try:
                    t_phonemes = align_audio(match[0], match[1], sentence_id, accent_target)
                    t_stops = extract_stop_features(t_audio, t_sr, t_phonemes)
                    t_sa = _build_sentence_analysis(t_audio, t_sr, t_phonemes, t_stops, sentence_id)
                except Exception:  # noqa: BLE001
                    t_audio = t_sr = t_sa = None

    try:
        cs = score_consonants(
            sa, audio, sr,
            target_audio=t_audio, target_sr=t_sr, target=t_sa,
            accent_target=accent_target,
        )
    except Exception as e:  # noqa: BLE001
        print(f"    [score err] {wav.name}: {e}", file=sys.stderr)
        return None

    n_fric = sum(1 for p in phonemes if p.phoneme in {"s","z","ʃ","ʒ","θ","ð","f","v"})
    n_rh   = sum(1 for p in phonemes if p.phoneme == "r")
    n_lat  = sum(1 for p in phonemes if p.phoneme == "l")
    return {
        "composite": cs.score,
        "fricative":  cs.fricative_score,
        "stop_vot":   cs.stop_aspiration_score,
        "rhotic":     cs.rhotic_score,
        "lateral":    cs.lateral_score,
        "n_phonemes": len(phonemes),
        "n_stops":    len(stops),
        "n_fric": n_fric, "n_rh": n_rh, "n_lat": n_lat,
    }


def bench_group(
    label: str,
    entries: list[dict],
    manifest_path: Path,
    n: int,
    verbose: bool,
    accent_target: str,
    speaker_filter: str | None = None,
    target_index: dict[str, tuple[Path, str]] | None = None,
) -> dict:
    if speaker_filter:
        entries = [e for e in entries if e.get("speaker") == speaker_filter]

    # Prefer longer clips (more phoneme tokens = more reliable scores)
    def dur(e: dict) -> float:
        s, end = e.get("start_s", 0), e.get("end_s")
        return (end - s) if end else 3.0
    entries = sorted(entries, key=dur, reverse=True)

    sample = random.sample(entries[:max(n * 3, 30)], min(n, len(entries)))

    results: list[dict] = []
    for i, entry in enumerate(sample):
        wav = resolve_path(entry, manifest_path, REPO_ROOT)
        if wav is None:
            continue
        transcript = entry.get("transcript") or entry.get("prompt", "")
        if not transcript:
            continue
        r = _score_clip(wav, transcript, i, accent_target, target_index=target_index)
        if r is None:
            continue
        results.append(r)
        if verbose:
            fric  = f"{r['fricative']:5.1f}" if r["fricative"]  is not None else "  n/a"
            vot   = f"{r['stop_vot']:5.1f}"  if r["stop_vot"]   is not None else "  n/a"
            rh    = f"{r['rhotic']:5.1f}"    if r["rhotic"]     is not None else "  n/a"
            lat   = f"{r['lateral']:5.1f}"   if r["lateral"]    is not None else "  n/a"
            print(
                f"  {wav.name:<38} comp={r['composite']:5.1f} "
                f"fric={fric} vot={vot} rh={rh} lat={lat} "
                f"(ph={r['n_phonemes']} fr={r['n_fric']} rh={r['n_rh']} l={r['n_lat']})"
            )

    if not results:
        return {}

    agg: dict[str, float] = {"n": float(len(results))}
    for key in _COLS:
        vals = [r[key] for r in results if r.get(key) is not None]
        agg[key] = float(np.mean(vals)) if vals else float("nan")
    return agg


def _fmt(v: float) -> str:
    return f"{v:7.1f}" if not np.isnan(v) else "    n/a"


def _print_discrimination(summary: list[tuple[str, str, dict]]) -> None:
    """Report DISCRIMINATION, not just level: owner gap + inversion flag per metric.

    A level-only table hides ordering regressions (e.g. owner rising from lowest
    to highest while the absolute numbers all look 'believable').  For each metric
    this prints owner's value, the lowest OTHER group, the gap (lowest_other −
    owner; positive = owner correctly below everyone), and a ✗ when the owner is
    not strictly lowest — the invariant that actually matters for coaching.
    """
    by_label = {label: agg for label, _fam, agg in summary}
    if "owner" not in by_label:
        print("\n  (no owner group scored — cannot report discrimination)")
        return

    print(f"\n{'='*75}")
    print("  DISCRIMINATION  (owner must be STRICTLY lowest on every metric)")
    print(f"{'='*75}")
    print(f"{'metric':<12} {'owner':>7} {'lowest-other':>14} {'gap':>8}   owner-lowest?")
    print("─" * 60)
    all_ok = True
    for metric in ("composite", *_COLS[1:]):
        owner = by_label["owner"].get(metric, float("nan"))
        others = {
            lbl: agg.get(metric, float("nan"))
            for lbl, agg in by_label.items()
            if lbl != "owner"
        }
        others = {lbl: v for lbl, v in others.items() if not np.isnan(v)}
        if np.isnan(owner) or not others:
            print(f"{metric:<12} {_fmt(owner):>7}  {'n/a':>13}  {'n/a':>7}   —")
            continue
        low_lbl = min(others, key=lambda k: others[k])
        low_val = others[low_lbl]
        gap = low_val - owner
        ok = gap > 0  # owner strictly below the lowest other
        all_ok = all_ok and ok
        flag = "✓" if ok else "✗ INVERTED"
        print(
            f"{metric:<12} {owner:7.1f} {low_val:8.1f} ({low_lbl:<10}) {gap:+7.1f}   {flag}"
        )
    verdict = "GREEN — owner lowest everywhere" if all_ok else "RED — at least one inversion"
    print("─" * 60)
    print(f"  verdict: {verdict}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=12, help="clips per group (default 12)")
    p.add_argument("--accent", choices=["rp", "genam"], default="rp")
    p.add_argument("--group", default=None, help="comma-separated group labels to run")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--target-manifest", default=None,
        help="BC target manifest for comparison mode (§1D); clips matched by "
             "normalized transcript are scored against the BC clip of the same phrase.",
    )
    args = p.parse_args()

    random.seed(args.seed)

    target_index = _build_target_index(
        Path(args.target_manifest) if args.target_manifest else None
    )
    if target_index:
        print(f"  comparison mode ON — {len(target_index)} BC target phrases indexed")

    filter_labels = set(args.group.split(",")) if args.group else None

    active = [
        (label, path, family) for label, path, family in GROUPS
        if (filter_labels is None or label in filter_labels) and path.exists()
    ]
    if not active:
        print("No matching groups found.")
        sys.exit(1)

    print(f"\n{'='*75}")
    print(f"  Consonant bench — accent={args.accent}, n={args.n} clips/group")
    print(f"{'='*75}")

    summary: list[tuple[str, str, dict]] = []
    for label, manifest_path, family in active:
        entries = load_manifest(manifest_path)
        print(f"\n── {label}  [{family}]  ({len(entries)} entries in manifest)")
        agg = bench_group(
            label, entries, manifest_path,
            n=args.n, verbose=args.verbose, accent_target=args.accent,
            target_index=target_index or None,
        )
        if agg:
            summary.append((label, family, agg))
        else:
            print("   (no results)")

    if not summary:
        print("\nNo results.")
        return

    print(f"\n{'='*75}")
    print("  SUMMARY  (column = mean across clips scored; NaN = no tokens of that class)")
    print(f"{'='*75}")
    hdr = f"{'Label':<20} {'Family':<10} {'composite':>9} {'fricative':>9} {'stop_vot':>9} {'rhotic':>9} {'lateral':>9}  n"
    print(hdr)
    print("─" * len(hdr))
    for label, family, agg in summary:
        print(
            f"{label:<20} {family:<10} "
            f"{_fmt(agg.get('composite', float('nan')))} "
            f"{_fmt(agg.get('fricative', float('nan')))} "
            f"{_fmt(agg.get('stop_vot', float('nan')))} "
            f"{_fmt(agg.get('rhotic', float('nan')))} "
            f"{_fmt(agg.get('lateral', float('nan')))} "
            f" {int(agg.get('n', 0))}"
        )

    _print_discrimination(summary)

    # Print expected ordering note
    print(f"\n{'─'*75}")
    print("  Expected ordering:  RP/GA natives ≥ TTS ≥ real BC > owner")
    print("  A negative gap above = owner is NOT lowest = broken metric (CLAUDE.md).")
    print("  Rhotic/lateral may show NaN for groups whose clips have no /r/ or /l/ tokens.")


if __name__ == "__main__":
    main()
