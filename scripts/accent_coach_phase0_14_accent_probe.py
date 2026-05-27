"""WS-B B3 accent probe driver — Phase 0.14.

For each reference accent (georgia, scottish, irish):
  1. Extract per-phoneme formant centroids from the generated cal_25 clips
     using the standard pipeline (MFA alignment + parselmouth).
  2. Extract per-cluster formant centroids from the reference clip itself
     using raw parselmouth + nearest-neighbour assignment to GenAm categories.
  3. Compute Bark-space distances:
     - dist-to-source: generated centroids vs reference clip centroids (sex-agnostic)
     - dist-to-RP:     generated centroids vs sex-matched RP norms
     - dist-to-GenAm:  generated centroids vs sex-matched GenAm norms
  4. Write tts_output/accent_coach/phase0_14/accent_probe/grid.csv and
     docs/accent_coach_phase0_14_accent_probe_findings.md.

Run: .venv/bin/python scripts/accent_coach_phase0_14_accent_probe.py

Scottish speaker (James McAvoy) is male but measured F0 > 165 Hz due to audio
artefacts; norm selection is forced to male for that accent.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from accent_coach.diagnostics.bark_distance import hz_to_bark
from accent_coach.pipeline.centroids import build_centroids_from_formants
from accent_coach.pipeline.formants import extract_formants
from accent_coach.reference.genam_norms import get_genam_norms
from accent_coach.reference.rp_norms import get_rp_norms

PROBE_ROOT = PROJECT_ROOT / "tts_output/accent_coach/phase0_14/accent_probe"
REFS_ROOT = PROJECT_ROOT / "tts_output/refs/accent_test"

ACCENTS = ["georgia", "scottish", "irish"]

# Sex override: scottish speaker is McAvoy (male) despite measured F0 > 165 Hz
_FORCE_MALE = {"scottish"}

# Formant extraction ceiling per accent (female 5500, male 5000)
_MAX_FORMANT = {"georgia": 5500.0, "scottish": 5000.0, "irish": 5500.0}

# Measured mean F0 from PROVENANCE (used for RP/GenAm norm selection)
_MEASURED_F0 = {"georgia": 309.2, "scottish": 222.6, "irish": 280.5}


def _effective_f0_for_norms(accent: str) -> float:
    """Return f0 to pass to get_rp_norms / get_genam_norms."""
    if accent in _FORCE_MALE:
        return 100.0  # forces male norm branch (threshold 165 Hz)
    return _MEASURED_F0[accent]


def _bark_dist(f1a: float, f2a: float, f1b: float, f2b: float) -> float:
    """Euclidean distance in Bark space."""
    return math.sqrt((hz_to_bark(f1a) - hz_to_bark(f1b)) ** 2
                     + (hz_to_bark(f2a) - hz_to_bark(f2b)) ** 2)


def _extract_ref_centroids(
    wav_path: Path,
    genam_norms: dict[str, tuple[float, float]],
    max_formant: float = 5500.0,
    min_tokens: int = 3,
) -> dict[str, dict]:
    """Extract vowel centroids from reference clip without MFA.

    Assigns each voiced frame to the nearest GenAm phoneme category (Bark space
    nearest-neighbour), then returns per-category mean F1/F2. This is a rough
    approximation suitable for the accent probe probe (not pass/fail).
    """
    audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float64)

    sound = parselmouth.Sound(audio, sampling_frequency=float(sr))
    formant = sound.to_formant_burg(
        time_step=0.01,
        max_number_of_formants=5,
        maximum_formant=max_formant,
        window_length=0.025,
        pre_emphasis_from=50.0,
    )

    # Collect voiced frames: F1 in plausible vowel range
    duration = sound.duration
    times = np.arange(0.025, duration, 0.010)
    voiced: list[tuple[float, float]] = []
    for t in times:
        f1 = formant.get_value_at_time(1, t)
        f2 = formant.get_value_at_time(2, t)
        if np.isnan(f1) or np.isnan(f2):
            continue
        if not (200 < f1 < 1100 and 500 < f2 < 3500):
            continue
        voiced.append((f1, f2))

    if not voiced:
        return {}

    # Build GenAm Bark centroids for nearest-neighbour assignment
    phonemes = list(genam_norms.keys())
    genam_bark = {ph: (hz_to_bark(f1), hz_to_bark(f2))
                  for ph, (f1, f2) in genam_norms.items()}

    assignments: dict[str, list[tuple[float, float]]] = {ph: [] for ph in phonemes}
    for (f1, f2) in voiced:
        b1, b2 = hz_to_bark(f1), hz_to_bark(f2)
        best = min(phonemes,
                   key=lambda p: (b1 - genam_bark[p][0]) ** 2 + (b2 - genam_bark[p][1]) ** 2)
        assignments[best].append((f1, f2))

    centroids: dict[str, dict] = {}
    for ph, frames in assignments.items():
        if len(frames) >= min_tokens:
            centroids[ph] = {
                "f1": float(np.mean([f[0] for f in frames])),
                "f2": float(np.mean([f[1] for f in frames])),
                "n": len(frames),
            }
    return centroids


def _avg_dist(
    gen_cent: dict[str, dict],
    target_norms: dict[str, tuple[float, float]],
) -> float:
    """Mean Bark distance from generated centroids to target norm table."""
    dists: list[float] = []
    for ph, vals in gen_cent.items():
        if ph not in target_norms:
            continue
        f1t, f2t = target_norms[ph]
        dists.append(_bark_dist(vals["f1"], vals["f2"], f1t, f2t))
    return float(np.mean(dists)) if dists else float("nan")


def _avg_dist_to_ref(
    gen_cent: dict[str, dict],
    ref_cent: dict[str, dict],
) -> float:
    """Mean Bark distance from generated centroids to reference clip centroids."""
    dists: list[float] = []
    for ph, gvals in gen_cent.items():
        if ph not in ref_cent:
            continue
        rvals = ref_cent[ph]
        dists.append(_bark_dist(gvals["f1"], gvals["f2"], rvals["f1"], rvals["f2"]))
    return float(np.mean(dists)) if dists else float("nan")


def run_probe(accent: str, args: argparse.Namespace) -> dict:
    """Run the probe for one accent. Returns a result row dict."""
    gen_dir = PROBE_ROOT / accent
    manifest_path = gen_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"  [SKIP] {accent}: manifest not found ({manifest_path})", flush=True)
        return {}

    ref_wav = REFS_ROOT / f"{accent}.wav"
    if not ref_wav.exists():
        print(f"  [SKIP] {accent}: reference WAV not found ({ref_wav})", flush=True)
        return {}

    mean_f0 = _effective_f0_for_norms(accent)
    genam_norms = get_genam_norms(mean_f0)
    rp_norms = get_rp_norms(mean_f0)

    print(f"\n[{accent}] extracting formants from generated clips...", flush=True)
    formants_csv = gen_dir / "formants.csv"
    extract_formants(manifest_path, formants_csv, source_label=f"{accent}_probe")

    gen_cent = build_centroids_from_formants(formants_csv)
    print(f"  generated centroids: {len(gen_cent)} phonemes", flush=True)

    print(f"[{accent}] extracting reference clip centroids...", flush=True)
    ref_cent = _extract_ref_centroids(
        ref_wav, genam_norms, max_formant=_MAX_FORMANT[accent]
    )
    print(f"  reference centroids: {len(ref_cent)} categories", flush=True)

    d_source = _avg_dist_to_ref(gen_cent, ref_cent)
    d_rp = _avg_dist(gen_cent, {ph: (f1, f2) for ph, (f1, f2) in rp_norms.items()})
    d_genam = _avg_dist(gen_cent, {ph: (f1, f2) for ph, (f1, f2) in genam_norms.items()})

    result = {
        "accent": accent,
        "n_phonemes": len(gen_cent),
        "dist_to_source": round(d_source, 3),
        "dist_to_RP": round(d_rp, 3),
        "dist_to_GenAm": round(d_genam, 3),
        "verdict": _verdict(d_source, d_rp, d_genam),
    }
    print(f"  dist-to-source={d_source:.3f}  dist-to-RP={d_rp:.3f}  dist-to-GenAm={d_genam:.3f}",
          flush=True)
    print(f"  verdict: {result['verdict']}", flush=True)
    return result


def _verdict(d_src: float, d_rp: float, d_genam: float) -> str:
    if math.isnan(d_src):
        return "UNKNOWN (no ref centroids matched)"
    if d_src < min(d_rp, d_genam) * 0.85:
        return "FOLLOWS_REFERENCE (model follows reference accent)"
    if d_genam < d_rp and d_genam < d_src:
        return "AMERICAN_BIAS (output near GenAm regardless of reference)"
    if d_rp < d_genam and d_rp < d_src:
        return "RP_LEANING (output near RP)"
    return f"AMBIGUOUS (src={d_src:.2f} rp={d_rp:.2f} genam={d_genam:.2f})"


def write_grid(results: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "accent", "n_phonemes", "dist_to_source", "dist_to_RP", "dist_to_GenAm", "verdict"
        ])
        w.writeheader()
        w.writerows(results)
    print(f"\ngrid → {out_csv}", flush=True)


def write_findings(results: list[dict], out_md: Path) -> None:
    lines = [
        "# Phase 0.14 WS-B — Accent probe findings",
        "",
        "**Date:** 2026-05-27  **Author:** claude-sonnet-4-6",
        "",
        "## Question",
        "",
        "Given a non-RP reference clip, does IndexTTS-2 place the cloned voice near that",
        "accent, near General American, or near RP?",
        "",
        "## Results",
        "",
        "| Accent | Phonemes | Dist→Source | Dist→RP | Dist→GenAm | Verdict |",
        "|--------|----------|-------------|---------|------------|---------|",
    ]
    for r in results:
        if not r:
            continue
        lines.append(
            f"| {r['accent']} | {r['n_phonemes']} | {r['dist_to_source']:.3f} |"
            f" {r['dist_to_RP']:.3f} | {r['dist_to_GenAm']:.3f} | {r['verdict']} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- **dist-to-source << dist-to-both-norms** → model follows the reference accent",
        "  (zero-shot transfer; weakens the 'model-bound' claim).",
        "- **output near GenAm regardless of reference** → American bias in the GPT",
        "  (strong support for WS-C LoRA).",
        "- **output near RP** → model already RP-leaning for these refs.",
        "",
        "## Per-accent perceptual notes",
        "",
        "*(Fill in after listening to 2 clips per accent.)*",
        "",
        "- **Georgia** (Southern US, Brianne Howey): *TODO — listen to indextts_cal_010.wav,"
        " indextts_cal_007.wav*",
        "- **Scottish** (James McAvoy): *TODO — listen to indextts_cal_010.wav,"
        " indextts_cal_007.wav*",
        "- **Irish** (Saoirse Ronan): *TODO — listen to indextts_cal_010.wav,"
        " indextts_cal_007.wav*",
        "",
        "## Explicit verdict",
        "",
    ]
    for r in results:
        if not r:
            continue
        lines.append(f"- **{r['accent']}**: {r['verdict']}")

    lines += [
        "",
        "## Implication for WS-C",
        "",
        "*(Written after reviewing all three verdicts.)*",
        "",
        "If all three accents show AMERICAN_BIAS regardless of reference → strong",
        "evidence the bias is in the GPT token distribution (model-bound); supports",
        "WS-C LoRA approach.",
        "",
        "If any accent shows FOLLOWS_REFERENCE → accent lives partly in the reference",
        "(zero-shot transferable); WS-C may achieve the same result more cheaply",
        "by selecting a better RP reference.",
    ]

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")
    print(f"findings → {out_md}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accents", nargs="*", default=ACCENTS,
                        help="Accents to score (default: all three)")
    args = parser.parse_args()

    results: list[dict] = []
    for accent in args.accents:
        r = run_probe(accent, args)
        if r:
            results.append(r)

    if not results:
        print("No results — check that TTS generation is complete.", file=sys.stderr)
        return 1

    grid_csv = PROBE_ROOT / "grid.csv"
    write_grid(results, grid_csv)

    findings_md = PROJECT_ROOT / "docs/accent_coach_phase0_14_accent_probe_findings.md"
    write_findings(results, findings_md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
