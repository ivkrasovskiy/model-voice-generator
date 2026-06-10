"""Re-derive VOT reference (RP_VOT_MEAN_MS/SD, GA_VOT_MEAN_MS/SD) IN-DOMAIN.

The Lisker & Abramson (1964) reference (/p t k/ = 67.5/77.5/87.5 ms) is
textbook isolated-word VOT. extract_vot (accent_coach/pipeline/vot.py) measures
VOT through THIS pipeline on connected speech, where even native /p t k/ come
out at 0-22 ms — the same bandwidth/measurement-domain confound as the
pre-fix fricative CoG (Jongman 7000 Hz vs in-domain 5200 Hz).

This measures native /p t k/ VOT through the IDENTICAL pipeline
(normalize_audio -> align -> extract_stop_features, which already applies
filter_aspirating_stops) and reports per-phoneme native mean/SD, separately
for RP and GA natives plus pooled, so the reference can be set from the
native distribution (CLAUDE.md principle). Also reports the owner distribution
so the native-owner gap can be checked before/after the constant change.

    .venv/bin/python scripts/tools/measure_vot_reference.py --n 30
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from accent_coach.pipeline.alignment import align_audio
from accent_coach.pipeline.audio_io import load_standard_audio
from accent_coach.pipeline.vot import extract_stop_features
from scripts.lib.manifest import load_manifest, resolve_path

_VOICELESS_STOPS = ("p", "t", "k")

# Native English sources, grouped by accent family. Pool within family.
NATIVE_SOURCES: dict[str, list[tuple[str, Path]]] = {
    "rp": [
        ("fry",     REPO_ROOT / "tts_output/modern_rp_corpus/fry_manifest.json"),
        ("lindsey", REPO_ROOT / "tts_output/modern_rp_corpus/lindsey_manifest.json"),
        ("bbc",     REPO_ROOT / "tts_output/modern_rp_corpus/bbc_manifest.json"),
        ("real_bc", REPO_ROOT / "tts_output/real_bc_corpus/manifest.json"),
    ],
    "genam": [
        ("genam_lecture", REPO_ROOT / "tts_output/genam_lecture_corpus/manifest.json"),
        ("genam_corpus",  REPO_ROOT / "tts_output/genam_corpus/manifest.json"),
        ("vsauce",        REPO_ROOT / "tts_output/genam_corpus/manifest_vsauce.json"),
    ],
    "owner": [
        ("owner", REPO_ROOT / "tts_output/owner_cal_50/manifest.json"),
    ],
}


def _measure_source(manifest: Path, n: int, accent: str) -> dict[str, list[float]]:
    vot: dict[str, list[float]] = defaultdict(list)
    if not manifest.exists():
        print(f"  [missing] {manifest}", file=sys.stderr)
        return vot
    entries = load_manifest(manifest)
    entries = sorted(entries, key=lambda e: (e.get("end_s", 3) - e.get("start_s", 0)), reverse=True)
    sample = random.sample(entries[: max(n * 3, 40)], min(n, len(entries)))
    for i, e in enumerate(sample):
        wav = resolve_path(e, manifest, REPO_ROOT)
        tr = e.get("transcript") or e.get("prompt", "")
        if wav is None or not tr:
            continue
        loaded = load_standard_audio(wav)
        if loaded is None:
            continue
        audio, sr = loaded
        try:
            phon = align_audio(wav, tr, i, accent if accent != "owner" else "rp")
        except Exception:  # noqa: BLE001 — drop unalignable clip
            continue
        for s in extract_stop_features(audio, sr, phon):
            ph = s.phoneme.phoneme
            if ph in _VOICELESS_STOPS:
                vot[ph].append(s.vot_ms)
    return vot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="clips per source")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    def _report(title: str, pooled: dict[str, list[float]]) -> None:
        print(f"\n  {title}")
        print(f"  {'ph':<4} {'n':>4} {'mean':>7} {'median':>7} {'std':>7} {'%=0':>6} {'%>20ms':>7}")
        print("  " + "-" * 50)
        for ph in _VOICELESS_STOPS:
            v = np.array(pooled.get(ph, []), dtype=float)
            if len(v) == 0:
                print(f"  {ph:<4} {0:>4} {'-':>7} {'-':>7} {'-':>7} {'-':>6} {'-':>7}")
                continue
            pct_zero = 100.0 * np.mean(v == 0.0)
            pct_gt20 = 100.0 * np.mean(v > 20.0)
            print(
                f"  {ph:<4} {len(v):>4} {v.mean():>7.1f} {np.median(v):>7.1f} {v.std():>7.1f} "
                f"{pct_zero:>5.0f}% {pct_gt20:>6.0f}%"
            )

    all_natives: dict[str, list[float]] = defaultdict(list)
    family_pools: dict[str, dict[str, list[float]]] = {}
    for accent, sources in NATIVE_SOURCES.items():
        pooled: dict[str, list[float]] = defaultdict(list)
        print(f"\n{'='*70}\n  {accent.upper()}\n{'='*70}")
        for label, manifest in sources:
            vot = _measure_source(manifest, args.n, accent)
            print(f"  {label:<16} tokens={sum(len(v) for v in vot.values())}")
            _report(f"  -- {label} --", vot)
            for ph, vals in vot.items():
                pooled[ph].extend(vals)
                if accent != "owner":
                    all_natives[ph].extend(vals)
        family_pools[accent] = pooled
        _report(f"{accent.upper()} pooled (in-domain, 16 kHz):", pooled)

    print(f"\n{'='*70}\n  ALL NATIVES POOLED (RP + GA, accent-neutral reference candidate)\n{'='*70}")
    _report("pooled across RP + GA natives:", all_natives)

    if "owner" in family_pools and any(family_pools["owner"].values()):
        print(f"\n{'='*70}\n  NATIVE - OWNER GAP (raw VOT ms, sanity check)\n{'='*70}")
        print(f"  {'ph':<4} {'native_mean':>12} {'owner_mean':>11} {'gap':>7}")
        print("  " + "-" * 36)
        for ph in _VOICELESS_STOPS:
            nat = np.array(all_natives.get(ph, []), dtype=float)
            own = np.array(family_pools["owner"].get(ph, []), dtype=float)
            if len(nat) == 0 or len(own) == 0:
                continue
            print(f"  {ph:<4} {nat.mean():>12.1f} {own.mean():>11.1f} {nat.mean() - own.mean():>+7.1f}")


if __name__ == "__main__":
    main()
