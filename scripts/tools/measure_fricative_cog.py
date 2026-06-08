"""Re-derive fricative CoG reference IN-DOMAIN from native speakers.

The Jongman (2000) reference (e.g. /s/ = 7000 Hz) was measured on wide-band
(~22 kHz) studio recordings. Our pipeline normalizes everything to 16 kHz
(8 kHz Nyquist), so that target is uncapturable and natives score "too low".

This measures native fricative CoG through the IDENTICAL pipeline
(normalize_audio → align → _spectral_centroid with the frication gate) and
reports per-phoneme native means + spread, separately for RP and GA natives, so
the reference can be set from the native distribution (CLAUDE.md principle).

    .venv/bin/python scripts/tools/measure_fricative_cog.py --n 30
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

from accent_coach.comparison.consonants.fricatives import _FRICATIVES, _SKIP, _spectral_centroid
from accent_coach.pipeline.alignment import align_audio
from accent_coach.pipeline.audio_io import load_standard_audio
from scripts.lib.manifest import load_manifest, resolve_path

# Native English sources, grouped by accent family. Pool within family.
NATIVE_SOURCES: dict[str, list[tuple[str, Path, str | None]]] = {
    "rp": [
        ("fry",     REPO_ROOT / "tts_output/modern_rp_corpus/fry_manifest.json", None),
        ("lindsey", REPO_ROOT / "tts_output/modern_rp_corpus/lindsey_manifest.json", None),
        ("bbc",     REPO_ROOT / "tts_output/modern_rp_corpus/bbc_manifest.json", None),
        ("real_bc", REPO_ROOT / "tts_output/real_bc_corpus/manifest.json", None),
    ],
    "genam": [
        ("genam_lecture", REPO_ROOT / "tts_output/genam_lecture_corpus/manifest.json", None),
        ("genam_corpus",  REPO_ROOT / "tts_output/genam_corpus/manifest.json", None),
        ("vsauce",        REPO_ROOT / "tts_output/genam_corpus/manifest_vsauce.json", None),
    ],
}


def _measure_source(manifest: Path, n: int, accent: str) -> dict[str, list[float]]:
    cog: dict[str, list[float]] = defaultdict(list)
    if not manifest.exists():
        print(f"  [missing] {manifest}", file=sys.stderr)
        return cog
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
            phon = align_audio(wav, tr, i, accent)
        except Exception:  # noqa: BLE001 — drop unalignable clip
            continue
        for p in phon:
            if p.phoneme in _FRICATIVES and p.phoneme not in _SKIP:
                c = _spectral_centroid(audio, sr, p)  # None if not frication (gate)
                if c is not None:
                    cog[p.phoneme].append(c)
    return cog


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="clips per source")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    def _report(title: str, pooled: dict[str, list[float]]) -> None:
        print(f"\n  {title}")
        print(f"  {'ph':<4} {'n':>4} {'mean':>7} {'median':>7} {'std':>7}")
        print("  " + "-" * 36)
        for ph in sorted(pooled, key=lambda p: -np.mean(pooled[p]) if pooled[p] else 0):
            v = np.array(pooled[ph], dtype=float)
            if len(v) == 0:
                continue
            print(f"  {ph:<4} {len(v):>4} {v.mean():>7.0f} {np.median(v):>7.0f} {v.std():>7.0f}")

    all_natives: dict[str, list[float]] = defaultdict(list)
    for accent, sources in NATIVE_SOURCES.items():
        pooled: dict[str, list[float]] = defaultdict(list)
        print(f"\n{'='*70}\n  {accent.upper()} natives\n{'='*70}")
        for label, manifest, _spk in sources:
            cog = _measure_source(manifest, args.n, accent)
            print(f"  {label:<16} tokens={sum(len(v) for v in cog.values())}")
            for ph, vals in cog.items():
                pooled[ph].extend(vals)
                all_natives[ph].extend(vals)
        _report(f"{accent.upper()} pooled (in-domain, 16 kHz):", pooled)

    print(f"\n{'='*70}\n  ALL NATIVES POOLED (accent-neutral reference candidate)\n{'='*70}")
    _report("pooled across RP + GA:", all_natives)


if __name__ == "__main__":
    main()
