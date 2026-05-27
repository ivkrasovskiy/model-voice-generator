"""Phase 0.13c — F3-based normalization bench (independent of Lever A/B).

Tasks C2 + C4:
  C2: Re-extract formants with F3 column for 6 speakers.
  C4: Run 4 normalization methods, compute tightness / discrimination / BC-gap.

Output:
    tts_output/accent_coach/phase0_13c/formants_with_f3/{speaker}.csv  (C2)
    docs/accent_coach_phase0_13c_norm_bench.csv                         (C4)

Usage:
    .venv/bin/python scripts/accent_coach_phase0_13c_f3_bench.py

Hard constraints (§5.4):
  - Does NOT modify production scoring path.
  - accent_coach/diagnostics/f3_normalization.py must NOT import Bark symbols.
  - bark_control row uses existing bark_transform (read-only, in this script only).
  - Existing CSVs in cleaned_corpus/ are untouched.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

# bark_control ONLY — imported in this script, never in f3_normalization.py
from accent_coach.diagnostics.bark_distance import bark_transform
from accent_coach.diagnostics.f3_normalization import (
    f_ratios,
    nearey_intrinsic,
    syrdal_gopal,
)

PHASE13C_DIR     = PROJECT_ROOT / "tts_output/accent_coach/phase0_13c"
F3_CSV_DIR       = PHASE13C_DIR / "formants_with_f3"
CLEANED_DIR      = PROJECT_ROOT / "tts_output/accent_coach/cleaned_corpus"
MODERN_RP_CORPUS = PROJECT_ROOT / "tts_output/modern_rp_corpus"
REAL_BC_CORPUS   = PROJECT_ROOT / "tts_output/real_bc_corpus"
BC_CAL50_DIR     = PROJECT_ROOT / "tts_output/accent_coach/bc_cal_50"
CAL50_CSV        = PROJECT_ROOT / "tts_output/accent_coach/cal_50.csv"
OWNER_MANIFEST   = PROJECT_ROOT / "tts_output/owner_cal_50/manifest.json"
BENCH_OUT_CSV    = PROJECT_ROOT / "docs/accent_coach_phase0_13c_norm_bench.csv"
PROJECT_PYTHON   = PROJECT_ROOT / ".venv/bin/python"

# Target phonemes for the bench (same 17 as Phase 0.7/0.8/0.10)
TARGET_PHONEMES = [
    "iː", "ɪ", "ɛ", "æ", "ɑː", "ɒ", "ɔː", "ʊ", "uː",
    "ʌ", "ɜː", "ə", "eɪ", "aɪ", "ɔɪ", "əʊ", "aʊ",
]
NATIVE_SPEAKERS = ["fry", "lindsey", "bbc_male"]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# C2: build manifests and extract formants with F3
# ---------------------------------------------------------------------------

def _build_fry_manifest() -> Path:
    return CLEANED_DIR / "fry" / "manifest.json"


def _build_lindsey_manifest() -> Path:
    return CLEANED_DIR / "lindsey" / "manifest.json"


def _build_bbc_manifest() -> Path:
    """Filter modern_rp_corpus manifest to bbc label entries."""
    src = MODERN_RP_CORPUS / "manifest.json"
    manifest = json.loads(src.read_text())
    bbc_entries = [e for e in manifest if e.get("label") == "bbc"]
    # Normalize to standard format
    norm = []
    for e in bbc_entries:
        norm.append({
            "clip_id": e["clip_id"],
            "path": e["path"],
            "transcript": e.get("transcript", ""),
        })
    out = F3_CSV_DIR / "bbc_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(norm, indent=2))
    return out


def _build_real_bc_manifest() -> Path:
    return REAL_BC_CORPUS / "manifest.json"


def _build_owner_manifest() -> Path:
    return OWNER_MANIFEST


def _build_synth_bc_manifest() -> Path:
    """Build manifest from bc_cal_50 WAVs + cal_50 transcripts."""
    cal50 = list(csv.DictReader((CAL50_CSV).open()))
    slug_to_prompt = {r["slug"]: r["prompt"] for r in cal50}

    entries = []
    for wav_path in sorted(BC_CAL50_DIR.glob("indextts_*.wav")):
        stem = wav_path.stem  # e.g. indextts_cal_001
        # Extract slug: cal_001
        slug = stem.replace("indextts_", "")
        prompt = slug_to_prompt.get(slug, "")
        entries.append({
            "clip_id": slug,
            "path": str(wav_path),
            "transcript": prompt,
        })

    out = F3_CSV_DIR / "synth_bc_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(entries, indent=2))
    return out


_SPEAKER_MANIFEST_BUILDERS = {
    "fry":      _build_fry_manifest,
    "lindsey":  _build_lindsey_manifest,
    "bbc_male": _build_bbc_manifest,
    "real_BC":  _build_real_bc_manifest,
    "owner":    _build_owner_manifest,
    "synth_BC": _build_synth_bc_manifest,
}


def run_c2() -> None:
    """Extract F3-bearing formant CSVs for all 6 speakers."""
    F3_CSV_DIR.mkdir(parents=True, exist_ok=True)

    for speaker, manifest_builder in _SPEAKER_MANIFEST_BUILDERS.items():
        out_csv = F3_CSV_DIR / f"{speaker}.csv"
        if out_csv.exists():
            _log(f"C2 {speaker}: already exists ({out_csv.name}), skipping")
            continue

        _log(f"C2 {speaker}: building manifest…")
        manifest_path = manifest_builder()

        _log(f"C2 {speaker}: extracting formants → {out_csv.name}")
        r = subprocess.run(
            [str(PROJECT_PYTHON),
             str(PROJECT_ROOT / "scripts/accent_coach_extract_formants.py"),
             "--manifest", str(manifest_path),
             "--out", str(out_csv),
             "--source-label", speaker],
            cwd=str(PROJECT_ROOT),
        )
        if r.returncode != 0:
            raise RuntimeError(f"extract_formants failed for {speaker}")
        _log(f"C2 {speaker}: done → {out_csv}")


def _verify_c2() -> None:
    """Check F3 coverage ≥ 80% per speaker."""
    ok = True
    for speaker in _SPEAKER_MANIFEST_BUILDERS:
        csv_path = F3_CSV_DIR / f"{speaker}.csv"
        if not csv_path.exists():
            _log(f"MISSING: {csv_path}")
            ok = False
            continue
        rows = list(csv.DictReader(csv_path.open()))
        n_f3 = sum(1 for r in rows if r.get("F3", "").strip() not in ("", "nan"))
        pct = 100 * n_f3 / max(len(rows), 1)
        _log(f"  {speaker}: {len(rows)} rows, F3 present in {n_f3} ({pct:.0f}%)")
        if pct < 80:
            _log(f"  WARNING: F3 coverage < 80% for {speaker}")
    if not ok:
        raise RuntimeError("C2 verification failed — missing CSVs")


# ---------------------------------------------------------------------------
# C4: normalization bench
# ---------------------------------------------------------------------------

def _load_f3_rows(speaker: str) -> list[dict]:
    """Load F3-bearing formant rows, filtering invalid entries."""
    csv_path = F3_CSV_DIR / f"{speaker}.csv"
    rows = []
    for row in csv.DictReader(csv_path.open()):
        try:
            f1 = float(row["F1"])
            f2 = float(row["F2"])
            f3_str = row.get("F3", "").strip()
            if not f3_str or f3_str == "nan":
                continue
            f3 = float(f3_str)
            dur = float(row.get("duration_s", 0))
        except (ValueError, KeyError):
            continue
        if f1 > 900 or dur < 0.05 or f3 <= 0:
            continue
        rows.append({
            "phoneme": row["phoneme"],
            "F1": f1, "F2": f2, "F3": f3,
            "speaker": speaker,
        })
    return rows


def _build_centroids(rows: list[dict], method_fn) -> dict[str, dict[str, tuple[float, float]]]:
    """Build per-(phoneme, speaker) centroids in method's coordinate space.

    Returns {phoneme: {speaker: (dim1, dim2)}}.
    """
    # Accumulate (dim1, dim2) per (phoneme, speaker)
    acc: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for r in rows:
        ph = r["phoneme"]
        sp = r["speaker"]
        d1, d2 = method_fn(r["F1"], r["F2"], r["F3"])
        key = (ph, sp)
        acc.setdefault(key, []).append((float(d1), float(d2)))

    centroids: dict[str, dict[str, tuple[float, float]]] = {}
    for (ph, sp), vals in acc.items():
        centroids.setdefault(ph, {})[sp] = (
            float(np.mean([v[0] for v in vals])),
            float(np.mean([v[1] for v in vals])),
        )
    return centroids


def _build_bark_centroids(rows: list[dict]) -> dict[str, dict[str, tuple[float, float]]]:
    """Build centroids using the existing bark_transform (read-only, bench-only)."""
    # Build per-phoneme, per-speaker raw Hz centroids first
    raw_acc: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for r in rows:
        ph, sp = r["phoneme"], r["speaker"]
        raw_acc.setdefault((ph, sp), []).append((r["F1"], r["F2"]))

    raw_cents: dict[str, dict] = {}
    for (ph, sp), vals in raw_acc.items():
        if ph not in raw_cents:
            raw_cents[ph] = {}
        raw_cents[ph][sp] = {
            "f1": float(np.mean([v[0] for v in vals])),
            "f2": float(np.mean([v[1] for v in vals])),
            "n": len(vals),
        }

    bark_cents = bark_transform(raw_cents)

    # Convert to the same format as _build_centroids
    result: dict[str, dict[str, tuple[float, float]]] = {}
    for ph, spks in bark_cents.items():
        result[ph] = {}
        for sp, vals in spks.items():
            result[ph][sp] = (vals["f1"], vals["f2"])
    return result


def _compute_metrics(centroids: dict[str, dict[str, tuple[float, float]]]) -> dict:
    """Compute native_tightness, l2_discrimination, bc_gap."""
    native_tightnesses = []
    l2_discriminations = []
    bc_gaps = []

    for ph in TARGET_PHONEMES:
        if ph not in centroids:
            continue
        spks = centroids[ph]

        # Native tightness: std of native speakers' centroids
        native_pts = [spks[s] for s in NATIVE_SPEAKERS if s in spks]
        if len(native_pts) >= 2:
            d1s = [p[0] for p in native_pts]
            d2s = [p[1] for p in native_pts]
            tightness = float(np.std(d1s)) + float(np.std(d2s))
            native_tightnesses.append(tightness)

        # modern_rp = mean of native speakers
        if len(native_pts) >= 1:
            mrp = (
                float(np.mean([p[0] for p in native_pts])),
                float(np.mean([p[1] for p in native_pts])),
            )
        else:
            continue

        # L2 discrimination: |owner - modern_rp|
        if "owner" in spks:
            l2_discriminations.append(float(np.linalg.norm(
                np.array(spks["owner"]) - np.array(mrp)
            )))

        # BC gap: |real_BC - modern_rp|
        if "real_BC" in spks:
            bc_gaps.append(float(np.linalg.norm(
                np.array(spks["real_BC"]) - np.array(mrp)
            )))

    return {
        "native_tightness":   float(np.mean(native_tightnesses)) if native_tightnesses else float("nan"),
        "l2_discrimination":  float(np.mean(l2_discriminations)) if l2_discriminations else float("nan"),
        "bc_gap":             float(np.mean(bc_gaps))            if bc_gaps            else float("nan"),
    }


def run_c4(all_rows: dict[str, list[dict]]) -> None:
    """Run the bench: 4 methods × 3 metrics → docs/accent_coach_phase0_13c_norm_bench.csv."""
    # Flatten all rows
    flat: list[dict] = []
    for rows in all_rows.values():
        flat.extend(rows)

    methods: list[tuple[str, object]] = [
        ("bark_control",       None),        # handled specially
        ("syrdal_gopal",       syrdal_gopal),
        ("nearey_intrinsic",   nearey_intrinsic),
        ("f_ratios",           f_ratios),
    ]

    results: list[dict] = []
    bark_metrics: dict | None = None

    for method_name, method_fn in methods:
        _log(f"C4 method: {method_name}")
        if method_name == "bark_control":
            centroids = _build_bark_centroids(flat)
        else:
            centroids = _build_centroids(flat, method_fn)
        metrics = _compute_metrics(centroids)
        entry = {"method": method_name, **metrics}
        results.append(entry)
        if method_name == "bark_control":
            bark_metrics = metrics
        _log(f"  native_tightness={metrics['native_tightness']:.4f}  "
             f"l2_discrimination={metrics['l2_discrimination']:.4f}  "
             f"bc_gap={metrics['bc_gap']:.4f}")

    if bark_metrics is None:
        raise RuntimeError("bark_control metrics missing")

    # Add relative columns
    for row in results:
        for metric in ("native_tightness", "l2_discrimination", "bc_gap"):
            b = bark_metrics[metric]
            if b and not (isinstance(b, float) and b != b):  # not nan, not zero
                row[f"{metric}_rel_bark"] = round(row[metric] / b, 3)
            else:
                row[f"{metric}_rel_bark"] = float("nan")

    fieldnames = [
        "method", "native_tightness", "l2_discrimination", "bc_gap",
        "native_tightness_rel_bark", "l2_discrimination_rel_bark", "bc_gap_rel_bark",
    ]
    BENCH_OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with BENCH_OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in results:
            w.writerow({k: round(row[k], 4) if isinstance(row[k], float) else row[k]
                        for k in fieldnames})

    _log(f"Bench CSV written to {BENCH_OUT_CSV}")

    # Print table
    print("\n=== Phase 0.13c F3 Normalization Bench ===")
    print(f"{'Method':<22} {'nat_tight':>10} {'l2_disc':>8} {'bc_gap':>8} "
          f"{'nt_rel':>7} {'l2_rel':>7} {'bc_rel':>7}")
    for row in results:
        print(f"{row['method']:<22} "
              f"{row['native_tightness']:>10.4f} "
              f"{row['l2_discrimination']:>8.4f} "
              f"{row['bc_gap']:>8.4f} "
              f"{row['native_tightness_rel_bark']:>7.3f} "
              f"{row['l2_discrimination_rel_bark']:>7.3f} "
              f"{row['bc_gap_rel_bark']:>7.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    PHASE13C_DIR.mkdir(parents=True, exist_ok=True)

    # C2: re-extract formants with F3 column
    run_c2()
    _verify_c2()

    # Load F3 rows for all speakers
    all_rows: dict[str, list[dict]] = {}
    for speaker in _SPEAKER_MANIFEST_BUILDERS:
        rows = _load_f3_rows(speaker)
        all_rows[speaker] = rows
        _log(f"Loaded {len(rows)} valid F3 rows for {speaker}")

    # C4: run bench
    run_c4(all_rows)

    return 0


if __name__ == "__main__":
    sys.exit(main())
