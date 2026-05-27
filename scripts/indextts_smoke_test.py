"""
Regression smoke test for the IndexTTS-2 baseline capability.

Generates 2 short phrases with the locked baseline config and asserts they meet
WER + ECAPA thresholds vs the snapshotted baseline. Runs in ~2 minutes on CPU.

USE WHEN:
  - After installing/rebuilding vendor/index-tts/
  - After any experiment that touches IndexTTS-2 inference paths
  - Before declaring an experiment "done" (verify you didn't accidentally regress baseline)

EXIT CODE: 0 if all assertions pass, 1 otherwise.

Usage:
    vendor/index-tts/.venv/bin/python scripts/indextts_smoke_test.py
"""

import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "0")
PROJECT_ROOT = Path(__file__).parent.parent
INDEXTTS_ROOT = PROJECT_ROOT / "vendor" / "index-tts"

sys.path.insert(0, str(INDEXTTS_ROOT))

import soundfile as sf
from indextts.infer_v2 import IndexTTS2

_DEFAULT_DEVICE = "cpu"  # MPS unsupported: bigvgan alias_free conv_transpose1d fails >65536 channels

# Hard-coded baseline thresholds — derived from the 2026-05-19 zero-shot eval.
# Real baseline ECAPA was 0.78-0.85 across both clips. IndexTTS-2 has internal
# sampling (top_p/top_k/temperature in generation_kwargs) we don't seed from
# here, which causes per-clip ECAPA swings of ±0.08 between runs (observed
# range on these two clips: 0.71-0.85). The floor below sits 0.05 under the
# worst observed value so the test flags model breakage, not RNG luck.
THRESHOLDS = {
    "cas_01": {"max_wer": 0.10, "min_ecapa": 0.66},
    "sher_03": {"max_wer": 0.10, "min_ecapa": 0.66},
}

# Pick 2 phrases from eval_short.csv that should hit thresholds easily on baseline
SMOKE_SLUGS = list(THRESHOLDS.keys())

REF_AUDIO = str(PROJECT_ROOT / "tts_output/refs/indextts_baseline/ref_narrator.wav")
PHRASES_CSV = PROJECT_ROOT / "tts_output/cross_eval_50/eval_short.csv"
CFG_PATH = str(INDEXTTS_ROOT / "checkpoints/config.yaml")
MODEL_DIR = str(INDEXTTS_ROOT / "checkpoints")


def load_phrases() -> dict[str, dict]:
    with open(PHRASES_CSV) as f:
        return {r["slug"]: r for r in csv.DictReader(f) if r["slug"] in SMOKE_SLUGS}


def main() -> int:
    phrases = load_phrases()
    if len(phrases) != len(SMOKE_SLUGS):
        print(f"FAIL: only found {len(phrases)}/{len(SMOKE_SLUGS)} smoke phrases in {PHRASES_CSV}")
        return 1

    tmp_dir = Path(tempfile.mkdtemp(prefix="indextts_smoke_"))
    print(f"Smoke test workspace: {tmp_dir}")

    print(f"Loading IndexTTS-2 from {INDEXTTS_ROOT} (device={_DEFAULT_DEVICE})...")
    tts = IndexTTS2(cfg_path=CFG_PATH, model_dir=MODEL_DIR, device=_DEFAULT_DEVICE)

    manifest = []
    for slug in SMOKE_SLUGS:
        row = phrases[slug]
        out_path = tmp_dir / f"smoke_{slug}.wav"
        print(f"\n--- {slug}: {row['prompt']!r}")
        tts.infer(spk_audio_prompt=REF_AUDIO, text=row["prompt"], output_path=str(out_path))
        info = sf.info(str(out_path))
        print(f"  → {out_path.name} ({info.duration:.1f}s)")
        manifest.append({
            "label": "indextts_smoke",
            "step": 0,
            "slug": slug,
            "prompt": row["prompt"],
            "wav_path": str(out_path),
            "sr": int(info.samplerate),
        })

    # Write manifest in format posthoc_eval --score-only consumes
    import json
    (tmp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Score using main .venv (Whisper + ECAPA live there, not in vendor venv)
    print("\nScoring (running posthoc_eval.py --score-only)...")
    cmd = [
        str(PROJECT_ROOT / ".venv/bin/python"),
        str(PROJECT_ROOT / "scripts/posthoc_eval.py"),
        "--score-only",
        "--phrases-csv", str(PHRASES_CSV),
        "--out-dir", str(tmp_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAIL: scoring crashed\nstderr: {result.stderr[-500:]}")
        return 1

    # Parse per-clip scores
    scores_path = tmp_dir / "scores_detail.csv"
    if not scores_path.exists():
        print(f"FAIL: {scores_path} not produced")
        return 1

    with open(scores_path) as f:
        rows = {r["slug"]: r for r in csv.DictReader(f)}

    # Apply thresholds
    print("\n=== Regression check ===")
    fails = []
    for slug, t in THRESHOLDS.items():
        r = rows.get(slug)
        if r is None:
            fails.append(f"{slug}: missing from scores")
            continue
        wer = float(r["wer"])
        ecapa = float(r["ecapa_sim"])
        wer_ok = wer <= t["max_wer"]
        ecapa_ok = ecapa >= t["min_ecapa"]
        status = "✓" if (wer_ok and ecapa_ok) else "✗"
        print(f"  {status}  {slug}  wer={wer:.3f} (≤{t['max_wer']})  ecapa={ecapa:.3f} (≥{t['min_ecapa']})")
        if not wer_ok:
            fails.append(f"{slug}: WER {wer:.3f} > {t['max_wer']}")
        if not ecapa_ok:
            fails.append(f"{slug}: ECAPA {ecapa:.3f} < {t['min_ecapa']}")

    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)

    if fails:
        print(f"\nFAIL ({len(fails)} regressions):")
        for f in fails:
            print(f"  - {f}")
        print("\nThe baseline IndexTTS-2 capability has regressed. Do not declare experiments done.")
        print("Compare against tts_output/eval_indextts_v2/scores.regression_baseline.csv")
        return 1

    print("\nPASS — IndexTTS-2 baseline capability intact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
