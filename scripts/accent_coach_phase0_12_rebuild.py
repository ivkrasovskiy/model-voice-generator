"""Phase 0.12 — rebuild Lindsey/Fry/modern_rp centroids from ECAPA-filtered clips.

Pipeline:
  1. Load corpus_audit/{speaker}_audit.json (output of accent_coach_corpus_audit.py)
  2. Filter clips by ECAPA cosine sim >= THRESHOLD (default 0.50)
  3. Whisper-transcribe each kept clip (cached to transcripts.json — idempotent)
  4. Build manifest in extract_formants format
  5. Call accent_coach_extract_formants.py to get per-token vowel formants
  6. Build cleaned centroid per speaker
  7. Build cleaned modern_rp = mean(lindsey, fry, bbc_male) (bbc unchanged)
  8. Write speaker_centroids_cleaned.json overlay
  9. Re-score Phase 0.11 cells against cleaned targets → phase0_11_rescored.csv

Idempotent: skips any step whose output already exists.

Usage:
    .venv/bin/python scripts/accent_coach_phase0_12_rebuild.py
    .venv/bin/python scripts/accent_coach_phase0_12_rebuild.py --threshold 0.6
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from accent_coach.pipeline.experiment import (
    build_centroids_from_formants,
    load_baseline_centroids,
    score_against,
)

AUDIT_DIR    = PROJECT_ROOT / "tts_output/accent_coach/corpus_audit"
CLEANED_DIR  = PROJECT_ROOT / "tts_output/accent_coach/cleaned_corpus"
PHASE11_DIR  = PROJECT_ROOT / "tts_output/accent_coach/phase0_11"
CORPUS       = PROJECT_ROOT / "tts_output/modern_rp_corpus"

SPEAKERS = ["lindsey", "fry"]


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def filter_clips(speaker: str, threshold: float) -> list[dict]:
    """Read audit JSON, keep clips with sim >= threshold."""
    audit_path = AUDIT_DIR / f"{speaker}_audit.json"
    if not audit_path.exists():
        raise FileNotFoundError(f"audit missing: {audit_path} — run accent_coach_corpus_audit.py first")
    audit = json.loads(audit_path.read_text())
    kept = [c for c in audit["per_clip"]
            if c.get("sim") is not None and c["sim"] >= threshold]
    return kept


def transcribe_kept_clips(speaker: str, kept: list[dict],
                          clips_dir: Path, out_path: Path,
                          whisper_model_name: str = "base") -> dict[str, str]:
    """Whisper-transcribe each kept clip. Cached to out_path. Returns clip_name -> transcript."""
    if out_path.exists():
        cached = json.loads(out_path.read_text())
        _log(f"  [{speaker}] transcripts loaded from cache: {len(cached)} clips")
        # Add any new clips not in cache
        missing = [c for c in kept if c["clip"] not in cached]
        if not missing:
            return cached
        _log(f"  [{speaker}] {len(missing)} new clips to transcribe (cache miss)")
        transcripts = dict(cached)
    else:
        transcripts = {}
        missing = kept

    import whisper
    _log(f"  [{speaker}] loading whisper-{whisper_model_name}…")
    model = whisper.load_model(whisper_model_name)
    _log(f"  [{speaker}] loaded.")

    t0 = time.time()
    for i, clip in enumerate(missing, 1):
        clip_path = clips_dir / clip["clip"]
        try:
            r = model.transcribe(str(clip_path), language="en", fp16=False, verbose=False)
            transcripts[clip["clip"]] = r["text"].strip()
        except Exception as e:
            _log(f"    [{speaker}] {clip['clip']}: transcribe failed ({e}) — skipping")
            transcripts[clip["clip"]] = ""

        if i % 50 == 0 or i == len(missing):
            out_path.write_text(json.dumps(transcripts, indent=2))
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (len(missing) - i) / max(rate, 0.01)
            _log(f"  [{speaker}] [{i}/{len(missing)}] elapsed={elapsed:.0f}s, "
                 f"rate={rate:.2f}/s, eta={eta:.0f}s")

    out_path.write_text(json.dumps(transcripts, indent=2))
    return transcripts


def build_manifest(speaker: str, kept: list[dict],
                   clips_dir: Path, transcripts: dict[str, str],
                   out_path: Path) -> Path:
    """Build manifest.json in the format extract_formants expects."""
    manifest = []
    for clip in kept:
        clip_name = clip["clip"]
        wav_path = clips_dir / clip_name
        transcript = transcripts.get(clip_name, "").strip()
        if not transcript:
            continue  # skip clips with empty transcripts
        manifest.append({
            "clip_id":  f"{speaker}_{clip_name.replace('.wav', '')}",
            "path":     str(wav_path.relative_to(PROJECT_ROOT)),
            "transcript": transcript,
            "label":    speaker,
        })
    out_path.write_text(json.dumps(manifest, indent=2))
    return out_path


def extract_formants_subprocess(manifest_path: Path, out_csv: Path,
                                 source_label: str) -> Path:
    if out_csv.exists():
        _log(f"    formants already exist: {out_csv}")
        return out_csv
    _log(f"    extracting formants ({source_label})…")
    t0 = time.time()
    r = subprocess.run(
        [str(PROJECT_ROOT / ".venv/bin/python"),
         str(PROJECT_ROOT / "scripts/accent_coach_extract_formants.py"),
         "--manifest",      str(manifest_path),
         "--out",           str(out_csv),
         "--source-label",  source_label],
        cwd=str(PROJECT_ROOT),
    )
    elapsed = time.time() - t0
    if r.returncode != 0:
        raise RuntimeError(f"extract_formants failed (exit {r.returncode})")
    _log(f"    formants done in {elapsed:.0f}s → {out_csv.name}")
    return out_csv


def aggregate_modern_rp(lindsey_cent: dict, fry_cent: dict, bbc_cent: dict) -> dict:
    """Build new modern_rp centroid as mean of lindsey + fry + bbc per phoneme."""
    all_phonemes = set(lindsey_cent) | set(fry_cent) | set(bbc_cent)
    aggregated = {}
    for ph in all_phonemes:
        f1s, f2s, ns = [], [], 0
        for cent in [lindsey_cent, fry_cent, bbc_cent]:
            v = cent.get(ph)
            if v is None or "f1" not in v or "f2" not in v:
                continue
            f1s.append(v["f1"])
            f2s.append(v["f2"])
            ns += v.get("n", 0)
        if len(f1s) >= 2:  # require ≥2 speakers
            aggregated[ph] = {
                "f1": round(sum(f1s) / len(f1s), 1),
                "f2": round(sum(f2s) / len(f2s), 1),
                "n":  ns,
            }
    return aggregated


def rescore_phase11(cleaned_overlay: dict) -> list[dict]:
    """Re-score Phase 0.11 cells against cleaned targets, return rows."""
    cells_dir = PHASE11_DIR / "cells"
    rows = []
    for cell_dir in sorted(cells_dir.glob("cell_*")):
        cent_path = cell_dir / "centroids.json"
        result_path = cell_dir / "result.json"
        if not cent_path.exists() or not result_path.exists():
            continue
        synth_centroids = json.loads(cent_path.read_text())
        orig = json.loads(result_path.read_text())

        new_scores = {}
        for t in ["fry", "lindsey", "bbc_male", "modern_rp"]:
            try:
                new_scores[t] = round(
                    score_against(synth_centroids, t, baseline_centroids=cleaned_overlay)["composite"], 2
                )
            except Exception:
                new_scores[t] = None

        rows.append({
            "cell_id":           orig["cell_id"],
            "note":              orig["note"],
            "fry_orig":          orig.get("fry"),
            "fry_cleaned":       new_scores["fry"],
            "fry_delta":         (round(new_scores["fry"] - orig.get("fry", 0), 2)
                                  if new_scores["fry"] is not None else None),
            "lindsey_orig":      orig.get("lindsey"),
            "lindsey_cleaned":   new_scores["lindsey"],
            "lindsey_delta":     (round(new_scores["lindsey"] - orig.get("lindsey", 0), 2)
                                  if new_scores["lindsey"] is not None else None),
            "modern_rp_orig":    orig.get("modern_rp"),
            "modern_rp_cleaned": new_scores["modern_rp"],
            "modern_rp_delta":   (round(new_scores["modern_rp"] - orig.get("modern_rp", 0), 2)
                                  if new_scores["modern_rp"] is not None else None),
            "bbc_male_orig":     orig.get("bbc_male"),
            "bbc_male_cleaned":  new_scores["bbc_male"],
            "real_BC":           orig.get("real_BC"),  # unchanged
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="ECAPA cosine threshold for keeping clips (default 0.50)")
    parser.add_argument("--whisper-model", default="base",
                        help="Whisper model size for transcription (default base)")
    args = parser.parse_args()

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    _log(f"=== Phase 0.12 corpus rebuild (threshold={args.threshold}) ===")

    baseline = load_baseline_centroids()
    bbc_cent = {ph: spks["bbc_male"] for ph, spks in baseline.items() if "bbc_male" in spks}
    _log(f"BBC baseline (unchanged): {len(bbc_cent)} phonemes")

    cleaned_centroids = {}

    for speaker in SPEAKERS:
        _log(f"\n--- {speaker} ---")
        speaker_dir = CLEANED_DIR / speaker
        speaker_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: filter clips
        kept = filter_clips(speaker, args.threshold)
        kept_path = speaker_dir / "kept_clips.json"
        kept_path.write_text(json.dumps(kept, indent=2))
        _log(f"  [{speaker}] kept {len(kept)} clips (sim >= {args.threshold})")

        # Step 2: transcribe
        clips_dir = CORPUS / speaker / "clips"
        transcripts_path = speaker_dir / "transcripts.json"
        transcripts = transcribe_kept_clips(
            speaker, kept, clips_dir, transcripts_path,
            whisper_model_name=args.whisper_model)

        # Step 3: manifest
        manifest_path = speaker_dir / "manifest.json"
        build_manifest(speaker, kept, clips_dir, transcripts, manifest_path)
        n_in_manifest = len(json.loads(manifest_path.read_text()))
        _log(f"  [{speaker}] manifest: {n_in_manifest} clips with non-empty transcripts")

        # Step 4: formants
        formants_csv = speaker_dir / "formants.csv"
        extract_formants_subprocess(manifest_path, formants_csv, source_label=speaker)

        # Step 5: centroid
        cent = build_centroids_from_formants(formants_csv)
        (speaker_dir / "centroid.json").write_text(json.dumps(cent, indent=2))
        _log(f"  [{speaker}] cleaned centroid: {len(cent)} phonemes")
        cleaned_centroids[speaker] = cent

    # Step 6: aggregate modern_rp
    _log("\n--- aggregating cleaned modern_rp ---")
    cleaned_modern_rp = aggregate_modern_rp(
        cleaned_centroids["lindsey"], cleaned_centroids["fry"], bbc_cent)
    _log(f"  cleaned modern_rp: {len(cleaned_modern_rp)} phonemes")

    # Step 7: write overlay
    overlay = {ph: dict(spks) for ph, spks in baseline.items()}
    all_overlay_phonemes = (set(overlay) | set(cleaned_modern_rp)
                            | set(cleaned_centroids["lindsey"]) | set(cleaned_centroids["fry"]))
    for ph in all_overlay_phonemes:
        if ph not in overlay:
            overlay[ph] = {}
        if ph in cleaned_centroids["lindsey"]:
            overlay[ph]["lindsey"] = cleaned_centroids["lindsey"][ph]
        if ph in cleaned_centroids["fry"]:
            overlay[ph]["fry"] = cleaned_centroids["fry"][ph]
        if ph in cleaned_modern_rp:
            overlay[ph]["modern_rp"] = cleaned_modern_rp[ph]
    overlay_path = CLEANED_DIR / "speaker_centroids_cleaned.json"
    overlay_path.write_text(json.dumps(overlay, indent=2))
    _log(f"  wrote overlay: {overlay_path}")

    # Step 8: re-score Phase 0.11
    _log("\n--- re-scoring Phase 0.11 cells against cleaned targets ---")
    if (PHASE11_DIR / "cells").exists():
        rows = rescore_phase11(overlay)
        if rows:
            out_csv = CLEANED_DIR / "phase0_11_rescored.csv"
            fields = list(rows[0].keys())
            with out_csv.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
            _log(f"  wrote: {out_csv}")
            _log(f"\n  {'cell':<22s}  {'fry orig':>10s} {'cln':>6s} {'Δ':>6s}  "
                 f"{'modern_rp orig':>15s} {'cln':>6s} {'Δ':>6s}")
            for r in rows:
                print(f"  {r['cell_id']:<22s}  "
                      f"{r['fry_orig']:>10.2f} {r['fry_cleaned']:>6.2f} {r['fry_delta']:>+6.2f}  "
                      f"{r['modern_rp_orig']:>15.2f} {r['modern_rp_cleaned']:>6.2f} {r['modern_rp_delta']:>+6.2f}")
        else:
            _log("  no Phase 0.11 cells found to re-score")
    else:
        _log(f"  Phase 0.11 cells dir not found ({PHASE11_DIR / 'cells'}) — skipping re-score")

    _log("\n=== Phase 0.12 rebuild complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
