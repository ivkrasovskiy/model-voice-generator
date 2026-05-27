"""Phase 2 scoring extracted from posthoc_eval.py.

Kept separate to hold posthoc_eval.py under the 400-line limit.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import soundfile as sf

from lib.identity import cosine
from lib.identity import embed_wav as compute_ecapa_emb
from lib.posthoc_helpers import aggregate, compute_ref_centroid
from lib.scoring import load_scoring_models, score_single_wav


def phase2_score(
    manifest: list[dict],
    out_dir: Path,
    ref_audio: Path,
    phrase_real_audio: dict[str, str] | None = None,
    phrase_sources: dict[str, str] | None = None,
    centroid_dir: Path | None = None,
    centroid_n: int = 10,
    centroid_seed: int = 42,
) -> list[dict]:
    """Load Whisper + ECAPA + DNSMOS, score all WAVs in manifest.

    F5-TTS is fully unloaded at this point — no MPS pressure.
    Returns aggregated rows (one per label/step).
    """
    from lib.audio_io import read_wav_mono

    phrase_real_audio = phrase_real_audio or {}
    phrase_sources = phrase_sources or {}

    print("\n=== PHASE 2: Scoring (Whisper + ECAPA + DNSMOS) ===")
    whisper_model, ecapa, dnsmos_session = load_scoring_models(device="cpu")

    ref_wav, ref_sr = read_wav_mono(ref_audio)
    ecapa_ref_emb = compute_ecapa_emb(ref_wav, ref_sr, ecapa)
    print(f"  ECAPA ref embedding: dim={ecapa_ref_emb.shape[0]}, norm={np.linalg.norm(ecapa_ref_emb):.3f}")

    per_slug_emb: dict[str, np.ndarray] = {}
    if phrase_real_audio:
        print(f"  Pre-embedding {len(phrase_real_audio)} real val clips for per-phrase ECAPA...")
        for slug, path in phrase_real_audio.items():
            try:
                w, sr_ = read_wav_mono(path)
                per_slug_emb[slug] = compute_ecapa_emb(w, sr_, ecapa)
            except Exception as e:
                print(f"    WARN: {slug}: {e}")

    centroid_emb = None
    if centroid_dir is not None and not per_slug_emb:
        centroid_emb = compute_ref_centroid(centroid_dir, centroid_n, ecapa, centroid_seed)

    NAN_ROW = {"wer": np.nan, "ecapa_sim": np.nan, "ecapa_centroid_sim": np.nan,
               "dnsmos_sig": np.nan, "dnsmos_bak": np.nan, "dnsmos_ovr": np.nan,
               "transcript": ""}

    per_clip = []
    for entry in manifest:
        if not entry["wav_path"]:
            per_clip.append({**entry, **NAN_ROW})
            continue
        try:
            wav_np, sr = sf.read(entry["wav_path"])
            if wav_np.ndim > 1:
                wav_np = wav_np.mean(axis=1)
            wav_np = wav_np.astype(np.float32)
        except Exception as e:
            print(f"  {entry['label']}/{entry['slug']}: read failed: {e}")
            per_clip.append({**entry, **NAN_ROW})
            continue

        m = score_single_wav(wav_np, sr, entry["prompt"],
                             whisper_model, ecapa, dnsmos_session, ecapa_ref_emb)
        wer, hyp = m["wer"], m["transcript"]
        mos = {"sig": m["dnsmos_sig"], "bak": m["dnsmos_bak"], "ovr": m["dnsmos_ovr"]}

        ecapa_sim = m["ecapa_sim"]
        ecapa_centroid_sim = np.nan
        try:
            gen_emb = compute_ecapa_emb(wav_np, sr, ecapa)
            slug = entry.get("slug", "")
            if slug in per_slug_emb:
                ecapa_sim = cosine(per_slug_emb[slug], gen_emb)
            if centroid_emb is not None:
                ecapa_centroid_sim = cosine(centroid_emb, gen_emb)
        except Exception as e:
            print(f"    ECAPA centroid failed: {e}")

        ecapa_str = f"ecapa={ecapa_sim:.4f}"
        if centroid_emb is not None:
            ecapa_str += f" cent={ecapa_centroid_sim:.4f}"
        print(f"  {entry['label']}/{entry['slug']}: wer={wer:.3f} {ecapa_str} "
              f"sig={mos['sig']:.2f} bak={mos['bak']:.2f} ovr={mos['ovr']:.2f}")
        if isinstance(wer, float) and wer > 0.1:
            print(f"    (whisper heard: \"{hyp[:100]}\")")
        per_clip.append({**entry, "wer": wer, "ecapa_sim": ecapa_sim,
                         "ecapa_centroid_sim": ecapa_centroid_sim,
                         "dnsmos_sig": mos["sig"], "dnsmos_bak": mos["bak"],
                         "dnsmos_ovr": mos["ovr"], "transcript": hyp})

    by_label: dict[tuple, list] = {}
    for row in per_clip:
        by_label.setdefault((row["label"], row["step"]), []).append(row)
    agg_rows = []
    for (label, step), rows in by_label.items():
        a = aggregate(rows)
        a.update({"label": label, "step": step,
                  "listen_paths": ";".join(r["wav_path"] for r in rows if r["wav_path"])})
        agg_rows.append(a)
    agg_rows.sort(key=lambda r: r["step"])

    if phrase_sources:
        for row in per_clip:
            row["source"] = phrase_sources.get(row.get("slug", ""), "")

    detail_csv = out_dir / "scores_detail.csv"
    detail_fields = ["label", "step", "slug", "source", "prompt", "wer", "ecapa_sim",
                     "ecapa_centroid_sim",
                     "dnsmos_sig", "dnsmos_bak", "dnsmos_ovr", "transcript", "wav_path"]
    with detail_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=detail_fields)
        w.writeheader()
        for row in per_clip:
            w.writerow({k: row.get(k, "") for k in detail_fields})

    summary_csv = out_dir / "scores.csv"
    fields = ["label", "step", "n_scored", "wer_mean", "wer_std",
              "ecapa_sim_mean", "ecapa_sim_std",
              "ecapa_centroid_mean", "ecapa_centroid_std",
              "dnsmos_sig_mean", "dnsmos_bak_mean", "dnsmos_ovr_mean", "listen_paths"]
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in agg_rows:
            w.writerow({k: r.get(k, "") for k in fields})

    print(f"\n✓ Summary CSV → {summary_csv}")
    print(f"✓ Per-clip detail → {detail_csv}")
    print("\n=== Summary (WER ↓ better, ECAPA/CENT/DNSMOS ↑ better) ===")
    print(f"  {'label':<22} {'step':>6} {'n':>3}  "
          f"{'WER':>6}  {'ECAPA':>7}  {'CENT':>7}  {'SIG':>5}  {'BAK':>5}  {'OVR':>5}")
    for r in agg_rows:
        cent_str = f"{r['ecapa_centroid_mean']:>7.4f}" if not np.isnan(r["ecapa_centroid_mean"]) else "    n/a"
        print(f"  {r['label']:<22} {r['step']:>6} {r['n_scored']:>3}  "
              f"{r['wer_mean']:>6.3f}  {r['ecapa_sim_mean']:>7.4f}  {cent_str}  "
              f"{r['dnsmos_sig_mean']:>5.2f}  {r['dnsmos_bak_mean']:>5.2f}  {r['dnsmos_ovr_mean']:>5.2f}")
    return agg_rows
