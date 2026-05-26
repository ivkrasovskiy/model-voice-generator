"""Shared experiment infrastructure for Phase 0.10 and beyond.

Provides generate_clips, extract_formants, build_centroids_from_formants,
load_baseline_centroids, score_against, and score_posthoc_clips.
All functions emit timestamped progress lines so long runs are observable.
"""
from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from accent_coach.comparison.vowels import score_vowels_piecewise
from accent_coach.diagnostics.bark_distance import bark_transform
from accent_coach.diagnostics.cluster_eval import per_phoneme_sigma_rp
from accent_coach.reference.rp_norms import RP_VOWEL_F1_F2_MALE_LEGACY

PROJECT_ROOT = Path(__file__).parent.parent.parent
BASELINE_CENTROIDS_PATH = (
    PROJECT_ROOT / "tts_output/accent_coach/bench/phase0_7/speaker_centroids.json"
)
CLEANED_CENTROIDS_PATH = (
    PROJECT_ROOT / "tts_output/accent_coach/cleaned_corpus/speaker_centroids_cleaned.json"
)
INDEXTTS_PYTHON = PROJECT_ROOT / "vendor/index-tts/.venv/bin/python"
PROJECT_PYTHON = PROJECT_ROOT / ".venv/bin/python"

# ---------------------------------------------------------------------------
# Lazy singletons — loaded on first call, reused within the same process
# ---------------------------------------------------------------------------
_WHISPER = None
_ECAPA = None
_DNSMOS = None
_ECAPA_REF_EMB: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# generate_clips
# ---------------------------------------------------------------------------

def generate_clips(
    phrases_csv: Path,
    ref_audio: Path,
    out_dir: Path,
    gen_params: dict,
    label: str,
) -> Path:
    """Subprocess wrapper around indextts_gen.py. Idempotent if manifest is complete."""
    phrases_csv = Path(phrases_csv)
    ref_audio = Path(ref_audio)
    out_dir = Path(out_dir)

    manifest_path = out_dir / "manifest.json"
    n_expected = sum(1 for _ in phrases_csv.open()) - 1  # subtract header

    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if len(existing) == n_expected:
            _log(f"generate_clips: {n_expected} clips already present at {out_dir.name}, skipping")
            return manifest_path
        _log(f"generate_clips: partial manifest ({len(existing)}/{n_expected}), re-generating")

    out_dir.mkdir(parents=True, exist_ok=True)

    _defaults = {
        "temperature": 0.8, "top_p": 0.8, "top_k": 30, "num_beams": 3,
        "cfg_rate": 0.7, "diffusion_steps": 25,
    }
    flags: list[str] = []
    if gen_params.get("temperature",      _defaults["temperature"])      != _defaults["temperature"]:
        flags += ["--temperature",      str(gen_params["temperature"])]
    if gen_params.get("top_p",            _defaults["top_p"])            != _defaults["top_p"]:
        flags += ["--top-p",            str(gen_params["top_p"])]
    if gen_params.get("top_k",            _defaults["top_k"])            != _defaults["top_k"]:
        flags += ["--top-k",            str(gen_params["top_k"])]
    if gen_params.get("num_beams",        _defaults["num_beams"])        != _defaults["num_beams"]:
        flags += ["--num-beams",        str(gen_params["num_beams"])]
    if gen_params.get("cfg_rate",         _defaults["cfg_rate"])         != _defaults["cfg_rate"]:
        flags += ["--cfg-rate",         str(gen_params["cfg_rate"])]
    if gen_params.get("diffusion_steps",  _defaults["diffusion_steps"])  != _defaults["diffusion_steps"]:
        flags += ["--diffusion-steps",  str(gen_params["diffusion_steps"])]
    # Emo conditioning (Phase 0.11)
    if gen_params.get("emo_audio"):
        flags += ["--emo-audio",  str(gen_params["emo_audio"])]
    if gen_params.get("emo_alpha", 1.0) != 1.0:
        flags += ["--emo-alpha",  str(gen_params["emo_alpha"])]
    if gen_params.get("use_emo_text"):
        flags += ["--use-emo-text"]
        if gen_params.get("emo_text"):
            flags += ["--emo-text",  str(gen_params["emo_text"])]

    _log(f"generate_clips: {n_expected} phrases → {out_dir.name}  ref={ref_audio.name}  params={gen_params}")
    t0 = time.time()
    r = subprocess.run(
        [str(INDEXTTS_PYTHON), str(PROJECT_ROOT / "scripts/indextts_gen.py"),
         "--phrases-csv", str(phrases_csv),
         "--out-dir",     str(out_dir),
         "--ref-audio",   str(ref_audio),
         "--label",       label,
         *flags],
        cwd=str(PROJECT_ROOT),
    )
    elapsed = time.time() - t0
    if r.returncode != 0:
        raise RuntimeError(f"indextts_gen.py failed (exit {r.returncode})")
    _log(f"generate_clips: done in {elapsed:.0f}s → {manifest_path}")
    return manifest_path


# ---------------------------------------------------------------------------
# extract_formants
# ---------------------------------------------------------------------------

def extract_formants(
    manifest: Path,
    out_csv: Path,
    source_label: str = "synth_BC",
    n_workers: int | None = None,  # accepted for API compat; parallelism not yet in subprocess
) -> Path:
    """Subprocess wrapper around accent_coach_extract_formants.py."""
    manifest = Path(manifest)
    out_csv = Path(out_csv)

    if out_csv.exists():
        _log(f"extract_formants: already exists at {out_csv.name}, skipping")
        return out_csv

    _log(f"extract_formants: starting (source_label={source_label!r}, manifest={manifest.parent.name})")
    t0 = time.time()
    r = subprocess.run(
        [str(PROJECT_PYTHON), str(PROJECT_ROOT / "scripts/accent_coach_extract_formants.py"),
         "--manifest",      str(manifest),
         "--out",           str(out_csv),
         "--source-label",  source_label],
        cwd=str(PROJECT_ROOT),
    )
    elapsed = time.time() - t0
    if r.returncode != 0:
        raise RuntimeError(f"accent_coach_extract_formants.py failed (exit {r.returncode})")
    _log(f"extract_formants: done in {elapsed:.0f}s → {out_csv.name}")
    return out_csv


# ---------------------------------------------------------------------------
# build_centroids_from_formants
# ---------------------------------------------------------------------------

def build_centroids_from_formants(
    formants_csv: Path,
    min_duration_s: float = 0.050,
) -> dict[str, dict]:
    """Per-phoneme mean F1/F2 from formant CSV (duration≥min_duration_s filter)."""
    rows: list[dict] = []
    with Path(formants_csv).open() as f:
        for row in csv.DictReader(f):
            try:
                if float(row["duration_s"]) >= min_duration_s:
                    rows.append(row)
            except (ValueError, KeyError):
                continue

    by_phoneme: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        ph = row["phoneme"]
        try:
            f1, f2 = float(row["F1"]), float(row["F2"])
        except (ValueError, KeyError):
            continue
        by_phoneme.setdefault(ph, []).append((f1, f2))

    centroids: dict[str, dict] = {}
    for ph, pairs in by_phoneme.items():
        centroids[ph] = {
            "f1": round(float(np.mean([p[0] for p in pairs])), 1),
            "f2": round(float(np.mean([p[1] for p in pairs])), 1),
            "n":  len(pairs),
        }
    return centroids


# ---------------------------------------------------------------------------
# load_baseline_centroids
# ---------------------------------------------------------------------------

def load_baseline_centroids() -> dict:
    """Load speaker centroids + add deterding pseudo-speaker.

    Precedence: Phase 0.12 cleaned overlay (ECAPA-filtered lindsey/fry, rebuilt
    modern_rp) if present, else Phase 0.7 frozen baseline. Override either via
    env var ACCENT_COACH_CENTROIDS_PATH (absolute path).
    """
    import os
    override = os.environ.get("ACCENT_COACH_CENTROIDS_PATH")
    if override:
        path = Path(override)
    elif CLEANED_CENTROIDS_PATH.exists():
        path = CLEANED_CENTROIDS_PATH
    else:
        path = BASELINE_CENTROIDS_PATH
    with path.open() as f:
        raw = json.load(f)
    for phoneme, f1f2 in RP_VOWEL_F1_F2_MALE_LEGACY.items():
        if phoneme not in raw:
            raw[phoneme] = {}
        raw[phoneme]["deterding"] = {"f1": f1f2[0], "f2": f1f2[1], "n": 0}
    return raw


# ---------------------------------------------------------------------------
# score_against
# ---------------------------------------------------------------------------

def score_against(
    synth_centroids: dict[str, dict],
    target: str,
    baseline_centroids: dict | None = None,
) -> dict:
    """Apply bark_transform, compute sigma_rp from RP cluster, score synth_BC vs target.

    sigma_rp is always derived from {fry, lindsey, bbc_male} regardless of target.
    Returns {'composite': float, 'per_phoneme': {phoneme: score}}.
    """
    if baseline_centroids is None:
        baseline_centroids = load_baseline_centroids()

    # shallow-copy top level so we don't mutate the caller's dict
    centroids: dict[str, dict] = {ph: dict(spks) for ph, spks in baseline_centroids.items()}
    for phoneme, vals in synth_centroids.items():
        if phoneme not in centroids:
            centroids[phoneme] = {}
        centroids[phoneme]["synth_BC"] = vals

    bark_centroids = bark_transform(centroids)
    sigma_rp = per_phoneme_sigma_rp(bark_centroids)

    target_f1f2 = {
        ph: (spks[target]["f1"], spks[target]["f2"])
        for ph, spks in bark_centroids.items()
        if spks.get(target)
    }
    synth_f1f2 = {
        ph: (spks["synth_BC"]["f1"], spks["synth_BC"]["f2"])
        for ph, spks in bark_centroids.items()
        if spks.get("synth_BC")
    }

    scores = score_vowels_piecewise(synth_f1f2, target_f1f2, sigma_rp)
    return {"composite": scores["composite"], "per_phoneme": scores["per_phoneme"]}


# ---------------------------------------------------------------------------
# score_posthoc_clips
# ---------------------------------------------------------------------------

def _load_scoring_models() -> None:
    global _WHISPER, _ECAPA, _DNSMOS
    if _WHISPER is None:
        from lib.transcribe import load_whisper
        _log("Loading Whisper large-v3…")
        _WHISPER = load_whisper("large-v3", device="cpu")
    if _ECAPA is None:
        from lib.identity import load_ecapa
        _log("Loading ECAPA-TDNN…")
        _ECAPA = load_ecapa(device="cpu")
    if _DNSMOS is None:
        from lib.metrics import load_dnsmos
        _log("Loading DNSMOS…")
        _DNSMOS = load_dnsmos()


def _get_ecapa_ref_emb(ecapa_ref: Path) -> np.ndarray:
    global _ECAPA_REF_EMB
    if _ECAPA_REF_EMB is not None:
        return _ECAPA_REF_EMB
    from lib.identity import embed_file
    emb = embed_file(str(ecapa_ref), _ECAPA)
    if emb is None:
        raise RuntimeError(f"Failed to embed ECAPA reference: {ecapa_ref}")
    _ECAPA_REF_EMB = emb
    _log(f"ECAPA ref: {Path(ecapa_ref).name}  dim={emb.shape[0]}")
    return _ECAPA_REF_EMB


def score_posthoc_clips(
    manifest: Path,
    ecapa_ref: Path,
    out_csv: Path,
    n_workers: int | None = None,  # accepted for API compat; sequential for now
) -> dict:
    """Score clips for WER/ECAPA/DNSMOS. Returns {'WER', 'ECAPA', 'DNSMOS_OVR'}."""
    import soundfile as sf

    manifest = Path(manifest)
    ecapa_ref = Path(ecapa_ref)
    out_csv = Path(out_csv)

    if out_csv.exists():
        _log(f"score_posthoc_clips: cached at {out_csv.name}, skipping")
        existing = list(csv.DictReader(out_csv.open()))
        if existing:
            wers   = [float(r["wer"])       for r in existing if r.get("wer")       not in ("", "nan")]
            ecapas = [float(r["ecapa_sim"])  for r in existing if r.get("ecapa_sim") not in ("", "nan")]
            dnsmos = [float(r["dnsmos_ovr"]) for r in existing if r.get("dnsmos_ovr") not in ("", "nan")]
            return {
                "WER":        round(float(np.mean(wers)),   4) if wers   else float("nan"),
                "ECAPA":      round(float(np.mean(ecapas)), 4) if ecapas else float("nan"),
                "DNSMOS_OVR": round(float(np.mean(dnsmos)), 4) if dnsmos else float("nan"),
            }

    _load_scoring_models()
    ecapa_ref_emb = _get_ecapa_ref_emb(ecapa_ref)

    from lib.identity import cosine, embed_wav
    from lib.metrics import compute_dnsmos, compute_wer

    clips = json.loads(manifest.read_text())
    _log(f"score_posthoc_clips: scoring {len(clips)} clips…")

    rows = []
    for i, entry in enumerate(clips):
        wav_path = Path(entry["wav_path"])
        if not wav_path.exists():
            _log(f"  SKIP missing: {wav_path.name}")
            rows.append({"slug": entry["slug"], "prompt": entry["prompt"],
                         "wer": float("nan"), "ecapa_sim": float("nan"),
                         "dnsmos_ovr": float("nan"), "wav_path": str(wav_path)})
            continue

        audio, sr = sf.read(str(wav_path))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        wer = float("nan")
        try:
            wer, hyp = compute_wer(entry["prompt"], audio, sr, _WHISPER)
            if wer > 0.1:
                _log(f"  WER={wer:.3f} heard: {hyp[:80]!r}")
        except Exception as e:
            _log(f"  WER failed for {entry['slug']}: {e}")

        ecapa_sim = float("nan")
        try:
            gen_emb = embed_wav(audio, sr, _ECAPA)
            ecapa_sim = cosine(ecapa_ref_emb, gen_emb)
        except Exception as e:
            _log(f"  ECAPA failed for {entry['slug']}: {e}")

        dnsmos_ovr = float("nan")
        try:
            mos = compute_dnsmos(audio, sr, _DNSMOS)
            dnsmos_ovr = mos["ovr"]
        except Exception as e:
            _log(f"  DNSMOS failed for {entry['slug']}: {e}")

        rows.append({
            "slug":       entry["slug"],
            "prompt":     entry["prompt"],
            "wer":        round(wer, 4),
            "ecapa_sim":  round(ecapa_sim, 4),
            "dnsmos_ovr": round(dnsmos_ovr, 4),
            "wav_path":   str(wav_path),
        })
        _log(f"  [{i+1}/{len(clips)}] {entry['slug']}: "
             f"wer={wer:.3f} ecapa={ecapa_sim:.4f} dnsmos={dnsmos_ovr:.2f}")

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else ["slug"])
        w.writeheader()
        w.writerows(rows)

    wers_   = [r["wer"]        for r in rows if not (isinstance(r["wer"],        float) and math.isnan(r["wer"]))]
    ecapas_ = [r["ecapa_sim"]  for r in rows if not (isinstance(r["ecapa_sim"],  float) and math.isnan(r["ecapa_sim"]))]
    dnsmos_ = [r["dnsmos_ovr"] for r in rows if not (isinstance(r["dnsmos_ovr"], float) and math.isnan(r["dnsmos_ovr"]))]

    return {
        "WER":        round(float(np.mean(wers_)),   4) if wers_   else float("nan"),
        "ECAPA":      round(float(np.mean(ecapas_)), 4) if ecapas_ else float("nan"),
        "DNSMOS_OVR": round(float(np.mean(dnsmos_)), 4) if dnsmos_ else float("nan"),
    }
