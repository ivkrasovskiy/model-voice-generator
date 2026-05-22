"""Phase 0.9 driver: reference-clip swap experiment for synth_BC quality.

Tracks A (BC reference variants) and B (RP ceiling) — 8 generation runs total.

For each variant ID:
  1. Build reference clip if not yet present (A2/A3 via YouTube, B1/B2 via stitch)
  2. Generate 15-phrase eval set with IndexTTS-2 using that reference
  3. Extract per-phoneme vowel formants (alignment + formant extraction)
  4. Compute synth_BC_<ID> centroids; patch into Phase 0.7 baseline centroids
  5. H4 Bark piecewise score  (same metric as Phase 0.8 — must reproduce A1≈74.6)
  6. WER / ECAPA / DNSMOS
  7. Append row to experiment_log.csv

Usage:
    .venv/bin/python scripts/accent_coach_phase0_9_run.py
    .venv/bin/python scripts/accent_coach_phase0_9_run.py --variants A1 A2 B1
    .venv/bin/python scripts/accent_coach_phase0_9_run.py --variants A1 --skip-gen
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "0")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))  # for lib.*

import numpy as np
import soundfile as sf

from accent_coach.comparison.vowels import score_vowels_piecewise
from accent_coach.diagnostics.bark_distance import bark_transform
from accent_coach.diagnostics.cluster_eval import per_phoneme_sigma_rp
from accent_coach.reference.rp_norms import RP_VOWEL_F1_F2_MALE_LEGACY

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PHASE09_DIR = PROJECT_ROOT / "tts_output/accent_coach/phase0_9"
VARIANTS_DIR = PHASE09_DIR / "variants"
LOG_CSV = PHASE09_DIR / "experiment_log.csv"
BASELINE_CENTROIDS = PROJECT_ROOT / "tts_output/accent_coach/bench/phase0_7/speaker_centroids.json"
# Eval phrases: used for WER / ECAPA / DNSMOS scoring (15-phrase, compact)
EVAL_CSV = PROJECT_ROOT / "tts_output/cross_eval_50/eval_short.csv"
# Cal phrases: used for TTS generation + formant extraction (50-phrase, richer vowel coverage)
CAL_CSV  = PROJECT_ROOT / "tts_output/accent_coach/cal_50.csv"
# Fixed BC identity reference for ECAPA (per-clip real-BC refs absent on this machine)
ECAPA_REF = PROJECT_ROOT / "tts_output/ref_narrator.wav"
INDEXTTS_PYTHON = PROJECT_ROOT / "vendor/index-tts/.venv/bin/python"
PROJECT_PYTHON = PROJECT_ROOT / ".venv/bin/python"

YT_URL = "https://youtu.be/cHmkAStZBkc"

LOG_FIELDS = [
    "experiment_id", "track", "ref_clip", "ref_duration_s", "gen_params_json",
    "synth_BC_piecewise_H4Bark", "per_phoneme_csv", "WER", "ECAPA", "DNSMOS_OVR",
    "gen_time_min", "notes",
]

# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

# Track C gen-param sweep variants.  Key "gen_params" overrides IndexTTS-2 defaults.
# Run on two refs: A1 (interview, BC identity) and B2 (Lindsey, RP ceiling).
# Default params for reference: temperature=0.8, top_p=0.8, top_k=30, num_beams=3
_C_REFS = {
    "int": ("tts_output/ref_interview.wav",  "interview ref"),
    "lin": ("tts_output/ref_lindsey.wav",    "Lindsey RP ref"),
}
_C_PARAMS = {
    "t06": {"temperature": 0.6, "top_p": 0.8,  "top_k": 30, "num_beams": 3},
    "t04": {"temperature": 0.4, "top_p": 0.8,  "top_k": 30, "num_beams": 3},
    "b5":  {"temperature": 0.8, "top_p": 0.8,  "top_k": 30, "num_beams": 5},
    "t04b5": {"temperature": 0.4, "top_p": 0.8, "top_k": 30, "num_beams": 5},
}

_TRACK_C: dict[str, dict] = {}
for _ref_id, (_ref_path, _ref_note) in _C_REFS.items():
    for _p_id, _p_vals in _C_PARAMS.items():
        _vid = f"C_{_ref_id}_{_p_id}"
        _TRACK_C[_vid] = {
            "track": "C",
            "ref_clip": _ref_path,
            "gen_params": _p_vals,
            "notes": f"{_ref_note} | temp={_p_vals['temperature']} beams={_p_vals['num_beams']}",
        }

VARIANTS: dict[str, dict] = {
    "A1": {
        "track": "A",
        "ref_clip": "tts_output/ref_interview.wav",
        # Skip TTS generation AND formant extraction for A1.  The synth_BC entry already
        # stored in phase0_7/speaker_centroids.json is exactly what produced Phase 0.8's
        # 74.6 score.  Scoring it directly validates the bark+piecewise math without
        # re-extraction variance (~±3 pts from fresh alignments on the same clips).
        "use_phase07_centroids": True,
        "cal_manifest": "tts_output/accent_coach/bc_cal_50/manifest.json",
        "notes": "Baseline — must reproduce 74.6±2",
    },
    "A2": {
        "track": "A",
        "ref_clip": "tts_output/ref_interview_429_446.wav",
        "build_yt": {"url": YT_URL, "start": "00:04:29", "duration": "17"},
        "notes": "Interview 4:29-4:46 (17s) — owner-flagged candidate",
    },
    "A3": {
        "track": "A",
        "ref_clip": "tts_output/ref_interview_1450_1510.wav",
        "build_yt": {"url": YT_URL, "start": "00:14:50", "duration": "20"},
        "notes": "Interview 14:50-15:10 (20s) — owner-flagged candidate",
    },
    "A4": {
        "track": "A",
        "ref_clip": "tts_output/ref_sherlock.wav",
        "notes": "Sherlock dialogue (7s) — character speech, scripted",
    },
    "A5": {
        "track": "A",
        "ref_clip": "tts_output/ref_narrator.wav",
        "notes": "Casanova audiobook narrator (11s) — read register",
    },
    "A6": {
        "track": "A",
        "ref_clip": "tts_output/ref_combined.wav",
        "notes": "Casanova+Sherlock stitched (18s) — multi-register",
    },
    "B1": {
        "track": "B",
        "ref_clip": "tts_output/ref_fry.wav",
        "build_rp": "fry",
        "notes": "Fry RP reference (~17s) — zero-shot vowel ceiling",
    },
    "B2": {
        "track": "B",
        "ref_clip": "tts_output/ref_lindsey.wav",
        "build_rp": "lindsey",
        "notes": "Lindsey RP reference (~28s) — zero-shot vowel ceiling",
    },
}
VARIANTS.update(_TRACK_C)

# ---------------------------------------------------------------------------
# Reference clip builders
# ---------------------------------------------------------------------------

def build_youtube_ref(ref_path: Path, cfg: dict) -> None:
    """Download + trim a YouTube clip using build_podcast_ref.py."""
    print(f"  Building {ref_path.name} from YouTube ({cfg['start']} +{cfg['duration']}s)…")
    r = subprocess.run(
        [str(PROJECT_PYTHON), "scripts/build_podcast_ref.py",
         "--url", cfg["url"],
         "--start", cfg["start"],
         "--duration", str(cfg["duration"]),
         "--out", str(ref_path.relative_to(PROJECT_ROOT)),
         "--force"],
        cwd=str(PROJECT_ROOT),
    )
    if r.returncode != 0:
        raise RuntimeError(f"build_podcast_ref.py failed for {ref_path.name}")


def build_rp_ref(speaker: str, out_path: Path, target_sr: int = 22050, target_dur: float = 20.0) -> list[str]:
    """Stitch top clips from modern_rp_corpus/<speaker>/clips/ into a reference.

    Selection: largest clips by file size (proxy for voiced content).
    Concatenated with 100 ms silence between clips, resampled to 22050 Hz.
    Returns list of source clip paths for logging.
    """
    import torch
    import torchaudio

    clips_dir = PROJECT_ROOT / f"tts_output/modern_rp_corpus/{speaker}/clips"
    wavs = sorted(clips_dir.glob("*.wav"), key=lambda p: p.stat().st_size, reverse=True)
    if not wavs:
        raise RuntimeError(f"No clips found in {clips_dir}")

    chosen: list[Path] = []
    total_dur = 0.0
    for wav in wavs:
        info = sf.info(str(wav))
        chosen.append(wav)
        total_dur += info.duration
        if total_dur >= target_dur:
            break

    silence = np.zeros(int(0.1 * target_sr), dtype=np.float32)
    parts: list[np.ndarray] = []
    for i, wav_path in enumerate(chosen):
        audio, sr = sf.read(str(wav_path))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        if sr != target_sr:
            t = torch.from_numpy(audio).float().unsqueeze(0)
            audio = torchaudio.functional.resample(t, sr, target_sr).squeeze(0).numpy()
        parts.append(audio)
        if i < len(chosen) - 1:
            parts.append(silence)

    combined = np.concatenate(parts)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), combined, target_sr, subtype="PCM_16")
    actual_dur = len(combined) / target_sr
    print(f"  Built {out_path.name}: {len(chosen)} clips → {actual_dur:.1f}s @ {target_sr}Hz")
    return [str(c) for c in chosen]


def ensure_ref_clip(variant_id: str, cfg: dict) -> Path:
    """Return the reference clip path, building it if necessary."""
    ref_path = PROJECT_ROOT / cfg["ref_clip"]
    if ref_path.exists():
        return ref_path

    if "build_yt" in cfg:
        build_youtube_ref(ref_path, cfg["build_yt"])
    elif "build_rp" in cfg:
        build_rp_ref(cfg["build_rp"], ref_path)
    else:
        raise FileNotFoundError(f"Reference clip missing and no build config: {ref_path}")

    if not ref_path.exists():
        raise RuntimeError(f"Reference clip still missing after build: {ref_path}")
    return ref_path


def get_ref_duration(ref_path: Path) -> float:
    info = sf.info(str(ref_path))
    return float(info.duration)

# ---------------------------------------------------------------------------
# Step 2: Generate clips
# ---------------------------------------------------------------------------

def _gen_param_flags(gen_params: dict) -> list[str]:
    """Return CLI flags for non-default generation params."""
    flags = []
    if gen_params.get("temperature", 0.8) != 0.8:
        flags += ["--temperature", str(gen_params["temperature"])]
    if gen_params.get("top_p", 0.8) != 0.8:
        flags += ["--top-p", str(gen_params["top_p"])]
    if gen_params.get("top_k", 30) != 30:
        flags += ["--top-k", str(gen_params["top_k"])]
    if gen_params.get("num_beams", 3) != 3:
        flags += ["--num-beams", str(gen_params["num_beams"])]
    return flags


def generate_clips(variant_id: str, ref_path: Path, clips_dir: Path, cfg: dict) -> Path:
    """Run IndexTTS-2 on CAL_CSV (50 phrases). Returns manifest path.

    If cfg has 'cal_manifest', skips generation and returns that path directly
    (used for A1 which reuses the existing bc_cal_50 clips).
    Passes any gen_params from cfg as CLI flags to indextts_gen.py.
    """
    # A1 special case: reuse pre-existing bc_cal_50 clips
    if "cal_manifest" in cfg:
        cal_manifest = PROJECT_ROOT / cfg["cal_manifest"]
        print(f"  [{variant_id}] generation: reusing {cal_manifest.parent.name} (no new clips needed)")
        return cal_manifest

    manifest_path = clips_dir / "manifest.json"
    n_expected = sum(1 for _ in CAL_CSV.open()) - 1  # subtract header
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if len(existing) == n_expected:
            print(f"  [{variant_id}] generation: {len(existing)} clips already exist, skipping")
            return manifest_path
        print(f"  [{variant_id}] generation: partial manifest ({len(existing)}/{n_expected}), re-generating")

    clips_dir.mkdir(parents=True, exist_ok=True)
    extra_flags = _gen_param_flags(cfg.get("gen_params", {}))
    print(f"  [{variant_id}] generating {n_expected} cal phrases with ref={ref_path.name}"
          + (f"  params={extra_flags}" if extra_flags else "") + "…")
    r = subprocess.run(
        [str(INDEXTTS_PYTHON), "scripts/indextts_gen.py",
         "--phrases-csv", str(CAL_CSV),
         "--out-dir", str(clips_dir),
         "--ref-audio", str(ref_path),
         "--label", f"synth_BC_{variant_id}",
         *extra_flags],
        cwd=str(PROJECT_ROOT),
    )
    if r.returncode != 0:
        raise RuntimeError(f"indextts_gen.py failed for variant {variant_id}")
    return manifest_path


# ---------------------------------------------------------------------------
# Step 3: Extract formants
# ---------------------------------------------------------------------------

def extract_formants(variant_id: str, manifest_path: Path, var_dir: Path) -> Path:
    """Run accent_coach_extract_formants.py. Returns formants CSV path."""
    formants_csv = var_dir / "formants.csv"
    if formants_csv.exists():
        print(f"  [{variant_id}] formants: already extracted, skipping")
        return formants_csv

    print(f"  [{variant_id}] extracting formants…")
    r = subprocess.run(
        [str(PROJECT_PYTHON), "scripts/accent_coach_extract_formants.py",
         "--manifest", str(manifest_path),
         "--out", str(formants_csv),
         "--source-label", "synth_BC"],
        cwd=str(PROJECT_ROOT),
    )
    if r.returncode != 0:
        raise RuntimeError(f"accent_coach_extract_formants.py failed for variant {variant_id}")
    return formants_csv


# ---------------------------------------------------------------------------
# Step 4 + 5: Centroids → H4 Bark piecewise score
# ---------------------------------------------------------------------------

def build_synth_bc_centroids(formants_csv: Path) -> dict[str, dict]:
    """Compute per-phoneme mean F1/F2 from formant CSV (duration≥50ms filter)."""
    rows: list[dict] = []
    with formants_csv.open() as f:
        for row in csv.DictReader(f):
            try:
                if float(row["duration_s"]) >= 0.050:
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
        f1_mean = float(np.mean([p[0] for p in pairs]))
        f2_mean = float(np.mean([p[1] for p in pairs]))
        centroids[ph] = {"f1": round(f1_mean, 1), "f2": round(f2_mean, 1), "n": len(pairs)}
    return centroids


def _load_baseline_centroids() -> dict:
    """Load phase0_7 speaker_centroids.json and add deterding pseudo-speaker."""
    with BASELINE_CENTROIDS.open() as f:
        raw = json.load(f)
    for phoneme, f1f2 in RP_VOWEL_F1_F2_MALE_LEGACY.items():
        if phoneme not in raw:
            raw[phoneme] = {}
        raw[phoneme]["deterding"] = {"f1": f1f2[0], "f2": f1f2[1], "n": 0}
    return raw


def score_h4_bark(variant_id: str, new_synth_bc: dict[str, dict] | None, var_dir: Path) -> dict:
    """Apply H4 Bark transform and score piecewise against modern_rp.

    If new_synth_bc is None, uses the frozen synth_BC entry from the phase0_7
    baseline centroids (A1 validity check — reproduces 74.6 exactly).
    Otherwise patches synth_BC with the supplied centroids before scoring.

    Returns dict with keys: synth_BC_piecewise, per_phoneme_csv.
    """
    centroids = _load_baseline_centroids()

    # Patch synth_BC with new variant's centroids (skip for A1 no-patch mode)
    if new_synth_bc is not None:
        for phoneme, vals in new_synth_bc.items():
            if phoneme not in centroids:
                centroids[phoneme] = {}
            centroids[phoneme]["synth_BC"] = vals

    # H4 Bark transform
    bark_centroids = bark_transform(centroids)

    # sigma_rp — invariant across variants since RP speakers unchanged
    sigma_rp = per_phoneme_sigma_rp(bark_centroids)

    # modern_rp centroid in Bark space
    ref_f1f2 = {
        ph: (spks["modern_rp"]["f1"], spks["modern_rp"]["f2"])
        for ph, spks in bark_centroids.items()
        if spks.get("modern_rp")
    }

    def _speaker_f1f2(spk: str) -> dict[str, tuple[float, float]]:
        return {
            ph: (spks[spk]["f1"], spks[spk]["f2"])
            for ph, spks in bark_centroids.items()
            if spks.get(spk)
        }

    synth_scores = score_vowels_piecewise(_speaker_f1f2("synth_BC"), ref_f1f2, sigma_rp)
    real_scores  = score_vowels_piecewise(_speaker_f1f2("real_BC"),  ref_f1f2, sigma_rp)
    owner_scores = score_vowels_piecewise(_speaker_f1f2("owner"),    ref_f1f2, sigma_rp)

    result = {
        "variant": variant_id,
        "sigma_rp": {k: round(v, 3) for k, v in sigma_rp.items()},
        "scores": {
            "synth_BC": synth_scores,
            "real_BC":  real_scores,
            "owner":    owner_scores,
        },
    }
    bark_out = var_dir / "bark_scores.json"
    bark_out.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    # Write per-phoneme Bark-distance CSV
    per_phoneme_csv = var_dir / "per_phoneme.csv"
    pp_rows = []
    for ph, score in synth_scores["per_phoneme"].items():
        ref = ref_f1f2.get(ph)
        s = bark_centroids.get(ph, {}).get("synth_BC")
        pp_rows.append({
            "phoneme": ph,
            "synth_BC_f1_bark": round(s["f1"], 4) if s else "",
            "synth_BC_f2_bark": round(s["f2"], 4) if s else "",
            "modern_rp_f1_bark": round(ref[0], 4) if ref else "",
            "modern_rp_f2_bark": round(ref[1], 4) if ref else "",
            "sigma_rp": round(sigma_rp.get(ph, 0), 4),
            "score": score,
        })
    if pp_rows:
        with per_phoneme_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=pp_rows[0].keys())
            w.writeheader()
            w.writerows(pp_rows)

    # Write per-variant centroids JSON for reproducibility
    centroids_out = var_dir / "centroids.json"
    centroids_out.write_text(json.dumps(
        {ph: {spk: vals for spk, vals in spks.items()} for ph, spks in centroids.items()},
        indent=2, ensure_ascii=False,
    ))

    return {
        "synth_BC_piecewise": synth_scores["composite"],
        "per_phoneme_csv": str(per_phoneme_csv),
    }


# ---------------------------------------------------------------------------
# Step 6: WER / ECAPA / DNSMOS
# ---------------------------------------------------------------------------

# Lazy singletons — loaded on first call, reused across variants
_WHISPER = None
_ECAPA   = None
_DNSMOS  = None
# Fixed BC reference embedding for ECAPA (per-clip real-BC refs absent; use narrator clip)
_ECAPA_REF_EMB: np.ndarray | None = None


def _load_models():
    global _WHISPER, _ECAPA, _DNSMOS
    if _WHISPER is None:
        from lib.transcribe import load_whisper
        print("  Loading Whisper large-v3…")
        _WHISPER = load_whisper("large-v3", device="cpu")
    if _ECAPA is None:
        from lib.identity import load_ecapa
        print("  Loading ECAPA-TDNN…")
        _ECAPA = load_ecapa(device="cpu")
    if _DNSMOS is None:
        from lib.metrics import load_dnsmos
        print("  Loading DNSMOS…")
        _DNSMOS = load_dnsmos()


def _load_ecapa_ref() -> np.ndarray:
    global _ECAPA_REF_EMB
    if _ECAPA_REF_EMB is not None:
        return _ECAPA_REF_EMB
    from lib.identity import embed_file
    emb = embed_file(str(ECAPA_REF), _ECAPA)
    if emb is None:
        raise RuntimeError(f"Failed to embed ECAPA reference: {ECAPA_REF}")
    _ECAPA_REF_EMB = emb
    print(f"  ECAPA ref: {ECAPA_REF.name}  dim={emb.shape[0]}")
    return _ECAPA_REF_EMB


def _ensure_eval_clips(variant_id: str, ref_path: Path, eval_clips_dir: Path,
                       gen_params: dict | None = None) -> Path:
    """Generate 15-phrase eval clips (EVAL_CSV) for WER/ECAPA/DNSMOS scoring."""
    manifest_path = eval_clips_dir / "manifest.json"
    n_expected = sum(1 for _ in EVAL_CSV.open()) - 1
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if len(existing) == n_expected:
            return manifest_path
    eval_clips_dir.mkdir(parents=True, exist_ok=True)
    extra_flags = _gen_param_flags(gen_params or {})
    print(f"  [{variant_id}] generating {n_expected} eval phrases for posthoc scoring…")
    r = subprocess.run(
        [str(INDEXTTS_PYTHON), "scripts/indextts_gen.py",
         "--phrases-csv", str(EVAL_CSV),
         "--out-dir", str(eval_clips_dir),
         "--ref-audio", str(ref_path),
         "--label", f"synth_BC_{variant_id}",
         *extra_flags],
        cwd=str(PROJECT_ROOT),
    )
    if r.returncode != 0:
        raise RuntimeError(f"indextts_gen.py failed (eval) for variant {variant_id}")
    return manifest_path


def score_posthoc(variant_id: str, ref_path: Path, var_dir: Path,
                  gen_params: dict | None = None) -> dict:
    """Generate 15 eval clips then score WER/ECAPA/DNSMOS. Returns aggregate means.

    Uses a separate eval_clips/ subdirectory so cal clips (for formants) stay clean.
    ECAPA compares against ref_narrator.wav (fixed BC identity ref; per-clip refs absent).
    """
    eval_clips_dir = var_dir / "eval_clips"
    posthoc_csv = var_dir / "posthoc_scores.csv"
    if posthoc_csv.exists():
        print(f"  [{variant_id}] posthoc: already scored, skipping")
        existing = list(csv.DictReader(posthoc_csv.open()))
        if existing:
            wers   = [float(r["wer"])        for r in existing if r.get("wer")        not in ("", "nan")]
            ecapas = [float(r["ecapa_sim"])   for r in existing if r.get("ecapa_sim")  not in ("", "nan")]
            dnsmos = [float(r["dnsmos_ovr"])  for r in existing if r.get("dnsmos_ovr") not in ("", "nan")]
            return {
                "WER": round(float(np.mean(wers)),   4) if wers   else float("nan"),
                "ECAPA": round(float(np.mean(ecapas)), 4) if ecapas else float("nan"),
                "DNSMOS_OVR": round(float(np.mean(dnsmos)), 4) if dnsmos else float("nan"),
            }

    # Generate eval clips if not present
    manifest_path = _ensure_eval_clips(variant_id, ref_path, eval_clips_dir, gen_params)

    _load_models()
    ecapa_ref_emb = _load_ecapa_ref()

    from lib.identity import cosine, embed_wav
    from lib.metrics import compute_dnsmos, compute_wer

    manifest = json.loads(manifest_path.read_text())

    rows = []
    for entry in manifest:
        wav_path = Path(entry["wav_path"])
        if not wav_path.exists():
            print(f"    SKIP missing: {wav_path.name}")
            rows.append({**entry, "wer": float("nan"), "ecapa_sim": float("nan"), "dnsmos_ovr": float("nan")})
            continue

        audio, sr = sf.read(str(wav_path))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        wer = float("nan")
        try:
            wer, hyp = compute_wer(entry["prompt"], audio, sr, _WHISPER)
            if wer > 0.1:
                print(f"    WER={wer:.3f} heard: {hyp[:80]!r}")
        except Exception as e:
            print(f"    WER failed for {entry['slug']}: {e}")

        ecapa_sim = float("nan")
        try:
            gen_emb = embed_wav(audio, sr, _ECAPA)
            ecapa_sim = cosine(ecapa_ref_emb, gen_emb)
        except Exception as e:
            print(f"    ECAPA failed for {entry['slug']}: {e}")

        dnsmos_ovr = float("nan")
        try:
            mos = compute_dnsmos(audio, sr, _DNSMOS)
            dnsmos_ovr = mos["ovr"]
        except Exception as e:
            print(f"    DNSMOS failed for {entry['slug']}: {e}")

        rows.append({
            "slug": entry["slug"],
            "prompt": entry["prompt"],
            "wer": round(wer, 4),
            "ecapa_sim": round(ecapa_sim, 4),
            "dnsmos_ovr": round(dnsmos_ovr, 4),
            "wav_path": str(wav_path),
        })
        print(f"    {entry['slug']}: wer={wer:.3f} ecapa={ecapa_sim:.4f} dnsmos={dnsmos_ovr:.2f}")

    with posthoc_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else ["slug"])
        w.writeheader()
        w.writerows(rows)

    wers   = [r["wer"]        for r in rows if not (isinstance(r["wer"],       float) and math.isnan(r["wer"]))]
    ecapas = [r["ecapa_sim"]  for r in rows if not (isinstance(r["ecapa_sim"], float) and math.isnan(r["ecapa_sim"]))]
    dnsmos = [r["dnsmos_ovr"] for r in rows if not (isinstance(r["dnsmos_ovr"],float) and math.isnan(r["dnsmos_ovr"]))]

    return {
        "WER": round(float(np.mean(wers)),   4) if wers   else float("nan"),
        "ECAPA": round(float(np.mean(ecapas)), 4) if ecapas else float("nan"),
        "DNSMOS_OVR": round(float(np.mean(dnsmos)), 4) if dnsmos else float("nan"),
    }


# ---------------------------------------------------------------------------
# Step 7: Experiment log
# ---------------------------------------------------------------------------

def _init_log() -> None:
    PHASE09_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG_CSV.exists():
        with LOG_CSV.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()


def _log_row_exists(variant_id: str) -> bool:
    if not LOG_CSV.exists():
        return False
    with LOG_CSV.open() as f:
        return any(row["experiment_id"] == variant_id for row in csv.DictReader(f))


def _write_log_row(row: dict) -> None:
    _init_log()
    # Remove existing row for this variant (allow re-runs to overwrite)
    existing = []
    if LOG_CSV.exists():
        with LOG_CSV.open() as f:
            existing = [r for r in csv.DictReader(f) if r["experiment_id"] != row["experiment_id"]]
    all_rows = existing + [{f: row.get(f, "") for f in LOG_FIELDS}]
    with LOG_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        w.writeheader()
        w.writerows(all_rows)


# ---------------------------------------------------------------------------
# Main variant runner
# ---------------------------------------------------------------------------

def run_variant(
    variant_id: str,
    skip_gen: bool = False,
    skip_formants: bool = False,
    skip_bark: bool = False,
    skip_posthoc: bool = False,
) -> None:
    cfg = VARIANTS[variant_id]
    var_dir = VARIANTS_DIR / variant_id
    clips_dir = var_dir / "clips"      # cal clips (50 phrases, for formant extraction)
    var_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Variant {variant_id} ({cfg['track']}) — {cfg['notes']}")
    print(f"{'='*60}")

    # Step 1: ensure reference clip exists
    ref_path = ensure_ref_clip(variant_id, cfg)
    ref_dur = get_ref_duration(ref_path)
    print(f"  ref: {ref_path.name}  ({ref_dur:.1f}s)")

    # Step 2: generate cal clips (50 phrases) for formant extraction
    # A1: reuses bc_cal_50 (no new generation); others: generate into clips/
    t0 = time.time()
    if not skip_gen:
        cal_manifest_path = generate_clips(variant_id, ref_path, clips_dir, cfg)
    else:
        if "cal_manifest" in cfg:
            cal_manifest_path = PROJECT_ROOT / cfg["cal_manifest"]
        else:
            cal_manifest_path = clips_dir / "manifest.json"
        if not cal_manifest_path.exists():
            raise FileNotFoundError(f"--skip-gen but manifest not found: {cal_manifest_path}")
        print(f"  [{variant_id}] generation: skipped")
    gen_time_min = (time.time() - t0) / 60

    # Step 3: extract formants from cal clips
    # A1 skips re-extraction — uses frozen phase0_7 synth_BC centroids for exact 74.6 reproduction.
    use_existing = cfg.get("use_phase07_centroids", False)
    if not skip_formants and not use_existing:
        formants_csv = extract_formants(variant_id, cal_manifest_path, var_dir)
    elif use_existing:
        formants_csv = var_dir / "formants.csv"  # won't be read; centroids come from phase0_7
        print(f"  [{variant_id}] formants: using frozen phase0_7 synth_BC centroids (no re-extraction)")
    else:
        formants_csv = var_dir / "formants.csv"
        if not formants_csv.exists():
            raise FileNotFoundError(f"--skip-formants but formants.csv not found: {formants_csv}")
        print(f"  [{variant_id}] formants: skipped")

    # Step 4+5: centroids + H4 Bark score
    if not skip_bark:
        if use_existing:
            new_centroids = None  # score_h4_bark will use phase0_7 synth_BC as-is
            print(f"  [{variant_id}] scoring H4 Bark piecewise (frozen centroids)…")
        else:
            print(f"  [{variant_id}] computing synth_BC centroids…")
            new_centroids = build_synth_bc_centroids(formants_csv)
            n_phonemes = len(new_centroids)
            print(f"    {n_phonemes} phonemes with data: {sorted(new_centroids)}")
            if n_phonemes < 8:
                print(f"  WARNING: only {n_phonemes} phonemes — formant extraction may have failed badly")
            print(f"  [{variant_id}] scoring H4 Bark piecewise…")

        bark_result = score_h4_bark(variant_id, new_centroids, var_dir)
        piecewise = bark_result["synth_BC_piecewise"]
        per_phoneme_csv = bark_result["per_phoneme_csv"]
        print(f"  >>> synth_BC H4 Bark piecewise = {piecewise:.1f}")
    else:
        bark_out = var_dir / "bark_scores.json"
        if bark_out.exists():
            d = json.loads(bark_out.read_text())
            piecewise = d["scores"]["synth_BC"]["composite"]
            print(f"  [{variant_id}] bark: skipped (cached {piecewise:.1f})")
        else:
            piecewise = float("nan")
        per_phoneme_csv = str(var_dir / "per_phoneme.csv")

    # Step 6: posthoc scores (eval clips generated separately from cal clips)
    gen_params = cfg.get("gen_params", {})
    if not skip_posthoc:
        print(f"  [{variant_id}] scoring WER/ECAPA/DNSMOS…")
        posthoc = score_posthoc(variant_id, ref_path, var_dir, gen_params)
    else:
        posthoc = {"WER": float("nan"), "ECAPA": float("nan"), "DNSMOS_OVR": float("nan")}

    print(f"  WER={posthoc['WER']:.4f}  ECAPA={posthoc['ECAPA']:.4f}  DNSMOS={posthoc['DNSMOS_OVR']:.2f}")

    # Step 7: log
    _write_log_row({
        "experiment_id": variant_id,
        "track": cfg["track"],
        "ref_clip": str(ref_path),
        "ref_duration_s": round(ref_dur, 2),
        "gen_params_json": json.dumps({**{"ecapa_ref": ECAPA_REF.name}, **cfg.get("gen_params", {})}),
        "synth_BC_piecewise_H4Bark": round(piecewise, 2) if not math.isnan(piecewise) else "",
        "per_phoneme_csv": per_phoneme_csv,
        "WER": posthoc["WER"],
        "ECAPA": posthoc["ECAPA"],
        "DNSMOS_OVR": posthoc["DNSMOS_OVR"],
        "gen_time_min": round(gen_time_min, 2),
        "notes": cfg["notes"],
    })
    print(f"  Logged → {LOG_CSV.name}")

    # A1 pipeline-validity check
    if variant_id == "A1" and not math.isnan(piecewise):
        expected = 74.6
        delta = abs(piecewise - expected)
        if delta > 2.0:
            print("\n  *** PIPELINE VALIDITY FAILURE ***")
            print(f"  A1 expected {expected}±2  got {piecewise:.1f}  (Δ={delta:.1f})")
            print("  Do NOT trust subsequent variant scores until pipeline is diagnosed.")
            sys.exit(1)
        else:
            print(f"\n  A1 validity check PASSED: {piecewise:.1f} (expected {expected}±2, Δ={delta:.1f})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 0.9 reference-clip swap experiment")
    parser.add_argument(
        "--variants", nargs="+", default=list(VARIANTS.keys()),
        choices=list(VARIANTS.keys()), metavar="ID",
        help="Which variants to run (default: all A1-A6 B1-B2)",
    )
    parser.add_argument("--skip-gen",      action="store_true", help="Skip IndexTTS-2 generation")
    parser.add_argument("--skip-formants", action="store_true", help="Skip formant extraction")
    parser.add_argument("--skip-bark",     action="store_true", help="Skip H4 Bark scoring")
    parser.add_argument("--skip-posthoc",  action="store_true", help="Skip WER/ECAPA/DNSMOS")
    args = parser.parse_args()

    _init_log()
    print(f"Phase 0.9 — running variants: {args.variants}")
    print(f"Log: {LOG_CSV}")

    for vid in args.variants:
        run_variant(
            vid,
            skip_gen=args.skip_gen,
            skip_formants=args.skip_formants,
            skip_bark=args.skip_bark,
            skip_posthoc=args.skip_posthoc,
        )

    print(f"\n{'='*60}")
    print("All variants done. Results:")
    if LOG_CSV.exists():
        with LOG_CSV.open() as f:
            for row in csv.DictReader(f):
                if row["experiment_id"] in args.variants:
                    score = row["synth_BC_piecewise_H4Bark"] or "?"
                    wer   = row["WER"] or "?"
                    ecapa = row["ECAPA"] or "?"
                    print(f"  {row['experiment_id']:3s}  H4Bark={score:6s}  WER={wer:6s}  ECAPA={ecapa:6s}  {row['notes']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
