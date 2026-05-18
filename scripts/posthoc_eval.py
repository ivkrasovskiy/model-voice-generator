"""
Post-hoc eval with richer metrics:
  - Whisper large-v3 WER  → intelligibility ("did the words come out right?")
  - ECAPA-TDNN cosine sim → speaker identity (SOTA, replaces Resemblyzer)
  - DNSMOS SIG/BAK/OVR    → perceived audio quality (1-5 MOS)
  - Resemblyzer cosine    → kept for comparison with earlier numbers

All metrics are read-only: each takes the same generated WAV in, produces numbers out.
They cannot affect each other.

Usage:
    PYTHONHASHSEED=random .venv/bin/python scripts/posthoc_eval.py \
        --listen-checkpoints step_000500.pt
"""

import argparse
import csv
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _dotenv_init import init_env_then_reexec, kill_stale_python
init_env_then_reexec(__file__)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import soundfile as sf

PROJECT_ROOT = Path(__file__).parent.parent
DNSMOS_CACHE_DIR = Path.home() / ".cache" / "dnsmos"
DNSMOS_URL = "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"

REF_AUDIO = PROJECT_ROOT / "tts_output/ref_narrator.wav"
REF_TEXT = "this an ideal opportunity for obtaining from her everything I wished."

# Inference knobs — overridden from CLI in main(); module-level so the gen helpers
# can read them without threading kwargs through every call.
_CFG_STRENGTH = 2.0  # F5-TTS default
_NFE_STEP = 32       # F5-TTS default
_SEED = 42           # F5-TTS infer seed — fixed for reproducibility across runs
_FIRST_N = None      # if set, run only the first N phrases (subset early-stop)
_CENTROID_DIR: Path | None = None  # if set, compute "Cumberbatch identity centroid" from N random WAVs here
_CENTROID_N = 10     # number of clips averaged into the centroid
_SPEED_FIX = False         # if True, apply speed=0.3 for segments < 10 bytes (F5-TTS Issue #1155)
_SELECTIVE_CFG: float | None = None  # if set, selective CFG threshold t (arXiv 2509.19668)
PHRASE_SOURCES: dict[str, str] = {}  # slug -> source label, populated when --phrases-csv has source column

# Six eval phrases, all ≤ 80 chars (F5-TTS single-batch limit → no compounding NaN).
# Each targets a different axis of voice identity / generalization:
#  - stella_short: backward-compat with earlier scores (Speech Accent Archive control)
#  - rainbow:      Rainbow Passage opener — phonetically balanced classic eval
#  - casual:       conversational register — explicit non-narration test
#  - technical:    modern/OOD domain — far from 1700s Casanova training
#  - deep_vowels:  open vowels exercising chest resonance (Cumberbatch's signature)
#  - imperative:   short bursts — crisp RP plosives, dynamic range, no narrator cadence
EVAL_PHRASES = [
    ("stella_short",
     "Please call Stella; ask her to bring these things from the store."),
    ("rainbow",
     "When the sunlight strikes raindrops in the air, they act as a prism."),
    ("casual",
     "I haven't seen him since the last meeting, but I'll ask around tomorrow."),
    ("technical",
     "The algorithm processes each frame independently before merging results."),
    ("deep_vowels",
     "Whose woods these are I think I know; his house is in the village."),
    ("imperative",
     "Stop. Don't move. There's something behind you."),
]


# ---------- model loaders ----------

def apply_checkpoint(tts, ckpt_path: Path) -> int:
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = checkpoint["model"]
    model = tts.ema_model.transformer
    own = model.state_dict()
    n_loaded = 0
    for k, v in sd.items():
        key = k.replace("transformer.", "", 1) if k.startswith("transformer.") else k
        if key in own:
            own[key] = v.to(own[key].device)
            n_loaded += 1
    model.load_state_dict(own, strict=False)
    return n_loaded


def load_whisper(device: str = "cpu"):
    import whisper
    print(f"Loading Whisper large-v3 on {device}...")
    return whisper.load_model("large-v3", device=device)


def load_ecapa(device: str = "cpu"):
    from speechbrain.inference.speaker import EncoderClassifier
    print(f"Loading ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb) on {device}...")
    savedir = Path.home() / ".cache" / "speechbrain-ecapa"
    savedir.mkdir(parents=True, exist_ok=True)
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(savedir),
        run_opts={"device": device},
    )


def load_dnsmos():
    import onnxruntime as ort
    DNSMOS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = DNSMOS_CACHE_DIR / "sig_bak_ovr.onnx"
    if not model_path.exists():
        print(f"Downloading DNSMOS ONNX → {model_path}")
        urllib.request.urlretrieve(DNSMOS_URL, str(model_path))
    print(f"Loading DNSMOS ONNX (sig_bak_ovr) on CPU...")
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


# ---------- metric computations ----------

def split_to_short_segments(text: str, max_chars: int = 50) -> list[str]:
    """Split a phrase into ≤max_chars segments, preferring sentence/clause boundaries.

    Why: F5-TTS auto-chunks any phrase longer than ~58 chars (depending on ref-audio
    duration), and each chunk independently has 30-50% NaN rate on MPS. Multi-batch
    phrases compound to ~5% per-attempt success. By pre-splitting to ≤50 chars and
    calling F5-TTS separately per segment with its own retry, each gen is single-batch
    and retries actually help.
    """
    # First pass: split on sentence-ending punctuation
    sentences = re.split(r"(?<=[.!?])\s+", text)
    segments: list[str] = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(s) <= max_chars:
            segments.append(s)
            continue
        # Sentence too long — split on clause boundaries (; : ,)
        parts = re.split(r"(?<=[,;:])\s+", s)
        buf = ""
        for p in parts:
            p = p.strip()
            if not p:
                continue
            joined = (buf + " " + p).strip() if buf else p
            if len(joined) <= max_chars:
                buf = joined
            else:
                if buf:
                    segments.append(buf)
                if len(p) <= max_chars:
                    buf = p
                else:
                    # Clause too long — split on word boundaries
                    words = p.split()
                    buf2 = ""
                    for w in words:
                        joined2 = (buf2 + " " + w).strip() if buf2 else w
                        if len(joined2) <= max_chars:
                            buf2 = joined2
                        else:
                            if buf2:
                                segments.append(buf2)
                            buf2 = w
                    buf = buf2
        if buf:
            segments.append(buf)
    return segments


def _mps_reset():
    """Flush MPS memory pool to prevent fragmentation-induced NaN across segments."""
    import gc
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _gen_segment_with_retry(tts, segment: str, max_retries: int = 5):
    """Generate a single ≤50-char segment with retry. Returns (wav_np, sr) or (None, None).

    Each retry uses seed + attempt to vary the noise — same config still produces the
    same first-attempt waveform, so runs are reproducible.
    """
    speed = 0.3 if _SPEED_FIX and len(segment.encode("utf-8")) < 10 else 1.0
    if speed != 1.0:
        print(f"      speed-fix: len={len(segment.encode('utf-8'))} bytes → speed={speed}")
    for attempt in range(1, max_retries + 1):
        wav, sr, _ = tts.infer(
            ref_file=str(REF_AUDIO), ref_text=REF_TEXT, gen_text=segment,
            cfg_strength=_CFG_STRENGTH, nfe_step=_NFE_STEP,
            speed=speed,
            seed=_SEED + (attempt - 1),
            selective_cfg_threshold=_SELECTIVE_CFG,
        )
        w = wav.squeeze() if hasattr(wav, "squeeze") else wav
        w = np.asarray(w, dtype=np.float32)
        finite = np.isfinite(w).all()
        peak = float(np.abs(w[np.isfinite(w)]).max()) if finite else 0.0
        _mps_reset()  # flush MPS pool after every gen, success or fail
        if finite and peak > 0.01:
            return w, sr
        print(f"      attempt {attempt}: {'NaN/Inf' if not finite else f'SILENT peak={peak:.4f}'}")
    return None, None


def gen_with_retry(tts, text: str, max_retries: int = 5):
    """Generate audio for arbitrary-length text by pre-splitting to single-batch segments
    and concatenating with 150 ms silence between them.

    Returns (wav_np, sr) or (None, None) if ANY segment fails after retries.
    """
    segments = split_to_short_segments(text, max_chars=50)
    if len(segments) == 1:
        return _gen_segment_with_retry(tts, segments[0], max_retries=max_retries)
    print(f"    split into {len(segments)} segments (≤50 chars each)")
    pieces = []
    sr_out = None
    for i, seg in enumerate(segments, 1):
        wav_np, sr = _gen_segment_with_retry(tts, seg, max_retries=max_retries)
        if wav_np is None:
            print(f"    segment {i}/{len(segments)} FAILED — abort")
            return None, None
        print(f"    segment {i}/{len(segments)} ok  ({len(wav_np)/sr:.1f}s, peak={float(np.abs(wav_np).max()):.3f})")
        pieces.append(wav_np)
        if sr_out is None:
            sr_out = sr
        # Inter-segment silence (150 ms) for natural pause
        pieces.append(np.zeros(int(0.15 * sr), dtype=np.float32))
    return np.concatenate(pieces[:-1]), sr_out  # drop trailing silence


def _resample(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return wav
    import torchaudio
    t = torch.from_numpy(wav).float().unsqueeze(0)
    return torchaudio.functional.resample(t, sr, target_sr).squeeze(0).numpy()


def _normalize_text(t: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — for WER fairness."""
    t = t.lower()
    t = re.sub(r"[^\w\s']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def compute_wer(ref_text: str, wav: np.ndarray, sr: int, whisper_model) -> float:
    import jiwer
    wav16 = _resample(wav, sr, 16000)
    result = whisper_model.transcribe(wav16, language="en", fp16=False, verbose=False)
    hyp = result["text"]
    return float(jiwer.wer(_normalize_text(ref_text), _normalize_text(hyp))), hyp


def compute_ecapa_emb(wav: np.ndarray, sr: int, ecapa) -> np.ndarray:
    wav16 = _resample(wav, sr, 16000)
    t = torch.from_numpy(wav16).float().unsqueeze(0)
    with torch.no_grad():
        emb = ecapa.encode_batch(t).squeeze().cpu().numpy()
    return emb


def compute_ref_centroid(centroid_dir: Path, n_samples: int, ecapa, seed: int) -> np.ndarray:
    """Embed N random WAVs from centroid_dir and average → 'Cumberbatch identity centroid'.

    More robust speaker target than a single ref clip: gen-vs-centroid measures
    'how Cumberbatch-like' rather than 'how like this one clip'. Deterministic
    via seed (same N clips picked every run).
    """
    import random
    wav_dir = centroid_dir / "wavs" if (centroid_dir / "wavs").is_dir() else centroid_dir
    wavs = sorted(wav_dir.glob("*.wav"))
    if len(wavs) < n_samples:
        raise RuntimeError(f"Only {len(wavs)} WAVs in {wav_dir}, need {n_samples}")
    rng = random.Random(seed)
    picked = rng.sample(wavs, n_samples)
    print(f"  centroid: averaging {n_samples} ECAPA embeddings from {wav_dir.name}/")
    embs = []
    for p in picked:
        w, sr = sf.read(str(p))
        if w.ndim > 1:
            w = w.mean(axis=1)
        embs.append(compute_ecapa_emb(w.astype(np.float32), sr, ecapa))
    centroid = np.mean(np.stack(embs, axis=0), axis=0)
    print(f"  centroid norm={np.linalg.norm(centroid):.3f}  (single-clip avg norm={np.mean([np.linalg.norm(e) for e in embs]):.3f})")
    return centroid


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def compute_dnsmos(wav: np.ndarray, sr: int, session) -> dict:
    """Returns dict with SIG, BAK, OVR each in [1, 5] MOS range.

    Microsoft's sig_bak_ovr.onnx — read expected input length from the model itself.
    """
    wav16 = _resample(wav, sr, 16000)
    expected_len = session.get_inputs()[0].shape[1]  # read from ONNX metadata
    if len(wav16) < expected_len:
        wav16 = np.pad(wav16, (0, expected_len - len(wav16)), mode="constant")
    else:
        wav16 = wav16[:expected_len]
    inp = wav16.astype(np.float32).reshape(1, -1)
    out = session.run(None, {session.get_inputs()[0].name: inp})
    sig, bak, ovr = out[0][0]
    return {"sig": float(sig), "bak": float(bak), "ovr": float(ovr)}


# ---------- main pipeline ----------

def evaluate_one(label: str, phrases, tts, whisper_model, ecapa, ecapa_ref_emb,
                 dnsmos_session, out_dir: Path, save_wavs: bool):
    """Generate + score one config (baseline or one checkpoint).

    Returns dict with mean/std for each metric, plus saved WAV paths.
    """
    rows = []
    saved = []
    for slug, prompt in phrases:
        wav_np, sr = gen_with_retry(tts, prompt)
        if wav_np is None:
            print(f"  {label}/{slug}: GENERATION FAILED")
            rows.append({"slug": slug, "wer": np.nan, "ecapa_sim": np.nan,
                         "dnsmos_sig": np.nan, "dnsmos_bak": np.nan, "dnsmos_ovr": np.nan,
                         "transcript": ""})
            continue

        if save_wavs:
            wav_path = out_dir / f"{label}_{slug}.wav"
            sf.write(str(wav_path), wav_np, sr)
            saved.append(str(wav_path.relative_to(PROJECT_ROOT)))
            print(f"  saved → {wav_path.name}")

        # Metrics
        wer, hyp = compute_wer(prompt, wav_np, sr, whisper_model)
        gen_emb = compute_ecapa_emb(wav_np, sr, ecapa)
        ecapa_sim = cosine(ecapa_ref_emb, gen_emb)
        mos = compute_dnsmos(wav_np, sr, dnsmos_session)
        print(f"  {label}/{slug}: wer={wer:.3f} ecapa={ecapa_sim:.4f} "
              f"sig={mos['sig']:.2f} bak={mos['bak']:.2f} ovr={mos['ovr']:.2f}")
        if wer > 0.1:
            print(f"    (whisper heard: \"{hyp[:80]}\")")
        rows.append({"slug": slug, "wer": wer, "ecapa_sim": ecapa_sim,
                     "dnsmos_sig": mos["sig"], "dnsmos_bak": mos["bak"], "dnsmos_ovr": mos["ovr"],
                     "transcript": hyp})

    return rows, saved


def aggregate(rows: list[dict]) -> dict:
    def m(key):
        vals = [r[key] for r in rows if not np.isnan(r[key])]
        if not vals:
            return float("nan"), float("nan"), 0
        return float(np.mean(vals)), float(np.std(vals)), len(vals)

    w_mean, w_std, n = m("wer")
    e_mean, e_std, _ = m("ecapa_sim")
    ec_mean, ec_std, _ = m("ecapa_centroid_sim")
    sig_mean, _, _ = m("dnsmos_sig")
    bak_mean, _, _ = m("dnsmos_bak")
    ovr_mean, _, _ = m("dnsmos_ovr")
    return {
        "n_scored": n,
        "wer_mean": w_mean, "wer_std": w_std,
        "ecapa_sim_mean": e_mean, "ecapa_sim_std": e_std,
        "ecapa_centroid_mean": ec_mean, "ecapa_centroid_std": ec_std,
        "dnsmos_sig_mean": sig_mean, "dnsmos_bak_mean": bak_mean, "dnsmos_ovr_mean": ovr_mean,
    }


def phase1_generate(listen_ckpts: list[Path], out_dir: Path, device: str,
                    skip_baseline: bool) -> list[dict]:
    """Phase 1: Load ONLY F5-TTS + Vocos, generate all WAVs, save to disk.

    Memory pressure from Whisper/ECAPA/DNSMOS would push MPS into NaN territory
    on long phrases — load them only after F5-TTS is unloaded.

    Resume-safe: existing WAVs on disk are reused. Manifest is written
    incrementally after each gen, so a mid-run crash loses at most one clip.
    If everything is already on disk, F5-TTS is never loaded.

    Returns manifest: list of {label, step, slug, prompt, wav_path}
    """
    import gc
    import json

    # Build the full config plan: (label, step) for baseline + each checkpoint
    configs: list[tuple[str, int]] = []
    if not skip_baseline:
        configs.append(("baseline", 0))
    for ckpt in listen_ckpts:
        step_num = 0
        if ckpt.name.startswith("step_"):
            step_num = int(ckpt.stem.split("_")[1])
        elif ckpt.name == "final.pt":
            try:
                step_num = torch.load(str(ckpt), map_location="cpu", weights_only=False)["step"]
            except Exception:
                step_num = -1
        configs.append((ckpt.stem, step_num))

    # Scan disk for already-generated WAVs (source of truth — manifest may be stale)
    manifest: list[dict] = []
    for label, step in configs:
        for slug, prompt in EVAL_PHRASES:
            wav_path = out_dir / f"{label}_{slug}.wav"
            if not wav_path.exists():
                continue
            try:
                info = sf.info(str(wav_path))
                manifest.append({"label": label, "step": step, "slug": slug,
                                 "prompt": prompt, "wav_path": str(wav_path),
                                 "sr": int(info.samplerate)})
            except Exception:
                pass  # corrupted file → leave out so we regenerate

    done_keys = {(e["label"], e["slug"]) for e in manifest}
    manifest_path = out_dir / "manifest.json"

    def _persist():
        manifest_path.write_text(json.dumps(manifest, indent=2))

    # Work list, preserving config order (baseline first, then checkpoints in order)
    todo = [(label, step, slug, prompt)
            for (label, step) in configs
            for (slug, prompt) in EVAL_PHRASES
            if (label, slug) not in done_keys]

    if manifest:
        print(f"\n=== PHASE 1: Resume — {len(manifest)} WAVs on disk, {len(todo)} remaining ===")
    else:
        print(f"\n=== PHASE 1: Generation (F5-TTS only, {len(todo)} clips) ===")
    _persist()  # write initial manifest so phase 2 can see what's already done

    if not todo:
        print("  All WAVs already exist — skipping F5-TTS load")
        return manifest

    from f5_tts.api import F5TTS
    print(f"Loading F5TTS_v1_Base on {device}...")
    tts = F5TTS(model="F5TTS_v1_Base", device=device)

    current_label = None
    for label, step, slug, prompt in todo:
        if label != current_label:
            if label == "baseline":
                print(f"\n--- Baseline (zero-shot) ---")
            else:
                ckpt = next(c for c in listen_ckpts if c.stem == label)
                print(f"\n--- {ckpt.name} ---")
                n_loaded = apply_checkpoint(tts, ckpt)
                print(f"  applied {n_loaded} tensors")
            current_label = label

        wav_np, sr = gen_with_retry(tts, prompt)
        wav_path = out_dir / f"{label}_{slug}.wav"
        if wav_np is None:
            print(f"  {label}/{slug}: GENERATION FAILED")
            manifest.append({"label": label, "step": step, "slug": slug,
                             "prompt": prompt, "wav_path": "", "sr": 0})
            _persist()
            continue
        sf.write(str(wav_path), wav_np, sr)
        print(f"  saved → {wav_path.name}  ({len(wav_np)/sr:.1f}s)")
        manifest.append({"label": label, "step": step, "slug": slug,
                         "prompt": prompt, "wav_path": str(wav_path), "sr": int(sr)})
        _persist()

    # Drop F5-TTS before phase 2
    del tts
    gc.collect()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return manifest


def phase2_score(manifest: list[dict], out_dir: Path) -> list[dict]:
    """Phase 2: Load Whisper + ECAPA + DNSMOS, score all WAVs in manifest.

    F5-TTS is fully unloaded at this point — no MPS pressure.
    """
    print("\n=== PHASE 2: Scoring (Whisper + ECAPA + DNSMOS) ===")
    whisper_model = load_whisper(device="cpu")
    ecapa = load_ecapa(device="cpu")
    dnsmos_session = load_dnsmos()

    # Single-ref embedding (gen-vs-the-12s-narrator-clip)
    ref_wav, ref_sr = sf.read(str(REF_AUDIO))
    if ref_wav.ndim > 1:
        ref_wav = ref_wav.mean(axis=1)
    ref_wav = ref_wav.astype(np.float32)
    ecapa_ref_emb = compute_ecapa_emb(ref_wav, ref_sr, ecapa)
    print(f"  ECAPA ref embedding: dim={ecapa_ref_emb.shape[0]}, norm={np.linalg.norm(ecapa_ref_emb):.3f}")

    # Optional multi-ref centroid (gen-vs-'Cumberbatch identity'). More robust
    # speaker target than a single 12s clip.
    centroid_emb = None
    if _CENTROID_DIR is not None:
        centroid_emb = compute_ref_centroid(_CENTROID_DIR, _CENTROID_N, ecapa, _SEED)

    NAN_ROW = {"wer": np.nan, "ecapa_sim": np.nan, "ecapa_centroid_sim": np.nan,
               "dnsmos_sig": np.nan, "dnsmos_bak": np.nan, "dnsmos_ovr": np.nan,
               "transcript": ""}

    per_clip = []  # raw per-(config, phrase) rows for the detailed CSV
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

        # Each metric computed independently — one failure doesn't wipe the others
        wer, hyp = (np.nan, "")
        try:
            wer, hyp = compute_wer(entry["prompt"], wav_np, sr, whisper_model)
        except Exception as e:
            print(f"    WER failed: {e}")

        ecapa_sim = np.nan
        ecapa_centroid_sim = np.nan
        try:
            gen_emb = compute_ecapa_emb(wav_np, sr, ecapa)
            ecapa_sim = cosine(ecapa_ref_emb, gen_emb)
            if centroid_emb is not None:
                ecapa_centroid_sim = cosine(centroid_emb, gen_emb)
        except Exception as e:
            print(f"    ECAPA failed: {e}")

        mos = {"sig": np.nan, "bak": np.nan, "ovr": np.nan}
        try:
            mos = compute_dnsmos(wav_np, sr, dnsmos_session)
        except Exception as e:
            print(f"    DNSMOS failed: {e}")

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

    # Aggregate per label
    by_label = {}
    for row in per_clip:
        by_label.setdefault((row["label"], row["step"]), []).append(row)
    agg_rows = []
    for (label, step), rows in by_label.items():
        a = aggregate(rows)
        a.update({"label": label, "step": step,
                  "listen_paths": ";".join(r["wav_path"] for r in rows if r["wav_path"])})
        agg_rows.append(a)
    agg_rows.sort(key=lambda r: r["step"])

    # Annotate rows with source (if --phrases-csv had a source column)
    if PHRASE_SOURCES:
        for row in per_clip:
            row["source"] = PHRASE_SOURCES.get(row.get("slug", ""), "")

    # Write CSVs
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="finetune_casanova")
    parser.add_argument("--listen-checkpoints", nargs="+",
                        default=["step_000500.pt"])
    parser.add_argument("--out-dir", default="tts_output/posthoc")
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--only-slug", default=None,
                        help="Only generate/score this phrase slug (e.g. stella_short)")
    parser.add_argument("--score-only", action="store_true",
                        help="Skip phase 1; score existing WAVs in out-dir using manifest.json")
    parser.add_argument("--cfg-strength", type=float, default=2.0,
                        help="F5-TTS classifier-free guidance strength (default 2.0)")
    parser.add_argument("--nfe-step", type=int, default=32,
                        help="F5-TTS ODE solver steps (default 32)")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Skip checkpoint generation; score baseline only")
    parser.add_argument("--seed", type=int, default=42,
                        help="F5-TTS inference seed — fixed for reproducibility across runs")
    parser.add_argument("--first-n", type=int, default=None,
                        help="Run only the first N phrases (subset early-stop)")
    parser.add_argument("--centroid-dir", default=None,
                        help="Dir of Cumberbatch WAVs to embed for 'identity centroid' ECAPA target "
                             "(e.g. data/cumberbatch_casanova). Defaults to single-ref ECAPA only.")
    parser.add_argument("--centroid-n", type=int, default=10,
                        help="Number of clips to average into the identity centroid (default 10)")
    parser.add_argument("--speed-fix", action="store_true",
                        help="Apply speed=0.3 for segments < 10 bytes (F5-TTS Issue #1155 short-text fix)")
    parser.add_argument("--selective-cfg", action="store_true",
                        help="Enable selective CFG (arXiv 2509.19668): standard CFG for t<=threshold, "
                             "text-conditioned CFG thereafter to amplify speaker identity")
    parser.add_argument("--t-threshold", type=float, default=0.08,
                        help="Timestep threshold for selective CFG (default 0.08, ~first 9 steps with sway sampling)")
    parser.add_argument("--phrases-csv", default=None,
                        help="Override built-in 6 eval phrases with a CSV (slug,prompt[,source]). "
                             "Optional source column groups detail rows for cross-register comparison.")
    args = parser.parse_args()

    # Free MPS memory by killing any stale F5-TTS / eval procs from prior runs.
    # CLAUDE.md notes that 3 stale F5-TTS procs ≈ 21 GB → OOM on 18 GB M3 Pro.
    n_killed = kill_stale_python()
    if n_killed:
        print(f"Pre-launch: killed {n_killed} stale F5-TTS/eval process(es)")

    # Set module-level knobs before phase1_generate / phase2_score use them
    global _CFG_STRENGTH, _NFE_STEP, _SEED, _FIRST_N, _CENTROID_DIR, _CENTROID_N, _SPEED_FIX, _SELECTIVE_CFG
    _CFG_STRENGTH = args.cfg_strength
    _NFE_STEP = args.nfe_step
    _SEED = args.seed
    _FIRST_N = args.first_n
    _CENTROID_DIR = (PROJECT_ROOT / args.centroid_dir).resolve() if args.centroid_dir else None
    _CENTROID_N = args.centroid_n
    _SPEED_FIX = args.speed_fix
    _SELECTIVE_CFG = args.t_threshold if args.selective_cfg else None
    print(f"Inference knobs: cfg_strength={_CFG_STRENGTH}, nfe_step={_NFE_STEP}, seed={_SEED}, "
          f"speed_fix={_SPEED_FIX}, selective_cfg={_SELECTIVE_CFG}")
    if _FIRST_N is not None:
        print(f"Subset mode: first-{_FIRST_N} phrases only")
    if _CENTROID_DIR is not None:
        print(f"Identity centroid: {_CENTROID_N} clips from {_CENTROID_DIR}")

    run_dir = PROJECT_ROOT / "runs" / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    listen_ckpts = []
    if not args.baseline_only:
        for name in args.listen_checkpoints:
            p = ckpt_dir / name
            if not p.exists():
                print(f"  WARN: {p} missing — skipping")
                continue
            listen_ckpts.append(p)
        if not listen_ckpts:
            print("No checkpoints to eval — aborting.")
            return
    else:
        if args.skip_baseline:
            print("--baseline-only and --skip-baseline are mutually exclusive — aborting.")
            return
        print("Baseline-only mode: skipping checkpoint generation")

    # Filter phrases if requested
    global EVAL_PHRASES
    if args.phrases_csv:
        loaded = []
        global PHRASE_SOURCES
        PHRASE_SOURCES = {}
        with open(args.phrases_csv) as f:
            for row in csv.DictReader(f):
                loaded.append((row["slug"], row["prompt"]))
                if "source" in row and row["source"]:
                    PHRASE_SOURCES[row["slug"]] = row["source"]
        EVAL_PHRASES = loaded
        print(f"Loaded {len(EVAL_PHRASES)} phrases from {args.phrases_csv}")
        if PHRASE_SOURCES:
            from collections import Counter
            counts = Counter(PHRASE_SOURCES.values())
            print(f"  by source: {dict(counts)}")
    if args.only_slug:
        EVAL_PHRASES = [(s, p) for s, p in EVAL_PHRASES if s == args.only_slug]
        if not EVAL_PHRASES:
            print(f"No phrase with slug '{args.only_slug}' — available: {[s for s,_ in EVAL_PHRASES]}")
            return
        print(f"Running only phrase: {args.only_slug}")
    if _FIRST_N is not None:
        EVAL_PHRASES = EVAL_PHRASES[:_FIRST_N]
        print(f"Running first {len(EVAL_PHRASES)} phrases: {[s for s,_ in EVAL_PHRASES]}")

    manifest_path = out_dir / "manifest.json"
    if args.score_only and manifest_path.exists():
        import json
        manifest = json.loads(manifest_path.read_text())
        print(f"Resuming from manifest: {len(manifest)} clips")
    else:
        manifest = phase1_generate(listen_ckpts, out_dir, args.device, args.skip_baseline)
        import json
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"\n✓ Phase 1 done. Manifest → {manifest_path}")

    phase2_score(manifest, out_dir)


if __name__ == "__main__":
    main()
