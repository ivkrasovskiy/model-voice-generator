"""Audio generation via IndexTTS-2."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
INDEXTTS_PYTHON = PROJECT_ROOT / "vendor/index-tts/.venv/bin/python"


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


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
    n_expected = sum(1 for _ in phrases_csv.open()) - 1

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
