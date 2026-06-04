"""Build a GenAm corpus from YouTube lectures/monologues (Phase 0.16).

Isolates the dominant speaker (the lecturer) with an UNSUPERVISED robust ECAPA
centroid — no HF token / pyannote, no per-speaker reference needed. Handles the
noise an open lecture brings:

  - audience questions  → far from the lecturer centroid → dropped
  - applause / laughter → empty/garbage transcript + low speaker-sim → dropped
  - silence / music     → silence gate

Reuses download + transcription helpers from accent_coach_build_modern_rp.

Usage:
  .venv/bin/python scripts/accent_coach_build_genam.py \\
      --urls-json configs/accent_coach_phase0_16/genam_lecture_urls.json \\
      --out-dir tts_output/genam_lecture_corpus
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "build"))

import numpy as np
import soundfile as sf
from accent_coach_build_modern_rp import (  # noqa: E402
    _download_and_convert,
    _group_by_gap,
    _silent_fraction,
    _whisperx_transcribe,
)

from scripts.lib.identity import cosine, embed_wav, load_ecapa, robust_centroid  # noqa: E402

DEFAULT_URLS = "configs/accent_coach_phase0_16/genam_lecture_urls.json"
DEFAULT_OUT = "tts_output/genam_lecture_corpus"


def _segment_embeddings(wav: np.ndarray, sr: int, segments: list[dict]) -> np.ndarray:
    """ECAPA-embed each segment's audio."""
    ecapa = load_ecapa()
    embs = []
    for s in segments:
        seg = wav[int(s["start"] * sr):int(s["end"] * sr)]
        embs.append(embed_wav(seg, sr, ecapa))
    return np.array(embs)


def _dominant_speaker_filter(
    wav: np.ndarray, sr: int, segments: list[dict], keep_cos: float,
) -> tuple[list[dict], float]:
    """Keep only segments whose ECAPA embedding is near the robust (dominant)
    speaker centroid. Returns (kept_segments_with_sim, frac_kept)."""
    if len(segments) < 3:
        return segments, 1.0
    embs = _segment_embeddings(wav, sr, segments)
    centroid = robust_centroid(embs)
    sims = np.array([cosine(centroid, e) for e in embs])
    kept = [{**s, "spk_sim": round(float(sim), 4)}
            for s, sim in zip(segments, sims, strict=False) if sim >= keep_cos]
    return kept, len(kept) / len(segments)


def _process_source(
    i: int, entry: dict, out_dir: Path, whisper_model: str, keep_cos: float,
    min_words: int,
) -> list[dict]:
    import librosa

    speaker = entry.get("speaker", f"src{i}")
    url = entry["url"]
    start_s, end_s = entry.get("start_s"), entry.get("end_s")
    raw_dir = out_dir / speaker / "raw"
    clips_dir = out_dir / speaker / "clips"
    raw_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / f"{i}.wav"
    print(f"\n=== [{speaker}#{i}] {url}  (window {start_s}-{end_s}s) ===", flush=True)
    try:
        _download_and_convert(url, raw_path, 15, 15, start_s, end_s)
    except Exception as e:  # noqa: BLE001
        print(f"  [{speaker}] SKIP download failed: {e}", flush=True)
        return []

    dur = librosa.get_duration(path=str(raw_path))
    wav, sr = sf.read(str(raw_path))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    print(f"  [{speaker}] {dur:.0f}s, silent={_silent_fraction(wav, sr):.1%} — transcribing…", flush=True)

    words = _whisperx_transcribe(raw_path, whisper_model)
    segments = _group_by_gap(words, gap_ms=300.0)
    # First-pass noise reject: real connected speech only
    segments = [s for s in segments
                if (s["end"] - s["start"]) >= 1.0 and len(s["text"].split()) >= min_words]
    print(f"  [{speaker}] {len(segments)} speech segments after text/duration gate", flush=True)

    kept, frac = _dominant_speaker_filter(wav, sr, segments, keep_cos)
    print(f"  [{speaker}] dominant-speaker filter kept {len(kept)}/{len(segments)} "
          f"({frac:.0%}) at cos≥{keep_cos}", flush=True)

    manifest_rows = []
    for seg_idx, seg in enumerate(kept):
        clip_path = clips_dir / f"{i}_{seg_idx:03d}.wav"
        si, ei = int(seg["start"] * sr), int(seg["end"] * sr)
        sf.write(str(clip_path), wav[si:ei], sr, subtype="PCM_16")
        manifest_rows.append({
            "clip_id": f"{speaker}_{i}_{seg_idx:03d}",
            "path": str(clip_path.relative_to(PROJECT_ROOT)),
            "label": "genam",
            "speaker": speaker,
            "url": url,
            "start_s": round(seg["start"], 3),
            "end_s": round(seg["end"], 3),
            "spk_sim": seg.get("spk_sim"),
            "transcript": seg["text"],
        })

    # listen-check sample: first 8 s of kept audio
    verify_dir = out_dir / "samples_to_verify"
    verify_dir.mkdir(exist_ok=True)
    if kept:
        s0 = kept[0]
        sf.write(str(verify_dir / f"{speaker}.wav"),
                 wav[int(s0["start"] * sr):int(s0["start"] * sr) + 8 * sr], sr, subtype="PCM_16")
    print(f"  [{speaker}] Done — {len(manifest_rows)} clips", flush=True)
    return manifest_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build GenAm lecture corpus")
    parser.add_argument("--urls-json", default=DEFAULT_URLS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--whisper-model", default="medium")
    parser.add_argument("--keep-cos", type=float, default=0.45,
                        help="Min ECAPA cosine to dominant-speaker centroid (drop audience/noise)")
    parser.add_argument("--min-words", type=int, default=4,
                        help="Min words/segment (drops applause/laughter fragments)")
    args = parser.parse_args()

    urls_json = PROJECT_ROOT / args.urls_json if not Path(args.urls_json).is_absolute() else Path(args.urls_json)
    out_dir = PROJECT_ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(urls_json.read_text())
    print(f"Building GenAm corpus from {len(config)} sources → {out_dir}", flush=True)

    manifest: list[dict] = []
    for i, entry in enumerate(config):
        manifest.extend(_process_source(i, entry, out_dir, args.whisper_model,
                                        args.keep_cos, args.min_words))

    manifest.sort(key=lambda r: (r["speaker"], r["clip_id"]))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    by_spk: dict[str, int] = {}
    for r in manifest:
        by_spk[r["speaker"]] = by_spk.get(r["speaker"], 0) + 1
    print(f"\nGenAm lecture corpus: {len(manifest)} clips — {by_spk}")
    print(f"Manifest: {out_dir / 'manifest.json'}")
    print(f">>> listen-check: {out_dir / 'samples_to_verify'}/*.wav")
    return 0


if __name__ == "__main__":
    sys.exit(main())
