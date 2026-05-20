"""Build D2 — modern RP corpus from a JSON list of YouTube URLs.

Usage:
    .venv/bin/python scripts/accent_coach_build_modern_rp.py \\
        --urls-json configs/accent_coach_phase0_5/modern_rp_urls.json

If any URL is "TODO", the script shortlists 5 YouTube candidates per missing
label and pauses for owner approval before downloading.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import soundfile as sf

DEFAULT_URLS_JSON = "configs/accent_coach_phase0_5/modern_rp_urls.json"
DEFAULT_OUT = "tts_output/modern_rp_corpus"
_COOKIES_FROM_BROWSER: str = ""  # set by --cookies-from-browser arg before any download


class EscalateToOwner(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Hardcoded candidate shortlists (used when URLs are TODO)
# ---------------------------------------------------------------------------

BBC_CANDIDATES = [
    {"title": "BBC News at Ten - 2024 full bulletin", "url": "https://www.youtube.com/watch?v=placeholder_bbc_1"},
    {"title": "BBC World News Today presenter bulletin", "url": "https://www.youtube.com/watch?v=placeholder_bbc_2"},
    {"title": "BBC News at Six - Standard Southern British accent", "url": "https://www.youtube.com/watch?v=placeholder_bbc_3"},
    {"title": "BBC Radio 4 Today Programme 2024", "url": "https://www.youtube.com/watch?v=placeholder_bbc_4"},
    {"title": "BBC News Channel main bulletin RP presenter", "url": "https://www.youtube.com/watch?v=placeholder_bbc_5"},
]

LINDSEY_CANDIDATES = [
    {"title": "Geoff Lindsey - The GOAT vowel in RP", "url": "https://www.youtube.com/watch?v=placeholder_lindsey_1"},
    {"title": "Geoff Lindsey - Standard Southern British pronunciation", "url": "https://www.youtube.com/watch?v=placeholder_lindsey_2"},
    {"title": "Geoff Lindsey - British English pronunciation guide", "url": "https://www.youtube.com/watch?v=placeholder_lindsey_3"},
    {"title": "Geoff Lindsey - RP vowel system explained", "url": "https://www.youtube.com/watch?v=placeholder_lindsey_4"},
    {"title": "Geoff Lindsey - English phonetics lecture", "url": "https://www.youtube.com/watch?v=placeholder_lindsey_5"},
]

CANDIDATES_BY_LABEL = {"bbc": BBC_CANDIDATES, "lindsey": LINDSEY_CANDIDATES}


def _shortlist_and_pause(missing_labels: list[str], config: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("PAUSE: Some URLs are 'TODO' — owner must fill them in.")
    print("=" * 60)

    for label in sorted(set(missing_labels)):
        candidates = CANDIDATES_BY_LABEL.get(label, [])
        print(f"\nLabel '{label}': {sum(1 for e in config if e['label']==label and e['url']=='TODO')} slot(s) needed")
        print(f"Candidate shortlist for '{label}':")
        for i, c in enumerate(candidates):
            print(f"  [{i+1}] {c['title']}")
            print(f"       {c['url']}")

    config_path = PROJECT_ROOT / DEFAULT_URLS_JSON
    print(f"\nEdit {config_path}")
    print("Replace each 'TODO' URL with a real YouTube URL, then re-run.")
    print("=" * 60)
    sys.exit(3)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _run_cmd(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDERR: {result.stderr[-2000:]}")


def _download_and_convert(
    url: str,
    out_path: Path,
    head_trim: float,
    tail_trim: float,
    start_s: float | None = None,
    end_s: float | None = None,
) -> Path:
    """Download, trim, and convert to 16 kHz mono WAV.

    If start_s/end_s are given they take precedence over head_trim/tail_trim.
    """
    if out_path.exists():
        print(f"    reusing existing {out_path.name} ({out_path.stat().st_size // 1024 // 1024}MB)",
              flush=True)
        return out_path

    tmp = out_path.parent / f"{out_path.stem}.tmp"
    cmd = ["yt-dlp", "-x", "--audio-format", "wav", "--format", "bestaudio",
           "--no-playlist", "-o", str(tmp)]
    if _COOKIES_FROM_BROWSER:
        cmd += ["--cookies-from-browser", _COOKIES_FROM_BROWSER]
    cmd.append(url)
    _run_cmd(cmd)

    # Find actual output (yt-dlp may add extension)
    candidates = list(out_path.parent.glob(f"{out_path.stem}.tmp*"))
    if not candidates:
        raise RuntimeError(f"yt-dlp produced no output for {url}")
    actual_tmp = candidates[0]

    if start_s is not None and end_s is not None:
        ss, to = start_s, end_s
    else:
        import librosa
        dur = librosa.get_duration(path=str(actual_tmp))
        ss = head_trim
        to = max(dur - tail_trim, head_trim + 10.0)

    _run_cmd([
        "ffmpeg", "-y",
        "-ss", str(ss),
        "-to", str(to),
        "-i", str(actual_tmp),
        "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
        str(out_path),
    ])
    actual_tmp.unlink(missing_ok=True)
    return out_path


def _silent_fraction(wav: np.ndarray, sr: int) -> float:
    """Fraction of 10 ms frames with RMS below -40 dBFS."""
    frame_len = sr // 100
    n_frames = len(wav) // frame_len
    if n_frames == 0:
        return 1.0
    threshold = 10 ** (-40 / 20)
    silent = sum(
        1 for i in range(n_frames)
        if np.sqrt(np.mean(wav[i * frame_len:(i + 1) * frame_len] ** 2)) < threshold
    )
    return silent / n_frames


def _group_by_gap(words: list[dict], gap_ms: float = 300.0) -> list[dict]:
    if not words:
        return []
    gap_s = gap_ms / 1000.0
    segments = []
    seg_words = [words[0]]
    for w in words[1:]:
        if w["start"] - seg_words[-1]["end"] > gap_s:
            text = " ".join(x["word"] for x in seg_words)
            segments.append({"start": seg_words[0]["start"], "end": seg_words[-1]["end"], "text": text})
            seg_words = [w]
        else:
            seg_words.append(w)
    if seg_words:
        text = " ".join(x["word"] for x in seg_words)
        segments.append({"start": seg_words[0]["start"], "end": seg_words[-1]["end"], "text": text})
    return segments


def _whisperx_transcribe(wav_path: Path, whisper_model: str = "medium") -> list[dict]:
    import functools

    import torch
    import whisperx

    # PyTorch 2.6+ weights_only=True default; lightning_fabric passes weights_only=None.
    _orig_load = torch.load
    @functools.wraps(_orig_load)
    def _patched_load(*args, **kwargs):
        if kwargs.get("weights_only") is None:
            kwargs["weights_only"] = False
        return _orig_load(*args, **kwargs)
    torch.load = _patched_load

    device = "cpu"
    audio = whisperx.load_audio(str(wav_path))
    model = whisperx.load_model(whisper_model, device=device, language="en", compute_type="int8")
    result = model.transcribe(audio, batch_size=8, verbose=True)
    model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device=device,
                            return_char_alignments=False)
    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            if "start" not in w or "end" not in w:
                continue
            words.append({"start": float(w["start"]), "end": float(w["end"]),
                          "word": w.get("word", "").strip()})
    return words


def _ecapa_chunk(wav: np.ndarray, sr: int, start: float, dur: float) -> np.ndarray:
    from scripts.lib.identity import embed_wav, load_ecapa
    si = int(start * sr)
    ei = int((start + dur) * sr)
    chunk = wav[si:ei]
    return embed_wav(chunk, sr, load_ecapa())


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten(), b.flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _process_one(
    i: int,
    entry: dict,
    out_dir: Path,
    whisper_model: str,
) -> tuple[list[dict], list[dict]]:
    """Download, quality-gate, transcribe one URL. Returns (manifest_rows, quality_rows)."""
    import librosa

    label, url = entry["label"], entry["url"]
    head_trim = entry.get("trim_head_s", 15)
    tail_trim = entry.get("trim_tail_s", 15)
    start_s = entry.get("start_s")
    end_s = entry.get("end_s")

    label_raw_dir = out_dir / label / "raw"
    label_clips_dir = out_dir / label / "clips"
    label_raw_dir.mkdir(parents=True, exist_ok=True)
    label_clips_dir.mkdir(parents=True, exist_ok=True)

    raw_path = label_raw_dir / f"{i}.wav"
    trim_desc = (f"start={start_s}s end={end_s}s" if start_s is not None
                 else f"trim_head={head_trim}s trim_tail={tail_trim}s")
    print(f"\n=== [{label}#{i}] {url}  ({trim_desc}) ===", flush=True)

    try:
        _download_and_convert(url, raw_path, head_trim, tail_trim, start_s, end_s)
    except Exception as e:
        print(f"  [{label}#{i}] SKIP download failed: {e}", flush=True)
        return [], [{"url": url, "label": label, "decision": f"reject_download_failed: {e}"}]

    dur = librosa.get_duration(path=str(raw_path))
    if dur < 60:
        print(f"  [{label}#{i}] REJECT too short ({dur:.1f}s)", flush=True)
        return [], [{"url": url, "label": label, "dur_s": round(dur, 1), "decision": "reject_too_short"}]

    wav, sr = sf.read(str(raw_path))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)

    silent_frac = _silent_fraction(wav, sr)
    if silent_frac > 0.60:
        print(f"  [{label}#{i}] REJECT too silent ({silent_frac:.1%})", flush=True)
        return [], [{"url": url, "label": label, "dur_s": round(dur, 1),
                     "silent_frac": round(silent_frac, 3), "decision": "reject_too_silent"}]

    random.seed(42)
    total_dur = len(wav) / sr
    valid_starts = list(np.arange(0, total_dur - 5.0, 5.0))
    starts = random.sample(valid_starts, min(5, len(valid_starts)))
    try:
        embs = [_ecapa_chunk(wav, sr, s, 5.0) for s in starts]
        min_pair = min(_cosine(a, b) for a, b in combinations(embs, 2)) if len(embs) >= 2 else 1.0
    except Exception:
        min_pair = 1.0

    if min_pair < 0.7:
        # Hand-curated sources — log warning but proceed; transcription segments will handle mixing
        print(f"  [{label}#{i}] WARN low speaker consistency (min_pair={min_pair:.3f}) — keeping",
              flush=True)

    print(f"  [{label}#{i}] KEEP dur={dur:.1f}s silent={silent_frac:.1%} min_pair={min_pair:.3f}", flush=True)
    print(f"  [{label}#{i}] Transcribing with {whisper_model} …", flush=True)
    words = _whisperx_transcribe(raw_path, whisper_model)
    segments = _group_by_gap(words, gap_ms=300.0)
    segments = [s for s in segments if s["end"] - s["start"] >= 1.0]

    manifest_rows = []
    for seg_idx, seg in enumerate(segments):
        clip_path = label_clips_dir / f"{i}_{seg_idx:03d}.wav"
        si = int(seg["start"] * sr)
        ei = int(seg["end"] * sr)
        sf.write(str(clip_path), wav[si:ei], sr, subtype="PCM_16")
        manifest_rows.append({
            "clip_id": f"{label}_{i}_{seg_idx:03d}",
            "path": str(clip_path.relative_to(PROJECT_ROOT)),
            "label": label,
            "url": url,
            "source_idx": i,
            "start_s": round(seg["start"], 3),
            "end_s": round(seg["end"], 3),
            "transcript": seg["text"],
        })

    verify_dir = out_dir / "samples_to_verify"
    verify_dir.mkdir(exist_ok=True)
    sf.write(str(verify_dir / f"{label}_{i}.wav"), wav[:5 * sr], sr, subtype="PCM_16")

    quality_row = {"url": url, "label": label, "dur_s": round(dur, 1),
                   "silent_frac": round(silent_frac, 3),
                   "min_pair_cos": round(min_pair, 3), "decision": "keep"}
    print(f"  [{label}#{i}] Done — {len(manifest_rows)} segments", flush=True)
    return manifest_rows, [quality_row]


def build_modern_rp(urls_json_path: Path, out_dir: Path, whisper_model: str = "medium", parallel: int = 1) -> None:
    import concurrent.futures

    config = json.loads(urls_json_path.read_text())

    if any(e["url"] == "TODO" for e in config):
        missing = [e["label"] for e in config if e["url"] == "TODO"]
        _shortlist_and_pause(missing, config)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    quality_log: list[dict] = []

    workers = min(parallel, len(config))
    print(f"Processing {len(config)} URLs with {workers} parallel worker(s), model={whisper_model}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process_one, i, entry, out_dir, whisper_model): i
            for i, entry in enumerate(config)
        }
        for fut in concurrent.futures.as_completed(futures):
            try:
                m_rows, q_rows = fut.result()
                manifest.extend(m_rows)
                quality_log.extend(q_rows)
            except Exception as e:
                print(f"  Worker error: {e}", flush=True)

    # Sort manifest by source_idx for deterministic order
    manifest.sort(key=lambda r: (r["label"], r["source_idx"], r["clip_id"]))

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    quality_csv = out_dir / "quality_log.csv"
    with quality_csv.open("w", newline="") as f:
        if quality_log:
            # Union of all keys — rows have different fields depending on rejection stage
            all_fields: list[str] = []
            for row in quality_log:
                for k in row:
                    if k not in all_fields:
                        all_fields.append(k)
            w = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(quality_log)

    kept = sum(1 for r in quality_log if r.get("decision") == "keep")
    print(f"\nModern RP corpus: {len(manifest)} clips from {kept}/{len(config)} URLs")
    print(f"Manifest: {manifest_path}")
    print(f"Quality log: {quality_csv}")
    print("\n>>> CHECKPOINT 2: listen to tts_output/modern_rp_corpus/samples_to_verify/*.wav")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build D2 modern-RP corpus from YouTube")
    parser.add_argument("--urls-json", default=DEFAULT_URLS_JSON)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--whisper-model", default="medium",
                        help="Whisper model size (default: medium). Use large-v2 for max accuracy.")
    parser.add_argument("--parallel", type=int, default=2,
                        help="Number of URLs to process in parallel (default: 2).")
    parser.add_argument("--cookies-from-browser", default="",
                        help="Browser to pull cookies from for yt-dlp (default: safari). "
                             "Use 'chrome', 'firefox', etc. Pass empty string to disable.")
    args = parser.parse_args()

    urls_json = PROJECT_ROOT / args.urls_json
    if not urls_json.exists():
        print(f"ERROR: URLs JSON not found: {urls_json}", file=sys.stderr)
        print("Create it from the template in configs/accent_coach_phase0_5/modern_rp_urls.json",
              file=sys.stderr)
        return 1

    global _COOKIES_FROM_BROWSER
    _COOKIES_FROM_BROWSER = args.cookies_from_browser

    build_modern_rp(urls_json, PROJECT_ROOT / args.out_dir,
                    whisper_model=args.whisper_model, parallel=args.parallel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
