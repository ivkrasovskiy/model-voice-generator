"""VOT aspiration-detection AUDIT — bucket real tokens into TP/FN/FP/TN + save
audio slices to listen to.

The synthetic test_vot.py suite proves the *mechanism*; this proves it on real
audio AND lets you hear every decision. Ground truth comes from PHONETIC CONTEXT
on NATIVE speech (no hand-labels needed):

  should aspirate  = stressed, prevocalic, non-post-/s/ /p t k/  (filter_aspirating_stops)
  should NOT       = post-/s/ prevocalic /p t k/  (/sp st sk/ — English natives
                     produce these SHORT-LAG / unaspirated; a real negative control)

"Aspiration detected" = extract_vot returns VOT >= _DETECT_MS.

Four bucket folders (each clip = stop release + following vowel, so you can hear
the puff or its absence; filename carries the measured VOT):

  detected_should_aspirate/   TP — detected, and there should be aspiration
  missed_should_aspirate/     FN — DIDN'T detect, but there should be   <- #6 target
  detected_should_not/        FP — detected, but there should NOT be
  correct_no_aspiration/      TN — didn't detect, and there shouldn't be

Owner tokens (the L2 test subject, not ground truth) go to owner_aspirating_ctx/
so you can listen to genuine under-aspiration separately.

    .venv/bin/python scripts/tools/audit_vot_aspiration.py --n 10
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from accent_coach.models import PhonemeInstance
from accent_coach.pipeline.alignment import (
    IPA_VOWELS,
    align_audio,
    filter_aspirating_stops,
)
from accent_coach.pipeline.audio_io import load_standard_audio
from accent_coach.pipeline.vot import extract_vot
from scripts.lib.manifest import load_manifest, resolve_path

# VOT (ms) at/above which we call a token "aspiration detected". Native in-domain
# aspirated /p t k/ run ~10-22 ms through this pipeline; post-/s/ controls cluster
# near 0. 12 ms splits the two without landing on either mode. Adjustable.
_DETECT_MS = 12.0

# Audio slice saved per token: enough pre-context to hear the closure + the burst
# and the whole following vowel, so the puff (or its absence) is audible.
_SLICE_PRE_S = 0.15
_SLICE_POST_S = 0.35

NATIVE_SOURCES: list[tuple[str, str, Path]] = [
    ("rp", "fry", REPO_ROOT / "tts_output/modern_rp_corpus/fry_manifest.json"),
    ("rp", "lindsey", REPO_ROOT / "tts_output/modern_rp_corpus/lindsey_manifest.json"),
    ("rp", "bbc", REPO_ROOT / "tts_output/modern_rp_corpus/bbc_manifest.json"),
    ("rp", "real_bc", REPO_ROOT / "tts_output/real_bc_corpus/manifest.json"),
    ("genam", "genam_lecture", REPO_ROOT / "tts_output/genam_lecture_corpus/manifest.json"),
    ("genam", "genam_corpus", REPO_ROOT / "tts_output/genam_corpus/manifest.json"),
    ("genam", "vsauce", REPO_ROOT / "tts_output/genam_corpus/manifest_vsauce.json"),
]
OWNER_SOURCE = ("owner", "owner", REPO_ROOT / "tts_output/owner_cal_50/manifest.json")

_STOP_IPAS = frozenset({"p", "t", "k"})


def filter_post_s_stops(phonemes: list[PhonemeInstance]) -> list[PhonemeInstance]:
    """Prevocalic /p t k/ immediately after /s/ — the unaspirated negative control."""
    ordered = sorted(phonemes, key=lambda x: x.start_time)
    out: list[PhonemeInstance] = []
    for i, p in enumerate(ordered):
        if p.phoneme not in _STOP_IPAS:
            continue
        nxt = ordered[i + 1].phoneme if i + 1 < len(ordered) else None
        prev = ordered[i - 1].phoneme if i > 0 else None
        if nxt in IPA_VOWELS and prev == "s":
            out.append(p)
    return out


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", s)[:20] or "x"


def _save_slice(audio, sr: int, t0: float, path: Path) -> bool:
    a = audio[max(0, int((t0 - _SLICE_PRE_S) * sr)) : int((t0 + _SLICE_POST_S) * sr)]
    if len(a) < int(0.05 * sr):
        return False
    sf.write(str(path), a, sr)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="clips per source")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="tts_output/accent_coach/vot_audit")
    args = ap.parse_args()
    random.seed(args.seed)

    out_root = REPO_ROOT / args.out
    buckets = {
        "TP": out_root / "detected_should_aspirate",
        "FN": out_root / "missed_should_aspirate",
        "FP": out_root / "detected_should_not",
        "TN": out_root / "correct_no_aspiration",
        "OWN": out_root / "owner_aspirating_ctx",
    }
    for b in buckets.values():
        b.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {k: 0 for k in buckets}
    vots: dict[str, list[float]] = {"should": [], "should_not": [], "owner": []}

    def handle(label: str, stops, truth: str, audio, sr: int) -> None:
        """truth in {'should','should_not','owner'}."""
        for s in stops:
            vot = extract_vot(audio, sr, s)
            detected = vot is not None and vot >= _DETECT_MS
            if truth == "owner":
                key, vk = "OWN", "owner"
            elif truth == "should":
                key, vk = ("TP" if detected else "FN"), "should"
            else:
                key, vk = ("FP" if detected else "TN"), "should_not"
            if vot is not None:
                vots[vk].append(vot)
            vstr = f"{vot:.0f}" if vot is not None else "None"
            fname = f"{label}_{_safe(s.word)}_{s.phoneme}_vot{vstr}_{counts[key]:03d}.wav"
            if _save_slice(audio, sr, s.start_time, buckets[key] / fname):
                counts[key] += 1

    sources = NATIVE_SOURCES + [OWNER_SOURCE]
    for accent, label, manifest in sources:
        if not manifest.exists():
            print(f"[missing] {manifest}", file=sys.stderr)
            continue
        entries = load_manifest(manifest)
        entries = sorted(entries, key=lambda e: e.get("end_s", 3) - e.get("start_s", 0), reverse=True)
        sample = random.sample(entries[: max(args.n * 3, 40)], min(args.n, len(entries)))
        kept = 0
        for i, e in enumerate(sample):
            wav = resolve_path(e, manifest, REPO_ROOT)
            tr = e.get("transcript") or e.get("prompt", "")
            if wav is None or not tr:
                continue
            loaded = load_standard_audio(wav)
            if loaded is None:
                continue
            audio, sr = loaded
            try:
                phon = align_audio(wav, tr, i, accent if accent != "owner" else "rp")
            except Exception:  # noqa: BLE001 — drop unalignable clip
                continue
            if accent == "owner":
                handle(label, filter_aspirating_stops(phon), "owner", audio, sr)
            else:
                handle(label, filter_aspirating_stops(phon), "should", audio, sr)
                handle(label, filter_post_s_stops(phon), "should_not", audio, sr)
            kept += 1
        print(f"  {label:<16} clips_used={kept}")

    print(f"\n{'='*60}\n  VOT ASPIRATION-DETECTION CONFUSION MATRIX (native ground truth)\n{'='*60}")
    tp, fn, fp, tn = counts["TP"], counts["FN"], counts["FP"], counts["TN"]
    print(f"  should aspirate : detected(TP)={tp:>4}   missed(FN)={fn:>4}   "
          f"-> recall {tp/(tp+fn)*100:.0f}%" if (tp + fn) else "  should aspirate: no tokens")
    print(f"  should NOT      : false(FP)   ={fp:>4}   correct(TN)={tn:>4}   "
          f"-> false-alarm {fp/(fp+tn)*100:.0f}%" if (fp + tn) else "  should NOT: no tokens")
    print(f"  owner (listen-only, under-aspiration) : {counts['OWN']}")

    def _stat(name: str, xs: list[float]) -> None:
        if not xs:
            print(f"  {name:<12} (none)")
            return
        import numpy as np
        a = np.array(xs)
        print(f"  {name:<12} n={len(a):>3}  mean={a.mean():>5.1f}  median={np.median(a):>5.1f}  "
              f"%>= {_DETECT_MS:.0f}ms={100*np.mean(a>=_DETECT_MS):>3.0f}%  %==0={100*np.mean(a==0):>3.0f}%")

    print(f"\n  raw VOT distributions (detect threshold = {_DETECT_MS:.0f} ms):")
    _stat("should", vots["should"])
    _stat("should_not", vots["should_not"])
    _stat("owner", vots["owner"])
    print(f"\n  Listen: {out_root}")


if __name__ == "__main__":
    main()
