"""Fast VOT-detector oracle graded against a HUMAN listening session.

The owner listened to the aspiration audit (2026-06-12) and labelled tokens by
ear. Those labels are the ground truth here: a detector change is only an
improvement if it moves these toward the human verdict WITHOUT regressing the
clips the human confirmed as correctly-aspirated.

The saved audit slices put the stop's char-aligned start at 0.15 s by
construction, so extract_vot can be run directly on each slice (no re-alignment)
— this is the FAST inner loop for iterating on the detector. Run after every
detector change:

    .venv/bin/python scripts/tools/vot_oracle.py

Categories (keyed by word, so stable across audit re-runs):
  ASPIRATED      human heard a clear puff, detector currently reads ~0 -> RECALL bug. want VOT >= DETECT
  UNASPIRATED    human heard NO puff, detector currently over-reads        -> PRECISION bug. want VOT <  DETECT (or None)
  KEEP_POSITIVE  human confirmed aspirated AND detector already gets it     -> must NOT regress. want VOT >= DETECT
  NO_STOP        the /p t k/ is not even in the window (bad alignment)      -> ideally None (alignment-limited; reported, not gated)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from accent_coach.models import PhonemeInstance
from accent_coach.pipeline.vot import extract_vot

_DETECT_MS = 12.0
_SLICE_STOP_T0 = 0.15  # audit slice = [t0-0.15, t0+0.35] -> stop starts at 0.15 s

# Owner ear labels, 2026-06-12 (word -> category). See module docstring.
LABELS: dict[str, str] = {
    # clear puff, detector reads ~0 today (the recall failures to fix)
    "kind": "ASPIRATED", "cognitive": "ASPIRATED", "copy": "ASPIRATED",
    "coronets": "ASPIRATED", "could": "ASPIRATED",
    # no puff, detector over-reads today (precision failures)
    "can_as": "UNASPIRATED",        # real_bc "as can do" — no aspiration, read 63
    "conflict": "UNASPIRATED",      # speaker did not aspirate
    # human-confirmed aspirated AND already detected — must stay detected
    # (cover excluded: its slice ran off the clip end — 11K, truncated post-context)
    "attack": "KEEP_POSITIVE", "case": "KEEP_POSITIVE",
    "tongue": "KEEP_POSITIVE", "can": "KEEP_POSITIVE",
    # stop not in the analysed audio (alignment miss) — ideally None
    "communication": "NO_STOP", "pretentious": "NO_STOP",
}
# Words the human flagged ambiguous/clipped/fast — reported but not graded.
_EXCLUDE = {"covering", "come", "course", "keeping", "apartment", "fantastic", "kinda"}

_FN = re.compile(r"^(.*)_([A-Za-z']+)_([ptk])_vot(-?\d+|None)_\d+\.wav$")


def _key(source: str, word: str) -> str:
    w = word.lower()
    # disambiguate the two "can"s: real_bc "as can do" (unaspirated) vs bbc "can" (TP)
    if w == "can" and source.startswith("real_bc"):
        return "can_as"
    return w


def main() -> None:
    base = REPO_ROOT / (sys.argv[1] if len(sys.argv) > 1 else "tts_output/accent_coach/vot_labeled")
    if not base.exists():
        print(f"[missing] {base} — run audit_vot_aspiration.py first", file=sys.stderr)
        sys.exit(1)

    results: dict[str, list[tuple[str, float | None, bool]]] = {}
    for wav in sorted(base.rglob("*.wav")):
        m = _FN.match(wav.name)
        if not m:
            continue
        source, word, _ph, _ = m.groups()
        key = _key(source, word)
        cat = LABELS.get(key)
        if cat is None or word.lower() in _EXCLUDE:
            continue
        audio, sr = sf.read(str(wav))
        vot = extract_vot(np.asarray(audio, dtype=np.float64), sr,
                          PhonemeInstance(phoneme=_ph, arpabet=_ph.upper(), start_time=_SLICE_STOP_T0,
                                          end_time=_SLICE_STOP_T0 + 0.05, sentence_id=1, word=word,
                                          is_stressed=True))
        det = vot is not None and vot >= _DETECT_MS
        ok = {
            "ASPIRATED": det, "KEEP_POSITIVE": det,
            "UNASPIRATED": not det, "NO_STOP": vot is None,
        }[cat]
        results.setdefault(cat, []).append((word, vot, ok))

    print(f"VOT oracle (detect >= {_DETECT_MS:.0f} ms) vs owner ear labels\n" + "=" * 56)
    score: dict[str, tuple[int, int]] = {}
    for cat in ("ASPIRATED", "KEEP_POSITIVE", "UNASPIRATED", "NO_STOP"):
        rows = results.get(cat, [])
        passed = sum(1 for _, _, ok in rows if ok)
        score[cat] = (passed, len(rows))
        print(f"\n{cat}  ({passed}/{len(rows)})")
        for word, vot, ok in rows:
            vs = f"{vot:5.1f}" if vot is not None else " None"
            print(f"   {'PASS' if ok else 'FAIL'}  {word:<14} VOT={vs}")

    asp = score.get("ASPIRATED", (0, 0))
    keep = score.get("KEEP_POSITIVE", (0, 0))
    prec = score.get("UNASPIRATED", (0, 0))
    guard = score.get("NO_STOP", (0, 0))
    print("\n" + "=" * 56)
    print(f"RECALL (ASPIRATED)     {asp[0]}/{asp[1]}")
    print(f"NO-REGRESSION (KEEP)   {keep[0]}/{keep[1]}")
    print(f"PRECISION (UNASP)      {prec[0]}/{prec[1]}")
    print(f"GUARD (NO_STOP->None)  {guard[0]}/{guard[1]}  (alignment-limited)")


if __name__ == "__main__":
    main()
