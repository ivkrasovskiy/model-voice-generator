"""F5-TTS MPS helpers: text splitting and allocator reset."""

import gc
import re

import torch


def split_to_short_segments(text: str, max_chars: int = 50) -> list[str]:
    """Split a phrase into ≤max_chars segments, preferring sentence/clause boundaries.

    Splits hierarchically: sentence → clause (,;:) → words. Keeps every segment
    under max_chars, which is the F5-TTS single-batch ceiling for typical refs.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    segments: list[str] = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(s) <= max_chars:
            segments.append(s)
            continue
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


def mps_reset():
    """Flush MPS allocator + run Python GC.

    Call this between long-lived MPS operations:
      - per generation in inference loops (NaN prevention via fragmentation reset)
      - every N steps in training loops (high-water-mark control under KD,
        where the per-step activation footprint doubles)

    Cheap (~10 ms) but worth ~2–3 GB of working-set headroom on 18 GB Macs.
    """
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


