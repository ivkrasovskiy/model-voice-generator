"""H3 — Anchor-relative coordinate normalisation (Phase 0.8).

For each speaker, uses the corner vowels {iː, ɑː, uː} to define a
local coordinate frame:
    origin = centroid({F1_iː,F1_ɑː,F1_uː}, {F2_iː,F2_ɑː,F2_uː})
    F1_scale = F1_ɑː − F1_iː   (ɑː is lower/higher-F1 than iː)
    F2_scale = F2_iː − F2_uː   (iː is more front/higher-F2 than uː)

Every phoneme's F1/F2 is projected into this frame:
    y' = (F1 − F1_origin) / F1_scale
    x' = (F2 − F2_origin) / F2_scale

Corner vowels themselves will occupy fixed positions (by construction)
for all speakers; distances between speakers are entirely driven by
non-corner vowels.

FAIL condition (documented in Phase 0.8 plan): if corner vowels ARE the
primary accent signal, this normalisation will miss them — that failure
mode will appear as C3 not passing.
"""
from __future__ import annotations

Centroids = dict[str, dict[str, dict[str, float] | None]]


def anchor_normalize(centroids: Centroids) -> Centroids:
    """Return anchor-relative centroids.

    Speakers missing any corner vowel are left with f1=f2=NaN for all
    phonemes (no partial anchoring).
    """
    all_speakers: set[str] = set()
    for phon_dict in centroids.values():
        all_speakers.update(phon_dict.keys())

    speaker_frame: dict[str, dict[str, float] | None] = {}
    for spk in all_speakers:
        ii = (centroids.get("iː") or {}).get(spk)
        aa = (centroids.get("ɑː") or {}).get(spk)
        uu = (centroids.get("uː") or {}).get(spk)
        if not ii or not aa or not uu:
            speaker_frame[spk] = None
            continue
        f1_origin = (ii["f1"] + aa["f1"] + uu["f1"]) / 3.0
        f2_origin = (ii["f2"] + aa["f2"] + uu["f2"]) / 3.0
        f1_scale = aa["f1"] - ii["f1"]   # ɑː has higher F1 than iː
        f2_scale = ii["f2"] - uu["f2"]   # iː has higher F2 than uː
        if abs(f1_scale) < 1 or abs(f2_scale) < 1:
            speaker_frame[spk] = None
            continue
        speaker_frame[spk] = {
            "f1_origin": f1_origin,
            "f2_origin": f2_origin,
            "f1_scale": f1_scale,
            "f2_scale": f2_scale,
        }

    out: Centroids = {}
    for phoneme, speakers in centroids.items():
        out[phoneme] = {}
        for spk, vals in speakers.items():
            frame = speaker_frame.get(spk)
            if vals is None or frame is None:
                out[phoneme][spk] = None
                continue
            out[phoneme][spk] = {
                "f1": (vals["f1"] - frame["f1_origin"]) / frame["f1_scale"],
                "f2": (vals["f2"] - frame["f2_origin"]) / frame["f2_scale"],
                "n": vals.get("n", 0),
            }
    return out
