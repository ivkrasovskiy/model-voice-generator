from __future__ import annotations

import math

import numpy as np

from accent_coach.models import SentenceAnalysis, VowelDiagnostic, VowelFeatures

_F1_THRESHOLD = 50.0   # Hz
_F2_THRESHOLD = 100.0  # Hz


def _f1_advice(delta_f1: float) -> str:
    if delta_f1 > _F1_THRESHOLD:
        return "tongue too low; raise toward palate"
    if delta_f1 < -_F1_THRESHOLD:
        return "tongue too high; lower jaw slightly"
    return ""


def _f2_advice(delta_f2: float) -> str:
    if delta_f2 > _F2_THRESHOLD:
        return "tongue too far forward; retract"
    if delta_f2 < -_F2_THRESHOLD:
        return "tongue too far back; advance"
    return ""


def _combine(f1_msg: str, f2_msg: str) -> str:
    parts = [m for m in (f1_msg, f2_msg) if m]
    if not parts:
        return "formant placement is close to target"
    return "; ".join(parts).capitalize() + "."


def build_vowel_diagnostics(
    user_vowels: list[VowelFeatures],
    reference_norms: dict[str, tuple[float, float]],
    target: SentenceAnalysis | None = None,
) -> list[VowelDiagnostic]:
    # Build per-phoneme target means
    if target is not None:
        tgt: dict[str, tuple[float, float]] = {}
        by_phoneme: dict[str, list[VowelFeatures]] = {}
        for v in target.vowels:
            by_phoneme.setdefault(v.phoneme.phoneme, []).append(v)
        for ph, vlist in by_phoneme.items():
            tgt[ph] = (
                float(np.mean([v.f1 for v in vlist])),
                float(np.mean([v.f2 for v in vlist])),
            )
        norms = tgt
    else:
        norms = reference_norms

    # Group user vowels by phoneme
    user_by_phoneme: dict[str, list[VowelFeatures]] = {}
    for v in user_vowels:
        user_by_phoneme.setdefault(v.phoneme.phoneme, []).append(v)

    diagnostics: list[VowelDiagnostic] = []
    for phoneme, vlist in user_by_phoneme.items():
        ref = norms.get(phoneme)
        if ref is None:
            continue
        ref_f1, ref_f2 = ref
        user_f1 = float(np.mean([v.f1 for v in vlist]))
        user_f2 = float(np.mean([v.f2 for v in vlist]))
        delta_f1 = user_f1 - ref_f1
        delta_f2 = user_f2 - ref_f2
        magnitude = math.sqrt((delta_f1 / 500) ** 2 + (delta_f2 / 1000) ** 2)
        advice = _combine(_f1_advice(delta_f1), _f2_advice(delta_f2))
        diagnostics.append(
            VowelDiagnostic(
                phoneme=phoneme,
                target_f1=ref_f1,
                target_f2=ref_f2,
                user_f1=user_f1,
                user_f2=user_f2,
                deviation_magnitude=magnitude,
                articulatory_advice=advice,
            )
        )
    return diagnostics
