from __future__ import annotations

import math

import numpy as np

from accent_coach.models import SentenceAnalysis, VowelFeatures
from accent_coach.reference.normalize import LobanovParams, rp_lobanov_params
from accent_coach.reference.rp_norms import get_rp_norms

# Why 1.5: allows ±100 Hz F1 / ±200 Hz F2 natural variation to score ~80+ (native-speaker range).
# A +400 Hz F1 shift (strong accent) drops score to ~55, satisfying sensitivity requirement.
_SCALE = 1.5


def _score_pair(
    user_f1: float, user_f2: float,
    ref_f1: float, ref_f2: float,
    speaker_params: LobanovParams | None,
    ref_params: LobanovParams | None,
) -> float:
    if speaker_params is not None and ref_params is not None:
        # Compare in Lobanov z-score space — removes vocal-tract-length effect
        uz1, uz2 = speaker_params.normalise(user_f1, user_f2)
        rz1, rz2 = ref_params.normalise(ref_f1, ref_f2)
        d = math.sqrt((uz1 - rz1) ** 2 + (uz2 - rz2) ** 2)
    else:
        d = math.sqrt(((user_f1 - ref_f1) / 500) ** 2 + ((user_f2 - ref_f2) / 1000) ** 2)
    return 100.0 * math.exp(-d / _SCALE)


def score_vowels(
    user: SentenceAnalysis,
    reference_norms: dict[str, tuple[float, float]] | None = None,
    target: SentenceAnalysis | None = None,
    speaker_params: LobanovParams | None = None,
    ref_params: LobanovParams | None = None,
) -> float:
    """Score vowel accuracy.

    When speaker_params and ref_params are provided (Lobanov mode), distances are
    computed in z-score space, removing the vocal-tract-length effect that makes
    a low-pitched speaker (like BC TTS at ~76 Hz) look systematically off in raw Hz.
    """
    if not user.vowels:
        return 0.0

    if target is not None:
        tgt_by_phoneme: dict[str, list[VowelFeatures]] = {}
        for v in target.vowels:
            tgt_by_phoneme.setdefault(v.phoneme.phoneme, []).append(v)
        scores: list[float] = []
        for v in user.vowels:
            tgt_list = tgt_by_phoneme.get(v.phoneme.phoneme)
            if not tgt_list:
                continue
            tgt_f1 = float(np.mean([t.f1 for t in tgt_list]))
            tgt_f2 = float(np.mean([t.f2 for t in tgt_list]))
            # For user-vs-target mode, build per-target params on the fly if needed
            t_params = ref_params
            if t_params is None and speaker_params is not None:
                t_params = speaker_params  # same speaker space as fallback
            scores.append(_score_pair(v.f1, v.f2, tgt_f1, tgt_f2, speaker_params, t_params))
        return float(np.mean(scores)) if scores else 0.0

    norms = reference_norms or {}
    if not norms and user.vowels:
        mean_f0 = float(np.mean([v.pitch_mean for v in user.vowels if v.pitch_mean > 50] or [120]))
        norms = get_rp_norms(mean_f0)

    # Build ref_params from norms if Lobanov mode is requested but ref_params not supplied
    effective_ref_params = ref_params
    if speaker_params is not None and effective_ref_params is None and norms:
        effective_ref_params = rp_lobanov_params(norms)

    scores = []
    for v in user.vowels:
        ref = norms.get(v.phoneme.phoneme)
        if ref is None:
            continue
        scores.append(_score_pair(v.f1, v.f2, ref[0], ref[1], speaker_params, effective_ref_params))
    return float(np.mean(scores)) if scores else 0.0


def score_vowels_piecewise(
    user_f1f2: dict[str, tuple[float, float]],
    ref_f1f2: dict[str, tuple[float, float]],
    sigma_rp: dict[str, float],
) -> dict:
    """Piecewise-linear vowel score tied to within-RP variance (Phase 0.8).

    Replaces the exponential-decay composite for bench evaluation.
    Accepts pre-normalised (f1, f2) pairs so any Phase 0.8 normalisation
    can be applied before calling.

    Args:
        user_f1f2:  {phoneme: (f1, f2)} for the test speaker
        ref_f1f2:   {phoneme: (f1, f2)} for the modern-RP centroid
        sigma_rp:   {phoneme: sigma} — mean within-RP cluster distance,
                    computed by cluster_eval.per_phoneme_sigma_rp

    Returns:
        {"per_phoneme": {phoneme: score}, "composite": float}
    """
    per_phoneme: dict[str, float] = {}
    for phoneme, (uf1, uf2) in user_f1f2.items():
        ref = ref_f1f2.get(phoneme)
        sigma = sigma_rp.get(phoneme)
        if ref is None or sigma is None or sigma <= 0:
            continue
        d = math.sqrt((uf1 - ref[0]) ** 2 + (uf2 - ref[1]) ** 2)
        if d <= sigma:
            score = 100.0
        elif d <= 2 * sigma:
            score = 100.0 - 30.0 * (d - sigma) / sigma
        elif d <= 4 * sigma:
            score = 70.0 - 40.0 * (d - 2 * sigma) / (2 * sigma)
        else:
            score = 30.0
        per_phoneme[phoneme] = round(score, 1)
    composite = float(np.mean(list(per_phoneme.values()))) if per_phoneme else 0.0
    return {"per_phoneme": per_phoneme, "composite": round(composite, 1)}
