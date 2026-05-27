"""F3-based speaker-intrinsic vowel normalization methods.

All functions take raw F1, F2, F3 in Hz (scalars or numpy arrays) and
return (dim1, dim2) in the normalized coordinate space.

IMPORTANT: This module must NOT import from accent_coach.diagnostics.bark_distance,
accent_coach.diagnostics.cluster_eval, or accent_coach.comparison.vowels.
It starts from raw Hz and implements each method from first principles.
"""
from __future__ import annotations

import math

import numpy as np


def _bark(f_hz: float | np.ndarray) -> float | np.ndarray:
    """Traunmüller (1990) Bark scale: Z = 26.81·f / (1960 + f) − 0.53."""
    return (26.81 * f_hz) / (1960.0 + f_hz) - 0.53


def syrdal_gopal(
    f1_hz: float | np.ndarray,
    f2_hz: float | np.ndarray,
    f3_hz: float | np.ndarray,
) -> tuple:
    """Syrdal & Gopal (1986) Bark-difference metric.

    Returns (Z3 - Z1, Z3 - Z2) where Z_i is the Bark value of formant i.
    """
    z1 = _bark(f1_hz)
    z2 = _bark(f2_hz)
    z3 = _bark(f3_hz)
    return (z3 - z1, z3 - z2)


def nearey_intrinsic(
    f1_hz: float | np.ndarray,
    f2_hz: float | np.ndarray,
    f3_hz: float | np.ndarray,
) -> tuple:
    """Nearey (1978) CLIH intrinsic normalization.

    Per token: log_gm = mean(log(F1), log(F2), log(F3))
    Returns (log(F1) - log_gm, log(F2) - log_gm).
    """
    log_f1 = np.log(f1_hz)
    log_f2 = np.log(f2_hz)
    log_f3 = np.log(f3_hz)
    log_gm = (log_f1 + log_f2 + log_f3) / 3.0
    return (log_f1 - log_gm, log_f2 - log_gm)


def f_ratios(
    f1_hz: float | np.ndarray,
    f2_hz: float | np.ndarray,
    f3_hz: float | np.ndarray,
) -> tuple:
    """Speaker-size-invariant F-ratio normalization.

    Returns (F1 / F3, F2 / F3).
    """
    return (f1_hz / f3_hz, f2_hz / f3_hz)


# ---------------------------------------------------------------------------
# Self-test (run as script)
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Verify implementations against hand-computed reference values.

    Reference: adult male /ɑ/ at F1=730, F2=1090, F3=2440 Hz.
    Tolerance: 1% relative error.
    """
    F1, F2, F3 = 730.0, 1090.0, 2440.0

    # --- Syrdal-Gopal ---
    z1 = (26.81 * F1) / (1960 + F1) - 0.53  # 6.7457
    z2 = (26.81 * F2) / (1960 + F2) - 0.53  # 9.0515
    z3 = (26.81 * F3) / (1960 + F3) - 0.53  # 14.3372
    ref_sg = (z3 - z1, z3 - z2)

    got_sg = syrdal_gopal(F1, F2, F3)
    for got, ref, name in zip(got_sg, ref_sg, ("Z3-Z1", "Z3-Z2"), strict=False):
        err = abs(got - ref) / (abs(ref) + 1e-12)
        assert err < 0.01, f"syrdal_gopal {name}: got {got:.4f}, ref {ref:.4f}, err {err:.4%}"

    # --- Nearey intrinsic ---
    log_gm_ref = (math.log(F1) + math.log(F2) + math.log(F3)) / 3.0
    ref_ni = (math.log(F1) - log_gm_ref, math.log(F2) - log_gm_ref)

    got_ni = nearey_intrinsic(F1, F2, F3)
    for got, ref, name in zip(got_ni, ref_ni, ("logF1-lgm", "logF2-lgm"), strict=False):
        err = abs(got - ref) / (abs(ref) + 1e-12)
        assert err < 0.01, f"nearey_intrinsic {name}: got {got:.4f}, ref {ref:.4f}, err {err:.4%}"

    # --- F-ratios ---
    ref_fr = (F1 / F3, F2 / F3)
    got_fr = f_ratios(F1, F2, F3)
    for got, ref, name in zip(got_fr, ref_fr, ("F1/F3", "F2/F3"), strict=False):
        err = abs(got - ref) / (abs(ref) + 1e-12)
        assert err < 0.01, f"f_ratios {name}: got {got:.4f}, ref {ref:.4f}, err {err:.4%}"

    print("f3_normalization self-test PASSED")
    print(f"  syrdal_gopal   : {got_sg[0]:.4f}, {got_sg[1]:.4f}")
    print(f"  nearey_intr    : {got_ni[0]:.4f}, {got_ni[1]:.4f}")
    print(f"  f_ratios       : {got_fr[0]:.4f}, {got_fr[1]:.4f}")


if __name__ == "__main__":
    _self_test()
