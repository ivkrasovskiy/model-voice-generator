"""F3-based speaker-intrinsic vowel normalization methods.

All functions take raw F1, F2, F3 in Hz (scalars or numpy arrays) and
return (dim1, dim2) in the normalized coordinate space.

IMPORTANT: This module must NOT import from accent_coach.diagnostics.bark_distance,
accent_coach.diagnostics.cluster_eval, or accent_coach.comparison.vowels.
It starts from raw Hz and implements each method from first principles.
"""
from __future__ import annotations

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


# ---------------------------------------------------------------------------
# Self-test (run as script)
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Verify syrdal_gopal against hand-computed reference values.

    Reference: adult male /ɑ/ at F1=730, F2=1090, F3=2440 Hz.
    Tolerance: 1% relative error.
    """
    F1, F2, F3 = 730.0, 1090.0, 2440.0

    z1 = (26.81 * F1) / (1960 + F1) - 0.53
    z2 = (26.81 * F2) / (1960 + F2) - 0.53
    z3 = (26.81 * F3) / (1960 + F3) - 0.53
    ref_sg = (z3 - z1, z3 - z2)

    got_sg = syrdal_gopal(F1, F2, F3)
    for got, ref, name in zip(got_sg, ref_sg, ("Z3-Z1", "Z3-Z2"), strict=False):
        err = abs(got - ref) / (abs(ref) + 1e-12)
        assert err < 0.01, f"syrdal_gopal {name}: got {got:.4f}, ref {ref:.4f}, err {err:.4%}"

    print("f3_normalization self-test PASSED")
    print(f"  syrdal_gopal   : {got_sg[0]:.4f}, {got_sg[1]:.4f}")


if __name__ == "__main__":
    _self_test()
