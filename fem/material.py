"""material.py — isotropic linear-elastic constitutive matrix (Voigt form).

Data/convention contract (must match the B-matrix in fem/elements/common.py):

  Units      : consistent mm-N-MPa.  E in MPa, stresses in MPa, K in N/mm.
  Strain     : small (engineering) strain, 6-component Voigt vector.
  Voigt order: [xx, yy, zz, xy, yz, zx]   (this exact order everywhere).
  Shear      : ENGINEERING shear convention, i.e.
                   gamma_xy = 2 * eps_xy,
                   gamma_yz = 2 * eps_yz,
                   gamma_zx = 2 * eps_zx.

With the engineering-shear convention the strain-energy density is the plain
quadratic form  w = 1/2 * eps_voigt^T C eps_voigt, with NO factors of 2 on the
shear terms (they are absorbed by gamma = 2*eps).  The B-matrix is built to emit
gamma (not eps) on the shear rows so that B @ u_e is directly this Voigt vector.

C is symmetric positive-definite for 0 <= nu < 0.5 and E > 0.
"""

from __future__ import annotations

import numpy as np

# Voigt component order used throughout the engine.  Documented here so any
# reader can confirm the constitutive matrix and the B-matrix agree.
VOIGT_ORDER: tuple[str, ...] = ("xx", "yy", "zz", "xy", "yz", "zx")


def elastic_C(E: float, nu: float) -> np.ndarray:
    """Isotropic linear-elastic constitutive matrix in engineering-shear Voigt.

    Args:
        E:  Young's modulus (MPa for mm-N-MPa units). Must be > 0.
        nu: Poisson's ratio. Must satisfy -1 < nu < 0.5 (strictly, for SPD C).

    Returns:
        (6, 6) ndarray relating stress = C @ strain with both vectors ordered
        [xx, yy, zz, xy, yz, zx] and strain using engineering shear (gamma).
    """
    assert E > 0.0, f"Young's modulus must be positive, got E={E}"
    assert -1.0 < nu < 0.5, f"Poisson's ratio out of range (-1, 0.5): nu={nu}"

    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))  # Lame lambda
    mu = E / (2.0 * (1.0 + nu))                       # Lame mu (== shear modulus G)

    C = np.zeros((6, 6), dtype=np.float64)
    # Normal block (xx, yy, zz).
    C[0, 0] = C[1, 1] = C[2, 2] = lam + 2.0 * mu
    C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = lam
    # Shear block: with engineering shear (gamma), tau = mu * gamma, so the
    # shear diagonal entries are exactly mu (NOT 2*mu).
    C[3, 3] = C[4, 4] = C[5, 5] = mu

    return C
