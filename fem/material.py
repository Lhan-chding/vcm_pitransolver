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

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Voigt component order used throughout the engine.  Documented here so any
# reader can confirm the constitutive matrix and the B-matrix agree.
VOIGT_ORDER: tuple[str, ...] = ("xx", "yy", "zz", "xy", "yz", "zx")

# Default material config (the spring blade material C1990 lives here).
_MATERIAL_YAML = Path(__file__).resolve().parents[1] / "config" / "material.yaml"


@dataclass(frozen=True)
class Material:
    """Isotropic linear-elastic material in mm-N-MPa units."""

    name: str
    E_MPa: float
    nu: float
    density_kg_m3: float | None = None
    tensile_yield_MPa: float | None = None

    def C(self) -> np.ndarray:
        """Constitutive matrix for this material."""
        return elastic_C(self.E_MPa, self.nu)


def load_material(name: str | None = None, yaml_path: str | Path | None = None) -> Material:
    """Load a named material from config/material.yaml (default = the file's `default`).

    Only isotropic materials are supported by elastic_C; a non-isotropic entry raises.
    """
    import yaml  # local import keeps numpy-only code paths dependency-free

    path = Path(yaml_path) if yaml_path else _MATERIAL_YAML
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    name = name or cfg["default"]
    entry = cfg["materials"].get(name)
    if entry is None:
        raise KeyError(f"material {name!r} not in {path} (have: {list(cfg['materials'])})")
    if entry.get("behavior", "isotropic") != "isotropic":
        raise ValueError(f"material {name!r} is not isotropic; elastic_C supports isotropic only")
    return Material(
        name=name,
        E_MPa=float(entry["E_MPa"]),
        nu=float(entry["nu"]),
        density_kg_m3=entry.get("density_kg_m3"),
        tensile_yield_MPa=entry.get("tensile_yield_MPa"),
    )


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
