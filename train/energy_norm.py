"""energy_norm.py — per-variant energy normalization for multi-geometry training.

Single-geometry training minimizes raw U directly. Across many geometries that
fails: variants differ in volume, stiffness, thickness and line width, so raw U
spans orders of magnitude and a multi-geometry loss `mean(U_i)` is dominated by
the stiffest samples — the network optimizes the few big-energy variants and
ignores the rest.

The fix is a per-variant reference energy U_ref so the normalized loss
`L_i = U_i / U_ref_i` is ~O(1) for every variant. We provide three U_ref choices
of increasing fidelity:

  * "scale"      : U_ref = 1/2 * E * V * (delta / L_char)^2
                   A dimensional estimate from the material modulus, body volume
                   V, prescribed stroke delta, and a characteristic length L_char
                   (in-plane bbox span). Needs no FEM solve — usable from geometry
                   alone, so it works for brand-new variants at deploy time.
  * "stiffness"  : U_ref = 1/2 * K_ref * delta^2
                   If a reference stiffness K_ref is known (e.g. a direct-FEM or
                   ANSYS value), this is the energy that stiffness would store.
  * "directfem"  : U_ref = U_directFEM
                   The oracle energy itself; makes L_i == 1 at the true solution.
                   Most faithful, but requires the (expensive) FEM solve per variant.

This module only COMPUTES references and records the bookkeeping GPT Pro asked to
log (raw_U, normalized_U, volume, characteristic_length, energy_scale). It does
not change single-geometry training; Stage 3 wires it into the multi-geometry loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from parse_mesh import Mesh
from fem.elements import common, hex20, tet10, wedge15

_FAMILY = {10: tet10, 15: wedge15, 20: hex20}


@dataclass(frozen=True)
class EnergyScale:
    """Per-variant normalization record (all the fields GPT Pro asked to log)."""

    method: str                  # "scale" | "stiffness" | "directfem"
    U_ref: float                 # reference energy (N*mm)
    volume_mm3: float            # body volume
    characteristic_length_mm: float  # in-plane bbox span used as L_char
    energy_scale: float          # == U_ref (the divisor); kept named for logging


def mesh_volume(mesh: Mesh) -> float:
    """Total body volume via the element quadrature (sum of element volumes)."""
    coords = mesh.coords
    V = 0.0
    for nnode, module in _FAMILY.items():
        conn = mesh.elements_of(nnode)
        for row in conn:
            V += common.element_volume(coords[row], module.shape_grads, module.GAUSS)
    return float(V)


def characteristic_length(mesh: Mesh) -> float:
    """In-plane characteristic length = the larger of the two non-thin bbox spans.

    The plate is thin in one axis; the spring spans ~11 mm in the other two. The
    deformation scale is set by that in-plane span, not the 0.06 mm thickness.
    """
    lo, hi = mesh.bbox()
    spans = np.sort(hi - lo)        # ascending; spans[0] is the thin axis
    return float(spans[-1])         # largest in-plane span


def energy_scale(
    mesh: Mesh,
    *,
    E_MPa: float,
    delta_mm: float,
    method: str = "scale",
    K_ref: float | None = None,
    U_directfem: float | None = None,
) -> EnergyScale:
    """Compute the per-variant reference energy U_ref by the chosen method."""
    V = mesh_volume(mesh)
    L = characteristic_length(mesh)

    if method == "scale":
        strain_scale = delta_mm / max(L, 1e-12)
        U_ref = 0.5 * E_MPa * V * strain_scale * strain_scale
    elif method == "stiffness":
        if K_ref is None:
            raise ValueError("method='stiffness' requires K_ref")
        U_ref = 0.5 * K_ref * delta_mm * delta_mm
    elif method == "directfem":
        if U_directfem is None:
            raise ValueError("method='directfem' requires U_directfem")
        U_ref = float(U_directfem)
    else:
        raise ValueError(f"unknown method {method!r}; use scale|stiffness|directfem")

    U_ref = max(U_ref, 1e-30)       # never divide by zero
    return EnergyScale(
        method=method,
        U_ref=U_ref,
        volume_mm3=V,
        characteristic_length_mm=L,
        energy_scale=U_ref,
    )


def normalized_loss(raw_U: float, scale: EnergyScale) -> float:
    """L = U / U_ref. ~O(1) per variant, so no variant dominates the batch."""
    return raw_U / scale.U_ref
