"""postprocess.py — derive K stiffness and stresses from a solved displacement.

Two INDEPENDENT estimates of the axial stiffness K (force per unit move-face
displacement, N/mm):

  K_energy   = 2 U / delta^2
      Total elastic strain energy of the body. For a linear system under pure
      displacement control, U = 1/2 * F * delta (Clapeyron), so 2U/delta^2 = F/delta = K.

  K_reaction = (sum of move-face reaction forces along the move axis) / delta
      The direct force the move face must exert to hold the prescribed delta.

In exact linear elasticity these are EQUAL. Their agreement is the end-to-end
correctness check on assembly + BC application + the element library: if the
boundary node sets, the DOF mapping, or the constitutive matrix were wrong, the
two numbers would diverge. We report both and their relative gap.

Also computes the nodal von Mises stress field (Gauss-point stresses averaged to
nodes) for a physical sanity view and for the eventual stress-prediction target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from parse_mesh import Mesh

from fem.direct_solver import SolveResult
from fem.elements import common, hex20, tet10, wedge15

_FAMILY = {10: tet10, 15: wedge15, 20: hex20}


@dataclass(frozen=True)
class StiffnessResult:
    K_energy: float           # 2U/delta^2  (N/mm)
    K_reaction: float         # move-face axial reaction / delta  (N/mm)
    rel_gap: float            # |K_energy - K_reaction| / |K_energy|
    total_strain_energy: float  # U (N*mm = mJ)
    move_face_force: float    # axial reaction summed over move face (N)
    delta_mm: float


def total_strain_energy(mesh: Mesh, u: np.ndarray, C: np.ndarray) -> float:
    """Sum element strain energies U = sum_e 1/2 int eps^T C eps dV."""
    coords = mesh.coords
    U = 0.0
    for nnode, module in _FAMILY.items():
        conn = mesh.elements_of(nnode)
        for row in conn:
            U += common.element_strain_energy(
                coords[row], u[row], C, module.shape_grads, module.GAUSS
            )
    return float(U)


def compute_stiffness(
    mesh: Mesh, result: SolveResult, C: np.ndarray
) -> StiffnessResult:
    """Energy- and reaction-based stiffness, with their consistency gap."""
    delta = result.delta_mm
    assert delta > 0.0, "delta must be positive for a stiffness ratio"

    U = total_strain_energy(mesh, result.u, C)
    K_energy = 2.0 * U / (delta * delta)

    # Reaction along the move axis, summed over move-face DOFs only. The move-axis
    # DOF of each move node sits in constrained_dofs; pull those reactions.
    axis = result.move_axis
    # global DOFs of the move-axis component for constrained nodes that carry the
    # prescribed delta (value == delta, not the transverse 0s).
    is_move_axis = (result.constrained_dofs % 3) == axis
    is_prescribed_delta = np.isclose(result.prescribed, delta)
    sel = is_move_axis & is_prescribed_delta
    move_dofs = result.constrained_dofs[sel]
    move_face_force = float(result.reaction[move_dofs].sum())
    K_reaction = move_face_force / delta

    rel_gap = abs(K_energy - K_reaction) / max(abs(K_energy), 1e-30)
    return StiffnessResult(
        K_energy=K_energy,
        K_reaction=K_reaction,
        rel_gap=rel_gap,
        total_strain_energy=U,
        move_face_force=move_face_force,
        delta_mm=delta,
    )


def von_mises_nodal(mesh: Mesh, u: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Nodal von Mises stress field (MPa), Gauss stresses averaged to nodes.

    Each element contributes its mean Gauss-point stress to its nodes; we average
    over all elements touching a node. This is the standard, cheap recovery; it is
    smooth enough for a sanity check and as a first stress-target proxy.
    """
    coords = mesh.coords
    sigma_sum = np.zeros((mesh.num_nodes, 6), dtype=np.float64)
    count = np.zeros(mesh.num_nodes, dtype=np.float64)

    for nnode, module in _FAMILY.items():
        conn = mesh.elements_of(nnode)
        for row in conn:
            sig = common.element_stresses(
                coords[row], u[row], C, module.shape_grads, module.GAUSS
            )  # (NGAUSS, 6)
            sig_mean = sig.mean(axis=0)  # element-representative Voigt stress
            sigma_sum[row] += sig_mean
            count[row] += 1.0

    count = np.maximum(count, 1.0)  # untouched nodes (none expected) stay 0
    sigma = sigma_sum / count[:, None]  # (N, 6) Voigt [xx,yy,zz,xy,yz,zx]
    return _von_mises_from_voigt(sigma)


def _von_mises_from_voigt(sigma: np.ndarray) -> np.ndarray:
    """von Mises from Voigt stress [xx, yy, zz, xy, yz, zx] (engineering shear).

    sigma_vm = sqrt( 1/2[(sxx-syy)^2+(syy-szz)^2+(szz-sxx)^2] + 3(sxy^2+syz^2+szx^2) ).
    """
    sxx, syy, szz = sigma[:, 0], sigma[:, 1], sigma[:, 2]
    sxy, syz, szx = sigma[:, 3], sigma[:, 4], sigma[:, 5]
    dev = 0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
    shear = 3.0 * (sxy ** 2 + syz ** 2 + szx ** 2)
    return np.sqrt(dev + shear)
