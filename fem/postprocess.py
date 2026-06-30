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

from parse_mesh import ELEM_TYPES, Mesh

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
    Works on any (..., 6) array; the last axis is the Voigt stress.
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    sxx, syy, szz = sigma[..., 0], sigma[..., 1], sigma[..., 2]
    sxy, syz, szx = sigma[..., 3], sigma[..., 4], sigma[..., 5]
    dev = 0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
    shear = 3.0 * (sxy ** 2 + syz ** 2 + szx ** 2)
    return np.sqrt(dev + shear)


# --------------------------------------------------------------------------- #
# Gauss-point / element-level stress recovery.                                 #
#                                                                              #
# Nodal averaging (von_mises_nodal) SMEARS local peaks — an R-angle hotspot is #
# diluted by its smoother neighbours. For a true max-stress the unaveraged     #
# Gauss-point stresses, or their per-element maximum, are the right quantity.  #
# These are what training/analysis should use as "max stress"; the nodal field #
# stays for smooth visualization only.                                         #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GaussStress:
    """Per-Gauss-point stresses for one element family, with element/coord index."""

    family: str                  # 'tet10' | 'wedge15' | 'hex20'
    elem_rows: np.ndarray        # (M,) element index into mesh.connectivity for this family
    sigma: np.ndarray            # (M, NGAUSS, 6) Voigt stress at each Gauss point
    von_mises: np.ndarray        # (M, NGAUSS) von Mises at each Gauss point
    gauss_xyz: np.ndarray        # (M, NGAUSS, 3) physical coords of each Gauss point


def compute_stress_at_gauss_points(mesh: Mesh, u: np.ndarray, C: np.ndarray) -> list[GaussStress]:
    """Voigt stress at every Gauss point of every element, grouped by family.

    No averaging: this is the raw quadrature-point stress the element actually
    carries — the correct basis for max-stress and hotspot location.
    """
    coords = mesh.coords
    out: list[GaussStress] = []
    for nnode, module in _FAMILY.items():
        # local element index within this family (0..M-1), and the global rows
        fam_mask = np.nonzero(mesh.elem_nnode == nnode)[0]
        conn = mesh.elements_of(nnode)
        if conn.shape[0] == 0:
            continue
        gauss = module.GAUSS
        ng = len(gauss)
        m = conn.shape[0]
        sig = np.empty((m, ng, 6), dtype=np.float64)
        gxyz = np.empty((m, ng, 3), dtype=np.float64)
        # shape functions at Gauss points (constant) for physical-coord recovery
        N_at_g = np.stack([module.shape_functions(nat) for nat, _w in gauss])  # (ng, nnode)
        for e in range(m):
            row = conn[e]
            sig[e] = common.element_stresses(
                coords[row], u[row], C, module.shape_grads, gauss
            )
            gxyz[e] = N_at_g @ coords[row]   # (ng, nnode)@(nnode,3) -> (ng,3)
        vm = _von_mises_from_voigt(sig)       # (m, ng)
        out.append(GaussStress(ELEM_TYPES[nnode], fam_mask, sig, vm, gxyz))
    return out


def compute_von_mises_at_gauss_points(mesh: Mesh, u: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Flat array of von Mises at ALL Gauss points across all families (MPa)."""
    parts = [gs.von_mises.reshape(-1) for gs in compute_stress_at_gauss_points(mesh, u, C)]
    return np.concatenate(parts) if parts else np.empty(0)


def compute_element_max_von_mises(
    mesh: Mesh, u: np.ndarray, C: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-element max von Mises over its Gauss points.

    Returns (elem_rows, vm_max) where elem_rows are global element indices into
    mesh.connectivity and vm_max is the matching per-element peak von Mises.
    """
    elem_rows_parts, vm_parts = [], []
    for gs in compute_stress_at_gauss_points(mesh, u, C):
        elem_rows_parts.append(gs.elem_rows)
        vm_parts.append(gs.von_mises.max(axis=1))  # (M,)
    if not elem_rows_parts:
        return np.empty(0, dtype=np.int64), np.empty(0)
    elem_rows = np.concatenate(elem_rows_parts)
    vm_max = np.concatenate(vm_parts)
    order = np.argsort(elem_rows)               # return sorted by element index
    return elem_rows[order], vm_max[order]


def locate_hotspot_element(mesh: Mesh, u: np.ndarray, C: np.ndarray) -> dict:
    """Find the single highest-von-Mises Gauss point and its element.

    Returns a JSON-ready dict: peak von Mises, the global element index/id, the
    family, and the physical (x,y,z) of the worst Gauss point. This is the
    max-stress location an engineer cares about — unsmeared.
    """
    best = {"von_mises_max_MPa": -1.0}
    for gs in compute_stress_at_gauss_points(mesh, u, C):
        local_e, local_g = np.unravel_index(int(np.argmax(gs.von_mises)), gs.von_mises.shape)
        peak = float(gs.von_mises[local_e, local_g])
        if peak > best["von_mises_max_MPa"]:
            global_e = int(gs.elem_rows[local_e])
            best = {
                "von_mises_max_MPa": peak,
                "family": gs.family,
                "element_index": global_e,
                "element_id": int(mesh.elem_ids[global_e]),
                "gauss_point": int(local_g),
                "location_xyz_mm": [float(x) for x in gs.gauss_xyz[local_e, local_g]],
            }
    return best
