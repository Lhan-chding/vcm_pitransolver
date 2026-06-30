"""direct_solver.py — apply Dirichlet BCs and solve K u = f directly.

This is the FEM *oracle*: it produces the pseudo-label displacement field that
trains and validates the Transolver. There are no external body/traction loads;
the only forcing is the prescribed displacement on the move face (displacement
control, as required by config/physics.yaml so that min(U) == equilibrium).

Boundary conditions (from config/physics.yaml):
  fixed nodes : u = 0           (all three components)
  move nodes  : u_x = delta     (move_axis = x; voice-coil stroke, out of plane)
                u_y = u_z = 0    (move_transverse = rigid)

Solution method — DOF partitioning (the textbook reduced-system approach):
  Split DOFs into constrained (c, prescribed value) and free (f, unknown).
      [ K_ff  K_fc ] [ u_f ]   [ f_f ]
      [ K_cf  K_cc ] [ u_c ] = [ r_c ]
  With f_f = 0 (no free-DOF loads):  K_ff u_f = -K_fc u_c.
  Solve that SPD system for u_f, then the reaction at constrained DOFs is
      r_c = K_cf u_f + K_cc u_c.
  This keeps K_ff symmetric positive-definite (all rigid modes are removed by the
  supports), so scipy's sparse Cholesky-capable solver is well conditioned, and
  the reactions fall straight out — needed for the K_reaction cross-check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from parse_face_to_nodes import BoundarySets
from parse_mesh import Mesh

from fem.assembly import assemble_global_stiffness, num_dofs

# Map axis name -> component index, matching the [ux, uy, uz] DOF layout.
_AXIS = {"x": 0, "y": 1, "z": 2}

# Supported sparse-solve backends, in rough order of speed on large SPD systems.
# scipy is always available; the others are optional accelerators (faster on the
# ~430k-DOF systems if installed). "auto" picks the fastest available.
_BACKENDS = ("auto", "scipy", "pypardiso", "umfpack", "cholmod")


def _sparse_solve(Kff: sp.csc_matrix, rhs: np.ndarray, backend: str) -> tuple[np.ndarray, str]:
    """Solve Kff u = rhs with the requested backend; return (u, backend_used).

    Kff is SPD (all rigid modes removed by the supports), so a Cholesky-capable
    backend (cholmod) is ideal; pypardiso/umfpack are strong LU alternatives.
    Falls back to scipy.sparse spsolve, which is always present.
    """
    order = (("cholmod", "pypardiso", "umfpack", "scipy") if backend == "auto"
             else (backend,))
    last_err = None
    for name in order:
        try:
            if name == "scipy":
                return spla.spsolve(Kff, rhs), "scipy"
            if name == "pypardiso":
                import pypardiso  # type: ignore
                return pypardiso.spsolve(Kff, rhs), "pypardiso"
            if name == "umfpack":
                import scikits.umfpack as um  # type: ignore
                return um.spsolve(Kff.tocsc(), rhs), "umfpack"
            if name == "cholmod":
                from sksparse.cholmod import cholesky  # type: ignore
                factor = cholesky(Kff.tocsc())
                return factor(rhs), "cholmod"
        except ImportError as exc:
            last_err = exc
            continue  # backend not installed -> try the next one
    # if an explicit (non-auto) backend was requested but missing, fall back loudly
    if backend not in ("auto", "scipy"):
        import warnings
        warnings.warn(f"backend {backend!r} unavailable ({last_err}); using scipy spsolve")
    return spla.spsolve(Kff, rhs), "scipy"


@dataclass(frozen=True)
class SolveResult:
    """Full solved state plus the bookkeeping needed for post-processing."""

    u: np.ndarray                 # (N, 3) nodal displacement field (mm)
    K: sp.csr_matrix              # assembled global stiffness (pre-BC)
    reaction: np.ndarray          # (ndof,) reaction forces at constrained DOFs (N)
    constrained_dofs: np.ndarray  # (Nc,) global DOF indices held by Dirichlet BCs
    prescribed: np.ndarray        # (Nc,) prescribed values at those DOFs (mm)
    move_axis: int                # component index of the prescribed move direction
    delta_mm: float               # prescribed move-face displacement magnitude
    solver_backend: str = "scipy" # which sparse backend actually solved the system


def _build_dirichlet(
    bs: BoundarySets,
    move_axis: int,
    delta_mm: float,
    move_transverse_rigid: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (constrained_dof_indices, prescribed_values) for the BCs.

    fixed nodes pin all 3 DOFs to 0; move nodes prescribe the move-axis DOF to
    delta and (if rigid) the other two to 0.
    """
    dof: list[int] = []
    val: list[float] = []

    # Fixed: u = 0 on all components.
    for r in bs.fixed_nodes:
        for c in range(3):
            dof.append(3 * int(r) + c)
            val.append(0.0)

    # Move: prescribe move axis = delta; transverse components = 0 if rigid.
    for r in bs.move_nodes:
        for c in range(3):
            if c == move_axis:
                dof.append(3 * int(r) + c)
                val.append(delta_mm)
            elif move_transverse_rigid:
                dof.append(3 * int(r) + c)
                val.append(0.0)

    dof_arr = np.asarray(dof, dtype=np.int64)
    val_arr = np.asarray(val, dtype=np.float64)
    # A node could in principle appear in both sets; reconstruct_boundary already
    # guarantees disjoint move/fixed, so no duplicate DOFs here. Assert it anyway.
    assert len(np.unique(dof_arr)) == dof_arr.size, "duplicate constrained DOF"
    return dof_arr, val_arr


def solve_displacement(
    mesh: Mesh,
    bs: BoundarySets,
    C: np.ndarray,
    *,
    move_axis: str = "x",
    delta_mm: float = 0.005,
    move_transverse_rigid: bool = True,
    K: sp.csr_matrix | None = None,
    backend: str = "scipy",
) -> SolveResult:
    """Assemble (if needed), apply Dirichlet BCs, solve K u = f, return the field.

    Args:
        mesh:  parsed, orientation-normalized Mesh.
        bs:    reconstructed boundary sets (move / fixed node rows).
        C:     (6, 6) constitutive matrix.
        move_axis: 'x' | 'y' | 'z' prescribed-displacement direction.
        delta_mm:  prescribed move-face displacement (mm).
        move_transverse_rigid: pin the non-move components of move nodes to 0.
        K:     optionally reuse a pre-assembled global stiffness.

    Returns:
        SolveResult with the (N,3) displacement field and reaction vector.
    """
    axis = _AXIS[move_axis.lower()]
    if K is None:
        K = assemble_global_stiffness(mesh, C)

    ndof = num_dofs(mesh)
    assert K.shape == (ndof, ndof), "stiffness shape does not match mesh DOFs"

    c_dofs, c_vals = _build_dirichlet(bs, axis, delta_mm, move_transverse_rigid)

    # free DOFs = everything not constrained
    is_constrained = np.zeros(ndof, dtype=bool)
    is_constrained[c_dofs] = True
    f_dofs = np.nonzero(~is_constrained)[0]

    # Partition K. CSR slicing by fancy index is fine at this size.
    Kff = K[f_dofs][:, f_dofs].tocsc()
    Kfc = K[f_dofs][:, c_dofs]

    # K_ff u_f = -K_fc u_c   (no free-DOF external loads)
    rhs = -(Kfc @ c_vals)
    u_f, backend_used = _sparse_solve(Kff, rhs, backend)
    if not np.all(np.isfinite(u_f)):
        raise FloatingPointError(f"{backend_used} solve returned non-finite displacements")

    # Reassemble the full displacement vector.
    u_full = np.zeros(ndof, dtype=np.float64)
    u_full[f_dofs] = u_f
    u_full[c_dofs] = c_vals

    # Reaction forces: r = K u (nonzero only at constrained DOFs to balance loads).
    reaction = K @ u_full

    return SolveResult(
        u=u_full.reshape(mesh.num_nodes, 3),
        K=K,
        reaction=reaction,
        constrained_dofs=c_dofs,
        prescribed=c_vals,
        move_axis=axis,
        delta_mm=delta_mm,
        solver_backend=backend_used,
    )
