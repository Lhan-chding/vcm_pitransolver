"""assembly.py — global stiffness assembly for the mixed quadratic mesh.

Builds the global sparse stiffness matrix K (scipy CSR) by summing every
element's stiffness into the global DOF system. The mesh is heterogeneous
(tet10 / wedge15 / hex20), so we dispatch per family and reuse the generic
machinery in fem.elements.common.

DOF convention (must match fem.elements.common element DOF ordering):
  Each node row r owns 3 consecutive global DOFs:
      ux -> 3*r + 0,  uy -> 3*r + 1,  uz -> 3*r + 2.
  An element with node rows [r0, r1, ...] therefore maps its local DOF j
  (node-major: local node n, component c, j = 3*n + c) to global DOF
      3*rows[n] + c.

We collect (row, col, value) triplets per family in a vectorized way and hand
them to scipy.sparse.coo_matrix, which sums duplicate (row, col) entries on
conversion to CSR — exactly the assembly accumulation we want. K is symmetric
positive-semidefinite (rigid-body modes give the 6 zero eigenvalues) until
Dirichlet BCs are applied in direct_solver.py.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from parse_mesh import ELEM_TYPES, Mesh

from fem.elements import common, hex20, tet10, wedge15

# Element family modules keyed by node count, matching parse_mesh.ELEM_TYPES.
_FAMILY = {10: tet10, 15: wedge15, 20: hex20}


def num_dofs(mesh: Mesh) -> int:
    """Total scalar DOFs = 3 per node."""
    return 3 * mesh.num_nodes


def global_dofs(rows: np.ndarray) -> np.ndarray:
    """Map node row indices -> their global DOF indices, node-major.

    Args:
        rows: (..., nnode) int array of node row indices.

    Returns:
        (..., 3*nnode) int array of global DOFs [3r0, 3r0+1, 3r0+2, 3r1, ...].
    """
    rows = np.asarray(rows, dtype=np.int64)
    # (..., nnode, 3) -> flatten last two dims to (..., 3*nnode)
    base = 3 * rows[..., :, None] + np.arange(3, dtype=np.int64)
    return base.reshape(*rows.shape[:-1], 3 * rows.shape[-1])


def _family_triplets(
    mesh: Mesh, nnode: int, C: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """COO (row, col, val) triplets for every element of one family.

    Returns three flat arrays. Empty arrays if the family is absent.
    """
    conn = mesh.elements_of(nnode)  # (M, nnode) row indices
    M = conn.shape[0]
    if M == 0:
        empty_i = np.empty(0, dtype=np.int64)
        return empty_i, empty_i, np.empty(0, dtype=np.float64)

    module = _FAMILY[nnode]
    ndof_e = 3 * nnode

    # Global DOF list per element: (M, ndof_e).
    gdofs = global_dofs(conn)

    # Per element, the dense Ke is (ndof_e, ndof_e). The COO row/col indices for
    # that block are gdofs[e] broadcast: rows repeat each dof ndof_e times, cols
    # tile the dof vector ndof_e times. Build the index pattern once.
    rows_block = np.repeat(np.arange(ndof_e), ndof_e)   # (ndof_e^2,)
    cols_block = np.tile(np.arange(ndof_e), ndof_e)     # (ndof_e^2,)

    all_rows = np.empty(M * ndof_e * ndof_e, dtype=np.int64)
    all_cols = np.empty_like(all_rows)
    all_vals = np.empty(M * ndof_e * ndof_e, dtype=np.float64)

    block = ndof_e * ndof_e
    coords = mesh.coords
    for e in range(M):
        node_xyz = coords[conn[e]]
        Ke = common.element_stiffness(node_xyz, C, module.shape_grads, module.GAUSS)
        g = gdofs[e]
        s = e * block
        all_rows[s : s + block] = g[rows_block]
        all_cols[s : s + block] = g[cols_block]
        all_vals[s : s + block] = Ke.reshape(-1)
    return all_rows, all_cols, all_vals


def assemble_global_stiffness(mesh: Mesh, C: np.ndarray) -> sp.csr_matrix:
    """Assemble and return the global stiffness matrix K (CSR, ndof x ndof).

    Args:
        mesh: parsed, orientation-normalized Mesh.
        C:    (6, 6) constitutive matrix (fem.material.elastic_C / Material.C()).

    Returns:
        scipy.sparse.csr_matrix of shape (3N, 3N); symmetric, PSD before BCs.
    """
    C = np.asarray(C, dtype=np.float64)
    assert C.shape == (6, 6), f"C must be (6,6), got {C.shape}"

    rows_parts, cols_parts, vals_parts = [], [], []
    for nnode in sorted(ELEM_TYPES):  # 10, 15, 20
        r, c, v = _family_triplets(mesh, nnode, C)
        if r.size:
            rows_parts.append(r)
            cols_parts.append(c)
            vals_parts.append(v)

    ndof = num_dofs(mesh)
    if not rows_parts:
        raise ValueError("mesh has no supported elements; nothing to assemble")

    rows = np.concatenate(rows_parts)
    cols = np.concatenate(cols_parts)
    vals = np.concatenate(vals_parts)

    # coo_matrix sums duplicate (row, col) entries when converted to CSR — this
    # IS the assembly accumulation across shared element DOFs.
    K = sp.coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()
    K.eliminate_zeros()
    return K
