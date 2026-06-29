"""common.py — shared FE machinery used by every quadratic element family.

All routines are generic in the number of element nodes (NNODE) and are driven
by a callable shape_grads_fn(xi) -> (NNODE, 3) plus a Gauss rule (list of
(natural_coords, weight)). This keeps tet10 / wedge15 / hex20 to just their
shape functions; the B-matrix, stiffness, energy and stress logic live here once.

Conventions (must agree with fem/material.py):
  Voigt order  : [xx, yy, zz, xy, yz, zx]
  shear        : ENGINEERING shear, gamma_xy = 2*eps_xy, etc. The B-matrix emits
                 gamma on the shear rows, so B @ u_e is the Voigt strain vector
                 that pairs with elastic_C() with no extra factors of 2.

Element DOF ordering: node-major, [ux0, uy0, uz0, ux1, uy1, uz1, ...], i.e. the
i-th node's 3 dofs occupy columns 3*i : 3*i+3 of B and rows/cols of K.

Invariants are asserted, not hoped for: detJ > 0 at every Gauss point (a
non-positive Jacobian means a flipped or mis-numbered element).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# Type alias: a shape-gradient callable maps natural coords -> (NNODE, 3).
ShapeGradsFn = Callable[[tuple[float, ...]], np.ndarray]
GaussRule = list


def jacobian(
    node_xyz: np.ndarray, dN_dnat: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray]:
    """Map natural-coord shape gradients to physical gradients at one point.

    Args:
        node_xyz: (NNODE, 3) physical coordinates of the element's nodes.
        dN_dnat:  (NNODE, 3) shape-function derivatives w.r.t. natural coords.

    Returns:
        J      : (3, 3) Jacobian, J[i, j] = d x_i / d xi_j.
        detJ   : scalar determinant (asserted > 0).
        dN_dx  : (NNODE, 3) shape-function derivatives w.r.t. physical x, y, z.
    """
    node_xyz = np.asarray(node_xyz, dtype=np.float64)
    dN_dnat = np.asarray(dN_dnat, dtype=np.float64)
    assert node_xyz.shape[0] == dN_dnat.shape[0], "node count mismatch"
    assert node_xyz.shape[1] == 3 and dN_dnat.shape[1] == 3

    # J[i, j] = sum_n x_n,i * dN_n/dxi_j  =>  J = node_xyz^T @ dN_dnat
    J = node_xyz.T @ dN_dnat  # (3, 3)
    detJ = float(np.linalg.det(J))
    assert detJ > 0.0, (
        f"non-positive Jacobian detJ={detJ:.6e}; element is flipped or the "
        "node ordering is wrong"
    )
    Jinv = np.linalg.inv(J)
    # dN/dx = dN/dxi @ J^{-1}   (chain rule), shape (NNODE, 3).
    dN_dx = dN_dnat @ Jinv
    return J, detJ, dN_dx


def B_matrix(dN_dx: np.ndarray) -> np.ndarray:
    """Strain-displacement matrix B (6, 3*NNODE) in engineering-shear Voigt.

    Rows follow the Voigt order [xx, yy, zz, xy, yz, zx]; shear rows carry the
    engineering shear (factor-of-1 on each of the two displacement-gradient
    contributions, which together give gamma = 2*eps).

    Args:
        dN_dx: (NNODE, 3) physical shape-function gradients [dN/dx, dN/dy, dN/dz].

    Returns:
        (6, 3*NNODE) B-matrix.
    """
    dN_dx = np.asarray(dN_dx, dtype=np.float64)
    nnode = dN_dx.shape[0]
    assert dN_dx.shape == (nnode, 3)

    B = np.zeros((6, 3 * nnode), dtype=np.float64)
    dNx = dN_dx[:, 0]
    dNy = dN_dx[:, 1]
    dNz = dN_dx[:, 2]

    cx = slice(0, 3 * nnode, 3)  # ux columns
    cy = slice(1, 3 * nnode, 3)  # uy columns
    cz = slice(2, 3 * nnode, 3)  # uz columns

    B[0, cx] = dNx              # eps_xx = du/dx
    B[1, cy] = dNy              # eps_yy = dv/dy
    B[2, cz] = dNz              # eps_zz = dw/dz
    B[3, cx] = dNy              # gamma_xy = du/dy + dv/dx
    B[3, cy] = dNx
    B[4, cy] = dNz              # gamma_yz = dv/dz + dw/dy
    B[4, cz] = dNy
    B[5, cx] = dNz              # gamma_zx = du/dz + dw/dx
    B[5, cz] = dNx
    return B


def element_stiffness(
    node_xyz: np.ndarray,
    C: np.ndarray,
    shape_grads_fn: ShapeGradsFn,
    gauss: GaussRule,
) -> np.ndarray:
    """Element stiffness K_e = sum_g w_g * detJ_g * B_g^T C B_g.

    Args:
        node_xyz:       (NNODE, 3) physical node coords.
        C:              (6, 6) constitutive matrix (see fem.material.elastic_C).
        shape_grads_fn: callable natural-coords -> (NNODE, 3) dN/dnatural.
        gauss:          list of (natural_coords_tuple, weight).

    Returns:
        (3*NNODE, 3*NNODE) symmetric stiffness matrix.
    """
    node_xyz = np.asarray(node_xyz, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    assert C.shape == (6, 6)
    nnode = node_xyz.shape[0]
    ndof = 3 * nnode

    Ke = np.zeros((ndof, ndof), dtype=np.float64)
    for nat, w in gauss:
        dN_dnat = shape_grads_fn(nat)
        _, detJ, dN_dx = jacobian(node_xyz, dN_dnat)
        B = B_matrix(dN_dx)
        Ke += (w * detJ) * (B.T @ C @ B)
    return Ke


def element_stresses(
    node_xyz: np.ndarray,
    u_e: np.ndarray,
    C: np.ndarray,
    shape_grads_fn: ShapeGradsFn,
    gauss: GaussRule,
) -> np.ndarray:
    """Per-Gauss-point Voigt stresses sigma = C @ (B @ u_e).

    Args:
        node_xyz:       (NNODE, 3) physical node coords.
        u_e:            (NNODE, 3) nodal displacements for this element.
        C:              (6, 6) constitutive matrix.
        shape_grads_fn: callable natural-coords -> (NNODE, 3).
        gauss:          list of (natural_coords_tuple, weight).

    Returns:
        (NGAUSS, 6) array of Voigt stresses [xx, yy, zz, xy, yz, zx] per point.
    """
    node_xyz = np.asarray(node_xyz, dtype=np.float64)
    u_e = np.asarray(u_e, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    nnode = node_xyz.shape[0]
    assert u_e.shape == (nnode, 3), f"u_e must be (NNODE,3), got {u_e.shape}"

    u_flat = u_e.reshape(-1)  # node-major (3*NNODE,)
    out = np.empty((len(gauss), 6), dtype=np.float64)
    for g, (nat, _w) in enumerate(gauss):
        dN_dnat = shape_grads_fn(nat)
        _, _detJ, dN_dx = jacobian(node_xyz, dN_dnat)
        B = B_matrix(dN_dx)
        eps = B @ u_flat            # Voigt strain (engineering shear)
        out[g, :] = C @ eps         # Voigt stress
    return out


def element_strain_energy(
    node_xyz: np.ndarray,
    u_e: np.ndarray,
    C: np.ndarray,
    shape_grads_fn: ShapeGradsFn,
    gauss: GaussRule,
) -> float:
    """Strain energy U_e = sum_g w_g detJ_g * 1/2 eps_g^T C eps_g.

    Args:
        node_xyz:       (NNODE, 3) physical node coords.
        u_e:            (NNODE, 3) nodal displacements.
        C:              (6, 6) constitutive matrix.
        shape_grads_fn: callable natural-coords -> (NNODE, 3).
        gauss:          list of (natural_coords_tuple, weight).

    Returns:
        scalar strain energy.
    """
    node_xyz = np.asarray(node_xyz, dtype=np.float64)
    u_e = np.asarray(u_e, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    nnode = node_xyz.shape[0]
    assert u_e.shape == (nnode, 3), f"u_e must be (NNODE,3), got {u_e.shape}"

    u_flat = u_e.reshape(-1)
    U = 0.0
    for nat, w in gauss:
        dN_dnat = shape_grads_fn(nat)
        _, detJ, dN_dx = jacobian(node_xyz, dN_dnat)
        B = B_matrix(dN_dx)
        eps = B @ u_flat
        U += (w * detJ) * 0.5 * float(eps @ C @ eps)
    return U


def element_volume(
    node_xyz: np.ndarray,
    shape_grads_fn: ShapeGradsFn,
    gauss: GaussRule,
) -> float:
    """Element volume sum_g w_g * detJ_g (independent of any displacement)."""
    node_xyz = np.asarray(node_xyz, dtype=np.float64)
    vol = 0.0
    for nat, w in gauss:
        dN_dnat = shape_grads_fn(nat)
        _, detJ, _dN_dx = jacobian(node_xyz, dN_dnat)
        vol += w * detJ
    return vol
