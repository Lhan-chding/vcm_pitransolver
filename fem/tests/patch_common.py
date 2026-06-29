"""patch_common.py — shared constant-strain patch-test driver.

A linear displacement field u(x) = A @ x + b (A a constant 3x3, b constant)
produces a CONSTANT strain eps = sym(A) = 1/2 (A + A^T) everywhere inside any
element. A correct (shape-functions / B / quadrature / Voigt) implementation
must reproduce this exactly:

  * B @ u_e at every Gauss point  == the analytic constant Voigt strain,
  * element_stresses               == C @ eps_voigt at every Gauss point,
  * element_strain_energy          == 1/2 eps^T C eps * volume,
    with volume computed independently as sum_g w_g detJ_g.

If the patch test passes, the element is consistent. If the mid-edge node
ordering were wrong, B @ u_e would NOT be constant and this would fail.
"""

from __future__ import annotations

import numpy as np

from fem.elements import common
from fem.material import elastic_C

# A small but fully general constant displacement gradient and offset.
A_GRAD = np.array(
    [
        [0.0010, -0.0007, 0.0004],
        [0.0003, 0.0009, -0.0006],
        [-0.0005, 0.0002, 0.0011],
    ],
    dtype=np.float64,
)
B_OFFSET = np.array([0.05, -0.02, 0.03], dtype=np.float64)

E_MOD = 193_000.0  # MPa (stainless-steel-ish; arbitrary, material is a param)
NU = 0.29


def voigt_strain_from_gradient(A: np.ndarray) -> np.ndarray:
    """Constant engineering-shear Voigt strain from displacement gradient A.

    eps = sym(A); Voigt order [xx, yy, zz, xy, yz, zx] with engineering shear
    gamma_ij = 2 eps_ij = A_ij + A_ji.
    """
    eps = 0.5 * (A + A.T)
    return np.array(
        [
            eps[0, 0],
            eps[1, 1],
            eps[2, 2],
            2.0 * eps[0, 1],  # gamma_xy
            2.0 * eps[1, 2],  # gamma_yz
            2.0 * eps[2, 0],  # gamma_zx
        ],
        dtype=np.float64,
    )


def linear_field_at_nodes(node_xyz: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Evaluate u(x) = A @ x + b at every node -> (NNODE, 3)."""
    return node_xyz @ A.T + b  # (NNODE,3)


def run_patch_test(module, node_xyz: np.ndarray) -> None:
    """Assert the constant-strain patch test for one element instance."""
    C = elastic_C(E_MOD, NU)
    eps_voigt = voigt_strain_from_gradient(A_GRAD)
    sigma_expected = C @ eps_voigt

    u_e = linear_field_at_nodes(node_xyz, A_GRAD, B_OFFSET)

    # 1) B @ u_e == constant analytic strain at every Gauss point.
    u_flat = u_e.reshape(-1)
    for nat, _w in module.GAUSS:
        dN_dnat = module.shape_grads(nat)
        _, detJ, dN_dx = common.jacobian(node_xyz, dN_dnat)
        assert detJ > 0.0
        B = common.B_matrix(dN_dx)
        eps_g = B @ u_flat
        assert np.allclose(eps_g, eps_voigt, atol=1e-10), (
            f"strain not constant at gauss point {nat}: {eps_g} vs {eps_voigt}"
        )

    # 2) element_stresses == C @ eps everywhere.
    stresses = common.element_stresses(node_xyz, u_e, C, module.shape_grads, module.GAUSS)
    for s in stresses:
        assert np.allclose(s, sigma_expected, atol=1e-6), (
            f"stress mismatch: {s} vs {sigma_expected}"
        )

    # 3) energy == 1/2 eps^T C eps * volume (volume from independent quadrature).
    volume = common.element_volume(node_xyz, module.shape_grads, module.GAUSS)
    U_expected = 0.5 * float(eps_voigt @ C @ eps_voigt) * volume
    U = common.element_strain_energy(node_xyz, u_e, C, module.shape_grads, module.GAUSS)
    assert np.isclose(U, U_expected, rtol=1e-8, atol=0.0), (
        f"strain energy mismatch: {U} vs {U_expected} (volume={volume})"
    )
