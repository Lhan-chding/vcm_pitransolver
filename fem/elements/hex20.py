"""hex20.py — 20-node quadratic (serendipity) hexahedron (Abaqus C3D20 order).

Natural coordinates xi = (xi, eta, zeta), each in [-1, 1]. Reference element is
the cube [-1, 1]^3 (volume 8).

Node ordering (Abaqus C3D20 — ASSUMED; the patch test must confirm it):
  Corner nodes 0..7 — the two z-faces, each traversed counter-clockwise:
        bottom (zeta = -1):  0(-,-,-) 1(+,-,-) 2(+,+,-) 3(-,+,-)
        top    (zeta = +1):  4(-,-,+) 5(+,-,+) 6(+,+,+) 7(-,+,+)
  Mid-edge nodes 8..19 (each at the midpoint of a corner-corner edge):
        bottom-face edges:   8: 0-1   9: 1-2   10: 2-3   11: 3-0
        top-face edges:     12: 4-5  13: 5-6   14: 6-7   15: 7-4
        vertical edges:     16: 0-4  17: 1-5   18: 2-6   19: 3-7
This is the standard Abaqus C3D20 convention. A wrong mid-edge order makes the
B-matrix silently wrong; the constant-strain patch test is what catches it.

Serendipity shape functions, with (xi_i, eta_i, zeta_i) the natural coords of
node i and xi0 = xi*xi_i etc.:
    corner i:   N_i = 1/8 (1+xi0)(1+eta0)(1+zeta0)(xi0+eta0+zeta0 - 2)
    mid-edge with xi_i = 0:  N_i = 1/4 (1-xi^2)(1+eta0)(1+zeta0)
    mid-edge with eta_i = 0: N_i = 1/4 (1-eta^2)(1+xi0)(1+zeta0)
    mid-edge with zeta_i = 0:N_i = 1/4 (1-zeta^2)(1+xi0)(1+eta0)
"""

from __future__ import annotations

import numpy as np

from .quadrature import GAUSS_HEX20 as GAUSS  # noqa: F401  (re-exported)

NNODE = 20

# Natural coords of the 20 nodes (Abaqus C3D20 order).  Mid-edge nodes carry a
# zero in exactly one of the three components (the edge direction).
_NODES = np.array(
    [
        # corners 0..7
        [-1.0, -1.0, -1.0],  # 0
        [+1.0, -1.0, -1.0],  # 1
        [+1.0, +1.0, -1.0],  # 2
        [-1.0, +1.0, -1.0],  # 3
        [-1.0, -1.0, +1.0],  # 4
        [+1.0, -1.0, +1.0],  # 5
        [+1.0, +1.0, +1.0],  # 6
        [-1.0, +1.0, +1.0],  # 7
        # bottom-face mid-edges 8..11
        [0.0, -1.0, -1.0],   # 8  : edge 0-1
        [+1.0, 0.0, -1.0],   # 9  : edge 1-2
        [0.0, +1.0, -1.0],   # 10 : edge 2-3
        [-1.0, 0.0, -1.0],   # 11 : edge 3-0
        # top-face mid-edges 12..15
        [0.0, -1.0, +1.0],   # 12 : edge 4-5
        [+1.0, 0.0, +1.0],   # 13 : edge 5-6
        [0.0, +1.0, +1.0],   # 14 : edge 6-7
        [-1.0, 0.0, +1.0],   # 15 : edge 7-4
        # vertical mid-edges 16..19
        [-1.0, -1.0, 0.0],   # 16 : edge 0-4
        [+1.0, -1.0, 0.0],   # 17 : edge 1-5
        [+1.0, +1.0, 0.0],   # 18 : edge 2-6
        [-1.0, +1.0, 0.0],   # 19 : edge 3-7
    ],
    dtype=np.float64,
)


def shape_functions(xi: tuple[float, float, float]) -> np.ndarray:
    """Serendipity shape functions N (20,) at xi = (xi, eta, zeta)."""
    x, y, z = float(xi[0]), float(xi[1]), float(xi[2])
    N = np.empty(NNODE, dtype=np.float64)
    for i in range(NNODE):
        xi_i, eta_i, zeta_i = _NODES[i]
        x0 = x * xi_i
        y0 = y * eta_i
        z0 = z * zeta_i
        if xi_i != 0.0 and eta_i != 0.0 and zeta_i != 0.0:  # corner
            N[i] = 0.125 * (1 + x0) * (1 + y0) * (1 + z0) * (x0 + y0 + z0 - 2.0)
        elif xi_i == 0.0:  # mid-edge along xi
            N[i] = 0.25 * (1 - x * x) * (1 + y0) * (1 + z0)
        elif eta_i == 0.0:  # mid-edge along eta
            N[i] = 0.25 * (1 - y * y) * (1 + x0) * (1 + z0)
        else:  # zeta_i == 0.0, mid-edge along zeta
            N[i] = 0.25 * (1 - z * z) * (1 + x0) * (1 + y0)
    return N


def shape_grads(xi: tuple[float, float, float]) -> np.ndarray:
    """dN/d(xi, eta, zeta), shape (20, 3), at xi = (xi, eta, zeta)."""
    x, y, z = float(xi[0]), float(xi[1]), float(xi[2])
    dN = np.empty((NNODE, 3), dtype=np.float64)
    for i in range(NNODE):
        xi_i, eta_i, zeta_i = _NODES[i]
        x0 = x * xi_i
        y0 = y * eta_i
        z0 = z * zeta_i
        if xi_i != 0.0 and eta_i != 0.0 and zeta_i != 0.0:  # corner
            # N = 1/8 (1+x0)(1+y0)(1+z0)(x0+y0+z0-2)
            f = (1 + x0) * (1 + y0) * (1 + z0)
            g = x0 + y0 + z0 - 2.0
            # d/dx: chain rule on f*g, then *xi_i for x0=x*xi_i etc.
            dN[i, 0] = 0.125 * (xi_i * (1 + y0) * (1 + z0) * g + f * xi_i)
            dN[i, 1] = 0.125 * (eta_i * (1 + x0) * (1 + z0) * g + f * eta_i)
            dN[i, 2] = 0.125 * (zeta_i * (1 + x0) * (1 + y0) * g + f * zeta_i)
        elif xi_i == 0.0:  # N = 1/4 (1-x^2)(1+y0)(1+z0)
            dN[i, 0] = 0.25 * (-2.0 * x) * (1 + y0) * (1 + z0)
            dN[i, 1] = 0.25 * (1 - x * x) * eta_i * (1 + z0)
            dN[i, 2] = 0.25 * (1 - x * x) * (1 + y0) * zeta_i
        elif eta_i == 0.0:  # N = 1/4 (1-y^2)(1+x0)(1+z0)
            dN[i, 0] = 0.25 * (1 - y * y) * xi_i * (1 + z0)
            dN[i, 1] = 0.25 * (-2.0 * y) * (1 + x0) * (1 + z0)
            dN[i, 2] = 0.25 * (1 - y * y) * (1 + x0) * zeta_i
        else:  # zeta_i == 0.0, N = 1/4 (1-z^2)(1+x0)(1+y0)
            dN[i, 0] = 0.25 * (1 - z * z) * xi_i * (1 + y0)
            dN[i, 1] = 0.25 * (1 - z * z) * (1 + x0) * eta_i
            dN[i, 2] = 0.25 * (-2.0 * z) * (1 + x0) * (1 + y0)
    return dN
