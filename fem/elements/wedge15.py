"""wedge15.py — 15-node quadratic prism / pentahedron (Abaqus C3D15 order).

Natural coordinates: triangle area coords (L1, L2, L3 = 1 - L1 - L2) for the
cross-section, and zeta in [-1, 1] along the prism axis. The natural-coord
triple consumed by this module is xi = (L1, L2, zeta). Reference element:
L1, L2 >= 0, L1 + L2 <= 1, zeta in [-1, 1]  => volume = (1/2) * 2 = 1.0.

Node ordering (Abaqus C3D15 — ASSUMED; the patch test must confirm it):
  Corner nodes (triangle vertices on each z-face):
        bottom triangle (zeta = -1):  0(L1=1) 1(L2=1) 2(L3=1)
        top    triangle (zeta = +1):  3(L1=1) 4(L2=1) 5(L3=1)
  Mid-edge nodes:
        bottom-triangle edges:  6: 0-1   7: 1-2    8: 2-0
        top-triangle edges:     9: 3-4  10: 4-5   11: 5-3
        vertical edges:        12: 0-3  13: 1-4   14: 2-5
This matches Abaqus C3D15. A wrong mid-edge order silently corrupts the
B-matrix; the constant-strain patch test is the guardrail that catches it.

Shape functions (standard C3D15 / quadratic-prism serendipity form). With
area coords L1, L2, L3 and axis coord zeta, using g_minus = 1-zeta,
g_plus = 1+zeta:
  bottom corner i (i=0,1,2 with area coord Li):
        N = 1/2 * Li (2 Li - 1) * g_minus  -  1/2 * Li (1 - zeta^2)
  top corner i (i=3,4,5 with area coord Li):
        N = 1/2 * Li (2 Li - 1) * g_plus   -  1/2 * Li (1 - zeta^2)
  bottom mid-edge (between bottom corners with area coords La, Lb):
        N = 2 La Lb * g_minus
  top mid-edge (between top corners with area coords La, Lb):
        N = 2 La Lb * g_plus
  vertical mid-edge (over triangle vertex with area coord Li):
        N = Li (1 - zeta^2)
The 1/2-Li(1-zeta^2) subtraction on the corner terms is the serendipity
correction that makes the vertical mid-side nodes carry the quadratic-in-zeta
behaviour while keeping a partition of unity. Verified by the patch test.
"""

from __future__ import annotations

import numpy as np

from .quadrature import GAUSS_WEDGE15 as GAUSS  # noqa: F401  (re-exported)

NNODE = 15

# Which triangle vertex (area-coord index 0/1/2) each corner node sits on, and
# whether it is the bottom (zeta=-1) or top (zeta=+1) triangle.
_CORNER_VERT = (0, 1, 2, 0, 1, 2)          # area-coord index per corner 0..5
_CORNER_TOP = (False, False, False, True, True, True)

# Bottom/top mid-edge nodes: the two triangle-vertex indices they bridge.
_BOT_EDGES = ((0, 1), (1, 2), (2, 0))      # nodes 6,7,8
_TOP_EDGES = ((0, 1), (1, 2), (2, 0))      # nodes 9,10,11
# Vertical mid-edge nodes 12,13,14 sit over triangle vertices 0,1,2.
_VERT_VERT = (0, 1, 2)


def _area_coords(xi: tuple[float, float, float]) -> tuple[np.ndarray, float]:
    """Return (L (3,) area coords, zeta) from xi = (L1, L2, zeta)."""
    l1, l2, zeta = float(xi[0]), float(xi[1]), float(xi[2])
    L = np.array([l1, l2, 1.0 - l1 - l2], dtype=np.float64)
    return L, zeta


# dL/d(L1, L2) for the three area coords (L3 = 1 - L1 - L2). Axis derivative 0.
_dL_dnat = np.array(
    [
        [1.0, 0.0],   # dL1
        [0.0, 1.0],   # dL2
        [-1.0, -1.0],  # dL3
    ],
    dtype=np.float64,
)


def shape_functions(xi: tuple[float, float, float]) -> np.ndarray:
    """Quadratic-prism shape functions N (15,) at xi = (L1, L2, zeta)."""
    L, zeta = _area_coords(xi)
    gm = 1.0 - zeta
    gp = 1.0 + zeta
    q = 1.0 - zeta * zeta  # (1 - zeta^2)
    N = np.empty(NNODE, dtype=np.float64)

    for i in range(6):  # corner nodes 0..5
        v = _CORNER_VERT[i]
        Li = L[v]
        g = gp if _CORNER_TOP[i] else gm
        N[i] = 0.5 * Li * (2.0 * Li - 1.0) * g - 0.5 * Li * q

    for k, (a, b) in enumerate(_BOT_EDGES):   # nodes 6,7,8
        N[6 + k] = 2.0 * L[a] * L[b] * gm
    for k, (a, b) in enumerate(_TOP_EDGES):   # nodes 9,10,11
        N[9 + k] = 2.0 * L[a] * L[b] * gp
    for k, v in enumerate(_VERT_VERT):        # nodes 12,13,14
        N[12 + k] = L[v] * q

    return N


def shape_grads(xi: tuple[float, float, float]) -> np.ndarray:
    """dN/d(L1, L2, zeta), shape (15, 3), at xi = (L1, L2, zeta).

    Column 0 = d/dL1, column 1 = d/dL2, column 2 = d/dzeta. Derivatives w.r.t.
    L1/L2 go through the area coords via _dL_dnat (L3 depends on both).
    """
    L, zeta = _area_coords(xi)
    gm = 1.0 - zeta
    gp = 1.0 + zeta
    q = 1.0 - zeta * zeta
    dq_dz = -2.0 * zeta
    dg_dz = {True: 1.0, False: -1.0}  # d(1+zeta)/dz = +1 ; d(1-zeta)/dz = -1

    dN = np.zeros((NNODE, 3), dtype=np.float64)

    # Corner nodes 0..5.
    for i in range(6):
        v = _CORNER_VERT[i]
        Li = L[v]
        top = _CORNER_TOP[i]
        g = gp if top else gm
        # N = 0.5*Li*(2Li-1)*g - 0.5*Li*q
        # d/dLi: 0.5*(4Li-1)*g - 0.5*q  ; then spread over (L1,L2) via _dL_dnat.
        dN_dLi = 0.5 * (4.0 * Li - 1.0) * g - 0.5 * q
        dN[i, 0] = dN_dLi * _dL_dnat[v, 0]
        dN[i, 1] = dN_dLi * _dL_dnat[v, 1]
        # d/dzeta: 0.5*Li*(2Li-1)*dg - 0.5*Li*dq
        dN[i, 2] = 0.5 * Li * (2.0 * Li - 1.0) * dg_dz[top] - 0.5 * Li * dq_dz

    # Bottom mid-edges 6,7,8: N = 2 La Lb gm.
    for k, (a, b) in enumerate(_BOT_EDGES):
        n = 6 + k
        dN[n, 0] = 2.0 * gm * (L[a] * _dL_dnat[b, 0] + L[b] * _dL_dnat[a, 0])
        dN[n, 1] = 2.0 * gm * (L[a] * _dL_dnat[b, 1] + L[b] * _dL_dnat[a, 1])
        dN[n, 2] = 2.0 * L[a] * L[b] * dg_dz[False]

    # Top mid-edges 9,10,11: N = 2 La Lb gp.
    for k, (a, b) in enumerate(_TOP_EDGES):
        n = 9 + k
        dN[n, 0] = 2.0 * gp * (L[a] * _dL_dnat[b, 0] + L[b] * _dL_dnat[a, 0])
        dN[n, 1] = 2.0 * gp * (L[a] * _dL_dnat[b, 1] + L[b] * _dL_dnat[a, 1])
        dN[n, 2] = 2.0 * L[a] * L[b] * dg_dz[True]

    # Vertical mid-edges 12,13,14: N = Li q.
    for k, v in enumerate(_VERT_VERT):
        n = 12 + k
        dN[n, 0] = q * _dL_dnat[v, 0]
        dN[n, 1] = q * _dL_dnat[v, 1]
        dN[n, 2] = L[v] * dq_dz

    return dN
