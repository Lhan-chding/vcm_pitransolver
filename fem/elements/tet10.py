"""tet10.py — 10-node quadratic tetrahedron (Abaqus C3D10 ordering).

Natural coordinates: barycentric (volume) coords with
    L1 = r, L2 = s, L3 = t, L4 = 1 - r - s - t,
and natural-coord triple xi = (r, s, t). The reference tet occupies
r, s, t >= 0 and r + s + t <= 1 (volume 1/6).

Node ordering (Abaqus C3D10 — ASSUMED; the patch test must confirm it):
    Corner nodes (Li = 1 at corner i):
        0 -> L1 (r=1)          1 -> L2 (s=1)
        2 -> L3 (t=1)          3 -> L4 (r=s=t=0)
    Mid-edge nodes (midpoint of the edge joining two corners):
        4 -> edge 0-1   (between corners 0 and 1)
        5 -> edge 1-2   (between corners 1 and 2)
        6 -> edge 2-0   (between corners 2 and 0)
        7 -> edge 0-3   (between corners 0 and 3)
        8 -> edge 1-3   (between corners 1 and 3)
        9 -> edge 2-3   (between corners 2 and 3)
This is the standard Abaqus C3D10 convention. If a mesh uses a different
mid-edge order the B-matrix is silently wrong and the patch test will FAIL —
that is precisely what the test is there to catch.

Shape functions:
    corner i:   N_i = L_i (2 L_i - 1)
    mid-edge ij:N_ij = 4 L_i L_j
"""

from __future__ import annotations

import numpy as np

from .quadrature import GAUSS_TET10 as GAUSS  # noqa: F401  (re-exported)

NNODE = 10

# (corner_a, corner_b) pairs defining each mid-edge node, in C3D10 order.
_MID_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),  # node 4
    (1, 2),  # node 5
    (2, 0),  # node 6
    (0, 3),  # node 7
    (1, 3),  # node 8
    (2, 3),  # node 9
)


def _bary(xi: tuple[float, float, float]) -> np.ndarray:
    """Return the 4 barycentric coords (L1, L2, L3, L4) from xi = (r, s, t)."""
    r, s, t = float(xi[0]), float(xi[1]), float(xi[2])
    return np.array([r, s, t, 1.0 - r - s - t], dtype=np.float64)


def shape_functions(xi: tuple[float, float, float]) -> np.ndarray:
    """Quadratic shape functions N (10,) at natural coords xi = (r, s, t)."""
    L = _bary(xi)
    N = np.empty(NNODE, dtype=np.float64)
    for i in range(4):  # corner nodes
        N[i] = L[i] * (2.0 * L[i] - 1.0)
    for k, (a, b) in enumerate(_MID_EDGES):  # mid-edge nodes
        N[4 + k] = 4.0 * L[a] * L[b]
    return N


def shape_grads(xi: tuple[float, float, float]) -> np.ndarray:
    """dN/d(r, s, t), shape (10, 3), at natural coords xi.

    Uses dL/d(r,s,t):
        L1=r       -> dL1 = ( 1,  0,  0)
        L2=s       -> dL2 = ( 0,  1,  0)
        L3=t       -> dL3 = ( 0,  0,  1)
        L4=1-r-s-t -> dL4 = (-1, -1, -1)
    """
    L = _bary(xi)
    dL = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, -1.0, -1.0],
        ],
        dtype=np.float64,
    )
    dN = np.empty((NNODE, 3), dtype=np.float64)
    for i in range(4):  # d/dL [L_i(2L_i-1)] = (4L_i - 1)
        dN[i, :] = (4.0 * L[i] - 1.0) * dL[i, :]
    for k, (a, b) in enumerate(_MID_EDGES):  # d/d. [4 L_a L_b]
        dN[4 + k, :] = 4.0 * (L[a] * dL[b, :] + L[b] * dL[a, :])
    return dN
