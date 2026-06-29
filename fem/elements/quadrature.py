"""quadrature.py — hardcoded Gauss rules for the three quadratic families.

Each rule is exposed as a list of (natural_coords_tuple, weight). The natural
coordinate convention is element-specific and MUST match the shape-function
modules that consume the rule:

  tet10   : barycentric-like (r, s, t) with the 4th coord L4 = 1 - r - s - t.
            4-point rule, degree-of-precision 2 on the unit tetrahedron.
            Sum of weights = 1/6 = volume of the reference tet (r,s,t >= 0,
            r+s+t <= 1).
  wedge15 : (L1, L2, zeta) where (L1, L2, L3=1-L1-L2) are triangle area
            coords and zeta in [-1, 1] is the prism axis. 6-point rule =
            3-point triangle x 2-point Gauss line. Sum of weights = 1.0 =
            volume of the reference prism (unit-area? no: triangle area 1/2
            times height 2 => volume 1.0).
  hex20   : (xi, eta, zeta) each in [-1, 1]. 3x3x3 = 27-point tensor-product
            Gauss rule. Sum of weights = 8 = volume of the [-1,1]^3 cube.

These weight-sum identities are exactly what fem/tests/test_quadrature.py
checks, so they double as a self-documenting invariant.
"""

from __future__ import annotations

import math

# --------------------------------------------------------------------------- #
# tet10 : 4-point rule (degree 2), barycentric a/b parameters.                 #
# --------------------------------------------------------------------------- #
# Points sit at (a,b,b), (b,a,b), (b,b,a), (b,b,b) in (r,s,t); the 4th
# barycentric coord L4 = 1 - r - s - t takes the remaining value. Each weight
# is (1/6)/4 = 1/24 so the weights sum to the tet volume 1/6.
_TET_A = 0.58541020  # 0.5854102 (1+3/sqrt(5))/4 rounded to given precision
_TET_B = 0.13819660  # 0.1381966 (1-1/sqrt(5))/4
_TET_W = (1.0 / 6.0) / 4.0  # = 1/24

GAUSS_TET10: list[tuple[tuple[float, float, float], float]] = [
    ((_TET_A, _TET_B, _TET_B), _TET_W),
    ((_TET_B, _TET_A, _TET_B), _TET_W),
    ((_TET_B, _TET_B, _TET_A), _TET_W),
    ((_TET_B, _TET_B, _TET_B), _TET_W),
]


# --------------------------------------------------------------------------- #
# wedge15 : 6-point rule = 3-pt triangle (area coords) x 2-pt Gauss line.      #
# --------------------------------------------------------------------------- #
# 3-point triangle rule (degree 2), points at the edge midpoints in area
# coords, each weight 1/3 of the triangle area (1/2): w_tri = (1/2)/3 = 1/6.
_TRI3 = [
    ((0.5, 0.5), 1.0 / 6.0),  # (L1, L2) ; L3 = 1 - L1 - L2 = 0.0
    ((0.0, 0.5), 1.0 / 6.0),
    ((0.5, 0.0), 1.0 / 6.0),
]
# 2-point Gauss on [-1, 1]: nodes +-1/sqrt(3), weights 1 each.
_GL2 = [(-1.0 / math.sqrt(3.0), 1.0), (1.0 / math.sqrt(3.0), 1.0)]

GAUSS_WEDGE15: list[tuple[tuple[float, float, float], float]] = [
    ((l1, l2, z), w_tri * w_z)
    for (l1, l2), w_tri in _TRI3
    for z, w_z in _GL2
]


# --------------------------------------------------------------------------- #
# hex20 : 3x3x3 = 27-point tensor-product Gauss rule on [-1, 1]^3.            #
# --------------------------------------------------------------------------- #
_GL3_NODES = (-math.sqrt(3.0 / 5.0), 0.0, math.sqrt(3.0 / 5.0))
_GL3_WTS = (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0)

GAUSS_HEX20: list[tuple[tuple[float, float, float], float]] = [
    ((xi, eta, zeta), wx * wy * wz)
    for xi, wx in zip(_GL3_NODES, _GL3_WTS)
    for eta, wy in zip(_GL3_NODES, _GL3_WTS)
    for zeta, wz in zip(_GL3_NODES, _GL3_WTS)
]
