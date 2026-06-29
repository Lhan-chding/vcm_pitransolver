"""reference_elements.py — explicit node coordinates for well-shaped reference
elements of each quadratic family, used by the patch / rigid-body tests.

Each builder returns an (NNODE, 3) array of physical coordinates whose row order
matches the element module's assumed node ordering (corner nodes first, then
mid-edge nodes at the geometric midpoint of the corresponding edge). Mid-edge
coordinates are placed at the EXACT edge midpoints so the element is straight-
sided (constant Jacobian where applicable) and the constant-strain patch test
is exact.

Reference geometries:
  tet10   : unit tetrahedron, corners (0,0,0)(1,0,0)(0,1,0)(0,0,1) — but mapped
            to match module corner order (L1->c0, L2->c1, L3->c2, L4->c3).
  hex20   : unit cube [0,1]^3.
  wedge15 : unit triangular prism, triangle (0,0)(1,0)(0,1) extruded z in [0,1].

These deliberately differ from the *reference* natural-coordinate volumes
(tet 1/6, prism 1.0, cube 1.0... here cube volume 1.0, tet volume 1/6, prism
volume 1/2). The patch test computes volume independently from the Gauss rule,
so any consistent geometry works.
"""

from __future__ import annotations

import numpy as np

from fem.elements import hex20, tet10, wedge15


def _midpoints(corner_xyz: np.ndarray, edges) -> np.ndarray:
    return np.array(
        [0.5 * (corner_xyz[a] + corner_xyz[b]) for a, b in edges],
        dtype=np.float64,
    )


def tet10_reference() -> np.ndarray:
    """Unit tet, node rows ordered to match fem.elements.tet10.

    Corner natural coords: c0=L1(1,0,0) c1=L2(0,1,0) c2=L3(0,0,1) c3=L4(0,0,0).
    Map to physical corners (here identical to natural for a unit tet placed
    with c3 at the origin):
    """
    corners = np.array(
        [
            [1.0, 0.0, 0.0],  # c0 (L1)
            [0.0, 1.0, 0.0],  # c1 (L2)
            [0.0, 0.0, 1.0],  # c2 (L3)
            [0.0, 0.0, 0.0],  # c3 (L4)
        ],
        dtype=np.float64,
    )
    mids = _midpoints(corners, tet10._MID_EDGES)
    return np.vstack([corners, mids])


def hex20_reference() -> np.ndarray:
    """Unit cube [0,1]^3, node rows ordered to match fem.elements.hex20.

    Corner order mirrors the natural-coord layout in hex20._NODES, mapped from
    [-1,1] to [0,1] via (n+1)/2.
    """
    nat = hex20._NODES  # (20,3) in [-1,1]
    return (nat + 1.0) * 0.5  # map to [0,1]^3, mid-edges land on midpoints


def wedge15_reference() -> np.ndarray:
    """Unit triangular prism, node rows ordered to match fem.elements.wedge15.

    Triangle vertices (in x,y): v0=(0,0) v1=(1,0) v2=(0,1); z = 0 (bottom) and
    z = 1 (top). Corner order: bottom v0,v1,v2 then top v0,v1,v2 — matching the
    module's _CORNER_VERT / _CORNER_TOP.
    """
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    corners = []
    for top in (False, True):
        z = 1.0 if top else 0.0
        for v in range(3):
            corners.append([tri[v, 0], tri[v, 1], z])
    corners = np.array(corners, dtype=np.float64)  # rows 0..5

    # Mid-edge nodes in module order: bottom edges, top edges, vertical edges.
    bot = _midpoints(corners[0:3], wedge15._BOT_EDGES)      # nodes 6,7,8
    top = _midpoints(corners[3:6], wedge15._TOP_EDGES)      # nodes 9,10,11
    vert = np.array(
        [0.5 * (corners[v] + corners[3 + v]) for v in wedge15._VERT_VERT],
        dtype=np.float64,
    )  # nodes 12,13,14
    return np.vstack([corners, bot, top, vert])


# Convenience registry: name -> (module, reference-builder).
REFERENCES = {
    "tet10": (tet10, tet10_reference),
    "hex20": (hex20, hex20_reference),
    "wedge15": (wedge15, wedge15_reference),
}
