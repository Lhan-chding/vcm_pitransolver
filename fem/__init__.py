"""fem — Stage 1 finite-element engine (pure numpy/scipy, no torch).

Geometric/numerical foundation of the physics-informed structural solver.
Implements the three mixed-quadratic element families the real mesh contains
(tet10 / wedge15 / hex20), the isotropic constitutive matrix, and the shared
B-matrix / element-stiffness / strain-energy machinery they all use.

Correctness is proven by patch tests (constant-strain) and rigid-body tests,
not assumed. See fem/tests/.
"""

from __future__ import annotations
