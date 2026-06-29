"""fem.elements — quadratic element families and shared FE machinery.

Modules:
  quadrature  : hardcoded Gauss rules (natural coords + weights) per family.
  common      : jacobian / B_matrix / element_stiffness / strain energy / stress.
  tet10       : 10-node quadratic tetrahedron  (Abaqus C3D10 ordering).
  wedge15     : 15-node quadratic prism        (Abaqus C3D15 ordering).
  hex20       : 20-node quadratic hexahedron    (Abaqus C3D20 ordering).

Every element module exposes the same small interface:
    NNODE : int
    shape_functions(xi) -> (NNODE,)
    shape_grads(xi)     -> (NNODE, 3)   (derivatives w.r.t. natural coords)
    GAUSS               : list[(natural_coords_tuple, weight)]
so common.element_stiffness / element_stresses / element_strain_energy can drive
any of them generically.
"""

from __future__ import annotations
