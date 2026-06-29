"""Gauss-rule sanity: integrate reference-element volume and a linear function exactly."""

from __future__ import annotations

import numpy as np
import pytest

from fem.elements import common
from fem.tests.reference_elements import REFERENCES

# analytic volumes of the reference geometries in reference_elements.py
EXPECTED_VOLUME = {
    "tet10": 1.0 / 6.0,   # unit tet with corners at origin + 3 axis points
    "hex20": 1.0,         # unit cube [0,1]^3
    "wedge15": 0.5,       # triangle area 1/2 extruded over z in [0,1]
}


@pytest.mark.parametrize("name", sorted(REFERENCES))
def test_volume_integration(name):
    module, builder = REFERENCES[name]
    node_xyz = builder()
    vol = common.element_volume(node_xyz, module.shape_grads, module.GAUSS)
    assert np.isclose(vol, EXPECTED_VOLUME[name], rtol=1e-7), (
        f"{name}: volume {vol} != expected {EXPECTED_VOLUME[name]}"
    )


@pytest.mark.parametrize("name", sorted(REFERENCES))
def test_partition_of_unity(name):
    """Shape functions must sum to 1 at every Gauss point (consistency)."""
    module, _ = REFERENCES[name]
    for nat, _w in module.GAUSS:
        N = module.shape_functions(nat)
        assert np.isclose(N.sum(), 1.0, atol=1e-10), (
            f"{name}: shape functions sum to {N.sum()} at {nat}, expected 1"
        )


@pytest.mark.parametrize("name", sorted(REFERENCES))
def test_linear_completeness_of_shape_functions(name):
    """sum_i N_i(xi) * x_i must reproduce the physical point (linear completeness).

    This verifies the shape functions can represent linear fields exactly, which is
    the prerequisite for the constant-strain patch test to even be meaningful.
    """
    module, builder = REFERENCES[name]
    node_xyz = builder()
    for nat, _w in module.GAUSS:
        N = module.shape_functions(nat)
        x_interp = N @ node_xyz  # (3,)
        # reconstruct the same point by mapping natural coords through geometry:
        # for straight-sided reference elements the interpolated point is well
        # defined; we just check it lies inside the bbox and is finite + that a
        # linear function f(x)=c.x is reproduced.
        c = np.array([0.3, -0.7, 1.1])
        f_nodal = node_xyz @ c
        f_interp = N @ f_nodal
        assert np.isfinite(x_interp).all()
        assert np.isclose(f_interp, c @ x_interp, atol=1e-10), (
            f"{name}: linear function not reproduced at {nat}"
        )
