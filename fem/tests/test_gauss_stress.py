"""Gauss-point / element-level stress recovery (postprocess.py extensions).

Under uniform uniaxial strain on a single hex (u_x = delta*x, nu=0), the stress
field is constant: every Gauss point carries sigma_xx = E*delta (L=1), so von
Mises = E*delta everywhere. This pins the new unaveraged recovery exactly and
checks that element-max == that value and the hotspot lands inside the element.
"""

from __future__ import annotations

import numpy as np
import pytest

from parse_mesh import Mesh
from parse_face_to_nodes import BoundarySets
from fem.direct_solver import solve_displacement
from fem.material import elastic_C
from fem.postprocess import (
    compute_element_max_von_mises,
    compute_stress_at_gauss_points,
    compute_von_mises_at_gauss_points,
    locate_hotspot_element,
)
from fem.tests.reference_elements import hex20_reference

E = 127000.0
NU = 0.0
DELTA = 0.005


def _single_hex_mesh() -> Mesh:
    coords = hex20_reference()
    row = np.arange(20, dtype=np.int64)
    return Mesh(
        node_ids=np.arange(20, dtype=np.int64),
        coords=coords,
        elem_ids=np.array([42], dtype=np.int64),   # non-trivial id to test id mapping
        connectivity=[row],
        elem_nnode=np.array([20], dtype=np.int8),
        id_to_row={i: i for i in range(20)},
    )


def _x_boundary(coords):
    fixed = np.nonzero(np.isclose(coords[:, 0], 0.0))[0]
    move = np.nonzero(np.isclose(coords[:, 0], 1.0))[0]
    free = np.setdiff1d(np.arange(coords.shape[0]), np.union1d(fixed, move))
    return BoundarySets(move_nodes=move, fixed_nodes=fixed, free_nodes=free,
                        move_face_descriptor={})


@pytest.fixture
def solved():
    mesh = _single_hex_mesh()
    C = elastic_C(E, NU)
    bs = _x_boundary(mesh.coords)
    res = solve_displacement(mesh, bs, C, move_axis="x", delta_mm=DELTA,
                             move_transverse_rigid=False)
    return mesh, res.u, C


def test_gauss_stress_uniform_uniaxial(solved):
    mesh, u, C = solved
    groups = compute_stress_at_gauss_points(mesh, u, C)
    assert len(groups) == 1                       # only hex20 present
    gs = groups[0]
    assert gs.family == "hex20"
    assert gs.sigma.shape == (1, 27, 6)           # 1 element, 27 Gauss pts, Voigt
    # sigma_xx = E*delta on every Gauss point; other normal/shear ~ 0
    assert np.allclose(gs.sigma[..., 0], E * DELTA, rtol=1e-6)
    assert np.allclose(gs.sigma[..., 1:], 0.0, atol=1e-6)
    # von Mises uniform = E*delta
    assert np.allclose(gs.von_mises, E * DELTA, rtol=1e-6)
    # gauss points lie inside the unit cube
    assert (gs.gauss_xyz >= -1e-9).all() and (gs.gauss_xyz <= 1 + 1e-9).all()


def test_von_mises_gauss_flat_array(solved):
    mesh, u, C = solved
    vm = compute_von_mises_at_gauss_points(mesh, u, C)
    assert vm.shape == (27,)
    assert np.allclose(vm, E * DELTA, rtol=1e-6)


def test_element_max_von_mises(solved):
    mesh, u, C = solved
    rows, vm_max = compute_element_max_von_mises(mesh, u, C)
    assert rows.tolist() == [0]                   # single element, global index 0
    assert vm_max[0] == pytest.approx(E * DELTA, rel=1e-6)


def test_locate_hotspot(solved):
    mesh, u, C = solved
    hot = locate_hotspot_element(mesh, u, C)
    assert hot["family"] == "hex20"
    assert hot["element_index"] == 0
    assert hot["element_id"] == 42                # maps through mesh.elem_ids
    assert hot["von_mises_max_MPa"] == pytest.approx(E * DELTA, rel=1e-6)
    x, y, z = hot["location_xyz_mm"]
    assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= z <= 1.0


def test_element_max_geq_nodal_for_hotspot_sanity(solved):
    """Element/Gauss max must be >= nodal-averaged max (averaging never amplifies)."""
    from fem.postprocess import von_mises_nodal
    mesh, u, C = solved
    _, vm_elem = compute_element_max_von_mises(mesh, u, C)
    vm_nodal = von_mises_nodal(mesh, u, C)
    # uniform field => equal; the invariant is element-max >= nodal-max in general
    assert vm_elem.max() >= vm_nodal.max() - 1e-9
