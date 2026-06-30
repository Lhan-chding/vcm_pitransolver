"""End-to-end solver tests: assembly -> Dirichlet BC -> spsolve -> stiffness.

The cornerstone is a single hex20 unit cube with an ANALYTIC answer: fixing the
x=0 face and pulling the x=1 face to u_x = delta (lateral faces free) reproduces
the exact uniaxial solution u_x = delta * x, with reaction force F = E*A*delta/L.
For the unit cube (A=1, L=1) this gives K = F/delta = E exactly, and the strain
energy U = 1/2 * E * delta^2 so K_energy = 2U/delta^2 = E too. Any DOF-mapping,
assembly, or BC-application bug breaks this exact identity.

A second test confirms the two independent stiffness estimates (energy vs
reaction) agree on a less trivial config, which is the real correctness invariant
the pipeline relies on.
"""

from __future__ import annotations

import numpy as np
import pytest

from parse_mesh import Mesh
from parse_face_to_nodes import BoundarySets

from fem.assembly import assemble_global_stiffness, num_dofs
from fem.direct_solver import solve_displacement
from fem.material import elastic_C
from fem.postprocess import compute_stiffness, von_mises_nodal
from fem.tests.reference_elements import hex20_reference

E = 127000.0   # MPa  (use the real spring material modulus)
NU = 0.0       # nu=0 -> pure uniaxial, K == E exactly with no lateral coupling
DELTA = 0.005  # mm


def _single_hex_mesh() -> Mesh:
    """A one-element hex20 mesh on the unit cube [0,1]^3."""
    coords = hex20_reference()  # (20,3) on [0,1]^3, module node order
    row = np.arange(20, dtype=np.int64)
    return Mesh(
        node_ids=np.arange(20, dtype=np.int64),
        coords=coords,
        elem_ids=np.array([0], dtype=np.int64),
        connectivity=[row],
        elem_nnode=np.array([20], dtype=np.int8),
        id_to_row={i: i for i in range(20)},
    )


def _x_face_boundary(coords: np.ndarray) -> BoundarySets:
    """Fixed = nodes on x=0 face; move = nodes on x=1 face."""
    fixed = np.nonzero(np.isclose(coords[:, 0], 0.0))[0]
    move = np.nonzero(np.isclose(coords[:, 0], 1.0))[0]
    all_idx = np.arange(coords.shape[0])
    free = np.setdiff1d(all_idx, np.union1d(fixed, move))
    return BoundarySets(
        move_nodes=move,
        fixed_nodes=fixed,
        free_nodes=free,
        move_face_descriptor={},
    )


def test_assembled_stiffness_is_symmetric_psd():
    mesh = _single_hex_mesh()
    C = elastic_C(E, NU)
    K = assemble_global_stiffness(mesh, C)
    assert K.shape == (num_dofs(mesh), num_dofs(mesh))
    # symmetry
    assert abs((K - K.T)).max() < 1e-6 * abs(K).max()
    # PSD: smallest eigenvalue ~ 0 (rigid modes), none significantly negative.
    dense = K.toarray()
    eigmin = np.linalg.eigvalsh(dense).min()
    assert eigmin > -1e-6 * abs(K).max(), f"K not PSD; min eig {eigmin}"


def test_single_hex_uniaxial_stiffness_equals_E():
    """Unit cube, nu=0, pulled in x: K_energy == K_reaction == E (analytic)."""
    mesh = _single_hex_mesh()
    C = elastic_C(E, NU)
    bs = _x_face_boundary(mesh.coords)

    # transverse free so the bar can contract laterally (pure uniaxial). With
    # nu=0 there is no contraction anyway, but 'free' keeps it exact.
    res = solve_displacement(
        mesh, bs, C, move_axis="x", delta_mm=DELTA, move_transverse_rigid=False
    )
    stiff = compute_stiffness(mesh, res, C)

    assert stiff.K_energy == pytest.approx(E, rel=1e-6)
    assert stiff.K_reaction == pytest.approx(E, rel=1e-6)
    assert stiff.rel_gap < 1e-9


def test_displacement_field_matches_linear_solution():
    """u_x must equal delta * x exactly (linear field), u_y = u_z = 0 at nu=0."""
    mesh = _single_hex_mesh()
    C = elastic_C(E, NU)
    bs = _x_face_boundary(mesh.coords)
    res = solve_displacement(
        mesh, bs, C, move_axis="x", delta_mm=DELTA, move_transverse_rigid=False
    )
    expected_ux = DELTA * mesh.coords[:, 0]
    assert np.allclose(res.u[:, 0], expected_ux, atol=1e-9)
    assert np.allclose(res.u[:, 1], 0.0, atol=1e-9)
    assert np.allclose(res.u[:, 2], 0.0, atol=1e-9)


def test_bc_values_are_applied_exactly():
    """Constrained DOFs in the solution must hold their prescribed values."""
    mesh = _single_hex_mesh()
    C = elastic_C(E, NU)
    bs = _x_face_boundary(mesh.coords)
    res = solve_displacement(
        mesh, bs, C, move_axis="x", delta_mm=DELTA, move_transverse_rigid=True
    )
    u_flat = res.u.reshape(-1)
    assert np.allclose(u_flat[res.constrained_dofs], res.prescribed, atol=1e-12)
    # fixed face is pinned to zero
    for r in bs.fixed_nodes:
        assert np.allclose(res.u[r], 0.0, atol=1e-12)
    # move face x-component is delta
    for r in bs.move_nodes:
        assert res.u[r, 0] == pytest.approx(DELTA)


def test_energy_reaction_consistency_with_poisson():
    """With nu=0.33 (lateral coupling), K_energy and K_reaction must still agree.

    This is the real invariant: Clapeyron's theorem makes 2U/delta^2 == F/delta
    for ANY linear displacement-controlled solve, regardless of Poisson coupling.
    """
    mesh = _single_hex_mesh()
    C = elastic_C(E, 0.33)
    bs = _x_face_boundary(mesh.coords)
    res = solve_displacement(
        mesh, bs, C, move_axis="x", delta_mm=DELTA, move_transverse_rigid=False
    )
    stiff = compute_stiffness(mesh, res, C)
    assert stiff.rel_gap < 1e-8, (
        f"energy vs reaction K disagree: {stiff.K_energy} vs {stiff.K_reaction}"
    )


def test_von_mises_uniform_under_uniaxial_strain():
    """Uniform uniaxial strain => uniform von Mises = E*delta/L on every node."""
    mesh = _single_hex_mesh()
    C = elastic_C(E, NU)
    bs = _x_face_boundary(mesh.coords)
    res = solve_displacement(
        mesh, bs, C, move_axis="x", delta_mm=DELTA, move_transverse_rigid=False
    )
    vm = von_mises_nodal(mesh, res.u, C)
    # sigma_xx = E * eps_xx = E * delta (L=1); uniaxial -> von Mises = |sigma_xx|.
    expected = E * DELTA
    assert np.allclose(vm, expected, rtol=1e-6)
