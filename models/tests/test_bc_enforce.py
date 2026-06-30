"""Hard-Dirichlet mask blending: value correctness, gradient flow, FEM consistency.

The decisive test is DOF-set consistency: the DOFs pinned by build_bc_tensors
must be EXACTLY the DOFs fem.direct_solver._build_dirichlet constrains, so the
neural predictor and the FEM oracle solve the same boundary-value problem.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # models depend on torch; skip if absent

from parse_face_to_nodes import BoundarySets
from fem.direct_solver import _build_dirichlet
from models.bc_enforce import (
    build_bc_tensors,
    constrained_dofs_from_mask,
    enforce_bc,
)

DELTA = 0.005


def _toy_boundary() -> tuple[BoundarySets, int]:
    """8 nodes: 0,1 fixed; 6,7 move; 2-5 free."""
    n = 8
    fixed = np.array([0, 1], dtype=np.int64)
    move = np.array([6, 7], dtype=np.int64)
    free = np.array([2, 3, 4, 5], dtype=np.int64)
    bs = BoundarySets(move_nodes=move, fixed_nodes=fixed, free_nodes=free,
                      move_face_descriptor={})
    return bs, n


def test_fixed_nodes_pinned_to_zero():
    bs, n = _toy_boundary()
    fm, up = build_bc_tensors(bs, n, move_axis="x", delta_mm=DELTA)
    u_raw = torch.randn(n, 3, dtype=torch.float64)
    u = enforce_bc(u_raw, fm, up)
    for r in bs.fixed_nodes:
        assert torch.allclose(u[r], torch.zeros(3, dtype=torch.float64))


def test_move_nodes_get_delta_on_axis():
    bs, n = _toy_boundary()
    fm, up = build_bc_tensors(bs, n, move_axis="x", delta_mm=DELTA,
                              move_transverse_rigid=True)
    u_raw = torch.randn(n, 3, dtype=torch.float64)
    u = enforce_bc(u_raw, fm, up)
    for r in bs.move_nodes:
        assert u[r, 0] == pytest.approx(DELTA)     # move axis = x
        assert u[r, 1] == pytest.approx(0.0)        # transverse rigid
        assert u[r, 2] == pytest.approx(0.0)


def test_move_transverse_free_keeps_network_value():
    bs, n = _toy_boundary()
    fm, up = build_bc_tensors(bs, n, move_axis="x", delta_mm=DELTA,
                              move_transverse_rigid=False)
    u_raw = torch.randn(n, 3, dtype=torch.float64)
    u = enforce_bc(u_raw, fm, up)
    for r in bs.move_nodes:
        assert u[r, 0] == pytest.approx(DELTA)               # axis pinned
        assert u[r, 1] == pytest.approx(float(u_raw[r, 1]))  # transverse free
        assert u[r, 2] == pytest.approx(float(u_raw[r, 2]))


def test_gradient_flows_only_through_free_dofs():
    bs, n = _toy_boundary()
    fm, up = build_bc_tensors(bs, n, move_axis="x", delta_mm=DELTA,
                              move_transverse_rigid=True)
    u_raw = torch.randn(n, 3, dtype=torch.float64, requires_grad=True)
    u = enforce_bc(u_raw, fm, up)
    u.sum().backward()
    grad = u_raw.grad
    # free DOFs: grad == 1; pinned DOFs: grad == 0
    free = fm == 1.0
    assert torch.all(grad[free] == 1.0)
    assert torch.all(grad[~free] == 0.0)


def test_no_inplace_mutation_of_u_raw():
    """enforce_bc must not modify u_raw in place."""
    bs, n = _toy_boundary()
    fm, up = build_bc_tensors(bs, n)
    u_raw = torch.randn(n, 3, dtype=torch.float64)
    before = u_raw.clone()
    _ = enforce_bc(u_raw, fm, up)
    assert torch.equal(u_raw, before)


@pytest.mark.parametrize("rigid", [True, False])
@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_dof_set_matches_direct_solver(axis, rigid):
    """The pinned-DOF set must equal fem.direct_solver._build_dirichlet's set.

    This is the consistency guarantee: predictor and oracle constrain identically.
    """
    bs, n = _toy_boundary()
    fm, _up = build_bc_tensors(bs, n, move_axis=axis, delta_mm=DELTA,
                               move_transverse_rigid=rigid)
    mask_dofs = set(constrained_dofs_from_mask(fm).tolist())

    axis_i = {"x": 0, "y": 1, "z": 2}[axis]
    c_dofs, _vals = _build_dirichlet(bs, axis_i, DELTA, rigid)
    solver_dofs = set(c_dofs.tolist())

    assert mask_dofs == solver_dofs


def test_prescribed_values_match_direct_solver():
    """Not just the DOF set — the prescribed VALUES must match the oracle too."""
    bs, n = _toy_boundary()
    fm, up = build_bc_tensors(bs, n, move_axis="x", delta_mm=DELTA,
                              move_transverse_rigid=True)
    # build a value lookup from the mask path
    pinned = constrained_dofs_from_mask(fm)
    up_flat = up.reshape(-1).cpu().numpy()
    mask_vals = {int(d): float(up_flat[d]) for d in pinned}

    c_dofs, c_vals = _build_dirichlet(bs, 0, DELTA, True)
    solver_vals = {int(d): float(v) for d, v in zip(c_dofs, c_vals)}

    assert mask_vals == solver_vals
