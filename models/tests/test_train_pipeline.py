"""End-to-end pipeline test with a fake backbone (no PhysicsNeMo needed).

Validates the full Stage-2 training path on CPU: features -> fake backbone ->
enforce_bc -> energy -> loss -> backward -> optimizer step. The fake backbone is
not a physics model, but minimizing the elastic energy under hard BCs must still
DECREASE U over steps — proving the loop is wired correctly and the gradient
actually flows from energy back to the network parameters.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # models depend on torch; skip if absent

from parse_mesh import Mesh
from parse_face_to_nodes import BoundarySets
from fem.material import elastic_C
from fem.tests.reference_elements import hex20_reference
from models.transolver_wrap import FakeBackbone, forward_single
from models.features import FEATURE_DIM, build_node_inputs
from train.train_single import TrainConfig, train_single

E, NU, DELTA = 127000.0, 0.0, 0.005


def _single_hex_mesh() -> Mesh:
    coords = hex20_reference()
    row = np.arange(20, dtype=np.int64)
    return Mesh(
        node_ids=np.arange(20, dtype=np.int64), coords=coords,
        elem_ids=np.array([0], dtype=np.int64), connectivity=[row],
        elem_nnode=np.array([20], dtype=np.int8), id_to_row={i: i for i in range(20)},
    )


def _x_boundary(coords):
    fixed = np.nonzero(np.isclose(coords[:, 0], 0.0))[0]
    move = np.nonzero(np.isclose(coords[:, 0], 1.0))[0]
    free = np.setdiff1d(np.arange(coords.shape[0]), np.union1d(fixed, move))
    return BoundarySets(move_nodes=move, fixed_nodes=fixed, free_nodes=free,
                        move_face_descriptor={})


def test_fake_backbone_forward_shape():
    mesh = _single_hex_mesh()
    bs = _x_boundary(mesh.coords)
    ni = build_node_inputs(mesh.coords, bs, E_MPa=E, nu=NU, delta_mm=DELTA)
    assert ni.fx.shape == (20, FEATURE_DIM)
    model = FakeBackbone(FEATURE_DIM).to(torch.float64)
    u = forward_single(model, ni.fx, ni.coords_net)
    assert u.shape == (20, 3)


def test_training_decreases_energy():
    """U must go down over training — the loop and gradient path are correct."""
    torch.manual_seed(0)
    mesh = _single_hex_mesh()
    bs = _x_boundary(mesh.coords)
    C = elastic_C(E, NU)
    model = FakeBackbone(FEATURE_DIM)
    cfg = TrainConfig(move_axis="x", delta_mm=DELTA, move_transverse_rigid=False,
                      steps=300, lr=5e-3, log_every=50)
    hist = train_single(model, mesh, bs, C, E_MPa=E, nu=NU, cfg=cfg, verbose=False)
    assert hist.U[-1] < hist.U[0], f"energy did not decrease: {hist.U[0]} -> {hist.U[-1]}"
    # all energies finite and non-negative
    assert all(np.isfinite(hist.U)) and all(u >= 0 for u in hist.U)


def test_reaction_consistency_gap_improves():
    """K_energy vs K_reaction gap must shrink as energy is minimized.

    This is the unlabeled-training sanity check: an untrained net is wildly
    inconsistent (large gap); minimizing energy drives K_energy and K_reaction
    together (Clapeyron). On the analytic single-hex problem they both approach E.
    """
    torch.manual_seed(0)
    mesh = _single_hex_mesh()
    bs = _x_boundary(mesh.coords)
    C = elastic_C(E, NU)
    model = FakeBackbone(FEATURE_DIM)
    cfg = TrainConfig(move_axis="x", delta_mm=DELTA, move_transverse_rigid=False,
                      steps=600, lr=5e-3, log_every=100)
    hist = train_single(model, mesh, bs, C, E_MPa=E, nu=NU, cfg=cfg, verbose=False)
    assert hist.rel_gap[-1] < hist.rel_gap[0]            # gap shrinks
    assert hist.rel_gap[-1] < 1e-2                       # and gets genuinely small
    # both stiffness estimates approach the analytic E within a few percent
    assert abs(hist.K_energy[-1] - E) / E < 0.05
    assert abs(hist.K_reaction[-1] - E) / E < 0.05


def test_bc_satisfied_during_training():
    """Even with a random fake backbone, enforced u must satisfy the BCs exactly."""
    torch.manual_seed(1)
    mesh = _single_hex_mesh()
    bs = _x_boundary(mesh.coords)
    C = elastic_C(E, NU)
    ni = build_node_inputs(mesh.coords, bs, E_MPa=E, nu=NU, delta_mm=DELTA)
    from models.bc_enforce import build_bc_tensors, enforce_bc
    fm, up = build_bc_tensors(bs, mesh.num_nodes, move_axis="x", delta_mm=DELTA,
                              move_transverse_rigid=True)
    model = FakeBackbone(FEATURE_DIM).to(torch.float64)
    u_raw = forward_single(model, ni.fx, ni.coords_net)
    u = enforce_bc(u_raw, fm, up).detach().numpy()
    assert np.allclose(u[bs.fixed_nodes], 0.0)
    assert np.allclose(u[bs.move_nodes, 0], DELTA)
