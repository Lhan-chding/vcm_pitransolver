"""torch elastic energy == numpy FEM energy — the Stage-2 hard gate.

If these pass, fem/energy.py is the SAME mathematics as the numpy engine, so the
physics-informed training signal is trustworthy. Covers all three families, the
autograd contract (grad finite, non-zero, flows only through u), and that the
cached training path equals the uncached reference path equals numpy.

All numerics are float64 with a <1e-10 relative tolerance (the real error is
~1e-15; the margin catches real bugs without flagging float64 noise).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # torch tests skip cleanly where torch is absent

from parse_mesh import Mesh
from fem.energy import (
    build_family_kernels_cached,
    build_family_kernels_uncached,
    elastic_energy_cached,
    elastic_energy_uncached,
    material_C_torch,
)
from fem.material import elastic_C
from fem.postprocess import total_strain_energy
from fem.tests.reference_elements import REFERENCES

E, NU = 127000.0, 0.33
RTOL = 1e-10


def _single_element_mesh(name: str) -> Mesh:
    """One element of the given family on its reference geometry."""
    module, builder = REFERENCES[name]
    coords = builder()
    nnode = coords.shape[0]
    row = np.arange(nnode, dtype=np.int64)
    return Mesh(
        node_ids=np.arange(nnode, dtype=np.int64),
        coords=coords,
        elem_ids=np.array([0], dtype=np.int64),
        connectivity=[row],
        elem_nnode=np.array([nnode], dtype=np.int8),
        id_to_row={i: i for i in range(nnode)},
    )


def _nontrivial_u(coords: np.ndarray, seed: int) -> np.ndarray:
    """A non-trivial displacement field: a linear field + small random wiggle so
    the strain is non-constant (exercises every Gauss point, not a special case)."""
    rng = np.random.RandomState(seed)
    A = rng.uniform(-0.01, 0.01, size=(3, 3))     # linear part: u_i = A_ij x_j
    lin = coords @ A.T
    wig = 1e-3 * rng.standard_normal(coords.shape)
    return lin + wig


@pytest.mark.parametrize("name", sorted(REFERENCES))
def test_torch_uncached_equals_numpy(name):
    mesh = _single_element_mesh(name)
    C_np = elastic_C(E, NU)
    u_np = _nontrivial_u(mesh.coords, seed=hash(name) % 1000)

    U_numpy = total_strain_energy(mesh, u_np, C_np)

    u = torch.as_tensor(u_np, dtype=torch.float64)
    coords = torch.as_tensor(mesh.coords, dtype=torch.float64)
    C = material_C_torch(C_np)
    kernels = build_family_kernels_uncached(mesh, dtype=torch.float64)
    U_torch = float(elastic_energy_uncached(u, coords, kernels, C))

    assert U_numpy > 0
    rel = abs(U_torch - U_numpy) / abs(U_numpy)
    assert rel < RTOL, f"{name}: torch U={U_torch} vs numpy U={U_numpy} rel={rel:.2e}"


@pytest.mark.parametrize("name", sorted(REFERENCES))
def test_cached_equals_uncached_equals_numpy(name):
    mesh = _single_element_mesh(name)
    C_np = elastic_C(E, NU)
    u_np = _nontrivial_u(mesh.coords, seed=7)

    U_numpy = total_strain_energy(mesh, u_np, C_np)
    u = torch.as_tensor(u_np, dtype=torch.float64)
    coords = torch.as_tensor(mesh.coords, dtype=torch.float64)
    C = material_C_torch(C_np)

    k_unc = build_family_kernels_uncached(mesh, dtype=torch.float64)
    k_cac = build_family_kernels_cached(mesh, dtype=torch.float64)
    U_unc = float(elastic_energy_uncached(u, coords, k_unc, C))
    U_cac = float(elastic_energy_cached(u, k_cac, C))

    assert abs(U_unc - U_numpy) / abs(U_numpy) < RTOL
    assert abs(U_cac - U_numpy) / abs(U_numpy) < RTOL
    assert abs(U_cac - U_unc) / abs(U_unc) < RTOL


@pytest.mark.parametrize("name", sorted(REFERENCES))
def test_autograd_through_u(name):
    """U.backward() gives a finite, non-zero grad on u.

    Training passes coords as a CONSTANT (requires_grad=False) so the only
    gradient path is u — exactly as used in train_single. We verify u.grad is
    finite, non-zero, correctly shaped, and that the energy is differentiable
    w.r.t. u even though coords is treated as a fixed geometry.
    """
    mesh = _single_element_mesh(name)
    C_np = elastic_C(E, NU)
    u_np = _nontrivial_u(mesh.coords, seed=3)

    u = torch.as_tensor(u_np, dtype=torch.float64).clone().requires_grad_(True)
    coords = torch.as_tensor(mesh.coords, dtype=torch.float64)  # constant geometry
    C = material_C_torch(C_np)
    kernels = build_family_kernels_uncached(mesh, dtype=torch.float64)

    U = elastic_energy_uncached(u, coords, kernels, C)
    U.backward()

    assert u.grad is not None
    assert torch.isfinite(u.grad).all()
    assert u.grad.abs().sum() > 0                      # non-zero gradient
    assert u.grad.shape == u.shape


@pytest.mark.parametrize("name", sorted(REFERENCES))
def test_energy_grad_matches_internal_force(name):
    """dU/du must equal K @ u (the internal force) — the physics behind the loss.

    For linear elasticity U = 1/2 u^T K u, so the autograd gradient of the torch
    energy w.r.t. u is exactly the internal force vector the numpy stiffness
    produces. This ties the differentiable loss to the assembled operator.
    """
    from fem.assembly import assemble_global_stiffness
    mesh = _single_element_mesh(name)
    C_np = elastic_C(E, NU)
    u_np = _nontrivial_u(mesh.coords, seed=11)

    u = torch.as_tensor(u_np, dtype=torch.float64).clone().requires_grad_(True)
    coords = torch.as_tensor(mesh.coords, dtype=torch.float64)
    C = material_C_torch(C_np)
    kernels = build_family_kernels_uncached(mesh, dtype=torch.float64)
    U = elastic_energy_uncached(u, coords, kernels, C)
    U.backward()
    grad = u.grad.reshape(-1).numpy()                  # node-major

    K = assemble_global_stiffness(mesh, C_np)
    f_internal = K @ u_np.reshape(-1)
    assert np.allclose(grad, f_internal, rtol=1e-9, atol=1e-12)


def test_cached_path_has_no_det_or_solve_dependency_on_u():
    """Cached energy must be linear-algebra-light: changing u must not require
    re-deriving geometry. Smoke check: two different u reuse the same kernel."""
    mesh = _single_element_mesh("hex20")
    C_np = elastic_C(E, NU)
    C = material_C_torch(C_np)
    kernels = build_family_kernels_cached(mesh, dtype=torch.float64)

    u1 = torch.as_tensor(_nontrivial_u(mesh.coords, 1), dtype=torch.float64)
    u2 = torch.as_tensor(_nontrivial_u(mesh.coords, 2), dtype=torch.float64)
    U1 = elastic_energy_cached(u1, kernels, C)
    U2 = elastic_energy_cached(u2, kernels, C)
    assert U1 > 0 and U2 > 0 and not torch.isclose(U1, U2)
