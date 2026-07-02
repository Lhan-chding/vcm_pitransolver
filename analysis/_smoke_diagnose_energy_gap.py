"""_smoke_diagnose_energy_gap.py — local self-check for diagnose_energy_gap.py.

No real variant mesh is needed on the local box, so we exercise the diagnostic's
core functions on the analytic unit-cube hex20 whose answer is KNOWN:

    x=0 face FIXED (u=0), x=1 face MOVE (u_x=delta), nu=0
    exact energy  U = 1/2 * E * delta^2   (V=1, L=1, uniaxial)
    exact stiffness K = E                 (unit cube uniaxial)

This proves every interface the server script calls is wired correctly:
  * build_family_kernels_cached / elastic_energy_cached  (torch energy)
  * solve_displacement                                   (numpy oracle)
  * build_bc_tensors / enforce_bc                        (hard BCs)
  * _per_element_energy / _concentration                 (locality probe)

If check[1] here says "KERNEL OK" with U == 1/2 E delta^2 and K == E, the same
code path on the real variant is trustworthy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "data"))

from parse_face_to_nodes import BoundarySets  # noqa: E402
from parse_mesh import Mesh  # noqa: E402
from fem.tests.reference_elements import hex20_reference  # noqa: E402
from fem.material import elastic_C  # noqa: E402
from fem.direct_solver import solve_displacement  # noqa: E402
from fem.energy import (  # noqa: E402
    build_family_kernels_cached,
    elastic_energy_cached,
    material_C_torch,
)
from models.bc_enforce import build_bc_tensors, enforce_bc  # noqa: E402

# import the functions under test from the real script
from diagnose_energy_gap import _per_element_energy, _concentration  # noqa: E402

E, NU, DELTA = 127000.0, 0.0, 0.005
U_TRUE = 0.5 * E * DELTA * DELTA        # unit cube uniaxial, nu=0
K_TRUE = E


def _unit_cube_mesh() -> Mesh:
    coords = hex20_reference()          # [0,1]^3, 20 nodes
    row = np.arange(20, dtype=np.int64)
    return Mesh(
        node_ids=np.arange(20, dtype=np.int64), coords=coords,
        elem_ids=np.array([0], dtype=np.int64), connectivity=[row],
        elem_nnode=np.array([20], dtype=np.int8), id_to_row={i: i for i in range(20)},
    )


def _cube_boundary(coords: np.ndarray) -> BoundarySets:
    """x=0 face -> FIXED, x=1 face -> MOVE, remainder -> FREE."""
    x = coords[:, 0]
    fixed = np.nonzero(np.isclose(x, 0.0))[0].astype(np.int64)
    move = np.nonzero(np.isclose(x, 1.0))[0].astype(np.int64)
    free = np.setdiff1d(np.arange(coords.shape[0], dtype=np.int64),
                        np.union1d(move, fixed))
    return BoundarySets(
        move_nodes=move, fixed_nodes=fixed, free_nodes=free,
        move_face_descriptor={}, diagnostics={},
    )


def main() -> None:
    dtype, dev = torch.float64, "cpu"
    mesh = _unit_cube_mesh()
    bs = _cube_boundary(mesh.coords)
    bs.assert_valid(mesh.num_nodes)
    C_np = elastic_C(E, NU)
    print(f"unit cube: {mesh.num_nodes} nodes, move={bs.move_nodes.size} "
          f"fixed={bs.fixed_nodes.size} free={bs.free_nodes.size}")

    kernels = build_family_kernels_cached(mesh, device=dev, dtype=dtype)
    C = material_C_torch(C_np, device=dev, dtype=dtype)
    free_mask, u_pre = build_bc_tensors(
        bs, mesh.num_nodes, move_axis="x", delta_mm=DELTA,
        move_transverse_rigid=True, device=dev, dtype=dtype,
    )
    d2 = DELTA * DELTA

    # ---- [1] oracle scale ------------------------------------------------- #
    res = solve_displacement(mesh, bs, C_np, move_axis="x", delta_mm=DELTA,
                             move_transverse_rigid=True, backend="scipy")
    u_star = torch.as_tensor(res.u, dtype=dtype, device=dev)
    U_star = float(elastic_energy_cached(u_star, kernels, C))
    K_star = 2.0 * U_star / d2
    print("\n[1] ORACLE SCALE")
    print(f"    U*      = {U_star:.8e}   (true {U_TRUE:.8e})")
    print(f"    K_star  = {K_star:.8e}   (true {K_TRUE:.8e})")
    ok_U = np.isclose(U_star, U_TRUE, rtol=1e-6)
    ok_K = np.isclose(K_star, K_TRUE, rtol=1e-6)
    print(f"    U match={ok_U}  K match={ok_K}  -> "
          f"{'KERNEL OK' if ok_U and ok_K else 'MISMATCH!'}")

    # ---- [2] locality: oracle vs dirty ------------------------------------ #
    rng = np.random.default_rng(0)
    u_rand = torch.as_tensor(rng.normal(scale=DELTA, size=(mesh.num_nodes, 3)),
                             dtype=dtype, device=dev)
    u_dirty = enforce_bc(u_rand, free_mask, u_pre)
    U_dirty = float(elastic_energy_cached(u_dirty, kernels, C))
    pe_star = _per_element_energy(u_star, kernels, C)
    pe_dirty = _per_element_energy(u_dirty, kernels, C)
    tot_star = float(sum(v.sum() for v in pe_star.values()))
    print("\n[2] LOCALITY")
    print(f"    per-elem sum(oracle) = {tot_star:.8e}  (== U* ? "
          f"{np.isclose(tot_star, U_star, rtol=1e-9)})")
    print(f"    U_dirty = {U_dirty:.6e}  (dirty/oracle = {U_dirty/U_star:.3e}x)")
    print(f"    concentration(oracle) = {_concentration(pe_star)}")
    print(f"    concentration(dirty)  = {_concentration(pe_dirty)}")

    passed = ok_U and ok_K and np.isclose(tot_star, U_star, rtol=1e-9)
    print("\n" + ("SMOKE PASS: script interfaces verified on analytic cube."
                  if passed else "SMOKE FAIL: fix the script before server run."))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
