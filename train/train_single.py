"""train_single.py — single-geometry physics-informed (energy-loss) training.

Bare PyTorch loop, no PhysicsNeMo-Sym. The enforced order is mandatory:

    u_raw = backbone(fx, coords_net)            # network on NORMALIZED coords
    u     = enforce_bc(u_raw, free_mask, u_pre) # hard Dirichlet (blended)
    U     = elastic_energy(u, coords_phys, ...) # energy on PHYSICAL coords
    loss  = U                                   # min U == equilibrium (disp. control)

Under pure displacement control W_external = 0, so minimizing U IS the
equilibrium condition — no labels needed. We monitor physics diagnostics every
few steps (U, K_energy, |u|max) rather than trusting the loss curve alone, per
the project's hard-won lesson that low loss != correct physics.

Works with any backbone exposing forward(fx, embedding=coords) -> (1,N,3): the
local FakeBackbone for pipeline validation, or the real Transolver on the server.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from parse_face_to_nodes import BoundarySets
from parse_mesh import Mesh
from fem.energy import (
    build_family_kernels_cached,
    elastic_energy_cached,
    material_C_torch,
)
from models.bc_enforce import build_bc_tensors, enforce_bc
from models.features import build_node_inputs
from models.transolver_wrap import forward_single


@dataclass
class TrainConfig:
    move_axis: str = "x"
    delta_mm: float = 0.005
    move_transverse_rigid: bool = True
    steps: int = 2000
    lr: float = 1e-3
    log_every: int = 100
    dtype: torch.dtype = torch.float64
    device: str = "cpu"


@dataclass
class TrainHistory:
    step: list
    U: list
    K_energy: list
    u_max: list


def train_single(
    backbone: torch.nn.Module,
    mesh: Mesh,
    bs: BoundarySets,
    C_np: np.ndarray,
    *,
    E_MPa: float,
    nu: float,
    cfg: TrainConfig = TrainConfig(),
    verbose: bool = True,
) -> TrainHistory:
    """Train `backbone` to minimize elastic energy on one geometry.

    Returns the diagnostic history. The backbone is updated in place.
    """
    dev, dt = cfg.device, cfg.dtype
    ni = build_node_inputs(
        mesh.coords, bs, E_MPa=E_MPa, nu=nu,
        move_axis=cfg.move_axis, delta_mm=cfg.delta_mm, device=dev, dtype=dt,
    )
    free_mask, u_pre = build_bc_tensors(
        bs, mesh.num_nodes, move_axis=cfg.move_axis, delta_mm=cfg.delta_mm,
        move_transverse_rigid=cfg.move_transverse_rigid, device=dev, dtype=dt,
    )
    kernels = build_family_kernels_cached(mesh, device=dev, dtype=dt)
    C = material_C_torch(C_np, device=dev, dtype=dt)
    delta2 = cfg.delta_mm * cfg.delta_mm

    backbone = backbone.to(dev).to(dt)
    opt = torch.optim.Adam(backbone.parameters(), lr=cfg.lr)

    hist = TrainHistory(step=[], U=[], K_energy=[], u_max=[])
    for step in range(cfg.steps):
        opt.zero_grad()
        u_raw = forward_single(backbone, ni.fx, ni.coords_net)   # (N,3)
        u = enforce_bc(u_raw, free_mask, u_pre)                  # blended
        U = elastic_energy_cached(u, kernels, C)                 # physical coords baked in
        U.backward()
        opt.step()

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            U_val = float(U.detach())
            K_energy = 2.0 * U_val / delta2
            u_max = float(u.detach().norm(dim=1).max())
            hist.step.append(step)
            hist.U.append(U_val)
            hist.K_energy.append(K_energy)
            hist.u_max.append(u_max)
            if verbose:
                print(f"step {step:5d}  U={U_val:.6e}  "
                      f"K_energy={K_energy:.6e} N/mm  |u|max={u_max:.4e} mm",
                      flush=True)
    return hist
