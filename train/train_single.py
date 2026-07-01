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

_AXIS = {"x": 0, "y": 1, "z": 2}


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
    # High-precision second-order refinement. The elastic energy is a quadratic
    # form with a very ill-conditioned Hessian (thin plate, ~4e5 DOFs), where
    # first-order Adam stalls far above the true solution. Run `steps` Adam steps
    # to get into the basin, then `lbfgs_steps` L-BFGS steps with strong-Wolfe
    # line search to drive U (and the K_energy/K_reaction gap) down to machine
    # precision. lbfgs_steps=0 keeps the pure-Adam behavior (default, back-compat).
    lbfgs_steps: int = 0
    lbfgs_lr: float = 1.0                 # with strong_wolfe the step is line-searched
    lbfgs_max_iter: int = 20              # inner iterations per outer L-BFGS step
    lbfgs_history: int = 100
    lbfgs_tol_grad: float = 1e-16         # float64: push to machine precision
    lbfgs_tol_change: float = 1e-18


@dataclass
class TrainHistory:
    step: list
    U: list
    K_energy: list
    K_reaction: list
    rel_gap: list
    u_max: list


def _diagnostics(backbone, ni, free_mask, u_pre, kernels, C,
                 move_rows, axis, delta_mm, delta2) -> dict:
    """Eval-only physics diagnostics: U, K_energy, K_reaction, rel_gap, |u|max.

    K_reaction uses the identity dU/du == K@u (proven in test_energy_consistency):
    the internal force at the move-axis move DOFs IS the energy gradient there, so
    K_reaction = sum_move (dU/du)_axis / delta — no separate stiffness assembly.
    For a converged linear solve K_energy == K_reaction (Clapeyron); their gap is
    the key unlabeled-training sanity check (low loss alone is NOT enough).
    """
    u_raw = forward_single(backbone, ni.fx, ni.coords_net)
    u = enforce_bc(u_raw, free_mask, u_pre).detach().requires_grad_(True)
    U = elastic_energy_cached(u, kernels, C)
    (grad,) = torch.autograd.grad(U, u)            # dU/du == K@u (internal force)
    U_val = float(U.detach())
    K_energy = 2.0 * U_val / delta2
    move_force = float(grad[move_rows, axis].sum())
    K_reaction = move_force / delta_mm
    rel_gap = abs(K_energy - K_reaction) / max(abs(K_energy), 1e-30)
    u_max = float(u.detach().norm(dim=1).max())
    return {"U": U_val, "K_energy": K_energy, "K_reaction": K_reaction,
            "rel_gap": rel_gap, "u_max": u_max}


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
    axis = _AXIS[cfg.move_axis.lower()]
    move_rows = torch.as_tensor(np.asarray(bs.move_nodes), dtype=torch.long, device=dev)

    backbone = backbone.to(dev).to(dt)

    hist = TrainHistory(step=[], U=[], K_energy=[], K_reaction=[], rel_gap=[], u_max=[])

    def _record(step: int) -> dict:
        diag = _diagnostics(backbone, ni, free_mask, u_pre, kernels, C,
                            move_rows, axis, cfg.delta_mm, delta2)
        hist.step.append(step)
        hist.U.append(diag["U"])
        hist.K_energy.append(diag["K_energy"])
        hist.K_reaction.append(diag["K_reaction"])
        hist.rel_gap.append(diag["rel_gap"])
        hist.u_max.append(diag["u_max"])
        if verbose:
            print(f"step {step:5d}  U={diag['U']:.6e}  "
                  f"K_energy={diag['K_energy']:.6e}  K_reaction={diag['K_reaction']:.6e} "
                  f"gap={diag['rel_gap']:.3e}  |u|max={diag['u_max']:.4e} mm",
                  flush=True)
        return diag

    def _forward_U() -> torch.Tensor:
        u_raw = forward_single(backbone, ni.fx, ni.coords_net)   # (N,3)
        u = enforce_bc(u_raw, free_mask, u_pre)                  # blended
        return elastic_energy_cached(u, kernels, C)              # physical coords baked in

    # ---- Phase 1: Adam (get into the basin) ------------------------------- #
    opt = torch.optim.Adam(backbone.parameters(), lr=cfg.lr)
    for step in range(cfg.steps):
        opt.zero_grad()
        U = _forward_U()
        U.backward()
        opt.step()
        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            _record(step)

    # ---- Phase 2: L-BFGS (second-order refinement to machine precision) --- #
    # The energy is a quadratic form with an ill-conditioned Hessian; strong-Wolfe
    # L-BFGS handles that far better than first-order Adam, driving U and the
    # K_energy/K_reaction gap down toward float64 precision. Each optimizer.step
    # runs up to lbfgs_max_iter inner line-search iterations, re-evaluating the
    # closure — so we log per outer step, offset past the Adam steps.
    if cfg.lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(
            backbone.parameters(),
            lr=cfg.lbfgs_lr,
            max_iter=cfg.lbfgs_max_iter,
            history_size=cfg.lbfgs_history,
            tolerance_grad=cfg.lbfgs_tol_grad,
            tolerance_change=cfg.lbfgs_tol_change,
            line_search_fn="strong_wolfe",
        )

        def _closure() -> torch.Tensor:
            lbfgs.zero_grad()
            U = _forward_U()
            U.backward()
            return U

        for j in range(cfg.lbfgs_steps):
            lbfgs.step(_closure)
            _record(cfg.steps + j)

    return hist
