"""transolver_wrap.py — backbone wrapper: real PhysicsNeMo Transolver (server) or
an injectable fake backbone (local pipeline test).

PhysicsNeMo is only present in the A800 container, so the real Transolver import
is LAZY (importing this module must not require physicsnemo). For local
development we use a small MLP backbone with the same call signature so the whole
pipeline — features -> backbone -> enforce_bc -> energy -> loss -> backward — is
exercised on CPU. On the server, swap in make_transolver().

API note (PhysicsNeMo 1.2.0, from source research, VERIFY IN CONTAINER before
training, see docs/stage2_energy_transolver_impl.md milestone-4a):
  Transolver(functional_dim, out_dim, embedding_dim, n_layers, n_hidden, n_head,
             slice_num, unified_pos=False, structured_shape=None, use_te=...)
  forward(fx, embedding=coords) -> (B, N, out_dim)   (fx and coords passed apart)
  1.2.0 has NO `plus` parameter (Transolver++ is 2.0.0+).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def make_transolver(
    functional_dim: int,
    *,
    n_layers: int = 8,
    n_hidden: int = 256,
    n_head: int = 8,
    slice_num: int = 64,
    use_te: bool = False,
) -> nn.Module:
    """Construct the real PhysicsNeMo Transolver (server only; lazy import)."""
    from physicsnemo.models.transolver import Transolver  # noqa: PLC0415
    return Transolver(
        functional_dim=functional_dim,
        out_dim=3,
        embedding_dim=3,
        structured_shape=None,
        unified_pos=False,
        n_layers=n_layers,
        n_hidden=n_hidden,
        n_head=n_head,
        slice_num=slice_num,
        use_te=use_te,
    )


class FakeBackbone(nn.Module):
    """A small MLP with the Transolver call signature, for local pipeline tests.

    forward(fx, embedding=coords) -> (B, N, 3). It concatenates features and
    coords and maps per node — enough to drive enforce_bc/energy/backward without
    PhysicsNeMo. NOT a physics model; it only has to be differentiable and shaped
    right so the training loop is validated end to end.
    """

    def __init__(self, functional_dim: int, embedding_dim: int = 3, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(functional_dim + embedding_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, fx: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        x = torch.cat([fx, embedding], dim=-1)
        return self.net(x)


def forward_single(model: nn.Module, fx: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """B=1 single-geometry forward: fx (N,F), coords (N,3) -> u_raw (N,3).

    Adds/removes the batch dim so the same call works for the real Transolver
    (which is batched) and the fake backbone.
    """
    u = model(fx.unsqueeze(0), embedding=coords.unsqueeze(0))  # (1,N,3)
    return u.squeeze(0)
