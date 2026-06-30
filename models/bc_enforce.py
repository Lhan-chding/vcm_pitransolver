"""bc_enforce.py — hard Dirichlet boundary conditions via mask blending.

The network outputs a raw displacement field u_raw; the physics requires fixed
nodes at u=0 and move nodes at the prescribed stroke. We impose this WITHOUT
in-place assignment (which would sever autograd and mutate a leaf):

    u = free_mask * u_raw + (1 - free_mask) * u_prescribed

so the gradient flows ONLY through the free DOFs — exactly the DOFs the network
is allowed to choose. The energy must then be computed on this blended u, never
on u_raw (computing it on u_raw would ignore the boundary conditions entirely).

The constrained-DOF set produced here is identical to the one
fem.direct_solver._build_dirichlet pins, so the neural predictor and the FEM
oracle solve the SAME boundary-value problem. A test asserts that equality.
"""

from __future__ import annotations

import numpy as np
import torch

from parse_face_to_nodes import BoundarySets

_AXIS = {"x": 0, "y": 1, "z": 2}


def build_bc_tensors(
    bs: BoundarySets,
    num_nodes: int,
    *,
    move_axis: str = "x",
    delta_mm: float = 0.005,
    move_transverse_rigid: bool = True,
    device="cpu",
    dtype=torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (free_mask, u_prescribed), both (N, 3).

    free_mask[i, c] = 1.0 if node i's component c is FREE (network decides),
                      0.0 if it is pinned by a Dirichlet condition.
    u_prescribed[i, c] = the value a pinned DOF must take (0 for fixed, delta for
                      the move axis of move nodes, 0 for move-transverse if rigid).
                      Free DOFs hold 0 here (they are multiplied out by free_mask).

    The pinned DOFs are: every component of every fixed node; the move-axis
    component of every move node; and (if move_transverse_rigid) the other two
    components of every move node. This mirrors direct_solver._build_dirichlet.
    """
    axis = _AXIS[move_axis.lower()]
    free_mask = torch.ones((num_nodes, 3), dtype=dtype, device=device)
    u_prescribed = torch.zeros((num_nodes, 3), dtype=dtype, device=device)

    fixed = np.asarray(bs.fixed_nodes, dtype=np.int64)
    move = np.asarray(bs.move_nodes, dtype=np.int64)

    # fixed: all three components pinned to 0.
    free_mask[fixed, :] = 0.0
    # move: move-axis pinned to delta.
    free_mask[move, axis] = 0.0
    u_prescribed[move, axis] = float(delta_mm)
    # move transverse: pin the other two components to 0 if rigid.
    if move_transverse_rigid:
        for c in range(3):
            if c != axis:
                free_mask[move, c] = 0.0
                # u_prescribed already 0 there
    return free_mask, u_prescribed


def enforce_bc(
    u_raw: torch.Tensor,
    free_mask: torch.Tensor,
    u_prescribed: torch.Tensor,
) -> torch.Tensor:
    """Blend network output with the prescribed BCs (no in-place).

    u = free_mask * u_raw + (1 - free_mask) * u_prescribed

    Gradient flows only where free_mask == 1; pinned DOFs are constant.
    """
    return free_mask * u_raw + (1.0 - free_mask) * u_prescribed


def constrained_dofs_from_mask(free_mask: torch.Tensor) -> np.ndarray:
    """Global DOF indices pinned by the mask (free_mask == 0), node-major.

    Lets a test compare this set against fem.direct_solver's constrained_dofs.
    """
    pinned = (free_mask == 0.0).reshape(-1).cpu().numpy()  # (3N,) node-major
    return np.nonzero(pinned)[0]
