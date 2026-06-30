"""features.py — node features and the coords_net / coords_phys dual track.

The network sees NORMALIZED coordinates (zero-centered, unit-scaled) for
conditioning; the FEM energy is ALWAYS computed on the original physical mm
coordinates. Mixing these up silently rescales the energy, so they are kept as
two explicit tensors and never conflated.

Node feature vector fx (per node):
  [fixed_flag, move_flag, free_flag,
   dist_to_fixed, dist_to_move,
   E_norm, nu,
   prescribed_ux_norm, prescribed_uy_norm, prescribed_uz_norm]
A design token (thickness/width/... broadcast to nodes) is deferred to the
multi-geometry stage; single-geometry training does not need it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from parse_face_to_nodes import BoundarySets

_AXIS = {"x": 0, "y": 1, "z": 2}
# A reference modulus to normalize E into ~O(1); C1990 is 127 GPa.
_E_REF_MPa = 127000.0


@dataclass(frozen=True)
class NodeInputs:
    coords_net: torch.Tensor     # (N, 3) normalized coords -> network embedding
    coords_phys: torch.Tensor    # (N, 3) physical mm coords -> FEM energy
    fx: torch.Tensor             # (N, F) node feature matrix
    center: np.ndarray           # (3,) bbox center used for normalization
    scale: float                 # scalar used for normalization


def _nearest_dist(coords: np.ndarray, target_rows: np.ndarray) -> np.ndarray:
    """Min Euclidean distance from every node to the nearest target node."""
    if target_rows.size == 0:
        return np.full(coords.shape[0], np.inf)
    from scipy.spatial import cKDTree
    tree = cKDTree(coords[target_rows])
    d, _ = tree.query(coords, k=1)
    return d


def build_node_inputs(
    coords: np.ndarray,
    bs: BoundarySets,
    *,
    E_MPa: float,
    nu: float,
    move_axis: str = "x",
    delta_mm: float = 0.005,
    device="cpu",
    dtype=torch.float64,
) -> NodeInputs:
    """Construct coords_net, coords_phys, and the node feature matrix fx."""
    n = coords.shape[0]
    axis = _AXIS[move_axis.lower()]

    # coordinate dual-track
    lo, hi = coords.min(0), coords.max(0)
    center = 0.5 * (lo + hi)
    scale = float(np.max(hi - lo)) or 1.0
    coords_net_np = (coords - center) / scale

    # boundary flags
    fixed_flag = np.zeros(n); fixed_flag[bs.fixed_nodes] = 1.0
    move_flag = np.zeros(n); move_flag[bs.move_nodes] = 1.0
    free_flag = 1.0 - fixed_flag - move_flag

    # distances (normalized by scale so they are ~O(1))
    d_fixed = _nearest_dist(coords, np.asarray(bs.fixed_nodes)) / scale
    d_move = _nearest_dist(coords, np.asarray(bs.move_nodes)) / scale
    d_fixed = np.where(np.isfinite(d_fixed), d_fixed, 0.0)
    d_move = np.where(np.isfinite(d_move), d_move, 0.0)

    # prescribed displacement (normalized by delta so move axis ~ 1.0)
    presc = np.zeros((n, 3))
    presc[bs.move_nodes, axis] = 1.0   # already normalized by delta

    E_col = np.full(n, E_MPa / _E_REF_MPa)
    nu_col = np.full(n, nu)

    fx_np = np.column_stack([
        fixed_flag, move_flag, free_flag,
        d_fixed, d_move,
        E_col, nu_col,
        presc[:, 0], presc[:, 1], presc[:, 2],
    ])

    return NodeInputs(
        coords_net=torch.as_tensor(coords_net_np, dtype=dtype, device=device),
        coords_phys=torch.as_tensor(coords, dtype=dtype, device=device),
        fx=torch.as_tensor(fx_np, dtype=dtype, device=device),
        center=center,
        scale=scale,
    )


FEATURE_DIM = 10  # length of fx above; transolver functional_dim must match
