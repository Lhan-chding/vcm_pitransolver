"""features.py — node features and the coords_net / coords_phys dual track.

The network sees NORMALIZED coordinates (zero-centered, PER-AXIS unit-scaled)
for conditioning; the FEM energy is ALWAYS computed on the original physical mm
coordinates. Mixing these up silently rescales the energy, so they are kept as
two explicit tensors and never conflated.

Normalization is per-axis (each axis divided by its own bbox extent), not by a
single scalar. VCM parts are extreme thin plates (thickness:width ~1:190), and a
single-scalar scale would crush the thin axis to near-zero in coords_net,
blinding the network to thickness-direction position — the root cause of the
~2.2e4x stiffness overshoot seen in the first energy-trained run.

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
    scale: np.ndarray            # (3,) PER-AXIS extent used to normalize coords_net
    scale_iso: float             # scalar (max extent) used to normalize distance feats


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

    # coordinate dual-track. PER-AXIS normalization: a VCM part is an extreme
    # thin plate (variant 0001: x=0.06mm vs y=z=11.42mm, ratio ~190). A single
    # scalar scale = max(extent) crushes the thin (move) axis to ~5e-3 in
    # coords_net, so the network is effectively blind to thickness-direction
    # position and cannot express the steep thin-axis gradient — its "smooth"
    # field then maps to huge physical strain (energy ~ (L/t)^2 too large, the
    # observed ~2.2e4x K overshoot). Normalizing each axis by its OWN extent
    # puts every axis at O(1) span so the net can resolve all three directions.
    lo, hi = coords.min(0), coords.max(0)
    center = 0.5 * (lo + hi)
    extent = hi - lo
    scale = np.where(extent > 0.0, extent, 1.0)      # (3,) per-axis, guard zero
    scale_iso = float(np.max(extent)) or 1.0         # scalar for distance feats
    coords_net_np = (coords - center) / scale

    # boundary flags
    fixed_flag = np.zeros(n); fixed_flag[bs.fixed_nodes] = 1.0
    move_flag = np.zeros(n); move_flag[bs.move_nodes] = 1.0
    free_flag = 1.0 - fixed_flag - move_flag

    # distances (normalized by the ISOTROPIC scalar so they stay ~O(1)). A
    # Euclidean distance is a single length, not a per-axis quantity, so it must
    # be divided by a scalar — dividing by the per-axis `scale` vector would be
    # dimensionally wrong. scale_iso == the old scalar scale, so this feature is
    # numerically unchanged; only coords_net gains the per-axis treatment.
    d_fixed = _nearest_dist(coords, np.asarray(bs.fixed_nodes)) / scale_iso
    d_move = _nearest_dist(coords, np.asarray(bs.move_nodes)) / scale_iso
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
        scale_iso=scale_iso,
    )


FEATURE_DIM = 10  # length of fx above; transolver functional_dim must match
