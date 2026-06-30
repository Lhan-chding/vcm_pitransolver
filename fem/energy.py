"""energy.py — torch differentiable elastic strain energy (Transolver loss).

This is the physics-informed training signal: U(u) for a displacement field u,
differentiable w.r.t. u so the network can minimize it. It reuses the EXACT same
B/C/Gauss mathematics as the numpy engine (fem.elements.common /
fem.postprocess.total_strain_energy); a consistency test pins the two to <1e-10.

Conventions (must match fem.elements.common, verified against it):
  Voigt order      : [xx, yy, zz, xy, yz, zx]
  shear            : ENGINEERING shear, gamma = 2*eps; energy is 1/2 eps^T C eps
                     with NO extra factor of 2 on shear (C shear diagonal = mu).
  element DOF order : node-major; per element u_e is (nnode, 3), gathered as u[conn].
  energy           : U = sum_family sum_elem sum_gauss  w_g * detJ_g * 1/2 eps^T C eps,
                     eps = strain from the physical-gradient of u.

Only u carries gradients. coords and the natural-coord shape gradients are
constants, so for the single-geometry training regime the geometry-derived
tensors (detJ, dN_dx) are constant across steps and can be precomputed once — the
cached path does exactly that; the uncached path recomputes them and is the
correctness reference that mirrors the numpy engine step for step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from parse_mesh import ELEM_TYPES, Mesh
from fem.elements import hex20, tet10, wedge15

_FAMILY = {10: tet10, 15: wedge15, 20: hex20}


# --------------------------------------------------------------------------- #
# kernels                                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class FamilyKernelUncached:
    """Per-family constants that do NOT depend on geometry-derived J: only the
    natural-coord shape gradients, Gauss weights, and connectivity. J/detJ/dN_dx
    are recomputed from coords each call (the numpy-equivalent reference path)."""

    family: str
    conn: torch.Tensor          # (M, nnode) long
    dN_dnat: torch.Tensor       # (G, nnode, 3) natural-coord shape gradients
    weights: torch.Tensor       # (G,)


@dataclass
class FamilyKernelCached:
    """Per-family constants WITH the geometry baked in: detJ and physical-coord
    shape gradients dN_dx are precomputed (valid while coords are fixed). Training
    then only gathers u[conn] and contracts — no det/solve per step."""

    family: str
    conn: torch.Tensor          # (M, nnode) long
    dN_dx: torch.Tensor         # (M, G, nnode, 3) physical shape gradients
    w_detJ: torch.Tensor        # (M, G) = weight_g * detJ_g  (the quadrature factor)


def _stack_dN_dnat(module, dtype, device) -> torch.Tensor:
    dN = np.stack([module.shape_grads(nat) for nat, _w in module.GAUSS])  # (G,nnode,3)
    return torch.as_tensor(dN, dtype=dtype, device=device)


def _weights(module, dtype, device) -> torch.Tensor:
    w = np.array([w for _nat, w in module.GAUSS], dtype=np.float64)
    return torch.as_tensor(w, dtype=dtype, device=device)


def build_family_kernels_uncached(
    mesh: Mesh, device="cpu", dtype=torch.float64
) -> dict[int, FamilyKernelUncached]:
    """Per-family uncached kernels (correctness reference; recompute J each call)."""
    out: dict[int, FamilyKernelUncached] = {}
    for nnode, module in _FAMILY.items():
        conn = mesh.elements_of(nnode)
        if conn.shape[0] == 0:
            continue
        out[nnode] = FamilyKernelUncached(
            family=ELEM_TYPES[nnode],
            conn=torch.as_tensor(conn, dtype=torch.long, device=device),
            dN_dnat=_stack_dN_dnat(module, dtype, device),
            weights=_weights(module, dtype, device),
        )
    return out


def build_family_kernels_cached(
    mesh: Mesh, device="cpu", dtype=torch.float64
) -> dict[int, FamilyKernelCached]:
    """Per-family cached kernels: precompute dN_dx and w*detJ from coords once.

    Use for single-geometry training where coords are fixed across steps.
    """
    coords = torch.as_tensor(mesh.coords, dtype=dtype, device=device)
    out: dict[int, FamilyKernelCached] = {}
    for nnode, module in _FAMILY.items():
        conn_np = mesh.elements_of(nnode)
        if conn_np.shape[0] == 0:
            continue
        conn = torch.as_tensor(conn_np, dtype=torch.long, device=device)
        dN_dnat = _stack_dN_dnat(module, dtype, device)     # (G,nnode,3)
        weights = _weights(module, dtype, device)           # (G,)
        xe = coords[conn]                                   # (M,nnode,3)
        dN_dx, detJ = _phys_grads_and_detj(xe, dN_dnat)     # (M,G,nnode,3),(M,G)
        out[nnode] = FamilyKernelCached(
            family=ELEM_TYPES[nnode],
            conn=conn,
            dN_dx=dN_dx,
            w_detJ=weights[None, :] * detJ,
        )
    return out


# --------------------------------------------------------------------------- #
# geometry + strain                                                            #
# --------------------------------------------------------------------------- #
def _phys_grads_and_detj(
    xe: torch.Tensor, dN_dnat: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Physical shape gradients dN_dx and detJ for a batch of elements.

    xe:      (M, nnode, 3) element node coords
    dN_dnat: (G, nnode, 3) natural-coord shape gradients
    returns  dN_dx (M, G, nnode, 3), detJ (M, G)

    J[m,g,i,j] = d x_i / d xi_j = sum_n xe[m,n,i] * dN_dnat[g,n,j]
    dN/dx = dN/dxi @ J^{-1}; solved as J^T X = dN_dnat^T (no explicit inverse).
    """
    J = torch.einsum("mni,gnj->mgij", xe, dN_dnat)          # (M,G,3,3)
    detJ = torch.linalg.det(J)                              # (M,G)
    M = xe.shape[0]
    dN_b = dN_dnat.unsqueeze(0).expand(M, -1, -1, -1)       # (M,G,nnode,3)
    # solve J^T y = dN_dnat^T  ->  y = J^{-T} dN_dnat^T ; transpose back to (...,nnode,3)
    dN_dx = torch.linalg.solve(
        J.transpose(-1, -2).unsqueeze(2),                   # (M,G,1,3,3)
        dN_b.unsqueeze(-1),                                 # (M,G,nnode,3,1)
    ).squeeze(-1)                                           # (M,G,nnode,3)
    return dN_dx, detJ


def _strain_from_grad(dN_dx: torch.Tensor, ue: torch.Tensor) -> torch.Tensor:
    """Engineering-shear Voigt strain from physical gradients.

    dN_dx: (M, G, nnode, 3), ue: (M, nnode, 3) -> eps: (M, G, 6).
    grad[i,j] = du_i/dx_j = sum_n dN_dx[...,n,j] * ue[...,n,i]
    Voigt rows match common.B_matrix exactly: gamma_xy = du/dy + dv/dx, etc.
    """
    grad = torch.einsum("mgnj,mni->mgij", dN_dx, ue)        # (M,G,3,3)
    exx, eyy, ezz = grad[..., 0, 0], grad[..., 1, 1], grad[..., 2, 2]
    gxy = grad[..., 0, 1] + grad[..., 1, 0]
    gyz = grad[..., 1, 2] + grad[..., 2, 1]
    gzx = grad[..., 2, 0] + grad[..., 0, 2]
    return torch.stack([exx, eyy, ezz, gxy, gyz, gzx], dim=-1)


# --------------------------------------------------------------------------- #
# energy                                                                       #
# --------------------------------------------------------------------------- #
def elastic_energy_uncached(
    u: torch.Tensor,
    coords: torch.Tensor,
    kernels: dict[int, FamilyKernelUncached],
    C: torch.Tensor,
) -> torch.Tensor:
    """U(u) recomputing J/detJ/dN_dx each call. Reference path (mirrors numpy)."""
    total = u.new_zeros(())
    for k in kernels.values():
        xe = coords[k.conn]                                 # (M,nnode,3)
        ue = u[k.conn]                                      # (M,nnode,3)
        dN_dx, detJ = _phys_grads_and_detj(xe, k.dN_dnat)
        eps = _strain_from_grad(dN_dx, ue)                  # (M,G,6)
        sig = torch.einsum("ij,mgj->mgi", C, eps)           # (M,G,6)
        dens = 0.5 * (eps * sig).sum(-1)                    # (M,G)
        total = total + (k.weights[None, :] * detJ * dens).sum()
    return total


def elastic_energy_cached(
    u: torch.Tensor,
    kernels: dict[int, FamilyKernelCached],
    C: torch.Tensor,
) -> torch.Tensor:
    """U(u) using precomputed dN_dx and w*detJ. Training path (no det/solve)."""
    total = u.new_zeros(())
    for k in kernels.values():
        ue = u[k.conn]                                      # (M,nnode,3)
        eps = _strain_from_grad(k.dN_dx, ue)                # (M,G,6)
        sig = torch.einsum("ij,mgj->mgi", C, eps)           # (M,G,6)
        dens = 0.5 * (eps * sig).sum(-1)                    # (M,G)
        total = total + (k.w_detJ * dens).sum()
    return total


def material_C_torch(C_np: np.ndarray, device="cpu", dtype=torch.float64) -> torch.Tensor:
    """Reuse the numpy (6,6) constitutive matrix as a tensor (exact, no recompute)."""
    return torch.as_tensor(np.asarray(C_np, dtype=np.float64), dtype=dtype, device=device)
