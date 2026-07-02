"""diagnose_energy_gap.py — pin down WHY trained K is ~22000x the oracle.

The Adam+L-BFGS run converged (U dropped 2e5x, K_energy/K_reaction gap ~5%) but
landed at K ~ 727 N/mm vs oracle ~0.032 N/mm — a 22439x overshoot, with |u|max
still ~= delta. That signature (displacement magnitude RIGHT, energy 22439x too
BIG, both estimators agreeing) says the network found a self-consistent but
UNDER-CONSTRAINED equilibrium: a rough interior field that satisfies the BCs yet
carries huge spurious internal strain. This script tests that hypothesis without
touching training or PhysicsNeMo — it uses ONLY the numpy oracle + the torch
energy kernel that Stage-1 already validated to <1e-10.

Four checks, each falsifies one root cause:

  [1] ORACLE SCALE   feed the direct-FEM u* into elastic_energy_cached.
                     Expect U* ~= 4.05e-7, K ~= oracle. If yes -> energy kernel
                     is correct, blame is 100% the network field. If no -> a
                     unit/scale bug (fix that FIRST).

  [2] ENERGY LOCALITY  per-element energy of the oracle field vs a "dirty" field
                     (same BCs, random interior). Shows how a rough interior
                     inflates U and concentrates it in a thin element layer.

  [3] DISPLACEMENT PROFILE  u along the move axis for the oracle: the smooth
                     low-frequency transition the network FAILED to find.

  [4] ANISOTROPY     bbox extents per axis. features.py normalizes coords by a
                     single scalar max(extent); a thin plate then gets crushed
                     on its thin axis, so a "smooth" net field maps to steep
                     physical gradients -> inflated energy.

Usage (run in the A800 container, where the real variant mesh lives):
    python analysis/diagnose_energy_gap.py <variant_dir> \
        [--k-oracle 0.03242570] [--device cpu] [--dtype float64] [--seed 0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "data"))

from parse_face_to_nodes import _load_named_selections, reconstruct_boundary  # noqa: E402
from parse_mesh import load_mesh  # noqa: E402
from fem.material import load_material  # noqa: E402
from fem.pipeline import load_physics  # noqa: E402
from fem.direct_solver import solve_displacement  # noqa: E402
from fem.energy import (  # noqa: E402
    build_family_kernels_cached,
    elastic_energy_cached,
    material_C_torch,
)
from models.bc_enforce import build_bc_tensors, enforce_bc  # noqa: E402

_AXIS = {"x": 0, "y": 1, "z": 2}


def _per_element_energy(u, kernels, C):
    """Return {family: (M,) per-element strain energy} for a displacement field.

    Mirrors elastic_energy_cached but keeps the sum at element granularity so we
    can see WHERE the energy lives (locality is the tell for a rough field).
    """
    from fem.energy import _strain_from_grad  # local import: private helper
    out = {}
    for nnode, k in kernels.items():
        ue = u[k.conn]                                  # (M,nnode,3)
        eps = _strain_from_grad(k.dN_dx, ue)            # (M,G,6)
        sig = torch.einsum("ij,mgj->mgi", C, eps)       # (M,G,6)
        dens = 0.5 * (eps * sig).sum(-1)                # (M,G)
        u_e = (k.w_detJ * dens).sum(dim=1)              # (M,) per-element energy
        out[k.family] = u_e
    return out


def _total_from_per_elem(per_elem) -> float:
    return float(sum(v.sum() for v in per_elem.values()))


def _concentration(per_elem) -> dict:
    """How concentrated is the energy? Fraction carried by the top-k% of elements."""
    all_e = torch.cat([v.reshape(-1) for v in per_elem.values()])
    all_e = torch.clamp(all_e, min=0.0)                 # guard tiny negatives (roundoff)
    total = float(all_e.sum())
    if total <= 0:
        return {"n_elem": int(all_e.numel()), "total": total}
    srt, _ = torch.sort(all_e, descending=True)
    n = srt.numel()
    frac = {}
    for pct in (0.1, 1.0, 5.0):
        k = max(1, int(n * pct / 100.0))
        frac[f"top_{pct}pct"] = float(srt[:k].sum()) / total
    return {"n_elem": n, "total": total, **frac}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("variant", help="path to the variant directory")
    ap.add_argument("--k-oracle", type=float, default=None,
                    help="direct-FEM K for this variant (N/mm) to compare against")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    dev = args.device
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    mat = load_material()
    phys = load_physics()
    delta = float(phys["delta_mm"])
    move_axis = str(phys["move_axis"])
    axis = _AXIS[move_axis.lower()]
    transverse_rigid = phys.get("move_transverse", "rigid") == "rigid"

    variant = Path(args.variant)
    mesh = load_mesh(variant)
    ns = _load_named_selections(variant)
    bs = reconstruct_boundary(mesh, ns)
    print(f"variant {variant.name}: {mesh.num_nodes} nodes, {mesh.num_elements} elems, "
          f"{3*mesh.num_nodes} DOFs")
    print(f"  move={bs.move_nodes.size} fixed={bs.fixed_nodes.size} "
          f"delta={delta:.3e} mm  axis={move_axis}")

    # ---- shared torch kernels + BC tensors (same problem the trainer solves) -- #
    kernels = build_family_kernels_cached(mesh, device=dev, dtype=dtype)
    C = material_C_torch(mat.C(), device=dev, dtype=dtype)
    free_mask, u_pre = build_bc_tensors(
        bs, mesh.num_nodes, move_axis=move_axis, delta_mm=delta,
        move_transverse_rigid=transverse_rigid, device=dev, dtype=dtype,
    )
    delta2 = delta * delta

    def K_energy_of(u_tensor) -> tuple[float, float]:
        """(U, K_energy) for an (N,3) displacement tensor."""
        U = float(elastic_energy_cached(u_tensor, kernels, C))
        return U, 2.0 * U / delta2

    # ======================================================================== #
    # [1] ORACLE SCALE — the decisive cut                                       #
    # ======================================================================== #
    print("\n" + "=" * 70)
    print("[1] ORACLE SCALE CHECK  (energy kernel vs network field)")
    print("=" * 70)
    res = solve_displacement(
        mesh, bs, mat.C(), move_axis=move_axis, delta_mm=delta,
        move_transverse_rigid=transverse_rigid, backend="auto",
    )
    u_star = torch.as_tensor(res.u, dtype=dtype, device=dev)   # (N,3) oracle field
    U_star, K_star = K_energy_of(u_star)
    # reaction-based K from the oracle's own reaction vector (independent path)
    move_rows = np.asarray(bs.move_nodes, dtype=np.int64)
    react = res.reaction.reshape(mesh.num_nodes, 3)
    K_react_oracle = float(react[move_rows, axis].sum()) / delta

    print(f"  U*  (oracle field through torch energy) = {U_star:.6e}  N*mm")
    print(f"  K_energy   from U*                       = {K_star:.6e}  N/mm")
    print(f"  K_reaction from oracle reaction          = {K_react_oracle:.6e}  N/mm")
    if args.k_oracle:
        print(f"  K_oracle (reported)                      = {args.k_oracle:.6e}  N/mm")
        err = abs(K_star - args.k_oracle) / args.k_oracle
        print(f"  |K_energy(U*) - K_oracle| / K_oracle     = {err*100:.4f}%")
        verdict = ("KERNEL OK -> blame the NETWORK FIELD" if err < 1e-2
                   else "KERNEL/SCALE BUG -> fix energy or units FIRST")
        print(f"  VERDICT: {verdict}")

    # ======================================================================== #
    # [2] ENERGY LOCALITY — oracle vs a BC-satisfying DIRTY field               #
    # ======================================================================== #
    print("\n" + "=" * 70)
    print("[2] ENERGY LOCALITY  (smooth oracle vs BC-correct, rough interior)")
    print("=" * 70)
    # dirty field: prescribed at the SAME O(delta) magnitude on free interior
    # nodes but spatially random -> satisfies BC after enforce_bc, mimics the
    # 'edges pinned, interior wanders' failure mode the trainer can fall into.
    u_rand = torch.as_tensor(
        rng.normal(scale=delta, size=(mesh.num_nodes, 3)), dtype=dtype, device=dev,
    )
    u_dirty = enforce_bc(u_rand, free_mask, u_pre)     # BCs hard-enforced, same as trainer
    U_dirty, K_dirty = K_energy_of(u_dirty)

    pe_star = _per_element_energy(u_star, kernels, C)
    pe_dirty = _per_element_energy(u_dirty, kernels, C)
    conc_star = _concentration(pe_star)
    conc_dirty = _concentration(pe_dirty)

    print(f"  oracle field : U={U_star:.6e}  K={K_star:.6e}")
    print(f"  dirty field  : U={U_dirty:.6e}  K={K_dirty:.6e}  "
          f"(K_dirty / K_oracle_field = {K_dirty/max(K_star,1e-30):.3e}x)")
    print("  energy concentration (fraction of U carried by hottest elements):")
    for tag, c in (("oracle", conc_star), ("dirty ", conc_dirty)):
        keys = [k for k in c if k.startswith("top_")]
        s = "  ".join(f"{k}={c[k]*100:6.2f}%" for k in keys)
        print(f"    {tag}: n_elem={c['n_elem']:>7d}  {s}")
    print("  -> if the trained field behaves like 'dirty' (energy in a few hot "
          "elements),\n     the fix is smoothness/PDE-residual regularization, "
          "not a scale change.")

    # ======================================================================== #
    # [3] DISPLACEMENT PROFILE — the smooth transition the net should learn      #
    # ======================================================================== #
    print("\n" + "=" * 70)
    print("[3] DISPLACEMENT PROFILE  (oracle u along move axis)")
    print("=" * 70)
    coords = mesh.coords
    pos = coords[:, axis]
    u_ax = res.u[:, axis]
    order = np.argsort(pos)
    nb = 12
    edges = np.linspace(pos.min(), pos.max(), nb + 1)
    print(f"  binning nodes along axis '{move_axis}' into {nb} slabs "
          f"(pos in mm, u_axis in mm):")
    print(f"    {'pos_lo':>10} {'pos_hi':>10} {'n':>7} {'u_mean':>12} "
          f"{'u_min':>12} {'u_max':>12}")
    for i in range(nb):
        m = (pos >= edges[i]) & (pos <= edges[i + 1] if i == nb - 1 else pos < edges[i + 1])
        if not m.any():
            continue
        seg = u_ax[m]
        print(f"    {edges[i]:10.4f} {edges[i+1]:10.4f} {m.sum():7d} "
              f"{seg.mean():12.4e} {seg.min():12.4e} {seg.max():12.4e}")
    print("  -> a healthy field ramps ~monotonically from 0 (fixed) to delta "
          "(move);\n     kinks/oscillation here would confirm a rough solution.")

    # ======================================================================== #
    # [4] ANISOTROPY — single-scalar normalization health                       #
    # ======================================================================== #
    print("\n" + "=" * 70)
    print("[4] ANISOTROPY  (features.py normalizes by a SINGLE scalar)")
    print("=" * 70)
    lo, hi = coords.min(0), coords.max(0)
    ext = hi - lo
    smax = float(ext.max())
    print(f"  bbox extents (mm): x={ext[0]:.4f}  y={ext[1]:.4f}  z={ext[2]:.4f}")
    print(f"  normalization scale = max extent = {smax:.4f} mm")
    print(f"  per-axis normalized span after /scale: "
          f"x={ext[0]/smax:.3e}  y={ext[1]/smax:.3e}  z={ext[2]/smax:.3e}")
    ratio = ext.max() / max(ext.min(), 1e-30)
    print(f"  anisotropy ratio (max/min extent) = {ratio:.2f}")
    if ratio > 10:
        print("  -> STRONG anisotropy: the thin axis is crushed to ~0 in "
              "coords_net.\n     A 'smooth' network field maps to steep physical "
              "gradients on that axis\n     -> inflated energy. Consider per-axis "
              "normalization in features.py.")
    else:
        print("  -> mild anisotropy; single-scalar normalization is probably "
              "not the main driver.")

    print("\n" + "=" * 70)
    print("DONE. Read [1] first: it decides kernel-vs-network. Then [2]/[4] "
          "tell you\nwhich fix (regularization vs per-axis norm) to reach for.")
    print("=" * 70)


if __name__ == "__main__":
    main()
