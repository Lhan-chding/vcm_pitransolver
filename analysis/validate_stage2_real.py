"""validate_stage2_real.py — run the full Stage-2 pipeline on a REAL variant mesh.

Proves the energy-loss training path works on the actual 430k-DOF spring mesh
(not just the analytic single hex): load_mesh -> features -> FakeBackbone ->
enforce_bc -> cached energy -> train_single, with physics diagnostics. This is
the local pre-server gate; the only thing left for the server is swapping the
FakeBackbone for the real PhysicsNeMo Transolver.

Usage:
    python analysis/validate_stage2_real.py [variant_dir] [--steps N]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "data"))

from parse_face_to_nodes import _load_named_selections, reconstruct_boundary  # noqa: E402
from parse_mesh import load_mesh  # noqa: E402
from fem.material import load_material  # noqa: E402
from fem.pipeline import load_physics  # noqa: E402
from models.features import FEATURE_DIM  # noqa: E402
from models.transolver_wrap import FakeBackbone  # noqa: E402
from train.train_single import TrainConfig, train_single  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("variant", nargs="?",
                    default=str(_REPO / "_devdata" / "VCM_COMPLEX_0001"))
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    torch.manual_seed(0)
    mat = load_material()
    phys = load_physics()
    variant = Path(args.variant)

    t0 = time.perf_counter()
    mesh = load_mesh(variant)
    ns = _load_named_selections(variant)
    bs = reconstruct_boundary(mesh, ns)
    print(f"loaded {variant.name}: {mesh.num_nodes} nodes, {mesh.num_elements} elems, "
          f"{3*mesh.num_nodes} DOFs  ({time.perf_counter()-t0:.1f}s)")
    print(f"  move={bs.move_nodes.size} fixed={bs.fixed_nodes.size}")

    model = FakeBackbone(FEATURE_DIM)
    cfg = TrainConfig(
        move_axis=str(phys["move_axis"]),
        delta_mm=float(phys["delta_mm"]),
        move_transverse_rigid=phys.get("move_transverse", "rigid") == "rigid",
        steps=args.steps, lr=args.lr, log_every=max(1, args.steps // 10),
        dtype=torch.float64,
    )
    t0 = time.perf_counter()
    hist = train_single(model, mesh, bs, mat.C(), E_MPa=mat.E_MPa, nu=mat.nu,
                        cfg=cfg, verbose=True)
    print(f"\ntrained {args.steps} steps in {time.perf_counter()-t0:.1f}s")
    print(f"U: {hist.U[0]:.4e} -> {hist.U[-1]:.4e}  "
          f"(decreased: {hist.U[-1] < hist.U[0]})")
    print(f"K_energy final: {hist.K_energy[-1]:.4e} N/mm   "
          f"K_reaction final: {hist.K_reaction[-1]:.4e} N/mm   "
          f"gap: {hist.rel_gap[-1]:.3e}")
    print("NOTE: FakeBackbone is not a physics model — this validates the PIPELINE "
          "runs on a real mesh, not that it predicts the right K. Real K comes from "
          "the Transolver (server) or direct-FEM (analysis/solve_variant.py).")


if __name__ == "__main__":
    main()
