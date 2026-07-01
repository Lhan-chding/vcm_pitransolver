"""train_transolver_single.py — P7: real PhysicsNeMo Transolver, single geometry.

This is the server counterpart to analysis/validate_stage2_real.py: it swaps the
FakeBackbone for the real Transolver and trains it, on one real variant mesh, to
minimize the FEM elastic energy under hard displacement-control BCs (no labels).

The convergence target is the direct-FEM oracle for the same variant:
    K_energy, K_reaction -> K_oracle  (variant 0001: 3.242570e-02 N/mm)
    rel_gap (|K_energy - K_reaction| / K) -> small   [the unlabeled sanity check]
    |u|max -> delta

Precision policy (A800, user requires high precision): everything runs in
float64 on CUDA — the Transolver, the energy kernels, and the cached geometry
factors. use_te=False (TransformerEngine targets fp8/bf16, incompatible with the
float64 bring-up path). Per the project lesson we watch the physics diagnostics
(U, K, gap, |u|max), never the loss curve alone.

Usage:
    python train/train_transolver_single.py <variant_dir> \
        [--steps N] [--lr LR] [--device cuda] [--dtype float64] \
        [--n-layers L] [--n-hidden H] [--n-head A] [--slice-num S] \
        [--k-oracle 0.03242570] [--out report.json]
"""

from __future__ import annotations

import argparse
import json
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
from models.transolver_wrap import ScaledBackbone, make_transolver  # noqa: E402
from train.train_single import TrainConfig, train_single  # noqa: E402

_DTYPES = {"float64": torch.float64, "float32": torch.float32}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("variant", help="path to the variant directory")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=list(_DTYPES), default="float64")
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--n-hidden", type=int, default=256)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--slice-num", type=int, default=64)
    ap.add_argument("--use-te", action="store_true",
                    help="enable TransformerEngine (leave off for float64 bring-up)")
    ap.add_argument("--k-oracle", type=float, default=None,
                    help="direct-FEM K for this variant (N/mm) to report error against")
    ap.add_argument("--out", default=None, help="write a JSON report to this path")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    dtype = _DTYPES[args.dtype]
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")

    torch.manual_seed(0)
    mat = load_material()
    phys = load_physics()
    variant = Path(args.variant)
    delta = float(phys["delta_mm"])

    t0 = time.perf_counter()
    mesh = load_mesh(variant)
    ns = _load_named_selections(variant)
    bs = reconstruct_boundary(mesh, ns)
    print(f"loaded {variant.name}: {mesh.num_nodes} nodes, {mesh.num_elements} elems, "
          f"{3*mesh.num_nodes} DOFs  ({time.perf_counter()-t0:.1f}s)", flush=True)
    print(f"  move={bs.move_nodes.size} fixed={bs.fixed_nodes.size}  "
          f"delta={delta*1e3:.3f} um axis={phys['move_axis']}", flush=True)

    # Real Transolver, wrapped so its O(1) output is scaled to O(delta) physical mm.
    core = make_transolver(
        FEATURE_DIM,
        n_layers=args.n_layers, n_hidden=args.n_hidden,
        n_head=args.n_head, slice_num=args.slice_num, use_te=args.use_te,
    )
    model = ScaledBackbone(core, scale=delta)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Transolver: n_layers={args.n_layers} n_hidden={args.n_hidden} "
          f"n_head={args.n_head} slice_num={args.slice_num} use_te={args.use_te} "
          f"params={n_params/1e6:.2f}M  dtype={args.dtype} device={device}", flush=True)

    cfg = TrainConfig(
        move_axis=str(phys["move_axis"]),
        delta_mm=delta,
        move_transverse_rigid=phys.get("move_transverse", "rigid") == "rigid",
        steps=args.steps, lr=args.lr, log_every=max(1, args.steps // 20),
        dtype=dtype, device=device,
    )

    t0 = time.perf_counter()
    hist = train_single(model, mesh, bs, mat.C(), E_MPa=mat.E_MPa, nu=mat.nu,
                        cfg=cfg, verbose=True)
    train_s = time.perf_counter() - t0

    k_energy = hist.K_energy[-1]
    k_reaction = hist.K_reaction[-1]
    print("\n" + "=" * 64)
    print(f"trained {args.steps} steps in {train_s:.1f}s "
          f"({1e3*train_s/max(args.steps,1):.1f} ms/step)")
    print(f"U:          {hist.U[0]:.6e} -> {hist.U[-1]:.6e}")
    print(f"K_energy:   {k_energy:.6e} N/mm")
    print(f"K_reaction: {k_reaction:.6e} N/mm")
    print(f"rel_gap:    {hist.rel_gap[-1]:.3e}")
    print(f"|u|max:     {hist.u_max[-1]:.6e} mm  (delta={delta:.3e})")
    if args.k_oracle:
        e_e = abs(k_energy - args.k_oracle) / args.k_oracle
        e_r = abs(k_reaction - args.k_oracle) / args.k_oracle
        print(f"vs oracle K={args.k_oracle:.6e} N/mm: "
              f"K_energy err {e_e*100:.3f}%  K_reaction err {e_r*100:.3f}%")
    print("=" * 64)

    if args.out:
        report = {
            "variant": variant.name,
            "num_nodes": mesh.num_nodes, "num_dofs": 3 * mesh.num_nodes,
            "steps": args.steps, "lr": args.lr,
            "dtype": args.dtype, "device": device,
            "model": {"n_layers": args.n_layers, "n_hidden": args.n_hidden,
                      "n_head": args.n_head, "slice_num": args.slice_num,
                      "use_te": args.use_te, "params": n_params},
            "train_seconds": train_s,
            "history": {
                "step": hist.step, "U": hist.U,
                "K_energy": hist.K_energy, "K_reaction": hist.K_reaction,
                "rel_gap": hist.rel_gap, "u_max": hist.u_max,
            },
            "final": {"K_energy": k_energy, "K_reaction": k_reaction,
                      "rel_gap": hist.rel_gap[-1], "u_max": hist.u_max[-1]},
            "k_oracle": args.k_oracle,
        }
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"wrote report -> {args.out}")


if __name__ == "__main__":
    main()
