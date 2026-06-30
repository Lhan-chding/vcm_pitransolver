"""solve_variant.py — end-to-end FEM solve for one variant; print the K value.

This is the project's first physical result: load a real mesh, reconstruct the
boundary node sets, assemble the global stiffness with the confirmed C1990
material, apply displacement-control BCs from config/physics.yaml, solve K u = f,
and report the axial stiffness K (two independent estimates) plus stress range.

Usage:
    python analysis/solve_variant.py [variant_dir]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "data"))

from parse_face_to_nodes import _load_named_selections, reconstruct_boundary  # noqa: E402
from parse_mesh import load_mesh  # noqa: E402

from fem.assembly import assemble_global_stiffness, num_dofs  # noqa: E402
from fem.direct_solver import solve_displacement  # noqa: E402
from fem.material import load_material  # noqa: E402
from fem.postprocess import compute_stiffness, von_mises_nodal  # noqa: E402


def _load_physics() -> dict:
    with open(REPO / "config" / "physics.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)["loading"]


def main() -> None:
    variant = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "_devdata" / "VCM_COMPLEX_0001"
    phys = _load_physics()
    mat = load_material()  # C1990

    t0 = time.perf_counter()
    mesh = load_mesh(variant)
    ns = _load_named_selections(variant)
    bs = reconstruct_boundary(mesh, ns)
    t_mesh = time.perf_counter() - t0

    C = mat.C()
    t0 = time.perf_counter()
    K = assemble_global_stiffness(mesh, C)
    t_assemble = time.perf_counter() - t0

    move_transverse_rigid = phys.get("move_transverse", "rigid") == "rigid"
    t0 = time.perf_counter()
    res = solve_displacement(
        mesh, bs, C,
        move_axis=phys["move_axis"],
        delta_mm=float(phys["delta_mm"]),
        move_transverse_rigid=move_transverse_rigid,
        K=K,
    )
    t_solve = time.perf_counter() - t0

    stiff = compute_stiffness(mesh, res, C)
    vm = von_mises_nodal(mesh, res.u, C)
    u_mag = np.linalg.norm(res.u, axis=1)

    print("=" * 64)
    print(f"VARIANT : {variant.name}")
    print(f"material: {mat.name}  E={mat.E_MPa:.0f} MPa  nu={mat.nu}")
    print("-" * 64)
    print(f"nodes / elements    : {mesh.num_nodes} / {mesh.num_elements}")
    print(f"element families    : {mesh.type_histogram()}")
    print(f"global DOFs         : {num_dofs(mesh)}")
    print(f"move / fixed nodes  : {bs.move_nodes.size} / {bs.fixed_nodes.size}")
    print(f"move face area ok   : {bs.diagnostics['move_area_ok']} "
          f"(rel err {bs.diagnostics['move_area_rel_err']:.3f})")
    print("-" * 64)
    print(f"prescribed delta    : {res.delta_mm*1e3:.3f} um (axis {phys['move_axis']})")
    print(f"K_energy   = 2U/d^2 : {stiff.K_energy:.6e} N/mm")
    print(f"K_reaction = F/d    : {stiff.K_reaction:.6e} N/mm")
    print(f"  energy/reaction gap: {stiff.rel_gap:.3e}  (should be ~0)")
    print(f"strain energy U     : {stiff.total_strain_energy:.6e} N*mm")
    print(f"move-face force     : {stiff.move_face_force:.6e} N")
    print("-" * 64)
    print(f"|u| max / mean      : {u_mag.max():.4e} / {u_mag.mean():.4e} mm")
    print(f"von Mises max/mean  : {vm.max():.4e} / {vm.mean():.4e} MPa")
    print(f"  (yield {mat.tensile_yield_MPa} MPa -> "
          f"{100*vm.max()/mat.tensile_yield_MPa:.2f}% of yield at this delta)")
    print("-" * 64)
    print(f"timing: mesh+bc {t_mesh:.2f}s | assemble {t_assemble:.2f}s | "
          f"solve {t_solve:.2f}s")
    print("=" * 64)


if __name__ == "__main__":
    main()
