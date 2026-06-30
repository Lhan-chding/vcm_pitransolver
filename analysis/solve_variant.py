"""solve_variant.py — end-to-end FEM solve for ONE variant; print K + diagnostics.

Thin verbose wrapper over fem.pipeline.solve_variant (the single shared solve
path, also used by analysis/batch_solve.py). This is the human-facing view: it
prints the K value, the energy/reaction consistency check, displacement range and
von Mises vs yield for a quick physical sanity read on a single variant.

Usage:
    python analysis/solve_variant.py [variant_dir]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "data"))

from fem.material import load_material  # noqa: E402
from fem.pipeline import load_physics, solve_variant  # noqa: E402


def main() -> None:
    variant = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "_devdata" / "VCM_COMPLEX_0001"
    mat = load_material()
    phys = load_physics()

    t0 = time.perf_counter()
    vs = solve_variant(variant, material=mat, physics=phys, keep_fields=False)
    elapsed = time.perf_counter() - t0

    print("=" * 64)
    print(f"VARIANT : {vs.variant_id}")
    print(f"material: {mat.name}  E={mat.E_MPa:.0f} MPa  nu={mat.nu}")
    print("-" * 64)
    print(f"nodes / elements    : {vs.num_nodes} / {vs.num_elements}")
    print(f"element families    : {vs.type_histogram}")
    print(f"global DOFs         : {vs.num_dofs}")
    print(f"mesh components     : {vs.n_components}  (must be 1 for a load path)")
    print(f"move / fixed nodes  : {vs.n_move_nodes} / {vs.n_fixed_nodes}")
    print(f"move face area ok   : {vs.move_area_ok} (rel err {vs.move_area_rel_err:.3f})")
    print("-" * 64)
    print(f"prescribed delta    : {vs.delta_mm*1e3:.3f} um (axis {vs.move_axis})")
    print(f"K_energy   = 2U/d^2 : {vs.K_energy:.6e} N/mm")
    print(f"K_reaction = F/d    : {vs.K_reaction:.6e} N/mm")
    print(f"  energy/reaction gap: {vs.rel_gap:.3e}  (should be ~0)")
    print(f"strain energy U     : {vs.total_strain_energy:.6e} N*mm")
    print(f"move-face force     : {vs.move_face_force:.6e} N")
    print("-" * 64)
    print(f"|u| max / mean      : {vs.u_max:.4e} / {vs.u_mean:.4e} mm")
    print(f"von Mises max/mean  : {vs.vm_max:.4e} / {vs.vm_mean:.4e} MPa")
    print(f"  ({vs.vm_frac_yield*100:.2f}% of yield {mat.tensile_yield_MPa} MPa at this delta)")
    print("-" * 64)
    print(f"total solve time    : {elapsed:.2f}s")
    print("=" * 64)


if __name__ == "__main__":
    main()
