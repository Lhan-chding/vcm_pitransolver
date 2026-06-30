"""pipeline.py — one-call FEM solve for a variant, with health diagnostics.

Wraps the Stage-1 steps (load -> reconstruct boundary -> assemble -> solve ->
post-process) into a single function returning a structured result, plus the
sanity diagnostics that decide whether a variant's pseudo-label is trustworthy:

  * connectivity      : merged mesh must be a single component (else the solve is
                        a zero-energy mechanism — the bug found on variant 0001).
  * energy/reaction gap: must be ~0 (Clapeyron); a large gap means a broken solve.
  * move-face area err : reconstructed move strip vs reported STEP face area.
  * von Mises vs yield : should stay in the linear-elastic range at the chosen Δ.

Both analysis/solve_variant.py (single, verbose) and the batch runner build on
this so there is one solve path, not two that can drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import yaml
from scipy.sparse.csgraph import connected_components

from parse_face_to_nodes import _load_named_selections, reconstruct_boundary
from parse_mesh import Mesh, load_mesh

from fem.assembly import assemble_global_stiffness, num_dofs
from fem.direct_solver import solve_displacement
from fem.material import Material, load_material
from fem.postprocess import compute_element_max_von_mises, compute_stiffness, von_mises_nodal

_REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class VariantSolve:
    """Everything one variant's FEM solve produces, fields + diagnostics."""

    variant_id: str
    num_nodes: int
    num_elements: int
    num_dofs: int
    type_histogram: dict
    n_move_nodes: int
    n_fixed_nodes: int
    n_components: int               # connected components of the (merged) mesh
    move_area_rel_err: float
    move_area_ok: bool
    delta_mm: float
    move_axis: str
    K_energy: float
    K_reaction: float
    rel_gap: float
    total_strain_energy: float
    move_face_force: float
    u_max: float
    u_mean: float
    vm_max: float                   # nodal-averaged peak (smoothed; visualization)
    vm_mean: float
    vm_frac_yield: float            # based on nodal vm_max
    vm_max_element: float           # element/Gauss peak (UNSMEARED; the real max stress)
    vm_frac_yield_element: float    # based on the element-level peak — use this for safety
    # heavy fields (not for CSV): the full solved displacement / stress.
    u: np.ndarray | None = field(default=None, repr=False)
    von_mises: np.ndarray | None = field(default=None, repr=False)


def load_physics(path: str | Path | None = None) -> dict:
    p = Path(path) if path else _REPO / "config" / "physics.yaml"
    with open(p, encoding="utf-8") as fh:
        return yaml.safe_load(fh)["loading"]


def count_components(mesh: Mesh) -> int:
    """Number of connected components of the mesh node graph (via element edges)."""
    n = mesh.num_nodes
    r, c = [], []
    for row in mesh.connectivity:
        row = np.asarray(row)
        hub = int(row[0])
        for a in row:
            r.append(int(a))
            c.append(hub)
    A = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    A = A + A.T
    return int(connected_components(A, directed=False)[0])


def solve_variant(
    variant_dir: str | Path,
    *,
    material: Material | None = None,
    physics: dict | None = None,
    keep_fields: bool = False,
) -> VariantSolve:
    """Run the full FEM pipeline for one variant and return a VariantSolve.

    Args:
        variant_dir: directory with nodes.csv / elements.csv / named_selections.json.
        material:    Material to use; defaults to load_material() (C1990).
        physics:     loading config dict; defaults to config/physics.yaml.
        keep_fields: if True, attach the (N,3) displacement and nodal von Mises to
                     the result (for saving). If False, only scalars are kept.

    Raises:
        Whatever load_mesh / reconstruct_boundary / solve raise on bad data — the
        batch runner catches per-variant so one bad variant doesn't abort the run.
    """
    variant_dir = Path(variant_dir)
    mat = material or load_material()
    phys = physics or load_physics()
    C = mat.C()

    mesh = load_mesh(variant_dir)               # merge_coincident=True by default
    ns = _load_named_selections(variant_dir)
    bs = reconstruct_boundary(mesh, ns)
    ncomp = count_components(mesh)

    delta = float(phys["delta_mm"])
    axis = str(phys["move_axis"])
    transverse_rigid = phys.get("move_transverse", "rigid") == "rigid"

    res = solve_displacement(
        mesh, bs, C,
        move_axis=axis,
        delta_mm=delta,
        move_transverse_rigid=transverse_rigid,
    )
    stiff = compute_stiffness(mesh, res, C)
    vm = von_mises_nodal(mesh, res.u, C)
    _, vm_elem = compute_element_max_von_mises(mesh, res.u, C)
    vm_max_elem = float(vm_elem.max()) if vm_elem.size else float("nan")
    u_mag = np.linalg.norm(res.u, axis=1)

    return VariantSolve(
        variant_id=variant_dir.name,
        num_nodes=mesh.num_nodes,
        num_elements=mesh.num_elements,
        num_dofs=num_dofs(mesh),
        type_histogram=mesh.type_histogram(),
        n_move_nodes=int(bs.move_nodes.size),
        n_fixed_nodes=int(bs.fixed_nodes.size),
        n_components=ncomp,
        move_area_rel_err=float(bs.diagnostics["move_area_rel_err"]),
        move_area_ok=bool(bs.diagnostics["move_area_ok"]),
        delta_mm=delta,
        move_axis=axis,
        K_energy=stiff.K_energy,
        K_reaction=stiff.K_reaction,
        rel_gap=stiff.rel_gap,
        total_strain_energy=stiff.total_strain_energy,
        move_face_force=stiff.move_face_force,
        u_max=float(u_mag.max()),
        u_mean=float(u_mag.mean()),
        vm_max=float(vm.max()),
        vm_mean=float(vm.mean()),
        vm_frac_yield=(float(vm.max()) / mat.tensile_yield_MPa
                       if mat.tensile_yield_MPa else float("nan")),
        vm_max_element=vm_max_elem,
        vm_frac_yield_element=(vm_max_elem / mat.tensile_yield_MPa
                               if mat.tensile_yield_MPa else float("nan")),
        u=res.u if keep_fields else None,
        von_mises=vm if keep_fields else None,
    )
