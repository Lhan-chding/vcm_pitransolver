"""bench_direct_fem.py — per-stage runtime + peak-memory benchmark for direct-FEM.

Measures the real cost of each stage (load_mesh / node_merge / boundary /
assembly / solve / postprocess) plus peak RSS, on baseline + representative
variants. The "direct-FEM is seconds" assumption is NOT taken on faith — this
reports the actual numbers so the pseudo-label generation cost (and whether a
faster sparse backend is warranted) is grounded in data.

Node-merge time is isolated by timing a raw (no-merge) load and a merged load and
differencing. The solver backend is recorded (scipy by default; pypardiso /
umfpack / cholmod are tried via --backend auto if installed).

Outputs reports/direct_fem_performance.csv.

Usage:
    python analysis/bench_direct_fem.py [zip_path] [--variants A,B,...] [--backend auto]
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import psutil

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "data"))

from parse_face_to_nodes import _load_named_selections, reconstruct_boundary  # noqa: E402
from parse_mesh import load_mesh  # noqa: E402
from fem.assembly import assemble_global_stiffness, num_dofs  # noqa: E402
from fem.direct_solver import solve_displacement  # noqa: E402
from fem.material import load_material  # noqa: E402
from fem.pipeline import load_physics  # noqa: E402
from fem.postprocess import (  # noqa: E402
    compute_element_max_von_mises, compute_stiffness, von_mises_nodal,
)

_NEEDED = ("nodes.csv", "elements.csv", "named_selections.json", "params.json")
_FIELDS = [
    "variant_id", "num_nodes", "num_elements", "num_dofs",
    "load_mesh_time_s", "node_merge_time_s", "boundary_time_s",
    "assembly_time_s", "solve_time_s", "postprocess_time_s",
    "total_time_s", "peak_memory_gb", "solver_backend",
]


def _peak_gb() -> float:
    return psutil.Process().memory_info().rss / 1e9


def _extract(zf, vid, dest):
    for fname in _NEEDED:
        try:
            with zf.open(f"variants/{vid}/{fname}") as s, open(dest / fname, "wb") as d:
                shutil.copyfileobj(s, d)
        except KeyError:
            pass


def bench_one(variant_dir: Path, backend: str) -> dict:
    mat = load_material()
    phys = load_physics()
    C = mat.C()
    axis = str(phys["move_axis"])
    delta = float(phys["delta_mm"])
    rigid = phys.get("move_transverse", "rigid") == "rigid"

    peak = 0.0

    # load_mesh (merged) — the real path used in production.
    t = time.perf_counter()
    mesh = load_mesh(variant_dir)                       # merge + orient (defaults)
    load_time = time.perf_counter() - t
    peak = max(peak, _peak_gb())

    # isolate node-merge cost: raw (no merge) load, diff against merged load.
    t = time.perf_counter()
    _ = load_mesh(variant_dir, merge_coincident=False, normalize_orientation=True)
    raw_time = time.perf_counter() - t
    merge_time = max(load_time - raw_time, 0.0)         # merge ~ the extra over raw

    t = time.perf_counter()
    ns = _load_named_selections(variant_dir)
    bs = reconstruct_boundary(mesh, ns)
    boundary_time = time.perf_counter() - t

    t = time.perf_counter()
    K = assemble_global_stiffness(mesh, C)
    assembly_time = time.perf_counter() - t
    peak = max(peak, _peak_gb())

    t = time.perf_counter()
    res = solve_displacement(mesh, bs, C, move_axis=axis, delta_mm=delta,
                             move_transverse_rigid=rigid, K=K, backend=backend)
    solve_time = time.perf_counter() - t
    peak = max(peak, _peak_gb())

    t = time.perf_counter()
    compute_stiffness(mesh, res, C)
    von_mises_nodal(mesh, res.u, C)
    compute_element_max_von_mises(mesh, res.u, C)
    post_time = time.perf_counter() - t
    peak = max(peak, _peak_gb())

    total = load_time + boundary_time + assembly_time + solve_time + post_time
    return {
        "variant_id": variant_dir.name,
        "num_nodes": mesh.num_nodes,
        "num_elements": mesh.num_elements,
        "num_dofs": num_dofs(mesh),
        "load_mesh_time_s": round(load_time, 3),
        "node_merge_time_s": round(merge_time, 3),
        "boundary_time_s": round(boundary_time, 3),
        "assembly_time_s": round(assembly_time, 3),
        "solve_time_s": round(solve_time, 3),
        "postprocess_time_s": round(post_time, 3),
        "total_time_s": round(total, 3),
        "peak_memory_gb": round(peak, 3),
        "solver_backend": res.solver_backend,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("zip_path", nargs="?",
                    default=str(_REPO.parent / "vcm_spring_valid_300_variants.zip"))
    ap.add_argument("--variants", default="",
                    help="comma-separated variant ids; default = first 3 + a large one")
    ap.add_argument("--backend", default="scipy", help="scipy|auto|pypardiso|umfpack|cholmod")
    ap.add_argument("--out", default=str(_REPO / "reports" / "direct_fem_performance.csv"))
    args = ap.parse_args()

    import re
    with zipfile.ZipFile(args.zip_path) as zf:
        all_vids = sorted(set(
            re.match(r"variants/([^/]+)/", n).group(1)
            for n in zf.namelist() if n.startswith("variants/")
        ))
        if args.variants:
            vids = [v for v in args.variants.split(",") if v in all_vids]
        else:
            vids = all_vids[:3]   # a small representative default; --variants to extend
        print(f"benchmarking {len(vids)} variant(s), backend={args.backend}")

        rows = []
        for vid in vids:
            with tempfile.TemporaryDirectory(prefix="vcm_bench_") as td:
                tdp = Path(td)
                _extract(zf, vid, tdp)
                row = bench_one(tdp, args.backend)
                rows.append(row)
                print(f"  {vid}: dofs={row['num_dofs']} "
                      f"assemble={row['assembly_time_s']}s solve={row['solve_time_s']}s "
                      f"total={row['total_time_s']}s peak={row['peak_memory_gb']}GB "
                      f"[{row['solver_backend']}]", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
