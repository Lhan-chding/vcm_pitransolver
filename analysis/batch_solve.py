"""batch_solve.py — run direct-FEM over every extracted variant; build K pseudo-labels.

Scans a root (default _devdata/) for variant directories, solves each with the
Stage-1 FEM pipeline, and writes:

  reports/fem_pseudolabels.csv   one row per variant: K, diagnostics, health gate
  reports/fields/<variant>.npz   the solved displacement + nodal von Mises (heavy;
                                  gitignored). Skipped with --no-fields.

Robust by design: each variant is solved in isolation; a failure (bad mesh,
disconnected after merge, solver blowup) is recorded as status=error with the
message and does NOT abort the batch. A health gate flags variants whose
pseudo-label should not be trusted (multi-component mesh, large energy/reaction
gap, von Mises past yield) so they can be excluded from training downstream.

Usage:
    python analysis/batch_solve.py                 # all variants in _devdata/
    python analysis/batch_solve.py --root DIR      # a different extraction root
    python analysis/batch_solve.py --jobs 4        # parallel (process per variant)
    python analysis/batch_solve.py --no-fields     # CSV only, skip .npz fields
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "data"))

from fem.material import load_material  # noqa: E402
from fem.pipeline import load_physics, solve_variant  # noqa: E402

# Health-gate thresholds. A variant failing any of these gets trustworthy=False.
MAX_REL_GAP = 1e-4          # energy vs reaction stiffness (Clapeyron); slack for big meshes
MAX_VM_FRAC_YIELD = 1.0     # stay in linear-elastic range at the chosen delta
REQUIRE_SINGLE_COMPONENT = True

# CSV columns: the scalar diagnostics (drop the heavy array fields).
_SCALAR_FIELDS = [
    "variant_id", "status", "subset",
    "num_nodes", "num_elements", "num_dofs",
    "n_move_nodes", "n_fixed_nodes", "n_components",
    "move_area_rel_err", "move_area_ok",
    "delta_mm", "move_axis",
    "K_energy", "K_reaction", "rel_gap",
    "total_strain_energy", "move_face_force",
    "u_max", "u_mean", "vm_max", "vm_mean", "vm_frac_yield",
    "vm_max_element", "vm_frac_yield_element",
    "trustworthy", "elapsed_s", "error",
]


def _subset_of(vid: str) -> str:
    if "STRICT" in vid:
        return "STRICT"
    if "COMPLEX" in vid:
        return "COMPLEX"
    return "UNKNOWN"


def _is_variant_dir(d: Path) -> bool:
    return (d / "nodes.csv").exists() and (d / "elements.csv").exists()


def find_variants(root: Path) -> list[Path]:
    """Variant dirs directly under root (and root itself if it is one)."""
    if _is_variant_dir(root):
        return [root]
    return sorted(d for d in root.iterdir() if d.is_dir() and _is_variant_dir(d))


def _gate(row: dict) -> bool:
    """Health gate -> is this pseudo-label trustworthy?"""
    if row.get("status") != "ok":
        return False
    if REQUIRE_SINGLE_COMPONENT and row.get("n_components", 99) != 1:
        return False
    if row.get("rel_gap", 1.0) > MAX_REL_GAP:
        return False
    vfy = row.get("vm_frac_yield")
    if vfy is None or (vfy == vfy and vfy > MAX_VM_FRAC_YIELD):  # nan-safe
        return False if vfy is not None and vfy == vfy else True
    return True


def solve_one(variant_dir: str, fields_dir: str | None) -> dict:
    """Solve a single variant -> a flat CSV-ready dict. Never raises."""
    vdir = Path(variant_dir)
    t0 = time.perf_counter()
    row: dict = {k: "" for k in _SCALAR_FIELDS}
    row["variant_id"] = vdir.name
    row["subset"] = _subset_of(vdir.name)
    try:
        mat = load_material()
        phys = load_physics()
        keep = fields_dir is not None
        vs = solve_variant(vdir, material=mat, physics=phys, keep_fields=keep)
        d = asdict(vs)
        d.pop("u", None)
        d.pop("type_histogram", None)
        d.pop("von_mises", None)
        row.update({k: d[k] for k in d if k in row})
        row["status"] = "ok"
        if keep:
            Path(fields_dir).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                Path(fields_dir) / f"{vdir.name}.npz",
                u=vs.u.astype(np.float32),
                von_mises=vs.von_mises.astype(np.float32),
            )
    except Exception as exc:  # noqa: BLE001 — per-variant isolation is the point
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
        # full traceback to stderr for debugging; CSV keeps the one-liner.
        traceback.print_exc()
    row["elapsed_s"] = round(time.perf_counter() - t0, 2)
    row["trustworthy"] = _gate(row)
    return row


def run(root: Path, out_csv: Path, fields_dir: Path | None, jobs: int) -> list[dict]:
    variants = find_variants(root)
    if not variants:
        raise SystemExit(f"no variant directories found under {root}")
    print(f"found {len(variants)} variant(s) under {root}; jobs={jobs}")

    rows: list[dict] = []
    if jobs <= 1:
        for v in variants:
            print(f"  solving {v.name} ...", flush=True)
            rows.append(solve_one(str(v), str(fields_dir) if fields_dir else None))
    else:
        fd = str(fields_dir) if fields_dir else None
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(solve_one, str(v), fd): v for v in variants}
            for fut in as_completed(futs):
                row = fut.result()
                rows.append(row)
                print(f"  done {row['variant_id']}: status={row['status']} "
                      f"K={row.get('K_energy', '')}", flush=True)

    rows.sort(key=lambda r: r["variant_id"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_SCALAR_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


def _print_summary(rows: list[dict], out_csv: Path) -> None:
    n = len(rows)
    ok = [r for r in rows if r["status"] == "ok"]
    trust = [r for r in rows if r["trustworthy"] is True]
    errs = [r for r in rows if r["status"] == "error"]
    print("=" * 60)
    print(f"variants            : {n}")
    print(f"  solved ok         : {len(ok)}")
    print(f"  trustworthy       : {len(trust)}")
    print(f"  errored           : {len(errs)}")
    if errs:
        print("  error variants    :")
        for r in errs[:10]:
            print(f"    {r['variant_id']}: {r['error']}")
    if trust:
        ks = np.array([float(r["K_energy"]) for r in trust])
        print("-" * 60)
        print(f"K (trustworthy) N/mm: min {ks.min():.4e}  "
              f"median {np.median(ks):.4e}  max {ks.max():.4e}")
        gaps = np.array([float(r["rel_gap"]) for r in trust])
        print(f"max energy/reaction gap among trustworthy: {gaps.max():.2e}")
    print("-" * 60)
    print(f"index -> {out_csv}")
    print("=" * 60)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(_REPO / "_devdata"),
                    help="extraction root containing variant dirs")
    ap.add_argument("--out", default=str(_REPO / "reports" / "fem_pseudolabels.csv"))
    ap.add_argument("--jobs", type=int, default=1, help="parallel processes")
    ap.add_argument("--no-fields", action="store_true",
                    help="skip saving per-variant displacement/stress .npz")
    args = ap.parse_args()

    fields_dir = None if args.no_fields else _REPO / "reports" / "fields"
    rows = run(Path(args.root), Path(args.out), fields_dir, args.jobs)
    _print_summary(rows, Path(args.out))


if __name__ == "__main__":
    main()
