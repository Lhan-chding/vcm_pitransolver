"""audit_health_300.py — full-dataset (300-variant) mesh / FEM health audit.

Runs the four P0 audits that gate a variant into physics-informed training,
streaming each variant out of the dataset zip one at a time (peak disk ~one
variant, ~9 MB) so the 2.4 GB archive need not be fully extracted:

  1. element-type audit   -> reports/element_type_audit_300.csv  (+ .md)
       nodes/elements readable, every element node-count in {10,15,20}, no unknown.
  2. detJ / Jacobian audit -> reports/detj_audit_300.csv
       EVERY element, EVERY Gauss point, using the SAME quadrature the energy /
       training code uses (fem.elements.*.GAUSS). num_negative / num_near_zero
       must be 0 for a trainable variant.
  3. node-merge / connectivity audit -> reports/node_merge_audit_300.csv
       raw vs merged node count, components before/after merge, degenerate
       elements after merge. Single component after merge == valid load path.
  4. boundary-node audit  -> reports/boundary_node_audit_300.csv
       fixed/move/free counts, overlap, bboxes, per-corner fixed counts.

A variant is solver-/training-ready only if ALL four audits pass. The audits are
deliberately independent CSVs (one concern each) but share a single parse so the
zip is read once. A failure on one variant never aborts the run.

Usage:
    python data/audit_health_300.py [zip_path] [--limit N] [--out reports]
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
import traceback
import zipfile
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "data"))

from parse_face_to_nodes import _load_named_selections, reconstruct_boundary  # noqa: E402
from parse_mesh import ELEM_TYPES, load_mesh  # noqa: E402
from fem.elements import hex20, tet10, wedge15  # noqa: E402
from fem.pipeline import count_components  # noqa: E402

_FAMILY = {10: tet10, 15: wedge15, 20: hex20}
_MERGE_TOL = 1e-6          # must match parse_mesh.load_mesh default
_NEAR_ZERO_DETJ = 1e-12    # |detJ| below this is treated as degenerate

# Files we need from each variant; STEP/scdocx/png are skipped to save IO.
_NEEDED = ("nodes.csv", "elements.csv", "named_selections.json", "params.json")


# --------------------------------------------------------------------------- #
# 1. element-type audit                                                        #
# --------------------------------------------------------------------------- #
def audit_element_types(variant_id: str, raw_counts: dict) -> dict:
    """raw_counts: node-count(int) -> #elements, plus a "_num_nodes" meta key."""
    num_nodes = raw_counts.get("_num_nodes", 0)
    # element-count buckets are the INTEGER keys only; "_num_nodes" is metadata.
    elem_buckets = {k: c for k, c in raw_counts.items() if isinstance(k, int)}
    n_tet = elem_buckets.get(10, 0)
    n_wedge = elem_buckets.get(15, 0)
    n_hex = elem_buckets.get(20, 0)
    n_unknown = sum(c for k, c in elem_buckets.items() if k not in ELEM_TYPES)
    total = sum(elem_buckets.values())
    return {
        "variant_id": variant_id,
        "num_nodes": num_nodes,
        "num_elements": total,
        "num_tet10": n_tet,
        "num_wedge15": n_wedge,
        "num_hex20": n_hex,
        "num_unknown": n_unknown,
        "element_type_ok": n_unknown == 0 and total > 0,
    }


# --------------------------------------------------------------------------- #
# 2. detJ / Jacobian audit (every element, every Gauss point, same quadrature) #
# --------------------------------------------------------------------------- #
def audit_detj(variant_id: str, mesh) -> list[dict]:
    """One row per element family present. Uses each family's training GAUSS."""
    rows = []
    coords = mesh.coords
    for nnode, module in _FAMILY.items():
        conn = mesh.elements_of(nnode)
        if conn.shape[0] == 0:
            continue
        gauss = module.GAUSS
        # Precompute shape-grads at every Gauss point once (constant per family).
        dN = np.stack([module.shape_grads(nat) for nat, _w in gauss])  # (G,nnode,3)
        xe = coords[conn]                                              # (M,nnode,3)
        # J[m,g,i,j] = sum_n xe[m,n,i] * dN[g,n,j]
        J = np.einsum("mni,gnj->mgij", xe, dN)                        # (M,G,3,3)
        detJ = np.linalg.det(J)                                       # (M,G)
        n_neg = int((detJ < 0.0).sum())
        n_near = int((np.abs(detJ) < _NEAR_ZERO_DETJ).sum())
        rows.append({
            "variant_id": variant_id,
            "element_family": ELEM_TYPES[nnode],
            "num_elements": int(conn.shape[0]),
            "num_gauss_points_checked": int(detJ.size),
            "min_detJ": float(detJ.min()),
            "max_detJ": float(detJ.max()),
            "num_negative_detJ": n_neg,
            "num_near_zero_detJ": n_near,
            "detj_ok": n_neg == 0 and n_near == 0,
        })
    return rows


# --------------------------------------------------------------------------- #
# 3. node-merge / connectivity audit                                           #
# --------------------------------------------------------------------------- #
def audit_node_merge(variant_id: str, variant_dir: Path) -> dict:
    """Compare raw (unmerged) vs merged mesh: nodes, components, degeneracy."""
    raw = load_mesh(variant_dir, merge_coincident=False, normalize_orientation=False)
    comp_before = count_components(raw)

    # Merge stats: how many coincident node pairs / their distances.
    from scipy.spatial import cKDTree
    pairs = cKDTree(raw.coords).query_pairs(r=_MERGE_TOL, output_type="ndarray")
    if len(pairs):
        d = np.linalg.norm(raw.coords[pairs[:, 0]] - raw.coords[pairs[:, 1]], axis=1)
        min_d, max_d = float(d.min()), float(d.max())
    else:
        min_d = max_d = 0.0

    merged = load_mesh(variant_dir, merge_coincident=True, normalize_orientation=True)
    comp_after = count_components(merged)

    # Degenerate elements after merge: any element with a repeated node row.
    n_degen = sum(
        len(set(row.tolist())) != len(row) for row in merged.connectivity
    )

    return {
        "variant_id": variant_id,
        "raw_num_nodes": raw.num_nodes,
        "merged_num_nodes": merged.num_nodes,
        "num_merged_nodes": raw.num_nodes - merged.num_nodes,
        "num_components_before_merge": comp_before,
        "num_components_after_merge": comp_after,
        "num_degenerate_elements_after_merge": int(n_degen),
        "min_merge_distance": min_d,
        "max_merge_distance": max_d,
        "connectivity_ok": comp_after == 1 and n_degen == 0,
    }


# --------------------------------------------------------------------------- #
# 4. boundary-node audit                                                       #
# --------------------------------------------------------------------------- #
def _bbox_str(coords: np.ndarray) -> str:
    if coords.size == 0:
        return ""
    lo, hi = coords.min(0), coords.max(0)
    return (f"x[{lo[0]:.3f},{hi[0]:.3f}] "
            f"y[{lo[1]:.3f},{hi[1]:.3f}] "
            f"z[{lo[2]:.3f},{hi[2]:.3f}]")


def _per_corner_counts(coords: np.ndarray, fixed_rows: np.ndarray, thin_axis: int) -> list[int]:
    """Count fixed nodes near each of the 4 (in-plane) bbox corners."""
    lo, hi = coords.min(0), coords.max(0)
    other = [a for a in (0, 1, 2) if a != thin_axis]
    a1, a2 = other
    fc = coords[fixed_rows]
    corners = [(lo[a1], lo[a2]), (lo[a1], hi[a2]), (hi[a1], lo[a2]), (hi[a1], hi[a2])]
    # assign each fixed node to its nearest corner; count per corner
    counts = [0, 0, 0, 0]
    if fc.size:
        c1 = fc[:, a1]
        c2 = fc[:, a2]
        for i, (x1, x2) in enumerate(corners):
            counts[i] = int(np.sum((np.abs(c1 - x1) < np.abs(c1 - (lo[a1] + hi[a1] - x1))) &
                                   (np.abs(c2 - x2) < np.abs(c2 - (lo[a2] + hi[a2] - x2)))))
    return counts


def audit_boundary(variant_id: str, mesh, variant_dir: Path) -> dict:
    ns = _load_named_selections(variant_dir)
    bs = reconstruct_boundary(mesh, ns)
    coords = mesh.coords
    lo, hi = coords.min(0), coords.max(0)
    thin_axis = int(np.argmin(hi - lo))
    overlap = int(np.intersect1d(bs.move_nodes, bs.fixed_nodes).size)
    corner = _per_corner_counts(coords, bs.fixed_nodes, thin_axis)
    boundary_ok = (
        bs.move_nodes.size > 0
        and bs.fixed_nodes.size > 0
        and overlap == 0
        and (bs.move_nodes.size + bs.fixed_nodes.size + bs.free_nodes.size) == mesh.num_nodes
        and all(c > 0 for c in corner)   # all four corners actually captured
    )
    return {
        "variant_id": variant_id,
        "n_fixed_nodes": int(bs.fixed_nodes.size),
        "n_move_nodes": int(bs.move_nodes.size),
        "n_free_nodes": int(bs.free_nodes.size),
        "fixed_move_overlap_count": overlap,
        "fixed_bbox": _bbox_str(coords[bs.fixed_nodes]),
        "move_bbox": _bbox_str(coords[bs.move_nodes]),
        "fixed_nodes_per_corner_1": corner[0],
        "fixed_nodes_per_corner_2": corner[1],
        "fixed_nodes_per_corner_3": corner[2],
        "fixed_nodes_per_corner_4": corner[3],
        "boundary_ok": boundary_ok,
    }


# --------------------------------------------------------------------------- #
# driver                                                                       #
# --------------------------------------------------------------------------- #
def _raw_element_counts(variant_dir: Path) -> dict:
    """Parse elements.csv WITHOUT the supported-only filter, so unknown counts show.

    parse_mesh raises on unsupported node-counts; here we want to COUNT them, so
    we read elements.csv directly and tally node-counts ourselves.
    """
    import ast
    counts: dict = {}
    nodes_csv = variant_dir / "nodes.csv"
    with open(variant_dir / "elements.csv", newline="") as fh:
        reader = csv.reader(fh)
        next(reader)  # header
        for row in reader:
            if not row:
                continue
            k = len(ast.literal_eval(row[1]))
            counts[k] = counts.get(k, 0) + 1
    # node count
    with open(nodes_csv, newline="") as fh:
        counts["_num_nodes"] = sum(1 for _ in fh) - 1
    return counts


def _variant_dirs_in_zip(zf: zipfile.ZipFile) -> dict[str, str]:
    """variant_id -> in-zip prefix, for variants that have the needed files."""
    import re
    out: dict[str, str] = {}
    for n in zf.namelist():
        m = re.match(r"variants/([^/]+)/", n)
        if m:
            vid = m.group(1)
            out.setdefault(vid, f"variants/{vid}/")
    return out


def _extract_variant(zf: zipfile.ZipFile, prefix: str, dest: Path) -> None:
    for fname in _NEEDED:
        member = prefix + fname
        try:
            with zf.open(member) as src, open(dest / fname, "wb") as dst:
                shutil.copyfileobj(src, dst)
        except KeyError:
            pass  # missing file surfaces later as a parse error


def run(zip_path: Path, out_dir: Path, limit: int | None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    et_rows, dj_rows, nm_rows, bd_rows = [], [], [], []
    errors: list[dict] = []

    with zipfile.ZipFile(zip_path) as zf:
        variants = _variant_dirs_in_zip(zf)
        vids = sorted(variants)
        if limit:
            vids = vids[:limit]
        print(f"auditing {len(vids)} variant(s) from {zip_path.name}")

        for i, vid in enumerate(vids, 1):
            with tempfile.TemporaryDirectory(prefix="vcm_audit_") as td:
                tdp = Path(td)
                _extract_variant(zf, variants[vid], tdp)
                try:
                    raw_counts = _raw_element_counts(tdp)
                    et_rows.append(audit_element_types(vid, raw_counts))
                    # merged mesh used by detJ + boundary (training-equivalent)
                    mesh = load_mesh(tdp)  # merge + orientation normalize (defaults)
                    dj_rows.extend(audit_detj(vid, mesh))
                    nm_rows.append(audit_node_merge(vid, tdp))
                    bd_rows.append(audit_boundary(vid, mesh, tdp))
                except Exception as exc:  # noqa: BLE001 — per-variant isolation
                    errors.append({"variant_id": vid, "error": f"{type(exc).__name__}: {exc}"})
                    traceback.print_exc()
            if i % 20 == 0 or i == len(vids):
                print(f"  {i}/{len(vids)} done ({len(errors)} errors so far)", flush=True)

    _write_csv(out_dir / "element_type_audit_300.csv", et_rows)
    _write_csv(out_dir / "detj_audit_300.csv", dj_rows)
    _write_csv(out_dir / "node_merge_audit_300.csv", nm_rows)
    _write_csv(out_dir / "boundary_node_audit_300.csv", bd_rows)
    if errors:
        _write_csv(out_dir / "audit_errors_300.csv", errors)

    summary = _summarize(et_rows, dj_rows, nm_rows, bd_rows, errors)
    _write_md(out_dir / "element_type_audit_300.md", summary, et_rows, dj_rows, nm_rows, bd_rows)
    return summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _summarize(et, dj, nm, bd, errors) -> dict:
    # per-variant detj_ok = all families ok
    dj_by_v: dict = {}
    for r in dj:
        dj_by_v.setdefault(r["variant_id"], True)
        dj_by_v[r["variant_id"]] &= bool(r["detj_ok"])
    et_ok = sum(bool(r["element_type_ok"]) for r in et)
    dj_ok = sum(1 for v in dj_by_v.values() if v)
    nm_ok = sum(bool(r["connectivity_ok"]) for r in nm)
    bd_ok = sum(bool(r["boundary_ok"]) for r in bd)
    # fully trainable = pass all four
    ok_sets = [
        {r["variant_id"] for r in et if r["element_type_ok"]},
        {v for v, ok in dj_by_v.items() if ok},
        {r["variant_id"] for r in nm if r["connectivity_ok"]},
        {r["variant_id"] for r in bd if r["boundary_ok"]},
    ]
    fully_ok = set.intersection(*ok_sets) if all(ok_sets) else set()
    return {
        "n_variants": len(et),
        "element_type_ok": et_ok,
        "detj_ok": dj_ok,
        "connectivity_ok": nm_ok,
        "boundary_ok": bd_ok,
        "fully_trainable": len(fully_ok),
        "n_errors": len(errors),
        "error_variants": [e["variant_id"] for e in errors],
    }


def _write_md(path, summary, et, dj, nm, bd) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Full-dataset health audit (300 variants)\n\n")
        fh.write(f"- variants audited: **{summary['n_variants']}**\n")
        fh.write(f"- element_type_ok : **{summary['element_type_ok']}**\n")
        fh.write(f"- detj_ok (all families): **{summary['detj_ok']}**\n")
        fh.write(f"- connectivity_ok (single component after merge): **{summary['connectivity_ok']}**\n")
        fh.write(f"- boundary_ok     : **{summary['boundary_ok']}**\n")
        fh.write(f"- **fully trainable (pass all 4)**: **{summary['fully_trainable']}**\n")
        fh.write(f"- errors          : **{summary['n_errors']}** {summary['error_variants'][:10]}\n\n")
        fh.write("CSV outputs: element_type_audit_300.csv, detj_audit_300.csv, "
                 "node_merge_audit_300.csv, boundary_node_audit_300.csv.\n\n")
        # element family distribution
        if et:
            tot = sum(r["num_tet10"] + r["num_wedge15"] + r["num_hex20"] for r in et)
            n_t = sum(r["num_tet10"] for r in et)
            n_w = sum(r["num_wedge15"] for r in et)
            n_h = sum(r["num_hex20"] for r in et)
            fh.write("## Element family totals (all variants)\n\n")
            fh.write(f"- tet10  : {n_t:,} ({100*n_t/tot:.1f}%)\n")
            fh.write(f"- wedge15: {n_w:,} ({100*n_w/tot:.1f}%)\n")
            fh.write(f"- hex20  : {n_h:,} ({100*n_h/tot:.1f}%)\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("zip_path", nargs="?",
                    default=str(_REPO.parent / "vcm_spring_valid_300_variants.zip"))
    ap.add_argument("--out", default=str(_REPO / "reports"))
    ap.add_argument("--limit", type=int, default=None, help="audit only first N variants")
    args = ap.parse_args()

    summary = run(Path(args.zip_path), Path(args.out), args.limit)
    print("=" * 60)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
