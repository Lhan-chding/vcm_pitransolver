"""
parse_face_to_nodes.py — Reconstruct boundary node sets from geometric descriptors.

WHY THIS EXISTS (verified 2026-06-29):
  The dataset's boundary definitions live in `named_selections.json` as STEP FACE
  IDs (e.g. MOVE_INNER_RING_FACE = [11566]). The mesh files contain NO node sets:
    - mesh.inp has only *NODE / *ELEMENT, no *NSET / *ELSET.
    - there is no .cdb file at all.
  So STEP face id -> mesh node mapping does NOT exist anywhere and must be rebuilt
  geometrically. The single usable signal is the `debug.move_candidates_top5` block,
  which gives each candidate face's centroid (cy,cz), outward normal (ny,nz) and area.

STRATEGY (priority order, matching the agreed plan):
  1. (future) if a mesh node-set / component is ever present -> read it directly.
     [not available in current dataset; kept as a hook]
  2. MOVE face: take the SELECTED move face descriptor (the one whose face_id matches
     named_selections["MOVE_INNER_RING_FACE"]) and select mesh nodes lying ON that
     plane (signed distance < tol) AND within the face's lateral extent.
  3. FIXED faces: the four corner "top" pads. Selected by geometry: nodes near the
     X-max surface (the plate's top face in the thin X direction) clustered at the
     four (y,z) corners. Descriptors for fixed faces are NOT in debug, so we use the
     plate geometry + the corner locations implied by symmetry.
  4. Emit everything needed for a human visual check (colored node export) and assert
     mutual-exclusivity / coverage invariants.

This module makes NO physics decision (e.g. which axis the load is on). It only
answers: "which mesh nodes belong to the named fixed / move faces?"
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from parse_mesh import Mesh, load_mesh


@dataclass(frozen=True)
class BoundarySets:
    move_nodes: np.ndarray              # (Nm,) row indices into coords
    fixed_nodes: np.ndarray             # (Nf,) row indices
    free_nodes: np.ndarray              # (Nfree,) row indices
    move_face_descriptor: dict          # the selected move face's geometric descriptor
    diagnostics: dict = field(default_factory=dict)

    def assert_valid(self, num_nodes: int) -> None:
        assert self.move_nodes.size > 0, "no MOVE nodes selected -- reconstruction failed"
        assert self.fixed_nodes.size > 0, "no FIXED nodes selected -- reconstruction failed"
        sm, sf = set(self.move_nodes.tolist()), set(self.fixed_nodes.tolist())
        assert sm.isdisjoint(sf), "MOVE and FIXED node sets overlap"
        covered = sm | sf | set(self.free_nodes.tolist())
        assert len(covered) == num_nodes, "fixed+move+free does not cover all nodes"


def _load_named_selections(variant_dir: Path) -> dict:
    with open(variant_dir / "named_selections.json") as fh:
        return json.load(fh)


def _selected_move_descriptor(ns: dict) -> dict:
    """Return the geometric descriptor of the face named MOVE_INNER_RING_FACE.

    named_selections["MOVE_INNER_RING_FACE"] holds the chosen face_id; the matching
    descriptor (centroid/normal/area) is in debug.move_candidates_top5.
    """
    move_ids = ns.get("MOVE_INNER_RING_FACE", [])
    if not move_ids:
        raise ValueError("named_selections has no MOVE_INNER_RING_FACE")
    target_id = int(move_ids[0])
    for cand in ns.get("debug", {}).get("move_candidates_top5", []):
        if int(cand["face_id"]) == target_id:
            return cand
    raise ValueError(
        f"move face_id {target_id} not found in debug.move_candidates_top5; "
        "cannot recover its geometry"
    )


def _select_planar_face_nodes(
    coords: np.ndarray,
    centroid_yz: tuple[float, float],
    normal_yz: tuple[float, float],
    dist_tol: float,
    lateral_halfspan: float,
) -> np.ndarray:
    """Select nodes on an in-plane (Y-Z) face strip.

    The move faces are thin vertical strips on the inner ring wall: their normal lies
    in the Y-Z plane (ny,nz), they span the full plate thickness in X, and a limited
    height along the in-plane tangent direction. We select nodes whose signed distance
    to the face plane is within dist_tol and whose tangential offset from the face
    centroid is within lateral_halfspan.
    """
    cy, cz = centroid_yz
    ny, nz = normal_yz
    nrm = np.hypot(ny, nz)
    if nrm == 0:
        raise ValueError("degenerate face normal (0,0) in Y-Z")
    ny, nz = ny / nrm, nz / nrm
    # in-plane tangent perpendicular to normal
    ty, tz = -nz, ny

    dy = coords[:, 1] - cy
    dz = coords[:, 2] - cz
    signed_dist = dy * ny + dz * nz          # distance along normal
    tangential = dy * ty + dz * tz           # offset along the face's in-plane tangent

    on_plane = np.abs(signed_dist) <= dist_tol
    within = np.abs(tangential) <= lateral_halfspan
    return np.nonzero(on_plane & within)[0]


def _select_fixed_corner_nodes(
    coords: np.ndarray,
    thin_axis: int,
    top_tol: float,
    corner_radius: float,
) -> np.ndarray:
    """Select nodes on the four corner top pads (X-max surface, four (y,z) corners).

    Fixed faces have no descriptor in debug, so we use plate geometry:
      - 'top' = the X-max surface (plate is thin in X) within top_tol of x_max,
      - 'four corners' = nodes near the four extreme (y,z) corners of the bbox.
    corner_radius controls how large each corner pad capture region is.
    """
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    other = [a for a in (0, 1, 2) if a != thin_axis]
    a1, a2 = other  # the two in-plane axes (y,z for this dataset)

    on_top = coords[:, thin_axis] >= (hi[thin_axis] - top_tol)

    # four corners in the in-plane axes
    corners = [
        (lo[a1], lo[a2]),
        (lo[a1], hi[a2]),
        (hi[a1], lo[a2]),
        (hi[a1], hi[a2]),
    ]
    near_corner = np.zeros(coords.shape[0], dtype=bool)
    for c1, c2 in corners:
        d = np.hypot(coords[:, a1] - c1, coords[:, a2] - c2)
        near_corner |= d <= corner_radius

    return np.nonzero(on_top & near_corner)[0]


def reconstruct_boundary(
    mesh: Mesh,
    ns: dict,
    *,
    dist_tol: float = 0.05,        # mm; ~ min_element_size from mesh_quality
    lateral_halfspan: float | None = None,  # mm; if None, derived from reported area
    top_tol: float = 0.02,         # mm; thickness slab for 'top' surface (~1/3 of 0.06)
    corner_radius: float = 1.6,    # mm; corner-pad capture radius
    area_tol_frac: float = 0.25,   # accept reconstructed strip area within this frac
) -> BoundarySets:
    coords = mesh.coords
    lo, hi = mesh.bbox()
    thin_axis = int(np.argmin(hi - lo))
    thickness = float((hi - lo)[thin_axis])

    desc = _selected_move_descriptor(ns)
    # The move face is a strip: area = thickness * height. Derive the tangent
    # half-span from the REPORTED face area instead of a magic number, so the
    # selection self-calibrates per variant. Pad slightly for node capture.
    if lateral_halfspan is None:
        face_height = float(desc["area"]) / max(thickness, 1e-9)
        lateral_halfspan = 0.5 * face_height * 1.10  # +10% capture margin
    move_nodes = _select_planar_face_nodes(
        coords,
        centroid_yz=(desc["cy"], desc["cz"]),
        normal_yz=(desc["ny"], desc["nz"]),
        dist_tol=dist_tol,
        lateral_halfspan=lateral_halfspan,
    )
    fixed_nodes = _select_fixed_corner_nodes(
        coords, thin_axis=thin_axis, top_tol=top_tol, corner_radius=corner_radius
    )

    # remove any accidental overlap (a node cannot be both); fixed wins (it is a
    # hard support), but we record the collision count for the audit.
    overlap = np.intersect1d(move_nodes, fixed_nodes)
    if overlap.size:
        move_nodes = np.setdiff1d(move_nodes, overlap)

    all_idx = np.arange(coords.shape[0])
    free_nodes = np.setdiff1d(all_idx, np.union1d(move_nodes, fixed_nodes))

    # Quantitative self-check: bounding-box area of the selected move strip should
    # match the reported STEP face area. A large mismatch means the tolerances (or
    # the chosen face) are wrong -- flag it instead of silently trusting the set.
    mv = coords[move_nodes]
    other = [a for a in (0, 1, 2) if a != thin_axis]
    recon_area = float(
        (mv[:, thin_axis].max() - mv[:, thin_axis].min())
        * max(
            mv[:, other[0]].max() - mv[:, other[0]].min(),
            mv[:, other[1]].max() - mv[:, other[1]].min(),
        )
    )
    area_rel_err = abs(recon_area - float(desc["area"])) / max(float(desc["area"]), 1e-12)
    area_ok = area_rel_err <= area_tol_frac

    diagnostics = {
        "thin_axis": "XYZ"[thin_axis],
        "plate_thickness_mm": thickness,
        "move_face_id": int(ns["MOVE_INNER_RING_FACE"][0]),
        "move_centroid_yz": [desc["cy"], desc["cz"]],
        "move_normal_yz": [desc["ny"], desc["nz"]],
        "move_face_area_reported": float(desc["area"]),
        "move_face_area_reconstructed": recon_area,
        "move_area_rel_err": area_rel_err,
        "move_area_ok": bool(area_ok),
        "n_move_nodes": int(move_nodes.size),
        "n_fixed_nodes": int(fixed_nodes.size),
        "n_free_nodes": int(free_nodes.size),
        "n_overlap_removed": int(overlap.size),
        "params": {
            "dist_tol": dist_tol,
            "lateral_halfspan": lateral_halfspan,
            "top_tol": top_tol,
            "corner_radius": corner_radius,
        },
    }

    bs = BoundarySets(
        move_nodes=move_nodes,
        fixed_nodes=fixed_nodes,
        free_nodes=free_nodes,
        move_face_descriptor=desc,
        diagnostics=diagnostics,
    )
    bs.assert_valid(coords.shape[0])
    return bs


def export_for_visual_check(mesh: Mesh, bs: BoundarySets, out_csv: str | Path) -> None:
    """Write a CSV of (x,y,z,label) so the selection can be eyeballed / rendered.

    label: 0=free, 1=fixed, 2=move. Human sign-off on this is a Stage-0 gate.
    """
    label = np.zeros(mesh.num_nodes, dtype=np.int8)
    label[bs.fixed_nodes] = 1
    label[bs.move_nodes] = 2
    out = np.column_stack([mesh.coords, label.astype(np.float64)])
    header = "x,y,z,label  # label: 0=free 1=fixed 2=move"
    np.savetxt(out_csv, out, delimiter=",", header=header, comments="", fmt="%.6f")


if __name__ == "__main__":
    import sys

    d = Path(sys.argv[1] if len(sys.argv) > 1 else "_devdata/VCM_COMPLEX_0001")
    mesh = load_mesh(d)
    ns = _load_named_selections(d)
    bs = reconstruct_boundary(mesh, ns)
    print(json.dumps(bs.diagnostics, indent=2))
    out = Path("reports") / f"bc_check_{d.name}.csv"
    out.parent.mkdir(exist_ok=True)
    export_for_visual_check(mesh, bs, out)
    print(f"\nvisual-check export -> {out}")
    print("  (render with label as color; confirm move=ring inner wall, fixed=4 corner top pads)")
