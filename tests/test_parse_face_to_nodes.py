"""Tests for data/parse_face_to_nodes.py — synthetic invariants + real-data check."""

from __future__ import annotations

import json

import numpy as np
import pytest

import parse_face_to_nodes as pf
from parse_mesh import load_mesh


def _planar_grid_plate(tmp_path):
    """Build a thin plate (X-thin, Y-Z plane) as a node cloud + one dummy tet10.

    We only need coords for boundary reconstruction; one valid element keeps the
    Mesh loader happy. Returns (variant_dir, ns_dict).
    """
    xs = [-0.02, 0.01, 0.04]            # 3 layers across 0.06 mm thickness
    ys = np.linspace(-5.0, 5.0, 21)
    zs = np.linspace(-5.0, 5.0, 21)
    coords = []
    for x in xs:
        for y in ys:
            for z in zs:
                coords.append((x, float(y), float(z)))
    coords = np.asarray(coords)
    node_ids = list(range(1, len(coords) + 1))

    # one tet10 from the first 10 nodes (non-degenerate enough for the loader)
    nodes = tmp_path / "nodes.csv"
    with open(nodes, "w", newline="") as fh:
        fh.write("node_id,x,y,z\n")
        for nid, (x, y, z) in zip(node_ids, coords):
            fh.write(f"{nid},{x},{y},{z}\n")
    elems = tmp_path / "elements.csv"
    with open(elems, "w", newline="") as fh:
        fh.write("element_id,node_ids\n")
        fh.write(f'1,"{node_ids[:10]}"\n')

    # named_selections: move face on z=-5 wall, normal -Z, height ~2 mm
    ns = {
        "MOVE_INNER_RING_FACE": [11566],
        "FIXED_TOP_FACES": [1, 2, 3, 4],
        "debug": {
            "move_candidates_top5": [
                {
                    "face_id": 11566,
                    "cy": 0.0,
                    "cz": -5.0,
                    "ny": 0.0,
                    "nz": -1.0,
                    "area": 0.06 * 2.0,  # thickness * height
                    "score": 1.0,
                }
            ]
        },
    }
    with open(tmp_path / "named_selections.json", "w") as fh:
        json.dump(ns, fh)
    return tmp_path, ns


def test_reconstruction_invariants(tmp_path):
    d, ns = _planar_grid_plate(tmp_path)
    mesh = load_mesh(d)
    bs = pf.reconstruct_boundary(mesh, ns)

    # invariants the BoundarySets must satisfy (also asserted internally)
    assert bs.move_nodes.size > 0
    assert bs.fixed_nodes.size > 0
    sm, sf = set(bs.move_nodes.tolist()), set(bs.fixed_nodes.tolist())
    assert sm.isdisjoint(sf)
    covered = sm | sf | set(bs.free_nodes.tolist())
    assert len(covered) == mesh.num_nodes

    # move nodes must actually lie on the z=-5 plane
    mv = mesh.coords[bs.move_nodes]
    assert np.all(np.abs(mv[:, 2] - (-5.0)) <= 0.05)
    # and span the full thickness in X
    assert (mv[:, 0].max() - mv[:, 0].min()) == pytest.approx(0.06, abs=1e-6)


def test_area_self_check_passes_on_clean_geometry(tmp_path):
    d, ns = _planar_grid_plate(tmp_path)
    mesh = load_mesh(d)
    bs = pf.reconstruct_boundary(mesh, ns)
    assert bs.diagnostics["move_area_ok"] is True
    assert bs.diagnostics["move_area_rel_err"] < 0.25


def test_missing_move_face_raises(tmp_path):
    d, ns = _planar_grid_plate(tmp_path)
    mesh = load_mesh(d)
    ns_bad = dict(ns)
    ns_bad["MOVE_INNER_RING_FACE"] = []
    with pytest.raises(ValueError, match="no MOVE_INNER_RING_FACE"):
        pf.reconstruct_boundary(mesh, ns_bad)


def test_move_id_not_in_debug_raises(tmp_path):
    d, ns = _planar_grid_plate(tmp_path)
    mesh = load_mesh(d)
    ns_bad = json.loads(json.dumps(ns))
    ns_bad["MOVE_INNER_RING_FACE"] = [999999]  # id absent from debug candidates
    with pytest.raises(ValueError, match="not found in debug"):
        pf.reconstruct_boundary(mesh, ns_bad)


# ---- real-data integration ----

def test_real_variant_boundary(dev_variant):
    mesh = load_mesh(dev_variant)
    ns = pf._load_named_selections(dev_variant)
    bs = pf.reconstruct_boundary(mesh, ns)
    bs.assert_valid(mesh.num_nodes)
    # area self-check should pass within tolerance on real geometry
    assert bs.diagnostics["move_area_ok"] is True
    # four fixed corner pads -> nodes present in all four (y,z) quadrants
    fx = mesh.coords[bs.fixed_nodes]
    quadrants = {(int(np.sign(y)), int(np.sign(z))) for y, z in fx[:, 1:3]}
    assert len(quadrants) == 4
