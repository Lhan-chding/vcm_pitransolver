"""Tests for data/parse_mesh.py — synthetic (always run) + real-data (skip if absent)."""

from __future__ import annotations

import numpy as np
import pytest

import parse_mesh
from parse_mesh import ELEM_TYPES, Mesh, load_mesh


def _write_variant(tmp_path, node_ids, coords, elements):
    """Write minimal nodes.csv / elements.csv for a synthetic variant."""
    nodes = tmp_path / "nodes.csv"
    with open(nodes, "w", newline="") as fh:
        fh.write("node_id,x,y,z\n")
        for nid, (x, y, z) in zip(node_ids, coords):
            fh.write(f"{nid},{x},{y},{z}\n")
    elems = tmp_path / "elements.csv"
    with open(elems, "w", newline="") as fh:
        fh.write("element_id,node_ids\n")
        for eid, conn in enumerate(elements, start=1):
            fh.write(f'{eid},"{list(conn)}"\n')
    return tmp_path


def test_loads_tet10_and_indexes_noncontiguous_ids(tmp_path):
    # node ids deliberately non-contiguous and not zero-based
    node_ids = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
    coords = [(float(i), 0.0, 0.0) for i in range(10)]
    elements = [node_ids]  # one tet10
    d = _write_variant(tmp_path, node_ids, coords, elements)

    m = load_mesh(d)
    assert m.num_nodes == 10
    assert m.num_elements == 1
    assert m.elem_nnode.tolist() == [10]
    # connectivity must be ROW indices (0..9), not original node ids
    assert m.connectivity[0].tolist() == list(range(10))
    assert m.id_to_row[101] == 0 and m.id_to_row[1010] == 9


def test_mixed_quadratic_histogram(tmp_path):
    # 1 tet10 (10), 1 wedge15 (15), 1 hex20 (20) sharing a node pool of 45
    node_ids = list(range(1, 46))
    coords = [(float(i), float(i % 3), float(i % 5)) for i in range(45)]
    tet10 = node_ids[0:10]
    wedge15 = node_ids[10:25]
    hex20 = node_ids[25:45]
    d = _write_variant(tmp_path, node_ids, coords, [tet10, wedge15, hex20])

    m = load_mesh(d)
    hist = m.type_histogram()
    assert hist == {"tet10": 1, "wedge15": 1, "hex20": 1}
    assert set(ELEM_TYPES.values()) == {"tet10", "wedge15", "hex20"}
    # elements_of returns rectangular batch per family
    assert m.elements_of(10).shape == (1, 10)
    assert m.elements_of(20).shape == (1, 20)
    assert m.elements_of(15).shape == (1, 15)


def test_rejects_unsupported_element(tmp_path):
    node_ids = list(range(1, 9))
    coords = [(float(i), 0.0, 0.0) for i in range(8)]
    d = _write_variant(tmp_path, node_ids, coords, [node_ids])  # 8-node hex8: unsupported
    with pytest.raises(ValueError, match="unsupported element node-counts"):
        load_mesh(d)


def test_rejects_degenerate_element(tmp_path):
    node_ids = list(range(1, 11))
    coords = [(float(i), 0.0, 0.0) for i in range(10)]
    bad = node_ids[:9] + [node_ids[0]]  # repeated node -> degenerate tet10
    d = _write_variant(tmp_path, node_ids, coords, [bad])
    with pytest.raises(ValueError, match="degenerate"):
        load_mesh(d)


def test_rejects_duplicate_node_id(tmp_path):
    node_ids = [1, 1, 3, 4, 5, 6, 7, 8, 9, 10]  # duplicate id 1
    coords = [(float(i), 0.0, 0.0) for i in range(10)]
    d = _write_variant(tmp_path, node_ids, coords, [list(range(1, 11))])
    with pytest.raises(ValueError, match="duplicate node_id"):
        load_mesh(d)


# ---- coincident-node merging ----

def _two_tets_sharing_a_duplicated_face(tmp_path):
    """Two tet10 elements whose shared face uses DISTINCT node ids at identical
    coordinates — i.e. an unmerged interface like the dataset's corner pads.

    Tet A corners: P0,P1,P2,P3 ; Tet B corners: P0',P1',P2',P4 where P0',P1',P2'
    are duplicates (same xyz, different ids) of P0,P1,P2. Until merged the two
    tets share no node, so the mesh is two disconnected components.
    """
    # 4 corners of tet A and the extra apex of tet B
    P = {
        0: (0.0, 0.0, 0.0),
        1: (1.0, 0.0, 0.0),
        2: (0.0, 1.0, 0.0),
        3: (0.0, 0.0, 1.0),   # tet A apex
        4: (0.0, 0.0, -1.0),  # tet B apex (other side of the shared face)
    }

    def tet_nodes(c0, c1, c2, c3):
        corners = [c0, c1, c2, c3]
        # mid-edge midpoints in C3D10 order: (0-1)(1-2)(2-0)(0-3)(1-3)(2-3)
        edges = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
        mids = [tuple(0.5 * (np.array(corners[a]) + np.array(corners[b]))) for a, b in edges]
        return corners + mids

    node_ids, coords = [], []
    nid = 1

    def add(xyz):
        nonlocal nid
        node_ids.append(nid)
        coords.append(xyz)
        nid += 1
        return node_ids[-1]

    # Tet A: corners P0,P1,P2,P3
    a_corner_xyz = tet_nodes(P[0], P[1], P[2], P[3])
    a_ids = [add(xyz) for xyz in a_corner_xyz]
    # Tet B: corners are DUPLICATE coords of P0,P1,P2 plus apex P4 (new ids)
    b_corner_xyz = tet_nodes(P[0], P[1], P[2], P[4])
    b_ids = [add(xyz) for xyz in b_corner_xyz]

    d = _write_variant(tmp_path, node_ids, coords, [a_ids, b_ids])
    return d


def test_merge_ties_coincident_nodes(tmp_path):
    """Default load merges the duplicated shared-face nodes; raw load does not."""
    d = _two_tets_sharing_a_duplicated_face(tmp_path)

    raw = load_mesh(d, merge_coincident=False, normalize_orientation=False)
    assert raw.num_nodes == 20  # two independent tet10, nothing shared

    merged = load_mesh(d, merge_coincident=True, normalize_orientation=False)
    # the shared face has 3 coincident corners + 3 coincident mid-edge nodes = 6
    # duplicate pairs removed -> 20 - 6 = 14 unique nodes.
    assert merged.num_nodes == 14
    # both elements must now reference the SAME rows for the shared face: their
    # connectivity row-sets should intersect in exactly 6 nodes.
    shared = set(merged.connectivity[0].tolist()) & set(merged.connectivity[1].tolist())
    assert len(shared) == 6


def test_merge_makes_mesh_single_component(tmp_path):
    """After merging, the two tets form one connected component (a load path)."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    d = _two_tets_sharing_a_duplicated_face(tmp_path)

    def n_components(mesh):
        n = mesh.num_nodes
        r, c = [], []
        for row in mesh.connectivity:
            row = np.asarray(row)
            for a in row:
                r.append(int(a))
                c.append(int(row[0]))
        A = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
        A = A + A.T
        return connected_components(A, directed=False)[0]

    raw = load_mesh(d, merge_coincident=False, normalize_orientation=False)
    merged = load_mesh(d, merge_coincident=True, normalize_orientation=False)
    assert n_components(raw) == 2      # disconnected before merge
    assert n_components(merged) == 1   # one bonded solid after merge


def test_merge_no_op_when_no_duplicates(tmp_path):
    """A clean mesh with no coincident nodes is returned unchanged by the merge."""
    node_ids = list(range(1, 11))
    coords = [(float(i), 0.0, 0.0) for i in range(10)]
    d = _write_variant(tmp_path, node_ids, coords, [list(range(1, 11))])
    m = load_mesh(d, merge_coincident=True)
    assert m.num_nodes == 10
    assert m.connectivity[0].tolist() == list(range(10))


# ---- real-data integration ----

def test_real_variant_is_mixed_quadratic(dev_variant):
    m = load_mesh(dev_variant)
    hist = m.type_histogram()
    assert m.num_nodes > 100_000
    assert hist["hex20"] > 0 and hist["tet10"] > 0  # mixed mesh, not pure tet10
    # plate is thin in exactly one axis
    lo, hi = m.bbox()
    spans = hi - lo
    assert spans.min() < 0.2  # ~0.06 mm thickness
    assert spans.max() > 5.0  # ~11 mm in-plane


def test_real_variant_is_single_component_after_merge(dev_variant):
    """The bonded spring must be ONE connected component after node merging.

    The raw export splits it into 5 components (central body + 4 corner pads) via
    unmerged coincident interface nodes; merging must reunite them so the FEM
    load path exists. This is the fix that turned the zero-energy mechanism into a
    real K value.
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    def n_components(mesh):
        n = mesh.num_nodes
        r, c = [], []
        for row in mesh.connectivity:
            row = np.asarray(row)
            for a in row:
                r.append(int(a))
                c.append(int(row[0]))
        A = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
        A = A + A.T
        return connected_components(A, directed=False)[0]

    raw = load_mesh(dev_variant, merge_coincident=False)
    merged = load_mesh(dev_variant, merge_coincident=True)
    assert n_components(raw) > 1            # disconnected as exported
    assert n_components(merged) == 1        # single bonded solid after merge
    assert merged.num_nodes < raw.num_nodes  # duplicates were removed
