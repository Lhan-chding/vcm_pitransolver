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
