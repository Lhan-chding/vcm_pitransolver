"""Tests for data/audit_health_300.py — the full-dataset health audit logic.

Locks the audit unit logic on controlled inputs (no zip needed): element-type
tallying (incl. the _num_nodes-must-not-count-as-unknown bug), detJ sign
detection on good vs flipped elements, and node-merge degeneracy / component
accounting. The real-data integration runs separately on the dataset zip.
"""

from __future__ import annotations

import numpy as np
import pytest

import audit_health_300 as audit
from parse_mesh import Mesh


# --- element-type audit -----------------------------------------------------

def test_element_type_ok_clean_mix():
    counts = {10: 100, 15: 5, 20: 200, "_num_nodes": 1234}
    row = audit.audit_element_types("v", counts)
    assert row["num_nodes"] == 1234
    assert row["num_elements"] == 305            # _num_nodes must NOT be summed
    assert row["num_tet10"] == 100
    assert row["num_wedge15"] == 5
    assert row["num_hex20"] == 200
    assert row["num_unknown"] == 0
    assert row["element_type_ok"] is True


def test_element_type_flags_unknown_nodecount():
    counts = {10: 100, 8: 3, "_num_nodes": 50}    # 8-node element is unsupported
    row = audit.audit_element_types("v", counts)
    assert row["num_unknown"] == 3
    assert row["element_type_ok"] is False


def test_element_type_num_nodes_never_counts_as_unknown():
    """Regression: the _num_nodes meta key must not inflate num_unknown."""
    counts = {20: 10, "_num_nodes": 99999}
    row = audit.audit_element_types("v", counts)
    assert row["num_unknown"] == 0
    assert row["element_type_ok"] is True


# --- detJ audit -------------------------------------------------------------

def _single_hex_mesh(coords):
    row = np.arange(20, dtype=np.int64)
    return Mesh(
        node_ids=np.arange(20, dtype=np.int64),
        coords=coords,
        elem_ids=np.array([0], dtype=np.int64),
        connectivity=[row],
        elem_nnode=np.array([20], dtype=np.int8),
        id_to_row={i: i for i in range(20)},
    )


def test_detj_positive_on_good_hex():
    from fem.tests.reference_elements import hex20_reference
    mesh = _single_hex_mesh(hex20_reference())     # unit cube, well-oriented
    rows = audit.audit_detj("v", mesh)
    assert len(rows) == 1
    r = rows[0]
    assert r["element_family"] == "hex20"
    assert r["num_gauss_points_checked"] == 27     # one element, 27-pt rule
    assert r["num_negative_detJ"] == 0
    assert r["num_near_zero_detJ"] == 0
    assert r["detj_ok"] is True
    assert r["min_detJ"] > 0


def test_detj_detects_flipped_element():
    """A reflected cube (negative Jacobian) must be flagged, not passed."""
    from fem.tests.reference_elements import hex20_reference
    coords = hex20_reference().copy()
    coords[:, 0] *= -1.0                            # reflect across x -> flips winding
    mesh = _single_hex_mesh(coords)
    r = audit.audit_detj("v", mesh)[0]
    assert r["num_negative_detJ"] == 27
    assert r["detj_ok"] is False


# --- node-merge audit -------------------------------------------------------

def test_count_components_via_audit_pipeline():
    """count_components (reused by the merge audit) on a 2-component mesh."""
    from fem.pipeline import count_components
    coords = np.random.RandomState(0).rand(20, 3)
    conn = [np.arange(10, dtype=np.int64), np.arange(10, 20, dtype=np.int64)]
    mesh = Mesh(
        node_ids=np.arange(20, dtype=np.int64), coords=coords,
        elem_ids=np.array([0, 1], dtype=np.int64), connectivity=conn,
        elem_nnode=np.array([10, 10], dtype=np.int8),
        id_to_row={i: i for i in range(20)},
    )
    assert count_components(mesh) == 2
