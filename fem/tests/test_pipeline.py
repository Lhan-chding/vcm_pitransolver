"""Pipeline + batch-runner logic: component counting and the trust health gate.

These lock the new batch-pseudolabel logic: count_components must agree with the
mesh's true connectivity, and the health gate must reject the exact failure modes
we care about (multi-component mesh, large energy/reaction gap, past-yield stress)
while accepting a clean solve.
"""

from __future__ import annotations

import numpy as np
import pytest

from parse_mesh import Mesh
from fem.pipeline import count_components


def _two_disjoint_tets() -> Mesh:
    """Two tet10 elements sharing NO node — two components."""
    coords = np.random.RandomState(0).rand(20, 3)
    conn = [np.arange(10, dtype=np.int64), np.arange(10, 20, dtype=np.int64)]
    return Mesh(
        node_ids=np.arange(20, dtype=np.int64),
        coords=coords,
        elem_ids=np.array([0, 1], dtype=np.int64),
        connectivity=conn,
        elem_nnode=np.array([10, 10], dtype=np.int8),
        id_to_row={i: i for i in range(20)},
    )


def _two_linked_tets() -> Mesh:
    """Two tet10 sharing 3 nodes — one component."""
    coords = np.random.RandomState(1).rand(17, 3)
    conn = [
        np.arange(10, dtype=np.int64),
        np.array([0, 1, 2, 10, 11, 12, 13, 14, 15, 16], dtype=np.int64),  # share 0,1,2
    ]
    return Mesh(
        node_ids=np.arange(17, dtype=np.int64),
        coords=coords,
        elem_ids=np.array([0, 1], dtype=np.int64),
        connectivity=conn,
        elem_nnode=np.array([10, 10], dtype=np.int8),
        id_to_row={i: i for i in range(17)},
    )


def test_count_components_disjoint():
    assert count_components(_two_disjoint_tets()) == 2


def test_count_components_linked():
    assert count_components(_two_linked_tets()) == 1


# --- health gate (imported lazily to avoid argparse import cost) ---

def _gate(row):
    from analysis.batch_solve import _gate as gate  # noqa: PLC0415
    return gate(row)


def _ok_row(**over):
    row = dict(status="ok", n_components=1, rel_gap=1e-9, vm_frac_yield=0.002)
    row.update(over)
    return row


def test_gate_accepts_clean_solve():
    assert _gate(_ok_row()) is True


def test_gate_rejects_multicomponent():
    assert _gate(_ok_row(n_components=5)) is False


def test_gate_rejects_large_gap():
    assert _gate(_ok_row(rel_gap=1e-2)) is False


def test_gate_rejects_past_yield():
    assert _gate(_ok_row(vm_frac_yield=1.5)) is False


def test_gate_rejects_errored_variant():
    assert _gate(_ok_row(status="error")) is False
