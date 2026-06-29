"""Real-mesh integration: FEM engine must work on actual dataset elements.

The synthetic patch tests prove internal consistency; this proves the assumed C3D
node ordering + orientation normalization match the REAL mesh. The dataset's tet10
elements use the opposite winding from the standard C3D10 reference, so without
orientation normalization in load_mesh these would all have detJ<0.

Skips cleanly if _devdata is absent (CI).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "data"))

DEV_VARIANT = REPO / "_devdata" / "VCM_COMPLEX_0001"


@pytest.fixture
def mesh():
    if not (DEV_VARIANT / "nodes.csv").exists():
        pytest.skip(f"dev data not present at {DEV_VARIANT}")
    from parse_mesh import load_mesh
    return load_mesh(DEV_VARIANT)


def _modules():
    from fem.elements import tet10, hex20, wedge15
    return {10: tet10, 15: wedge15, 20: hex20}


def test_all_real_elements_positive_jacobian(mesh):
    """Every real element (all families) must have detJ>0 after normalization."""
    mods = _modules()
    for nnode, mod in mods.items():
        conn = mesh.elements_of(nnode)
        if conn.shape[0] == 0:
            continue
        # check a strided sample across the whole family (fast but representative)
        step = max(1, conn.shape[0] // 500)
        for e in conn[::step]:
            xyz = mesh.coords[e]
            for nat, _w in mod.GAUSS:
                J = xyz.T @ mod.shape_grads(nat)
                assert np.linalg.det(J) > 0.0, (
                    f"{mod.__name__} element has detJ<=0 after orientation fix"
                )


def test_real_element_constant_strain_patch(mesh):
    """A linear field on real elements must reproduce constant strain (per family)."""
    from fem.elements import common
    from fem.tests.patch_common import (
        A_GRAD, B_OFFSET, voigt_strain_from_gradient, linear_field_at_nodes,
    )
    eps_exp = voigt_strain_from_gradient(A_GRAD)
    mods = _modules()
    for nnode, mod in mods.items():
        conn = mesh.elements_of(nnode)
        if conn.shape[0] == 0:
            continue
        for e in conn[:50]:
            xyz = mesh.coords[e]
            u_e = linear_field_at_nodes(xyz, A_GRAD, B_OFFSET)
            for nat, _w in mod.GAUSS:
                _, _detj, dN_dx = common.jacobian(xyz, mod.shape_grads(nat))
                eps_g = common.B_matrix(dN_dx) @ u_e.reshape(-1)
                assert np.allclose(eps_g, eps_exp, atol=1e-8), (
                    f"{mod.__name__}: real element failed constant-strain patch"
                )


def test_orientation_normalization_changes_tet10(mesh):
    """Sanity: with normalization OFF, real tet10 are reversed (detJ<0); ON fixes them.

    Documents the dataset fact that tet10 winding is opposite to C3D10 reference.
    """
    from parse_mesh import load_mesh
    from fem.elements import tet10
    raw = load_mesh(DEV_VARIANT, normalize_orientation=False)
    conn = raw.elements_of(10)
    if conn.shape[0] == 0:
        pytest.skip("no tet10 in this variant")
    nat = tet10.GAUSS[0][0]
    n_neg = sum(
        np.linalg.det(raw.coords[e].T @ tet10.shape_grads(nat)) < 0
        for e in conn[:100]
    )
    assert n_neg == 100, "expected raw tet10 to be reversed vs C3D10 reference"
