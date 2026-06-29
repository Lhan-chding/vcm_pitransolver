"""Rigid-body tests: pure translation and infinitesimal rotation give zero strain energy.

These catch B-matrix errors that a constant-strain test can miss: a rigid motion has
zero true strain, so the element must report ~zero energy. A spurious coupling in B
(e.g. a shear row mis-wired) would leak energy here.
"""

from __future__ import annotations

import numpy as np
import pytest

from fem.elements import common
from fem.material import elastic_C
from fem.tests.patch_common import E_MOD, NU, linear_field_at_nodes
from fem.tests.reference_elements import REFERENCES


@pytest.mark.parametrize("name", sorted(REFERENCES))
def test_pure_translation_zero_energy(name):
    module, builder = REFERENCES[name]
    node_xyz = builder()
    C = elastic_C(E_MOD, NU)
    A = np.zeros((3, 3))
    b = np.array([0.012, -0.034, 0.021])
    u_e = linear_field_at_nodes(node_xyz, A, b)
    U = common.element_strain_energy(node_xyz, u_e, C, module.shape_grads, module.GAUSS)
    assert abs(U) < 1e-9, f"{name}: translation gave nonzero energy {U}"


@pytest.mark.parametrize("name", sorted(REFERENCES))
def test_infinitesimal_rotation_zero_energy(name):
    module, builder = REFERENCES[name]
    node_xyz = builder()
    C = elastic_C(E_MOD, NU)
    # skew-symmetric A => infinitesimal rotation => sym(A)=0 => zero strain.
    theta = 1e-4
    A = np.array(
        [
            [0.0, -theta, 0.5 * theta],
            [theta, 0.0, -0.3 * theta],
            [-0.5 * theta, 0.3 * theta, 0.0],
        ]
    )
    u_e = linear_field_at_nodes(node_xyz, A, np.zeros(3))
    U = common.element_strain_energy(node_xyz, u_e, C, module.shape_grads, module.GAUSS)
    assert abs(U) < 1e-12, f"{name}: rotation gave nonzero energy {U}"


@pytest.mark.parametrize("name", sorted(REFERENCES))
def test_stiffness_symmetric_and_psd(name):
    """K_e must be symmetric and positive semi-definite (6 rigid-body zero modes)."""
    module, builder = REFERENCES[name]
    node_xyz = builder()
    C = elastic_C(E_MOD, NU)
    Ke = common.element_stiffness(node_xyz, C, module.shape_grads, module.GAUSS)
    assert np.allclose(Ke, Ke.T, atol=1e-6), f"{name}: K not symmetric"
    evals = np.linalg.eigvalsh(0.5 * (Ke + Ke.T))
    # smallest eigenvalues should be ~0 (rigid modes), none significantly negative
    assert evals.min() > -1e-3 * abs(evals.max()), (
        f"{name}: K has a significantly negative eigenvalue {evals.min()}"
    )
    n_zero = int(np.sum(np.abs(evals) < 1e-6 * abs(evals.max())))
    assert n_zero >= 6, f"{name}: expected >=6 rigid-body modes, got {n_zero}"
